"""Read-only calibrated MAMMA overlays. One bounded GPU renderer, no inference."""
import bisect
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPen, QMatrix4x4, QOpenGLContext, QOffscreenSurface, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat, QOpenGLFunctions_4_3_Core
from .camera_tracks import at


def projection(k, size, matrix, near=.03, far=2000.):
    """CV (+Z forward, +Y down) to GL clip coordinates; retains lens shift/skew."""
    k=np.asarray(k,float);w,h=size
    p=np.zeros((4,4))
    p[0,:3]=2*k[0]/w;p[0,2]-=1
    p[1,:3]=-2*k[1]/h;p[1,2]+=1
    p[2,2]=(far+near)/(far-near);p[2,3]=-2*far*near/(far-near)
    p[3,2]=1
    return p@matrix


def scene_samples(view,seconds,index,capture_space=False):
    """Pure lookup: secondary video time must not mutate the main scene's windows."""
    if view.people_mode:
        scene=view.people_scene
        if scene is None:return []
        preferred=scene.by_project.get(view.path.parent) if view.path else None
        result=[]
        for ident in sorted(view.scene_visible):
            take=preferred if preferred is not None and preferred.ident==ident else None
            frame=take.frame_at(seconds) if take is not None else None
            if frame is None:
                slot=bisect.bisect_right(scene.starts.get(ident,()),seconds+1e-7)-1
                if slot<0:continue
                first,end,take=scene.spans[ident][slot]
                if not first-1e-7<=seconds<end-1e-7:continue
                frame=take.frame_at(seconds)
            if frame is not None:
                result.append((str(take.project),take.arrays,take.asset,take.raw_asset,
                    take.target_asset,take.capture_transform if capture_space else take.transform,frame,
                    view.scene_colors.get(ident,'#b5b5b5'),take.native_required))
        return result
    if not view.single_actor_visible or view.joints is None or not 0<=index<len(view.joints):return []
    arrays=dict(mesh=view.mesh_frames,native=view.native_vertices,rig=view.target_frames)
    return [(str(view.path),arrays,view.mesh_asset,view.raw_asset,view.target_asset,
             np.eye(4) if capture_space else view.single_transform,index,view.body_color.name(),view.native_required)]


class OverlayRenderer:
    """One context/FBO reused for all cameras; at most eight actor buffers."""
    def __init__(self):
        fmt=QSurfaceFormat();fmt.setVersion(4,3);fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24);fmt.setAlphaBufferSize(8)
        self.surface=QOffscreenSurface();self.surface.setFormat(fmt);self.surface.create()
        self.context=QOpenGLContext();self.context.setFormat(fmt)
        if not self.context.create():raise RuntimeError('Mesh overlay requires OpenGL 4.3')
        self.renderers={};self.fbo=None;self.size=None

    def render(self, camera, matrix, samples, size):
        old=QOpenGLContext.currentContext();old_surface=old.surface() if old else None
        if not self.context.makeCurrent(self.surface):raise RuntimeError('Cannot activate mesh overlay renderer')
        try:
            if self.size!=size:
                self.fbo=None
                fmt=QOpenGLFramebufferObjectFormat();fmt.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
                self.fbo=QOpenGLFramebufferObject(*size,fmt);self.size=size
                if not self.fbo.isValid():raise RuntimeError('Cannot allocate mesh overlay')
            self.fbo.bind()
            gl=QOpenGLFunctions_4_3_Core();gl.initializeOpenGLFunctions()
            gl.glViewport(0,0,*size);gl.glDisable(0x0BE2);gl.glDisable(0x0C11)
            gl.glClearColor(0,0,0,0);gl.glClear(0x4000|0x100)
            mvp=QMatrix4x4(*projection(camera['K'],camera['size'],matrix).reshape(-1).tolist())
            used=set()
            from .mesh_gl import MeshRenderer
            from .rig_gl import RigRenderer
            for slot,arrays,asset,raw,rig,transform,index,color,native_required in samples[:8]:
                native=arrays.get('native')
                if native is not None or native_required:
                    if native is None or raw is None:raise ValueError('Prepare the selected body mesh preview first')
                    asset=raw['raw_asset'];frames=native;kind='native';cls=RigRenderer
                else:
                    frames=arrays.get('mesh');kind='skin';cls=MeshRenderer
                if asset is None or frames is None:raise ValueError('Prepare the selected body mesh preview first')
                key=(slot,kind);used.add(key)
                if key not in self.renderers:self.renderers[key]=cls()
                renderer=self.renderers[key];renderer.load(asset)
                renderer.deform(frames[index],(id(frames),asset['key'],index))
                c=QColor(color)
                renderer.draw(np.zeros(3),np.eye(3),*size,(c.redF(),c.greenF(),c.blueF()),
                              transform=transform,clear_depth=False,mvp=mvp)
            for key in list(self.renderers):
                if key not in used:self.renderers.pop(key).close()
            image=self.fbo.toImage();self.fbo.release()
            return image
        finally:
            self.context.doneCurrent()
            if old is not None:old.makeCurrent(old_surface)

    def close(self):
        old=QOpenGLContext.currentContext();surface=old.surface() if old else None
        if self.context.makeCurrent(self.surface):
            for renderer in self.renderers.values():renderer.close()
            self.renderers.clear();self.fbo=None;self.context.doneCurrent()
        self.surface.destroy()
        if old is not None:old.makeCurrent(surface)


class MammaOverlay:
    def __init__(self,panel):
        self.panel=panel;self.renderer=None;self.cache={};self.error=''

    def clear(self):
        self.cache.clear();self.error=''
        if self.renderer is not None:self.renderer.close();self.renderer=None

    def paint(self,video,painter,rect,camera_id):
        w=self.panel.owner
        if w.model_choice.currentData()!='mamma' or self.panel.mode.currentData()=='video':return
        if video.frame is None or video.external_mode:return
        if self.panel.opacity.value()==0:return
        tools=getattr(w,'camera_tools',None)
        message=''
        try:
            if getattr(w,'busy',False):raise ValueError('Overlay paused during processing')
            if tools is None or not tools.data or tools.data['model']!='mamma':
                raise ValueError(getattr(tools,'message','') or 'Prepare the MAMMA preview to load calibrated cameras')
            camera=next((c for c in tools.data['cameras'] if c['name']==camera_id),None)
            if camera is None:raise ValueError('No saved calibration for this camera')
            if tools.data.get('warning'):raise ValueError(tools.data['warning'])
            view=w.motion_view
            if w.project is None or view.path is None or view.path.parent!=w.project:
                raise ValueError('Loading selected motion preview')
            seconds=w.source_offset+video.index/w.info['fps']
            matrix=at(camera,seconds,0)
            if matrix is None:raise ValueError('Outside saved camera timing')
            samples=scene_samples(view,seconds,video.index,capture_space=True)
            if not samples:raise ValueError('No checked person has motion at this time')
            painter.save();painter.setClipRect(rect)
            try:
                painter.setOpacity(self.panel.opacity.value()/100.)
                if self.panel.mode.currentData()=='mesh':
                    # Bounded readback resolution; full topology stays on the GPU.
                    scale=min(1.,640/max(camera['size']))
                    size=tuple(max(1,round(v*scale)) for v in camera['size'])
                    stamp=tuple((s[0],id(s[1].get('mesh')),id(s[1].get('native')),s[6],s[7],s[5].tobytes()) for s in samples)
                    key=(id(tools.data),camera_id,size,stamp)
                    cached=self.cache.get(id(video))
                    if cached is None or cached[0]!=key:
                        if self.renderer is None:self.renderer=OverlayRenderer()
                        image=self.renderer.render(camera,matrix,samples,size)
                        self.cache[id(video)]=(key,image)
                    else:image=cached[1]
                    painter.drawImage(rect,image)
                else:
                    for _,arrays,asset,raw,rig,transform,index,color,_ in samples:
                        frames=arrays.get('rig')
                        if frames is None or rig is None:raise ValueError('Prepare skeleton preview first')
                        points=np.asarray(frames[index])[:,:3,3]
                        points=points@transform[:3,:3].T+transform[:3,3]
                        cam=points@matrix[:3,:3].T+matrix[:3,3]
                        painter.setPen(QPen(QColor(color),2,Qt.SolidLine,Qt.RoundCap))
                        for j,parent in enumerate(rig['parents']):
                            if parent<0:continue
                            a,b=cam[parent].copy(),cam[j].copy()
                            if not np.isfinite([a,b]).all() or max(a[2],b[2])<=.03:continue
                            if a[2]<.03:a+=(b-a)*((.03-a[2])/(b[2]-a[2]))
                            if b[2]<.03:b+=(a-b)*((.03-b[2])/(a[2]-b[2]))
                            q=np.array([a,b])@camera['K'].T;q=q[:,:2]/q[:,2:]
                            q[:,0]=rect.left()+q[:,0]*rect.width()/camera['size'][0]
                            q[:,1]=rect.top()+q[:,1]*rect.height()/camera['size'][1]
                            painter.drawLine(QPointF(*q[0]),QPointF(*q[1]))
            finally:painter.restore()
        except Exception as exc:
            message=str(exc)
        if message:
            painter.save();painter.setPen(QColor('#ffd397'))
            painter.drawText(rect.adjusted(8,8,-8,-8),Qt.AlignTop|Qt.TextWordWrap,message)
            painter.restore()
