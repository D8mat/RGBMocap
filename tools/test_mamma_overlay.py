"""Projection, GPU occlusion, four decoders and timing regression tests.

Run with the installed GUI runtime activated (real OpenGL 4.3 required).
"""
import sys,time,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication,QWidget,QComboBox,QSpinBox
from PySide6.QtGui import QImage,QPainter,QColor
from PySide6.QtCore import QRectF
from mocap.mamma_overlay import projection,OverlayRenderer,scene_samples,MammaOverlay
from mocap.camera_tracks import record
from mocap.mesh_data import adjacency
from mocap.mamma_video_grid import MammaVideoGrid
from mocap.viewer import VideoView

app=QApplication.instance() or QApplication([])
k=np.array([[200.,3.,107.],[0.,190.,95.],[0,0,1.]])
m=np.eye(4);m[:3,3]=[.1,.2,.3]
points=np.array([[0.,0.,2.,1.],[.4,-.2,3.,1.]])
clip=points@projection(k,(240,180),m).T
ndc=clip[:,:2]/clip[:,3:]
pixels=np.c_[(ndc[:,0]+1)*120,(1-ndc[:,1])*90]
camera_points=points@m.T;q=camera_points[:,:3]@k.T
np.testing.assert_allclose(pixels,q[:,:2]/q[:,2:])
print('PASS shifted/skewed intrinsics and extrinsics')

faces=np.array([[0,1,2]],np.uint32);offsets,neighbors=adjacency(faces,3)
asset=dict(key='triangle',mode='raw',vertex_count=3,faces=faces,offsets=offsets,neighbors=neighbors)
def sample(name,z,color):
    vertices=np.array([[[-.7,-.7,z],[.7,-.7,z],[0,.7,z]]],np.float32)
    return (name,dict(native=vertices),None,dict(raw_asset=asset),None,np.eye(4),0,color,True)
renderer=OverlayRenderer();camera=record('one',[[100,0,100],[0,100,100],[0,0,1]],(200,200),np.eye(4),0,30,'test')
front=sample('front',2,'#ff0000');back=sample('back',3,'#0000ff')
for samples in [[front,back],[back,front]]:
    image=renderer.render(camera,np.eye(4),samples,(200,200))
    center=image.pixelColor(100,100);assert center.red()>center.blue()+40,center.name()
    assert image.pixelColor(5,5).alpha()==0
renderer.render(camera,np.eye(4),[front],(300,200))
assert len(renderer.renderers)==1
renderer.close();print('PASS GPU mesh projection, transparency, depth occlusion, resize and cleanup')

class Take:
    ident=1;project=Path('test');arrays={};asset=None;raw_asset=None;target_asset=None
    transform=np.eye(4);capture_transform=np.eye(4);native_required=False
    def frame_at(self,t):return int((t-2)*30+1e-6) if 2<=t<3 else None
take=Take();scene=SimpleNamespace(by_project={},starts={1:[2]},spans={1:[(2,3,take)]})
view=SimpleNamespace(people_mode=True,people_scene=scene,path=None,scene_visible={1},scene_colors={1:'#ff0000'})
assert scene_samples(view,1.9,0)==[]
assert scene_samples(view,2.5,0)[0][6]==15
assert scene_samples(view,3,0)==[]
take.transform=np.eye(4);take.transform[:3,3]=[3,0,5.5]
np.testing.assert_array_equal(scene_samples(view,2.5,0,True)[0][5],np.eye(4))
np.testing.assert_array_equal(scene_samples(view,2.5,0)[0][5],take.transform)
print('PASS take offsets and half-open timing')

# Real paint path: skeleton opacity, no calibration fallback, and frame boundaries.
rig=np.broadcast_to(np.eye(4),(2,2,4,4)).copy()
rig[:,:,2,3]=2;rig[:,0,0,3]=-.5;rig[:,1,0,3]=.5
motion=SimpleNamespace(people_mode=False,single_actor_visible=True,joints=np.zeros((2,2,3)),
 path=Path('test/motion.json'),mesh_frames=None,mesh_asset=None,native_vertices=None,raw_asset=None,
 target_frames=rig,target_asset={'parents':[-1,0]},single_transform=np.eye(4),body_color=QColor('red'),native_required=False)
mode=QComboBox();mode.addItem('Skeleton','skeleton');opacity=QSpinBox();opacity.setRange(0,100);opacity.setValue(50)
choice=QComboBox();choice.addItem('MAMMA','mamma')
tools=SimpleNamespace(data={'model':'mamma','cameras':[camera]},message='')
owner=SimpleNamespace(model_choice=choice,camera_tools=tools,motion_view=motion,project=Path('test'),source_offset=0,info={'fps':30},busy=False)
overlay=MammaOverlay(SimpleNamespace(owner=owner,mode=mode,opacity=opacity))
video=SimpleNamespace(frame=QImage(200,200,QImage.Format_RGB32),external_mode=False,index=0)
def paint():
 image=QImage(200,200,QImage.Format_ARGB32);image.fill(QColor('black'))
 painter=QPainter(image);overlay.paint(video,painter,QRectF(0,0,200,200),'one');painter.end();return image
pixel=paint().pixelColor(100,100);assert 100<pixel.red()<150 and pixel.green()==0
opacity.setValue(0);assert paint().pixelColor(100,100).red()==0
opacity.setValue(100);video.index=2;assert paint().pixelColor(100,100).red()==0
video.index=0;tools.data=None;assert paint().pixelColor(100,100).red()==0
overlay.clear();print('PASS skeleton compositing, opacity and absent/stale frame guards')

def wait(pred):
    end=time.monotonic()+12
    while time.monotonic()<end:
        app.processEvents()
        if pred():return
        time.sleep(.01)
    raise AssertionError('Decoder timeout')

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);paths=[]
    for i,fps in enumerate([30,60,24,30]):
        p=root/f'cam{i}.avi';out=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),fps,(96,64))
        for f in range(fps*2):out.write(np.full((64,96,3),f,np.uint8))
        out.release();paths.append(p)
    owner=QWidget();owner.model_choice=QComboBox();owner.model_choice.addItem('MAMMA','mamma')
    owner.info={'fps':30};owner.project=None;owner.preview_data=None;owner.source_offset=0
    owner.mamma_inputs=[dict(id=f'cam{i}',path=str(p)) for i,p in enumerate(paths)]
    owner.mamma_active_camera=0
    primary=VideoView();grid=MammaVideoGrid(owner,primary);grid.resize(900,700);grid.show()
    grid.count.setCurrentIndex(3);primary.open(paths[0],timeline_fps=30)
    wait(lambda:all(v.native_fps for v in grid.all_videos))
    primary.seek(15)
    wait(lambda:all(v.index==15 for v in grid.all_videos))
    assert [v.requested_index for v in grid.all_videos]==[15,30,12,15]
    grid.extra_choices[0].setCurrentIndex(1)
    wait(lambda:grid.extra[0].native_fps==60 and grid.extra[0].index==15)
    assert grid.camera_id(2)=='cam1'
    grid.count.setCurrentIndex(0);app.processEvents()
    assert all(v.decoder is None for v in grid.all_videos[1:])
    grid.count.setCurrentIndex(3);wait(lambda:all(v.native_fps for v in grid.all_videos))
    owner.model_choice.addItem('ExMR','exmr');owner.model_choice.setCurrentIndex(1);grid.refresh()
    assert all(v.decoder is None for v in grid.all_videos[1:])
    grid.shutdown();primary.shutdown();grid.close();owner.close()
print('PASS four camera decoding, mixed FPS, camera switching, model switching and shutdown')
