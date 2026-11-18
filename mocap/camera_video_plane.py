"""Video reference planes sharing existing QImages, with no decoder or file I/O."""
import math
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF
from PySide6.QtGui import QPainter,QPainterPath,QTransform


def corners(camera,matrix,depth=.75):
    w,h=camera['size']
    pixels=np.array([[0.,0.,1.],[w,0.,1.],[w,h,1.],[0.,h,1.]])
    local=pixels@np.linalg.inv(camera['K']).T
    local*=depth/local[:,2,None]
    return (local-matrix[:3,3])@matrix[:3,:3]


def draw(view,painter,frames):
    eye,basis=view._paint_camera
    # Reference cards, drawn far-to-near behind the character overlays.
    planes=[(camera,image,(corners(camera,matrix)-eye)@basis.T) for camera,matrix,image in frames]
    planes.sort(key=lambda row:float(row[2][:,2].mean()),reverse=True)
    focal=view.height()*.5/math.tan(math.radians(20))
    cx,cy=view.width()*.5,view.height()*.5
    for camera,image,p in planes:
        if image.isNull():continue
        origin=p[0];u=(p[1]-p[0])/image.width();v=(p[3]-p[0])/image.height()
        if view.orthographic:
            k=focal/view.distance
            h=np.array([[k*u[0],k*v[0],k*origin[0]+cx],[-k*u[1],-k*v[1],-k*origin[1]+cy],[0.,0.,1.]])
        else:
            h=np.array([[focal*u[0]+cx*u[2],focal*v[0]+cx*v[2],focal*origin[0]+cx*origin[2]],
                        [-focal*u[1]+cy*u[2],-focal*v[1]+cy*v[2],-focal*origin[1]+cy*origin[2]],
                        [u[2],v[2],origin[2]]])
        if abs(np.linalg.det(h))<1e-10:continue
        clipped=[]
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            if a[2]>=.08:clipped.append(a)
            if (a[2]>=.08)!=(b[2]>=.08):clipped.append(a+(b-a)*((.08-a[2])/(b[2]-a[2])))
        if len(clipped)<3:continue
        points=np.array(clipped)
        xy=points[:,:2]*focal/(view.distance if view.orthographic else points[:,2:])
        xy[:,0]+=cx;xy[:,1]=cy-xy[:,1]
        path=QPainterPath();path.moveTo(QPointF(*xy[0]))
        for point in xy[1:]:path.lineTo(QPointF(*point))
        path.closeSubpath()
        painter.save()
        try:
            painter.setClipPath(path,Qt.IntersectClip)
            painter.setTransform(QTransform(h[0,0],h[1,0],h[2,0],h[0,1],h[1,1],h[2,1],h[0,2],h[1,2],h[2,2]),True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform,False)
            painter.drawImage(QRectF(0,0,image.width(),image.height()),image)
        finally:painter.restore()
