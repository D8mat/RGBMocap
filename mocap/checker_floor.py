"""Procedural checker floor, projected once per frame; no external texture."""
import math
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF
from PySide6.QtGui import QImage,QPainter,QPainterPath,QTransform


def texture(cells):
    image=QImage(cells*8,cells*8,QImage.Format_RGB32);image.fill(Qt.white)
    painter=QPainter(image)
    for z in range(cells):
        for x in range(cells):
            if (x+z)%2==0:painter.fillRect(x*8,z*8,8,8,Qt.black)
    painter.end();return image


def draw(painter,camera,width,height,floor,extent,spacing,image):
    eye,basis=camera
    # Texture coordinates map to app X/Z metres; Y is the existing grid height.
    scale=2*extent/image.width()
    origin=(np.array([-extent,floor,-extent])-eye)@basis.T
    u=np.array([scale,0,0])@basis.T;v=np.array([0,0,scale])@basis.T
    focal=height*.5/math.tan(math.radians(20));cx=width*.5;cy=height*.5
    h=np.array([[focal*u[0]+cx*u[2],focal*v[0]+cx*v[2],focal*origin[0]+cx*origin[2]],
                [-focal*u[1]+cy*u[2],-focal*v[1]+cy*v[2],-focal*origin[1]+cy*origin[2]],
                [u[2],v[2],origin[2]]])
    if abs(np.linalg.det(h))<1e-10:return
    corners=np.array([[-extent,floor,-extent],[extent,floor,-extent],
                      [extent,floor,extent],[-extent,floor,extent]])
    view=(corners-eye)@basis.T;clipped=[]
    for a,b in zip(view,np.roll(view,-1,axis=0)):
        if a[2]>=.08:clipped.append(a)
        if (a[2]>=.08)!=(b[2]>=.08):clipped.append(a+(b-a)*((.08-a[2])/(b[2]-a[2])))
    if len(clipped)<3:return
    p=np.array(clipped);xy=p[:,:2]*focal/p[:,2:];xy[:,0]+=cx;xy[:,1]=cy-xy[:,1]
    path=QPainterPath();path.moveTo(QPointF(*xy[0]))
    for point in xy[1:]:path.lineTo(QPointF(*point))
    path.closeSubpath()
    painter.save();painter.setClipPath(path,Qt.IntersectClip)
    painter.setTransform(QTransform(h[0,0],h[1,0],h[2,0],h[0,1],h[1,1],h[2,1],h[0,2],h[1,2],h[2,2]),True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform,False)
    painter.setOpacity(painter.opacity()*.5)
    painter.drawImage(QRectF(0,0,image.width(),image.height()),image)
    painter.restore()
