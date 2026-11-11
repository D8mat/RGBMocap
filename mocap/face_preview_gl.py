"""OpenGL morph preview. Geometry stays resident; playback uploads 52 floats."""
import math
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLShader
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from .mesh_gl import MeshRenderer

MORPH='''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model {float model[];};
struct Vertex {vec4 position;vec4 normal;};
layout(std430,binding=1) buffer Output {Vertex vertices[];};
layout(std430,binding=4) readonly buffer Frame {float pose[];};
uniform int vertexCount;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(vertexCount))return;
int stride=vertexCount*3;vec3 p=vec3(model[3*i],model[3*i+1],model[3*i+2]);
for(int k=0;k<52;k++){int o=(k+1)*stride+int(3*i);p+=pose[k]*vec3(model[o],model[o+1],model[o+2]);}
vertices[i].position=vec4(p,1);}
'''
VERTEX='''#version 430 core
layout(location=0) in vec3 position;layout(location=1) in vec3 normal;layout(location=2) in vec3 diffuse;
uniform mat4 mvp;uniform mat4 sceneTransform;out vec3 n;out vec3 c;
void main(){gl_Position=mvp*vec4(position,1);n=normal;c=diffuse;}
'''
FRAGMENT='''#version 430 core
in vec3 n;in vec3 c;out vec4 color;
void main(){float light=.42+.48*max(0,dot(normalize(n),normalize(vec3(-.4,-.8,.7))));
color=vec4(c*light,1);}
'''


class FaceRenderer(MeshRenderer):
    def __init__(self):
        super().__init__()
        self.skin.removeAllShaders();self.surface.removeAllShaders()
        self.skin=self.program(((QOpenGLShader.Compute,MORPH),))
        self.surface=self.program(((QOpenGLShader.Vertex,VERTEX),(QOpenGLShader.Fragment,FRAGMENT)))

    def load(self, asset):
        if self.asset_key==asset['key']:return
        super().load(asset)
        self.vao.bind();self.surface.bind()
        self.colors=self.buffer(asset['colors'])
        self.surface.enableAttributeArray(2);self.surface.setAttributeBuffer(2,0x1406,0,3,12)
        self.vao.release();self.surface.release()


class FaceView(QOpenGLWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        fmt=QSurfaceFormat();fmt.setVersion(4,3);fmt.setProfile(QSurfaceFormat.CoreProfile);fmt.setDepthBufferSize(24)
        self.setFormat(fmt);self.setMinimumSize(260,280)
        self.asset=None;self.renderer=None;self.values=np.zeros(52,np.float32);self.revision=0
        self.yaw=0.;self.pitch=.05;self.distance=1.8;self.anchor=None;self.error=''
        self.setToolTip('Drag to orbit. Wheel to zoom. Double-click to reset view.')

    def initializeGL(self):
        try:
            self.renderer=FaceRenderer()
            self.context().aboutToBeDestroyed.connect(self.release)
        except Exception as error:self.error=str(error)

    def set_asset(self,asset):
        self.asset=asset;self.revision+=1;self.update()

    def set_values(self,values):
        if np.array_equal(values,self.values):return
        self.values=np.array(values,np.float32,copy=True);self.revision+=1;self.update()

    def paintGL(self):
        if self.renderer is None:return
        gl=self.renderer.gl;gl.glClearColor(.075,.085,.10,1);gl.glClear(0x4000|0x100)
        if self.asset is None:return
        try:
            self.renderer.load(self.asset);self.renderer.deform(self.values,self.revision)
            direction=np.array([math.sin(self.yaw)*math.cos(self.pitch),-math.cos(self.yaw)*math.cos(self.pitch),math.sin(self.pitch)])
            eye=direction*self.distance*max(1.,.85*self.height()/max(1,self.width()));forward=-direction
            right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
            self.renderer.draw(eye,np.stack((right,up,forward)),self.width(),self.height())
        except Exception as error:self.error=str(error)

    def mousePressEvent(self,event):self.anchor=event.position()
    def mouseMoveEvent(self,event):
        if self.anchor is not None:
            delta=event.position()-self.anchor;self.anchor=event.position()
            self.yaw-=delta.x()*.01;self.pitch=max(-1.4,min(1.4,self.pitch+delta.y()*.01));self.update()
    def mouseReleaseEvent(self,event):self.anchor=None
    def wheelEvent(self,event):self.distance=max(.6,min(5,self.distance*math.exp(-event.angleDelta().y()*.001)));self.update()
    def mouseDoubleClickEvent(self,event):self.yaw=0.;self.pitch=.05;self.distance=1.8;self.update()
    def release(self):
        if self.renderer is not None:
            self.makeCurrent();self.renderer.close();self.renderer=None;self.doneCurrent()
