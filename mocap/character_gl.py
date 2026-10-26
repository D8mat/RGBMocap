"""GPU skinning for imported characters; only a compact bone palette changes per frame."""
import numpy as np
from PySide6.QtOpenGL import QOpenGLShader,QOpenGLBuffer
from .mesh_gl import MeshRenderer

SKIN='''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float pose[]; };
uniform int vertexCount;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=uint(vertexCount))return;
 vec4 v=vec4(model[3*i],model[3*i+1],model[3*i+2],1);
 int r=3*vertexCount+2*int(i);int a=int(model[r]);int n=int(model[r+1]);
 vec3 p=vec3(0);
 for(int k=0;k<n;k++){
  int o=5*vertexCount+2*(a+k);int b=16*int(model[o]);float w=model[o+1];
  p+=w*vec3(dot(vec4(pose[b],pose[b+1],pose[b+2],pose[b+3]),v),
  dot(vec4(pose[b+4],pose[b+5],pose[b+6],pose[b+7]),v),
  dot(vec4(pose[b+8],pose[b+9],pose[b+10],pose[b+11]),v));
 }
 vertices[i].position=vec4(p,1);
}'''
VERTEX='''#version 430 core
layout(location=0) in vec3 position;layout(location=1) in vec3 normal;layout(location=2) in vec3 diffuse;
uniform mat4 mvp;uniform mat4 sceneTransform;out vec3 n;out vec3 c;
void main(){gl_Position=mvp*sceneTransform*vec4(position,1);n=mat3(sceneTransform)*normal;c=diffuse;}'''
FRAGMENT='''#version 430 core
in vec3 n;in vec3 c;out vec4 color;
void main(){vec3 normal=normalize(n);float light=.60+.35*max(0,dot(normal,normalize(vec3(-.4,.8,-.6))));color=vec4(c*light,1);}'''

class CharacterRenderer(MeshRenderer):
    def __init__(self):
        super().__init__()
        self.skin.removeAllShaders();self.surface.removeAllShaders()
        self.skin=self.program(((QOpenGLShader.Compute,SKIN),))
        self.surface=self.program(((QOpenGLShader.Vertex,VERTEX),(QOpenGLShader.Fragment,FRAGMENT)))

    def load(self,asset,palette):
        if self.asset_key==asset['key']:return
        self.release_buffers();self.vertex_count=len(asset['vertices']);self.index_count=asset['faces'].size
        self.vao.bind()
        self.model=self.buffer(np.concatenate([asset['vertices'].ravel(),asset['rows'].ravel(),asset['weights'].ravel()]).astype('float32'))
        self.vertices=self.buffer(self.vertex_count*32)
        self.offsets=self.buffer(asset['offsets']);self.neighbors=self.buffer(asset['neighbors'])
        self.frame=self.buffer(palette*16*4)
        self.indices=self.buffer(asset['faces'],QOpenGLBuffer.IndexBuffer);self.indices.bind()
        self.surface.bind();self.vertices.bind()
        for i,offset in [(0,0),(1,16)]:
            self.surface.enableAttributeArray(i);self.surface.setAttributeBuffer(i,0x1406,offset,3,32)
        self.colors=self.buffer(np.asarray(asset['colors'],np.float32))
        self.surface.enableAttributeArray(2);self.surface.setAttributeBuffer(2,0x1406,0,3,12)
        self.vao.release();self.surface.release();self.asset_key=asset['key'];self.frame_key=None
        self.check('character load')
