"""CPU/PyTorch-only constructor helpers, avoiding unused native rasterization.

Uniform graph Laplacian follows the documented PyTorch3D definition. OBJ reader
supports triangle/polygon indices used by the supplied FLAME template only.
These helpers do not change the PEAR network or EHM forward skinning equations.
"""
from types import SimpleNamespace
import torch
class Meshes:
    def __init__(self,verts,faces):
        if len(verts)!=1 or len(faces)!=1:raise ValueError('Single mesh constructor helper only')
        self.verts=verts[0];self.faces=faces[0]
    def laplacian_packed(self):
        f=self.faces;n=len(self.verts)
        edges=torch.cat([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],0)
        edges=torch.unique(torch.sort(edges,1).values,dim=0)
        both=torch.cat([edges,edges.flip(1)],0).T
        deg=torch.bincount(both[0],minlength=n).float().clamp(min=1)
        idx=torch.arange(n,device=f.device)
        inds=torch.cat([both,torch.stack([idx,idx])],1)
        vals=torch.cat([1/deg[both[0]],-torch.ones(n,device=f.device)])
        return torch.sparse_coo_tensor(inds,vals,(n,n)).coalesce()
def load_obj(path,load_textures=False):
    verts=[];uv=[];faces=[];tex=[]
    for line in open(path,encoding='utf8'):
        a=line.split()
        if not a:continue
        if a[0]=='v':verts.append([float(v) for v in a[1:4]])
        elif a[0]=='vt':uv.append([float(v) for v in a[1:3]])
        elif a[0]=='f':
            poly=[v.split('/') for v in a[1:]]
            for j in range(1,len(poly)-1):
                t=[poly[0],poly[j],poly[j+1]]
                faces.append([int(v[0])-1 if int(v[0])>0 else len(verts)+int(v[0]) for v in t])
                tex.append([int(v[1])-1 if len(v)>1 and v[1] else -1 for v in t])
    return torch.tensor(verts,dtype=torch.float32),SimpleNamespace(verts_idx=torch.tensor(faces),textures_idx=torch.tensor(tex)),SimpleNamespace(verts_uvs=torch.tensor(uv,dtype=torch.float32))
