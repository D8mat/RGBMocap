"""Native FlowHMR SMPL-H skin and FK, without refitting the source motion."""
import numpy as np

def rotations6d(value):
    x=np.asarray(value).reshape(*value.shape[:-1],3,2)
    a=x[...,0];a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b=x[...,1]-(a*x[...,1]).sum(-1,keepdims=True)*a
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    r=np.stack([a,b,np.cross(a,b)],axis=-1)
    if not np.isfinite(r).all() or np.max(abs(np.linalg.det(r)-1))>1e-4:raise ValueError('Invalid FlowHMR rotations')
    return r

def prepare(parameters,model,folder,transform):
    import torch
    from .unreal_export import NAMES
    params=torch.load(parameters,map_location='cpu',weights_only=True,mmap=True)
    beta=params['shapes'].detach().numpy().reshape(-1)
    local=rotations6d(params['rot6d'].detach().numpy()[0]);trans=params['trans'].detach().numpy()[0]
    if beta.shape!=(16,) or local.shape[1:]!=(52,3,3) or trans.shape!=(len(local),3):raise ValueError('Invalid FlowHMR motion dimensions')
    with np.load(model,allow_pickle=False) as z:
        shaped=z['v_template']+np.einsum('vci,i->vc',z['shapedirs'],beta)
        joints=z['J_regressor']@shaped;parents=z['kintree_table'][0].astype(int);parents[0]=-1;weights=z['weights'];faces=z['f']
    keep=np.argsort(weights,axis=1)[:,-8:];bounded=np.zeros_like(weights)
    np.put_along_axis(bounded,keep,np.take_along_axis(weights,keep,axis=1),axis=1);weights=bounded/np.maximum(bounded.sum(1,keepdims=True),1e-12)
    world=np.tile(np.eye(4,dtype=np.float32),(len(local),52,1,1))
    for j,p in enumerate(parents):
        world[:,j,:3,:3]=local[:,j];world[:,j,:3,3]=joints[j] if p<0 else joints[j]-joints[p]
        if p>=0:world[:,j]=world[:,p]@world[:,j]
    world[:,:,:3,3]+=trans[:,None,:];world=transform@world
    if not np.isfinite(world).all() or not np.isfinite(shaped).all():raise ValueError('Non-finite FlowHMR export')
    np.save(folder/'motion.npy',world.astype(np.float32))
    np.savez(folder/'skin.npz',vertices=shaped,faces=faces,weights=weights,joints=joints,parents=parents)
    return NAMES[:22]+NAMES[25:],len(world)

def tpose(joints,parents,first):
    from .mixamo_rig import align
    count=len(parents);pose=np.tile(np.eye(4),(count,1,1))
    for j,p in enumerate(parents):
        if p<0:pose[j,:3,3]=joints[j]
        else:
            pose[j,:3,:3]=pose[p,:3,:3]
            pose[j,:3,3]=pose[p,:3,3]+pose[p,:3,:3]@(joints[j]-joints[p])
        if j in (16,17,18,19):
            child={16:18,17:19,18:20,19:21}[j]
            side=1 if joints[j,0]>=joints[0,0] else -1
            pose[j,:3,:3]=align(joints[child]-joints[j],np.array([side,0.,0.]))
    placement=np.eye(4);placement[:3,:3]=first[0,:3,:3]
    placement[:3,3]=first[0,:3,3]-placement[:3,:3]@joints[0]
    return placement@pose
