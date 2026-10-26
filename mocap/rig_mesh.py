"""Canonical skin binding and exact exported-pose frames, worker only.

Rebind the neutral skin instead of rotating the source spine to straighten it.
Large registered model buffers remain shared; the small binding is per actor.
"""
import numpy as np

FINGER_SOURCE=(37,38,39,25,26,27,28,29,30,34,35,36,31,32,33,
               52,53,54,40,41,42,43,44,45,49,50,51,46,47,48)


JOINT_SOURCE=tuple(range(22))+FINGER_SOURCE


def make_binding(body,rig,betas):
    import torch
    from smplx.lbs import batch_rodrigues,batch_rigid_transform
    body=getattr(body,'bm',body)
    source=np.tile(np.eye(3),(22,1,1))
    for j in (1,2,4,5,16,17,18,19):source[j]=rig['alignment'][j].T
    local=source.copy()
    for j,p in enumerate(rig['parents'][:22]):
        if p>=0:local[j]=source[p].T@source[j]
    with torch.no_grad():
        rotations=batch_rodrigues(body.pose_mean.reshape(-1,3)).reshape(1,55,3,3).clone()
        rotations[0,:22]=torch.as_tensor(local,dtype=torch.float32)
        beta=torch.as_tensor(betas,dtype=torch.float32).reshape(-1,10)[:1]
        shapes=torch.cat((beta,torch.zeros(1,10)),dim=-1)
        dirs=torch.cat((body.shapedirs,body.expr_dirs),dim=-1)
        shaped=body.v_template+torch.einsum('bi,vci->bvc',shapes,dirs)
        rest=torch.einsum('jv,bvc->bjc',body.J_regressor,shaped)
        joints,transforms=batch_rigid_transform(rotations,rest,body.parents)
        features=(rotations[:,1:]-torch.eye(3)).reshape(1,-1)
        posed=shaped+(features@body.posedirs).reshape(1,-1,3)
        blend=(body.lbs_weights@transforms.reshape(1,55,16)).reshape(1,-1,4,4)
        vertices=(blend[...,:3,:3]@posed[...,None])[...,0]+blend[...,:3,3]
        bind_world=rotations[0].clone()
        for j,p in enumerate(body.parents):
            if p>=0:bind_world[j]=bind_world[p]@rotations[0,j]
    joint=joints[0].numpy();canonical=joint.copy()
    canonical[list(JOINT_SOURCE)]=rig['rest']
    for j in (22,23,24):canonical[j]=canonical[15]+joint[j]-joint[15]
    vertices=vertices[0].numpy()+body.lbs_weights.numpy()@(canonical-joint)
    return dict(vertices=vertices.astype(np.float32),frames=bind_world.numpy(),local=rotations[0].numpy(),
        rest=canonical.astype(np.float32),parents=body.parents.numpy().copy(),betas=beta[0].numpy().copy())


def mesh_frames(body,rig,points,world,betas,binding=None):
    from .mixamo_rig import points_from_world
    binding=make_binding(body,rig,betas) if binding is None else binding
    beta=np.asarray(betas)
    if not np.allclose(beta,binding['betas'],rtol=0,atol=1e-5):
        raise ValueError('Canonical preview requires a consistent actor shape across frames.')
    points=np.asarray(points);world=np.asarray(world);n=len(points)
    error=float(np.linalg.norm(points_from_world(rig,points[:,0],world)-points,axis=-1).max())
    positions=np.empty((n,55,3));rotations=np.empty((n,55,3,3))
    positions[:,list(JOINT_SOURCE)]=points;rotations[:,list(JOINT_SOURCE)]=world
    for j in (22,23,24):
        rotations[:,j]=world[:,15]
        positions[:,j]=points[:,15]+np.einsum('tij,j->ti',world[:,15],binding['rest'][j]-binding['rest'][15])
    source=rotations@binding['frames'];local=source.copy()
    for j,p in enumerate(binding['parents']):
        if p>=0:local[:,j]=source[:,p].transpose(0,2,1)@source[:,j]
    transforms=np.tile(np.eye(4),(n,55,1,1));transforms[...,:3,:3]=rotations
    transforms[...,:3,3]=positions-np.einsum('tjik,jk->tji',rotations,binding['rest'])
    features=(local[:,1:]-binding['local'][1:]).reshape(n,486)
    frames=np.concatenate((transforms.reshape(n,880),features,np.zeros((n,23))),axis=1).astype(np.float32)
    if not np.isfinite(frames).all() or error>.00025:
        raise ValueError(f'Export mesh / bone alignment failed ({error*1000:.3f} mm). Preview not committed.')
    return frames,error


def reference_vertices(packed,frame,binding,count):
    """Independent CPU counterpart for diagnostic GPU comparisons only."""
    packed=np.asarray(packed);n=count*3
    pose=packed[n*21:n*507].reshape(486,n)
    weights=packed[n*507:].reshape(count,55)
    delta=(np.asarray(frame)[880:1366]@pose).reshape(count,3)
    transport=(weights@binding['frames'].reshape(55,9)).reshape(count,3,3)
    vertices=binding['vertices']+np.einsum('vij,vj->vi',transport,delta)
    transforms=(weights@np.asarray(frame)[:880].reshape(55,16)).reshape(count,4,4)
    return np.einsum('vij,vj->vi',transforms[:,:3,:3],vertices)+transforms[:,:3,3]


class BoundSoleGeometry:
    """Exact canonical skin over all foot-dominant vertices, worker only."""
    def __init__(self,body,binding):
        from .correction import SoleGeometry
        body=getattr(body,'bm',body);self.body=body;self.binding=binding
        geometry=SoleGeometry(body,binding['betas'])
        self.groups=geometry.groups;self.mean_groups=geometry.mean_groups
        self.ids=np.concatenate(self.groups)
        self.weights=body.lbs_weights.numpy()[self.ids]
        self.posedirs=body.posedirs.numpy().reshape(486,-1,3)[:,self.ids].reshape(486,-1)
        self.transport=(self.weights@binding['frames'].reshape(55,9)).reshape(-1,3,3)

    def sample(self,rig,positions,world):
        frames,_=mesh_frames(self.body,rig,positions,world,self.binding['betas'],self.binding)
        delta=(frames[:,880:1366]@self.posedirs).reshape(len(frames),-1,3)
        posed=self.binding['vertices'][self.ids]+np.einsum('vij,tvj->tvi',self.transport,delta)
        transforms=(self.weights@frames[:,:880].reshape(len(frames),55,16)).reshape(len(frames),-1,4,4)
        vertices=np.einsum('tvij,tvj->tvi',transforms[...,:3,:3],posed)+transforms[...,:3,3]
        probes=[];offset=0
        for ids,mean_ids in zip(self.groups,self.mean_groups):
            patch=vertices[:,offset:offset+len(ids)];offset+=len(ids)
            value=patch[:,mean_ids].mean(1);value[:,1]=patch[...,1].min(1);probes.append(value)
        return np.stack(probes,1),vertices
