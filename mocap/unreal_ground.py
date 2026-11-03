"""Measure actual skinned sole height; never use the pelvis/bone pivot as floor."""
import numpy as np

def sole_height(vertices,joints,weights,world=None):
    vertices=np.asarray(vertices,float);joints=np.asarray(joints,float);weights=np.asarray(weights,float)
    if weights.shape!=(len(vertices),len(joints)) or vertices.shape[1:]!=(3,):raise ValueError('Invalid grounding skin')
    foot=np.sum(weights[:,[7,8,10,11]],axis=1)>.25
    if not np.any(foot):raise ValueError('No weighted foot surface for grounding')
    v=vertices[foot]
    if world is not None:
        world=np.asarray(world,float)
        if world.shape!=(len(joints),4,4):raise ValueError('Invalid grounding pose')
        height=np.zeros(len(v))
        for j in range(len(joints)):
            height+=weights[foot,j]*((v-joints[j])@world[j,1,:3]+world[j,1,3])
    else:height=v[:,1]
    if not np.isfinite(height).all():raise ValueError('Non-finite foot height')
    return float(height.min())

def offsets(vertices,joints,weights,first,tpose=None,extra_cm=0.,enabled=True):
    if not np.isfinite(extra_cm):raise ValueError('Invalid Z offset')
    rest=motion=pose=float(extra_cm)
    if enabled:
        rest-=100*sole_height(vertices,joints,weights)
        motion-=100*sole_height(vertices,joints,weights,first)
        if tpose is not None:pose-=100*sole_height(vertices,joints,weights,tpose)
    return dict(rest=rest,motion=motion,tpose=pose)
