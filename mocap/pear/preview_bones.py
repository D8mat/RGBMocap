"""Shared bone appearance over native PEAR landmarks; display only."""
import numpy as np
from ..bone_overlay import geometry,reference_frame
from .bone_roll import SkeletonLayout


class PreviewBones(SkeletonLayout):
    def __init__(self,joints,parents,key=('pear-live',)):
        super().__init__(parents)
        p,ends=self.landmarks(joints)
        self.asset=geometry(p,self.parents,ends,key)
        self.bind_delta=np.asarray([s[2] for s in self.asset['segments']])
        self.bind_length=np.linalg.norm(self.bind_delta,axis=1)
        self.bind_inverse=np.asarray([reference_frame(d).T for d in self.bind_delta])

    def frames(self,joints,facing):
        p,ends=self.landmarks(joints)
        length=np.linalg.norm(self.directions(p,ends),axis=1)
        result=np.tile(np.eye(4,dtype=np.float32),(52,1,1))
        result[:,:3,:3]=(facing@self.bind_inverse)*(length/self.bind_length)[:,None,None]
        result[:,:3,3]=p
        return result
