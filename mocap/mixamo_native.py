"""Direct LOD mesh fitting. No per-frame SMPL-X pose conversion.

The registered rest template supplies anatomical landmarks and skin regions;
surface correspondences estimate segment frames, not distal-joint roll hints.
"""
import numpy as np
from .mixamo_rig import align,unit,BODY_PARENTS,fit_frame


def landmark_weights(vertices,joints,count=48):
    result=np.zeros((22,len(vertices)))
    for j,point in enumerate(joints):
        distance=np.linalg.norm(vertices-point,axis=1)
        ids=np.argsort(distance)[:count];delta=vertices[ids]-point
        a=np.row_stack((delta.T,np.ones(len(ids))))
        prior=1/np.maximum(distance[ids],.015)**2;prior/=prior.sum()
        cov=(a*prior)@a.T
        weights=prior*(a.T@np.linalg.solve(cov+np.diag([1e-10]*3+[1e-12]),[0,0,0,1]))
        if not np.isfinite(weights).all() or np.abs(weights).max()>5:raise ValueError('Native anatomical landmark calibration is ill-conditioned')
        result[j,ids]=weights/weights.sum()
    return result


def align_batch(a,b):
    a=unit(a);b=unit(b);v=np.cross(a,b);c=np.clip((a*b).sum(-1),-1,1)
    skew=np.zeros((len(a),3,3));skew[:,0,1]=-v[:,2];skew[:,0,2]=v[:,1]
    skew[:,1,0]=v[:,2];skew[:,1,2]=-v[:,0];skew[:,2,0]=-v[:,1];skew[:,2,1]=v[:,0]
    out=np.eye(3)+skew+skew@skew/np.maximum(1+c[:,None,None],1e-10)
    for i in np.flatnonzero(c < -1+1e-8):out[i]=align(a[i],b[i])
    return out


class NativeFitter:
    def __init__(self,vertices,joints,skin):
        self.vertices=np.asarray(vertices,dtype=float);self.rest=np.asarray(joints,dtype=float)
        self.regressor=landmark_weights(self.vertices,self.rest)
        self.regions=[]
        for j in range(22):
            weights=np.maximum(np.asarray(skin[:,j],float)-.15,0)**3
            if (weights>1e-5).sum()<6:
                distance=np.linalg.norm(self.vertices-self.rest[j],axis=1)
                ids=np.argsort(distance)[:18];weights=np.zeros(len(vertices));weights[ids]=1/np.maximum(distance[ids],.025)
            weights/=weights.sum();center=weights@self.vertices
            self.regions.append((weights,center,self.vertices-center))

    def proportions(self,frames):
        points=np.einsum('jv,tvc->tjc',self.regressor,np.asarray(frames))
        rest=self.rest.copy()
        for j,p in enumerate(BODY_PARENTS):
            if p<0:continue
            length=float(np.median(np.linalg.norm(points[:,j]-points[:,p],axis=-1)))
            if not .008<length<1.2:raise ValueError('Native body proportions could not be calibrated')
            rest[j]=rest[p]+unit(self.rest[j]-self.rest[p])*length
        return rest

    def fit(self,frames,rest):
        frames=np.asarray(frames,dtype=float)
        if frames.shape[1:]!=(595,3) or not np.isfinite(frames).all():raise ValueError('Invalid native mesh frames')
        points=np.einsum('jv,tvc->tjc',self.regressor,frames)
        rotations=np.empty((len(frames),22,3,3))
        for j,(w,center,reference) in enumerate(self.regions):
            y=frames-np.einsum('v,tvc->tc',w,frames)[:,None]
            covariance=np.einsum('vi,v,tvj->tij',reference,w,y)
            u,_,vt=np.linalg.svd(covariance)
            v=vt.transpose(0,2,1);sign=np.linalg.det(v@u.transpose(0,2,1))
            v[:,:,2]*=sign[:,None]
            rotations[:,j]=v@u.transpose(0,2,1)
        children={1:4,2:5,3:6,4:7,5:8,6:9,7:10,8:11,9:12,12:15,13:16,14:17,16:18,17:19,18:20,19:21}
        for j,child in children.items():
            reference=rest[child]-rest[j]
            predicted=np.einsum('tij,j->ti',rotations[:,j],reference)
            desired=points[:,child]-points[:,j]
            rotations[:,j]=align_batch(predicted,desired)@rotations[:,j]
        # Three independent tiny surface patches must not invent incompatible
        # waist frames. Distribute measured pelvis-to-chest motion continuously.
        from scipy.spatial.transform import Rotation
        # Set the actual pelvis frame first. Replacing it after distributing
        # the spine left the first segment in a different rotation basis.
        rotations[:,0]=fit_frame(rest[[1,2,3]]-rest[0],points[:,[1,2,3]]-points[:,0,None])
        relative=rotations[:,0].transpose(0,2,1)@rotations[:,9]
        vector=Rotation.from_matrix(relative).as_rotvec()
        angle=np.linalg.norm(vector,axis=1,keepdims=True)
        vector*=np.minimum(1,np.deg2rad(120)/np.maximum(angle,1e-10))
        for j,fraction in ((3,1/3),(6,2/3),(9,1)):
            rotations[:,j]=rotations[:,0]@Rotation.from_rotvec(vector*fraction).as_matrix()
        # Tiny independently deformed toe patches must not invent axial toe
        # roll. Keep their measured flexion in the foot's sagittal plane;
        # the ankle itself retains all three measured rotation axes.
        for foot,toe in ((7,10),(8,11)):
            relative=rotations[:,foot].transpose(0,2,1)@rotations[:,toe]
            flex=np.arctan2(relative[:,2,1]-relative[:,1,2],relative[:,1,1]+relative[:,2,2])
            flex=np.clip(flex,-np.pi/3,np.pi/3)
            vector=np.zeros((len(frames),3));vector[:,0]=flex
            rotations[:,toe]=rotations[:,foot]@Rotation.from_rotvec(vector).as_matrix()
        return points[:,0],rotations,points
