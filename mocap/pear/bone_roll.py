"""Source-relative live display frames, independent of PEAR inference/skin."""
from collections import deque
import numpy as np
from ..bone_overlay import reference_frame
from ..source_surface import layout,endpoints
from ..surface_roll import evidence,unit,smooth

ACTIVE=np.array([1,4,2,5,16,18,17,19])
# Both segments use the same knee/elbow material frame. A separate proximal
# neutral frame bakes in skin-contour bias as a permanent thigh/upper-arm roll.
PATCH=np.repeat(np.arange(4),2)
LIMIT=np.deg2rad(5.)
JOINTS=np.r_[np.arange(22),np.arange(25,55)]
# Pinned EHM: body joints, 68 face landmarks, then named surface landmarks.
TIPS={15:123,10:124,11:135,27:128,30:129,33:130,36:131,39:133,
      42:139,45:140,48:141,51:142,54:144}


class SkeletonLayout:
    def __init__(self,parents):
        lookup={int(j):i for i,j in enumerate(JOINTS)}
        self.parents=np.asarray([lookup.get(int(parents[j]),-1) for j in JOINTS],dtype=np.int32)
        self.children=[np.flatnonzero(self.parents==i) for i in range(52)]

    def landmarks(self,joints):
        joints=np.asarray(joints,dtype=np.float64);p=joints[JOINTS];ends=np.zeros_like(p)
        for i,native in enumerate(JOINTS):
            if len(self.children[i]):continue
            tip=TIPS.get(int(native))
            if tip is not None and tip<len(joints):ends[i]=joints[tip]-p[i]
            else:
                parent=self.parents[i]
                ends[i]=(p[i]-p[parent])*(1.4 if native==15 else .7) if parent>=0 else [0,.1,0]
        return p,ends

    def directions(self,p,ends):
        delta=np.stack([p[c].mean(0)-p[i] if len(c) else ends[i] for i,c in enumerate(self.children)])
        short=np.linalg.norm(delta,axis=1)<=.001;delta[short]+=[0.,.1,0.]
        return delta


def source_rotations(body,parents,camera):
    """Own-joint FK rotations used by EHM, including its neutral jaw/eyes."""
    local=np.concatenate([np.asarray(body['global_pose']).reshape(1,3,3),
        np.asarray(body['body_pose']).reshape(21,3,3),np.tile(np.eye(3),(3,1,1)),
        np.asarray(body['left_hand_pose']).reshape(15,3,3),
        np.asarray(body['right_hand_pose']).reshape(15,3,3)]).astype(np.float32)
    world=local.copy();world[0]=np.asarray(camera)[:3,:3]@local[0]
    for i in range(1,55):world[i]=world[int(parents[i])]@local[i]
    return world


def source_basis(rest,parents,children,neutral_surface):
    delta=np.stack([rest[c].mean(0)-rest[i] if len(c) else rest[i]-rest[parents[i]] for i,c in enumerate(children)])
    short=np.linalg.norm(delta,axis=1)<=.001;delta[short]+=[0.,.1,0.]
    basis=np.asarray([reference_frame(d) for d in delta])
    patches=np.asarray(neutral_surface)[PATCH]
    front=unit(patches[:,0]-patches[:,1]);side=unit(patches[:,2]-patches[:,3])
    y=unit(delta[ACTIVE]);x=basis[ACTIVE,:,0];z=basis[ACTIVE,:,2]
    c,t,strength,_,_=evidence(y,x,z,front,side)
    theta=np.where(strength>1e-6,np.arctan2(t,c),0.)
    x=x*np.cos(theta[:,None])+np.cross(y,x)*np.sin(theta[:,None])
    basis[ACTIVE]=np.stack([x,y,unit(np.cross(x,y))],axis=-1)
    # Shortest-arc zero roll is not a foot-width reference: the registered
    # ankle-to-toe pitch/yaw can tilt it into the sole. Calibrate both outgoing
    # foot segments to the neutral foot plane, retaining their own source FK.
    up=unit(rest[3]-rest[0]);lateral=unit(rest[1]-rest[2])
    for ankle,toe in ((7,10),(8,11)):
        forward=unit(rest[toe]-rest[ankle]);width=unit(np.cross(forward,up))
        if np.dot(width,lateral)<0:width=-width
        for joint in (ankle,toe):
            y=unit(delta[joint]);x=unit(width-y*np.dot(y,width))
            basis[joint]=np.stack([x,y,unit(np.cross(x,y))],axis=-1)
    return basis


class SurfaceRoll:
    def __init__(self):
        self.ids,self.lookup=layout('smplx');self.history=deque(maxlen=32);self.key=None;self.time=None

    def apply(self,y,x,vertices,timestamp=None,stream=None):
        z=unit(np.cross(x,y));x=unit(np.cross(y,z))
        patches=endpoints(np.asarray(vertices)[None,self.ids],self.lookup)[0][PATCH]
        front=unit(patches[:,0]-patches[:,1]);side=unit(patches[:,2]-patches[:,3])
        c,t,strength,_,_=evidence(y[ACTIVE],x[ACTIVE],z[ACTIVE],front,side)
        # Fixed neutral calibration carries the full source roll. Only the
        # additional skin-based correction has this small soft trust region.
        theta=np.where(strength>1e-6,np.arctan2(t,c),0.)
        # Opposed/ambiguous evidence fades to the source instead of flipping
        # between +/- the trust-region limit at the circular branch cut.
        angle=LIMIT*np.tanh(.25*np.sin(theta)/LIMIT)*smooth(strength,.005,.08)
        if timestamp is None:
            self.history.clear();self.time=None;self.key=None
        else:
            now=float(timestamp)
            if stream!=self.key or self.time is None or not 0<now-self.time<=.25:self.history.clear()
            self.key=stream;self.time=now
            while self.history and now-self.history[0][0]>.10:self.history.popleft()
            self.history.append((now,angle.copy()))
            values=np.asarray([v for _,v in self.history])
            angle=np.arctan2(np.sin(values).mean(0),np.cos(values).mean(0))
        x=x.copy();x[ACTIVE]=x[ACTIVE]*np.cos(angle[:,None])+np.cross(y[ACTIVE],x[ACTIVE])*np.sin(angle[:,None])
        return np.stack([x,y,unit(np.cross(x,y))],axis=-1)


class RollFrames(SkeletonLayout):
    """Small worker-side calculation; painting only consumes finished frames."""
    def __init__(self,parents,neutral_joints,neutral_surface):
        super().__init__(parents)
        self.source_basis=source_basis(np.asarray(neutral_joints)[JOINTS],self.parents,self.children,neutral_surface)
        self.roll=SurfaceRoll()

    def reset(self):self.roll.history.clear();self.roll.key=None;self.roll.time=None

    def frames(self,joints,rotations,vertices,timestamp=None,stream=None):
        p,ends=self.landmarks(joints);y=unit(self.directions(p,ends))
        prior=np.asarray(rotations)[JOINTS]@self.source_basis
        x=prior[:,:,0]-y*np.sum(y*prior[:,:,0],axis=-1,keepdims=True)
        fallback=np.cross(y,prior[:,:,2])
        x=unit(np.where(np.linalg.norm(x,axis=-1,keepdims=True)>.01,x,fallback))
        return self.roll.apply(y,x,vertices,timestamp,stream).astype(np.float32)
