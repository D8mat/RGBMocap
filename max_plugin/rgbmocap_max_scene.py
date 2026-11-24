"""Max scene adapter. All methods must be called on Max's main thread."""
import math
import pymxs
from pymxs import runtime as rt
from rgbmocap_max_protocol import matrix,rotation,position,multiply,transpose

def rows(tm):
    return [[float(v) for v in getattr(tm,'row'+str(i))] for i in (1,2,3,4)]
def normalized(tm):
    r=rows(tm)[:3];scale=[math.sqrt(sum(v*v for v in row)) for row in r]
    if min(scale)<1e-8:raise ValueError('Zero scale on target bone')
    unit=[[v/s for v in row] for row,s in zip(r,scale)]
    if any(abs(sum(a*b for a,b in zip(unit[i],unit[j])))>.001 for i in range(3) for j in range(i)):
        raise ValueError('Sheared target transforms are not supported')
    return transpose(unit),scale

def transform(rot,pos,scale):
    r=transpose(rot)
    return rt.matrix3(*[rt.point3(*[v*scale[i] for v in row]) for i,row in enumerate(r)],rt.point3(*pos))

def descendants(root):
    result=[];pending=[root]
    while pending:
        node=pending.pop(0);result.append(node);pending.extend(list(node.children))
    return result

def suffix(name):return str(name).rsplit(':',1)[-1].lower()

class Binding:
    def __init__(self,schema,nodes,root_motion_scale=1.):
        self.schema=schema;self.nodes=nodes;self.factor=float(root_motion_scale)
        self.source=[matrix(b[2]) for b in schema['bones']]
        self.original={i:rt.copy(n.controller) for i,n in nodes.items()}
        self.rest={i:rt.copy(n.transform) for i,n in nodes.items()}
        self.orientation={i:normalized(tm) for i,tm in self.rest.items()}
        names=[b[0] for b in schema['bones']]
        self.root=names.index('Hips') if 'Hips' in names else next(i for i,b in enumerate(schema['bones']) if b[1]<0)
        if self.root not in nodes:raise ValueError('Map Hips/root before applying motion')
        handles=[int(rt.getHandleByAnim(n)) for n in nodes.values()]
        if len(handles)!=len(set(handles)):raise ValueError('A target bone cannot be mapped twice')
        for n in nodes.values():
            cls=str(rt.classOf(n.controller)).lower()
            if cls!='prs':raise ValueError('Only ordinary FBX PRS controllers are supported, not Biped/CAT/constraints: '+str(n.name))
        # Preview uses temporary controller copies. Original animation is restored on Stop.
        for i,n in nodes.items():n.controller=rt.copy(self.original[i]);rt.invalidateTreeTM(n)
        self.units=float(rt.units.decodeValue('1m'))

    def restore(self):
        for i,n in self.nodes.items():
            if rt.isValidNode(n):n.controller=rt.copy(self.original[i]);rt.invalidateTreeTM(n)
    def save_current(self):return {i:rt.copy(n.controller) for i,n in self.nodes.items()}
    def restore_saved(self,saved):
        for i,n in self.nodes.items():
            if rt.isValidNode(n):n.controller=rt.copy(saved[i]);rt.invalidateTreeTM(n)
    def apply(self,pose,key=None):
        if len(pose)!=len(self.source):raise ValueError('Pose hierarchy changed; bind again')
        if not all(rt.isValidNode(n) for n in self.nodes.values()):raise ValueError('A mapped bone was deleted')
        context=pymxs.attime(float(key)) if key is not None else pymxs.attime(rt.sliderTime)
        with pymxs.animate(key is not None),context:
            for i in sorted(self.nodes):
                n=self.nodes[i];target_rest,scale=self.orientation[i]
                delta=multiply(rotation(pose[i]),transpose(rotation(self.source[i])))
                desired=multiply(delta,target_rest)
                # Parent rotation has already moved children; retain their local offsets.
                pos=rows(n.transform)[3]
                if i==self.root:
                    base=rows(self.rest[i])[3];src=position(self.source[i]);current=position(pose[i])
                    pos=[base[k]+(current[k]-src[k])*self.units*self.factor for k in range(3)]
                n.transform=transform(desired,pos,scale)

class Session:
    def __init__(self):self.bindings={};self.mode='idle';self.backup={};self.recorded=0
    def bind(self,actor,schema,nodes,scale=1.):
        if self.mode!='idle':raise ValueError('Stop recording/baking before mapping')
        used={int(rt.getHandleByAnim(n)) for a,b in self.bindings.items() if a!=actor for n in b.nodes.values()}
        if any(int(rt.getHandleByAnim(n)) in used for n in nodes.values()):raise ValueError('This skeleton is already assigned to another person')
        if actor in self.bindings:self.bindings[actor].restore()
        self.bindings[actor]=Binding(schema,nodes,scale)
    def start(self,mode,actors):
        if self.mode!='idle':raise ValueError('Another recording/bake is active')
        if not actors or any(a not in self.bindings for a in actors):raise ValueError('Map a character first')
        for b in self.bindings.values():b.restore()
        self.backup={a:self.bindings[a].save_current() for a in actors};self.mode=mode;self.recorded=0
    def finish(self,keep):
        for a,saved in self.backup.items():
            b=self.bindings[a]
            if keep:b.original=b.save_current()
            else:b.restore_saved(saved)
        self.backup={};self.mode='idle'
    def stop_preview(self):
        for b in self.bindings.values():b.restore()
