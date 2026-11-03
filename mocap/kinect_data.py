"""Versioned Kinect v2 packets, bounded reading and immutable capture sessions."""
from pathlib import Path
import hashlib, io, json, os, shutil, struct, subprocess
import numpy as np

NAMES = ['SpineBase','SpineMid','Neck','Head','ShoulderLeft','ElbowLeft','WristLeft','HandLeft','ShoulderRight','ElbowRight','WristRight','HandRight','HipLeft','KneeLeft','AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight','FootRight','SpineShoulder','HandTipLeft','ThumbLeft','HandTipRight','ThumbRight']
PARENTS = [-1,0,20,2,20,4,5,6,20,8,9,10,0,12,13,14,0,16,17,18,1,7,7,11,11]
MAX_PACKET = 16*1024**2
MAGIC = b'RGBKINECT1\n'

def exact(stream, n):
    result=bytearray()
    while len(result)<n:
        part=stream.read(n-len(result))
        if not part:raise EOFError('Incomplete Kinect frame')
        result.extend(part)
    return bytes(result)

def packet(stream):
    size=stream.read(4)
    if not size:return None
    if len(size)!=4:raise ValueError('Truncated Kinect packet length')
    n=struct.unpack('<I',size)[0]
    if not 32<=n<=MAX_PACKET:raise ValueError('Invalid Kinect packet size')
    return exact(stream,n)

def decode(raw):
    s=io.BytesIO(raw)
    def unpack(fmt):return struct.unpack(fmt,exact(s,struct.calcsize(fmt)))
    version,t,tc,tb=unpack('<iddd')
    if version!=1 or not np.isfinite([t,tc,tb]).all():raise ValueError('Invalid Kinect frame header')
    count,=unpack('<i')
    if not 0<=count<=6:raise ValueError('Invalid body count')
    bodies={}
    for _ in range(count):
        key,=unpack('<Q');joints=[];states=[];quats=[]
        for j in range(25):
            x,y,z,state,qx,qy,qz,qw=unpack('<ff f i ffff')
            joints.append([x,y,z]);states.append(state);quats.append([qx,qy,qz,qw])
        if str(key) in bodies or not np.isfinite(joints).all() or not np.isfinite(quats).all() or not set(states)<=set(range(3)):raise ValueError('Invalid tracked body')
        bodies[str(key)]=dict(joints=np.asarray(joints),states=np.asarray(states),quaternions=np.asarray(quats))
    n,=unpack('<i')
    if not 1<=n<=8*1024**2:raise ValueError('Invalid color image size')
    jpeg=exact(s,n);n,=unpack('<i')
    if n!=512*424:raise ValueError('Invalid depth dimensions')
    depth=np.frombuffer(exact(s,n*2),'<u2').reshape(424,512)
    n,=unpack('<i')
    if not 0<=n<=128*106:raise ValueError('Invalid point count')
    points=np.frombuffer(exact(s,n*16),dtype=[('xyz','<f4',3),('rgba','u1',4)])
    if s.read(1) or not np.isfinite(points['xyz']).all():raise ValueError('Invalid point cloud')
    return dict(time=t,color_time=tc,body_time=tb,bodies=bodies,jpeg=jpeg,depth=depth,points=points)

def bridge():
    from .paths import ROOT,HOME
    sdk=Path(os.environ.get('KINECTSDK20_DIR',r'C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409'))/'Assemblies/Microsoft.Kinect.dll'
    if not sdk.is_file():raise RuntimeError('Install Microsoft Kinect for Windows SDK 2.0 first.')
    source=ROOT/'tools/KinectCapture.cs';key=hashlib.sha256(source.read_bytes()+sdk.read_bytes()).hexdigest()[:20]
    folder=HOME/'cache/kinect/bridge'/key;folder.mkdir(parents=True,exist_ok=True)
    target=folder/'KinectCapture.exe'
    if not target.exists():
        compiler=Path(os.environ.get('WINDIR',r'C:\Windows'))/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        result=subprocess.run([str(compiler),'/nologo','/target:exe','/platform:x64','/reference:'+str(sdk),'/reference:System.Drawing.dll','/out:'+str(target),str(source)],capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(result.stdout.decode(errors='replace')+result.stderr.decode(errors='replace'))
    if not (folder/sdk.name).exists():shutil.copy2(sdk,folder/sdk.name)
    return target

class Recording:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False)
        self.file=(self.folder/'capture.krec').open('xb');self.file.write(MAGIC)
        self.index=[];self.start=None;self.last=None
        (self.folder/'recording.pending').write_text('Recording in progress',encoding='utf8')
    def append(self,raw,frame):
        t=frame['time']
        if self.last is not None and t<=self.last:raise ValueError('Kinect timestamps moved backwards')
        if len(self.index)%30==0 and shutil.disk_usage(self.folder).free<1024**3:raise RuntimeError('Recording stopped: less than 1 GiB free disk space.')
        offset=self.file.tell();self.file.write(struct.pack('<I',len(raw)));self.file.write(raw);self.index.append([offset,t]);self.last=t
    def close(self,error=None):
        self.file.close()
        meta=dict(version=1,frames=self.index,units='metres',axes='Kinect camera: X left, Y up, Z forward',joint_names=NAMES,parents=PARENTS,error=error)
        (self.folder/'session.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
        (self.folder/'recording.pending').unlink()

class Session:
    def __init__(self,folder):
        self.folder=Path(folder)
        if (self.folder/'recording.pending').exists():raise ValueError('Stop recording before opening the session.')
        p=self.folder/'session.json'
        if p.stat().st_size>32*1024**2:raise ValueError('Session index too large')
        self.meta=json.loads(p.read_text(encoding='utf8'));self.index=np.asarray(self.meta['frames'],dtype=np.float64)
        if self.meta['version']!=1 or self.index.ndim!=2 or self.index.shape[1]!=2 or not len(self.index) or not np.isfinite(self.index).all():raise ValueError('Empty or invalid session')
        if np.any(np.diff(self.index[:,1])<=0) or np.any(np.diff(self.index[:,0])<=0):raise ValueError('Invalid session ordering')
        with (self.folder/'capture.krec').open('rb') as f:
            if f.read(len(MAGIC))!=MAGIC:raise ValueError('Invalid capture file')
        self.times=self.index[:,1]-self.index[0,1]
    def frame(self,i):
        with (self.folder/'capture.krec').open('rb') as f:
            f.seek(int(self.index[i,0]));raw=packet(f)
        if raw is None:raise ValueError('Missing capture frame')
        result=decode(raw)
        if abs(result['time']-self.index[i,1])>1e-6:raise ValueError('Capture index mismatch')
        return result

def similarity(source,target):
    """Least-squares proper similarity, returning column-vector R, scale and offset."""
    a=np.asarray(source,float);b=np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<4 or not np.isfinite([a,b]).all():raise ValueError('Invalid alignment points')
    ac=a-a.mean(0);bc=b-b.mean(0)
    if np.linalg.matrix_rank(ac)<2:raise ValueError('Degenerate alignment pose')
    u,s,vt=np.linalg.svd(bc.T@ac);d=np.ones(3);d[-1]=np.linalg.det(u@vt);r=u@np.diag(d)@vt
    scale=float((s*d).sum()/(ac*ac).sum())
    if not .1<scale<10:raise ValueError('Implausible alignment scale; check matching person and time')
    shift=b.mean(0)-scale*r@a.mean(0)
    rms=float(np.sqrt(np.mean(np.sum((a@r.T*scale+shift-b)**2,axis=1))))
    return r,scale,shift,rms
