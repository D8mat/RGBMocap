"""Standalone Kinect scene export and explicit, non-destructive AI anchoring."""
from pathlib import Path
import json, subprocess, shutil, uuid
import numpy as np
from .kinect_data import Session,NAMES,PARENTS,similarity

MATCH = {'Hips':0,'Neck':2,'Head':3,'LeftArm':4,'LeftForeArm':5,'LeftHand':6,'RightArm':8,'RightForeArm':9,'RightHand':10,'LeftUpLeg':12,'LeftLeg':13,'LeftFoot':14,'RightUpLeg':16,'RightLeg':17,'RightFoot':18}

def anchor(session,meta_path,body_id,offset,follow,folder):
    p=Path(meta_path);meta=json.loads(p.read_text(encoding='utf8'))
    def child(key):
        name=meta[key]
        if Path(name).name!=name:raise ValueError('AI metadata must reference local files')
        return p.parent/name
    frames=np.load(child('frames_file'),mmap_mode='r',allow_pickle=False)
    if frames.shape[1:]!=(52,4,4) or not len(frames):raise ValueError('Select a prepared *_mixamo.json AI result')
    fps=float(meta['fps'])
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid AI frame rate')
    names=meta['names'];pairs=[(names.index(name),j) for name,j in MATCH.items() if name in names]
    output=np.lib.format.open_memmap(folder/'ai_frames.npy',mode='w+',dtype='float32',shape=frames.shape)
    valid=np.zeros(len(frames),bool);calibration=None;errors=[]
    for i,matrices in enumerate(frames):
        time=offset+i/fps;near=int(np.argmin(abs(session.times-time)))
        if abs(session.times[near]-time)>.06:continue
        body=session.frame(near)['bodies'].get(body_id)
        if body is None or body['states'][0]!=2:continue
        selected=[(a,b) for a,b in pairs if body['states'][b]==2]
        if calibration is None:
            if len(selected)<6:continue
            r,scale,shift,rms=similarity([matrices[a,:3,3] for a,b in selected],[body['joints'][b] for a,b in selected])
            if rms>.20:raise ValueError('Alignment error exceeds 20 cm. Check person, matching pose and time offset.')
            calibration=(r,scale,shift);errors.append(rms)
        r,scale,shift=calibration
        out=np.array(matrices,float);out[:,:3,:3]=r@matrices[:,:3,:3]
        out[:,:3,3]=(matrices[:,:3,3]@r.T)*scale+shift
        if follow:out[:,:3,3]+=body['joints'][0]-out[0,:3,3]
        if not np.isfinite(out).all():raise ValueError('Non-finite AI motion')
        output[i]=out;valid[i]=True
    if calibration is None:raise ValueError('No synchronized fully tracked pose for alignment. Adjust the time offset or capture the person.')
    output.flush();del output
    with np.load(child('asset_file'),allow_pickle=False) as asset:parents=asset['parents'].tolist()
    info=dict(names=names,parents=parents,fps=fps,offset=offset,valid=valid.tolist(),scale=calibration[1],calibration_rms_m=errors[0],follow_pelvis=follow,source=str(p.resolve()),body_id=body_id)
    (folder/'ai.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    return info

def export(folder,destination,snapshot=0,ai=None,progress=lambda text:None,cancel=lambda:False):
    from .paths import ROOT
    session=Session(folder);dest=Path(destination).resolve()
    if dest.exists():raise ValueError('Choose a new export folder')
    if not dest.parent.is_dir():raise ValueError('Output parent folder does not exist')
    if not 0<=snapshot<len(session.times):raise ValueError('Invalid point-cloud frame')
    if session.times[-1]>3600:raise ValueError('Export at most one hour per session')
    stage=dest.with_name('.'+dest.name+'.kinect-'+uuid.uuid4().hex);stage.mkdir()
    # Retain failed exports with their error; never remove an arbitrary destination.
    encoder=None
    try:
        f=session.frame(snapshot);cloud=f['points'];xyz=cloud['xyz'];rgb=cloud['rgba'][:,:3]
        with (stage/'environment.ply').open('wb') as out:
            out.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'%len(xyz)).encode())
            rows=np.empty(len(xyz),dtype=[('xyz','<f4',3),('rgb','u1',3)]);rows['xyz']=xyz;rows['rgb']=rgb;out.write(rows.tobytes())
        np.savez(stage/'environment.npz',xyz=xyz,rgb=rgb)
        fps=30;timeline=np.arange(int(np.floor(session.times[-1]*fps))+1)/fps
        ffmpeg=ROOT/'tools/bin/ffmpeg.exe';blender=ROOT/'runtime_character/blender.exe'
        if not ffmpeg.is_file() or not blender.is_file():raise ValueError('Install bundled FFmpeg and Blender using install.cmd')
        log=(stage/'export.log').open('wb')
        try:
            encoder=subprocess.Popen([str(ffmpeg),'-hide_banner','-loglevel','error','-f','image2pipe','-vcodec','mjpeg','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(stage/'video.mp4')],stdin=subprocess.PIPE,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            tracks={};last=-1;frame=None
            for i,t in enumerate(timeline):
                if cancel():raise RuntimeError('Export cancelled')
                k=int(np.searchsorted(session.times,t));k=min(k,len(session.times)-1)
                if k and abs(session.times[k-1]-t)<abs(session.times[k]-t):k-=1
                if k!=last:frame=session.frame(k);last=k
                encoder.stdin.write(frame['jpeg'])
                if abs(session.times[k]-t)<=.06:
                    for key,body in frame['bodies'].items():
                        tracks.setdefault(key,[]).append(dict(frame=i+1,joints=body['joints'].tolist(),states=body['states'].tolist(),quaternions=body['quaternions'].tolist()))
                if i%30==0:progress('Preparing video and skeletons: %d / %d'%(i+1,len(timeline)))
            encoder.stdin.close();encoder.wait()
            if encoder.returncode:raise RuntimeError('Video export failed; see export.log')
        finally:log.close()
        spec=dict(version=1,fps=fps,frames=len(timeline),names=NAMES,parents=PARENTS,bodies=tracks,snapshot=snapshot,source=str(session.folder.resolve()),coordinates='Kinect camera metres',ai=None)
        if ai:
            progress('Aligning AI motion with Kinect depth skeleton')
            spec['ai']=anchor(session,ai['path'],str(ai['body_id']),ai['offset'],ai['follow'],stage)
        (stage/'scene.json').write_text(json.dumps(spec),encoding='utf8')
        progress('Building Blender scene, FBX and BVH')
        with (stage/'blender.log').open('wb') as log:
            p=subprocess.Popen([str(blender),'-b','--factory-startup','--python',str(Path(__file__).with_name('kinect_blender.py')),'--',str(stage)],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while p.poll() is None:
                if cancel():p.terminate();p.wait();raise RuntimeError('Export cancelled')
                try:p.wait(timeout=.2)
                except subprocess.TimeoutExpired:pass
            if p.returncode or not (stage/'Kinect Scene.blend').exists():raise RuntimeError('Blender export failed; see '+str(stage/'blender.log'))
        if dest.exists():raise ValueError('Destination was created during export')
        stage.rename(dest);return dest
    except Exception as exc:
        (stage/'ERROR.txt').write_text(str(exc),encoding='utf8');raise
    finally:
        if encoder and encoder.poll() is None:encoder.kill();encoder.wait()
