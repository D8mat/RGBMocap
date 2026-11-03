"""Native SMPL-X skin/rig export, separate from reference-only Mixamo FBX."""
from pathlib import Path
import json,os,shutil,tempfile
import numpy as np
from .result_artifacts import file_stat

NAMES=['pelvis','thigh_l','thigh_r','spine_01','calf_l','calf_r','spine_02','foot_l','foot_r','spine_03','ball_l','ball_r','neck_01','clavicle_l','clavicle_r','head','upperarm_l','upperarm_r','lowerarm_l','lowerarm_r','hand_l','hand_r','jaw','eye_l','eye_r']
for side in ('l','r'):
    for finger in ('index','middle','pinky','ring','thumb'):
        NAMES.extend(f'{finger}_{i:02}_{side}' for i in (1,2,3))

def run(rows,emit=lambda *a,**k:None):
    from .motion_tools import selected_export,result_is_current,RESULTS
    from .mamma_body import body_path
    from .blend_support import build
    from .scene_export import context
    prepared=[];folders=[]
    try:
        for row in rows:
            project=Path(row['project']);kind=row['kind'];target=Path(row['target']).resolve()
            if target.suffix.lower()!='.fbx':raise ValueError('Unreal export requires an FBX filename')
            source=selected_export(project,kind)
            if source is None or source.resolve()!=Path(row['source']).resolve() or not result_is_current(project,kind):raise ValueError('Prepare the selected motion before Unreal export')
            meta=project/(kind+'_mesh.json')
            if not meta.exists():raise ValueError('Prepare the selected mesh preview before Unreal export.')
            info=json.loads(meta.read_text(encoding='utf8'));asset=Path(info['asset']);name=info['frames_file']
            if Path(name).name!=name:raise ValueError('Invalid mesh frame path')
            framefile=project/name
            smplh=info.get('representation')=='smplh'
            from .models import BY_KEY
            model=BY_KEY['smplh'].path if smplh else body_path(project)
            parameters=project/RESULTS[kind][0]
            tracked=[source,parameters,meta,asset,framefile,model];stamps=[file_stat(p) for p in tracked]
            if file_stat(parameters)!=[info['source_size'],info['source_mtime_ns']]:raise ValueError('Mesh preview is stale; prepare it again')
            if file_stat(model)!=[info['model_size'],info['model_mtime_ns']]:raise ValueError('Body model changed; prepare the mesh preview again')
            folder=Path(tempfile.mkdtemp(prefix='.unreal-',dir=target.parent));folders.append(folder)
            scene=row.get('scene');transform=np.eye(4)
            if scene:
                if context(scene['root'],project)!=scene:raise ValueError('Scene placement changed')
                transform=np.asarray(scene['transform'],float)
            frames=None
            if smplh:
                from .unreal_smplh import prepare
                names,length=prepare(parameters,model,folder,transform)
            else:
                with np.load(asset,allow_pickle=False) as z:packed=z['packed'];faces=z['faces'];count=int(z['vertex_count'])
                if count!=10475 or packed.size!=count*(3*507+55):raise ValueError('Unsupported mesh layout')
                frames=np.load(framefile,mmap_mode='r',allow_pickle=False)
                if frames.shape!=(info['frames'],1389):raise ValueError('Unsupported SMPL-X frame layout')
                with np.load(model,allow_pickle=False) as z:reg=z['J_regressor'];parents=z['kintree_table'][0].astype(int);parents[0]=-1
                shaped=packed[:count*3].reshape(count,3)+(frames[0,1366:1386]@packed[count*3:count*63].reshape(20,count*3)).reshape(count,3)
                joints=reg@shaped;weights=packed[count*1521:].reshape(count,55).copy()
                # Bound influences for UE's standard GPU skinning path, preserving normalization.
                keep=np.argsort(weights,axis=1)[:,-8:];mask=np.zeros_like(weights);np.put_along_axis(mask,keep,np.take_along_axis(weights,keep,axis=1),axis=1);weights=mask/np.maximum(mask.sum(1,keepdims=True),1e-12)
                if not np.isfinite(shaped).all() or not np.isfinite(weights).all():raise ValueError('Non-finite skin data')
                world=np.lib.format.open_memmap(folder/'motion.npy',mode='w+',dtype='float32',shape=(len(frames),55,4,4))
                bind=np.tile(np.eye(4),(55,1,1));bind[:,:3,3]=joints
                for i,frame in enumerate(frames):
                    world[i]=transform@(frame[:880].reshape(55,4,4)@bind)
                    world[i,:,:3,3]+=transform[:3,:3]@frame[-3:]
                    if i%100==0:emit('progress',stage='save_fbx',current=i,total=len(frames),message='Preparing Unreal skeletal animation')
                world.flush();del world
                np.savez(folder/'skin.npz',vertices=shaped,faces=faces,weights=weights,joints=joints,parents=parents)
                names,length=NAMES,len(frames)
            state=json.loads((project/'project.json').read_text(encoding='utf8'))
            if length!=state['frames']:raise ValueError('Motion length differs from the project')
            job=dict(target='character.fbx',names=names,fps=state['fps'],frames=length,prepend_tpose=row.get('prepend_tpose',False),z_offset_cm=float(row.get('unreal_z_offset_cm',0.)),ground_feet=bool(row.get('unreal_ground_feet',True)),align_heading=bool(row.get('unreal_align_heading',False)))
            if not np.isfinite(job['z_offset_cm']) or abs(job['z_offset_cm'])>100000:raise ValueError('Invalid Unreal Z offset')
            if job['prepend_tpose']:
                from .unreal_smplh import tpose
                with np.load(folder/'skin.npz',allow_pickle=False) as skin:
                    motion=np.load(folder/'motion.npy',mmap_mode='r',allow_pickle=False)
                    np.save(folder/'tpose.npy',tpose(skin['joints'],skin['parents'],motion[0]));motion._mmap.close()
            if frames is not None:frames._mmap.close()
            (folder/'unreal-job.json').write_text(json.dumps(job),encoding='utf8')
            build(folder,'unreal_scene.py',emit,'save_fbx')
            if not (folder/'character.fbx').exists():raise RuntimeError('Blender did not create Unreal FBX')
            if [file_stat(p) for p in tracked]!=stamps:raise ValueError('Source data changed during export')
            prepared.append((folder/'character.fbx',target))
        for src,dst in prepared:os.replace(src,dst)
        return len(prepared)
    finally:
        # Only directories freshly allocated by this call are removed.
        for folder in folders:shutil.rmtree(folder,ignore_errors=True)
