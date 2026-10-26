"""CPU-only FlowHMR shared-world receipts and bounded scene loading."""
from pathlib import Path
import json
import numpy as np
from .storage import atomic_json
from .result_artifacts import file_stat,artifacts_current


def prepare_cameras(root,rows):
    from .scene_camera import VERSION,camera_sources
    from .people_store import media_receipt
    from .models import BY_KEY
    root=Path(root)
    with np.load(root/'flow_camera.npz',allow_pickle=False) as data:
        world_to_camera=data['camera_RT_yup'][0].copy()
    from .flow_placement import wv_rotation,current
    # FlowHMR conditions on R only. VGGT's monocular translation/scale is not
    # metric and cannot be applied to independently generated metric bodies.
    world_to_camera[:3,3]=0
    rows=[(ident,p) for ident,p in rows if (p/'hmr4d_results.pt').is_file() and (p/'flow_placement.json').is_file()]
    calibrated=[json.loads((p/'flow_world.json').read_text(encoding='utf-8')) for _,p in rows if current(p)]
    if any((p/'flow_world.json').exists() or (p/'flow_world_pending.json').exists() for _,p in rows) and len(calibrated)!=len(rows):
        raise ValueError('FlowHMR scene contains different placement generations. Use Prepare Preview for all people.')
    # Equal actor/take estimates; no arbitrary inter-person spacing or body scale.
    estimates={}
    for ident,p in rows:
        if calibrated:estimates.setdefault(ident,[]).append(json.loads((p/'flow_world.json').read_text(encoding='utf-8'))['camera_height'])
    height=float(np.median([np.median(v) for v in estimates.values()])) if estimates else 0.
    if calibrated:world_to_camera[:3,:3]=world_to_camera[:3,:3]@wv_rotation(world_to_camera[:3,:3]).T
    world_to_camera[:3,3]=-world_to_camera[:3,:3]@np.array([0.,height,0.])
    # GUI/base and Flow workers use different NumPy/BLAS versions. Hashing
    # insignificant last-bit differences must not invalidate a shared floor.
    if calibrated:
        world_to_camera=np.round(world_to_camera.astype(np.float64),10)
        world_to_camera[np.abs(world_to_camera)<5e-11]=0.
    intrinsic=None
    for ident,project in rows:
        placement=json.loads((project/'flow_placement.json').read_text(encoding='utf-8'))
        k=np.asarray(placement['intrinsics'],float)
        if intrinsic is not None and not np.allclose(k,intrinsic,atol=1e-5):
            raise ValueError('People do not share original-frame camera intrinsics. Prepare Preview after reconstructing matching takes.')
        intrinsic=k
        model=BY_KEY['smplh'].path
        mesh=json.loads((project/'initial_mesh.json').read_text(encoding='utf-8')) if not model.exists() else {}
        atomic_json(project/'scene_camera.json',dict(version=VERSION,source_model='flowhmr',sources=camera_sources(project),
            model=file_stat(model) if model.exists() else [mesh['model_size'],mesh['model_mtime_ns']],
            world_to_camera=world_to_camera.tolist(),K=placement['intrinsics'],htd_compatible=False,
            placement_version=2 if calibrated else 1,camera_height=height,
            method='Shared FlowHMR WV world; SAM XZ placement; common camera height; native body height'))
    path=root/'people_scene.json'
    if rows:
        ident,project=min(rows,key=lambda row:(row[0],str(row[1])))
        camera=json.loads((project/'scene_camera.json').read_text(encoding='utf-8'))
        atomic_json(path,dict(version=VERSION,input=media_receipt(root),anchor=project.relative_to(root).as_posix(),anchor_id=ident,
            world_to_camera=world_to_camera.tolist(),sources=camera['sources'],floor=0.,K=camera['K'],camera_mode='shared_rotation',scale=1.,placement_version=2 if calibrated else 1,camera_height=height))
    return path


def load_scene(root,kind,cancelled):
    from .scene_data import PeopleScene,SceneTake
    from .scene_camera import take_projects
    from .people_store import load_people
    from .preview_data import mapped_array,motion_summary,LoadCancelled
    from .bone_overlay import open_preview
    from .motion_tools import result_is_current
    state=load_people(root);rows=take_projects(root,state)
    scene=PeopleScene(root,dict(floor=0.))
    usable=[(i,p) for i,p in rows if artifacts_current(p,kind) and result_is_current(p,kind,ignore_contacts=True)]
    if not usable:return scene
    prepare_cameras(root,rows)
    scene.reference=json.loads((root/'people_scene.json').read_text(encoding='utf-8'))
    from .scene_floor import current_resolution
    resolved=current_resolution(root)
    if resolved:scene.reference['floor']=resolved['floor'];scene.contact_floor=resolved['floor']
    interval_count=0
    try:
        for ident,project in usable:
            if cancelled():raise LoadCancelled()
            arrays={}
            try:
                loaded=open_preview(project,kind,cancelled)
                if not loaded:raise ValueError('FlowHMR native preview is unavailable. Use Prepare Preview.')
                arrays,asset,info=loaded
                arrays['joints']=mapped_array(project/(kind+'_joints.npz'),'joints',cancelled)
                arrays['joints'].max_blocks=1
                manifest=json.loads((project/'project.json').read_text(encoding='utf-8'))
                if len(arrays['joints'])!=manifest['frames']:raise ValueError('FlowHMR scene frame counts differ')
                take=SceneTake(ident,project,manifest,np.eye(4),arrays,motion_summary(arrays['joints']),None)
                take.native_required=True;take.raw_asset=asset;take.target_asset=asset
                take.target_stamp=take.raw_target_stamp=file_stat(project/(kind+'_mixamo.json'))
                from .preview_data import load_contact_track
                from .motion_tools import base_result
                track=load_contact_track(project,base_result(kind),cancelled)
                if track and track['context']['frames']==take.frames and track['context']['fps']==take.fps:
                    interval_count+=sum(map(len,track['ranges']))
                    if interval_count>65536:raise ValueError('Scene foot-lock metadata exceeds 65536 intervals. Trim the video.')
                    take.set_contacts(track['ranges']);take.contact_revision=track['revision']
                scene.takes.setdefault(ident,[]).append(take)
                for array in arrays.values():array.release_windows()
            except Exception:
                for array in arrays.values():array.close()
                raise
        scene.finish()
        return scene
    except Exception:scene.close();raise
