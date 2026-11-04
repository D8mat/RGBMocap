"""Non-destructive, per-take horizontal placement in application Y-up metres."""
import json
from pathlib import Path
import numpy as np
from .storage import atomic_json

FILE='character_position.json'


def offset(project):
    path=Path(project)/FILE
    if not path.exists():return np.zeros(3)
    value=np.asarray(json.loads(path.read_text(encoding='utf8'))['offset'],float)
    if value.shape!=(3,) or not np.isfinite(value).all() or value[1]!=0:
        raise ValueError('Invalid character placement')
    return value


def matrix(project):
    result=np.eye(4);result[:3,3]=offset(project);return result


def recenter_offset(previous,world_pelvis):
    pelvis=np.asarray(world_pelvis,float)
    if pelvis.shape!=(3,) or not np.isfinite(pelvis).all():raise ValueError('No finite pelvis position')
    result=np.asarray(previous,float).copy();result[[0,2]]-=pelvis[[0,2]];result[1]=0
    return result


def apply(window,reset=False):
    if window.busy:window.status.setText('Finish processing before changing character position.');return
    if window.project is None or window.model_choice.currentData()=='pear':
        window.status.setText('Select a reconstructed motion take first.');return
    if window.selected_character() is not None:
        window.status.setText('Select the source person to recenter its motion take.');return
    project=Path(window.project);view=window.motion_view
    try:
        value=offset(project)
        if reset:new=np.zeros(3)
        else:
            take=window.scene_data.by_project.get(project.resolve()) if window.scene_data else None
            if take is not None:
                frame=take.frame_at(view.scene_time)
                if frame is None:raise ValueError('Move the timeline inside the selected take')
                pelvis=view.scene_joint_points(take,frame)[0]
            else:
                if view.joints is None:raise ValueError('Prepare the selected motion preview first')
                pelvis=np.asarray(view.joints[min(view.index,len(view.joints)-1),0])+value
            new=recenter_offset(value,pelvis)
        atomic_json(project/FILE,dict(schema=1,units='metres',up='Y',offset=new.tolist()))
        # Update all already-loaded shared scene instances, then request fresh data.
        seen=set()
        for scene in [window.scene_data,*window.scene_cache.values()]:
            if scene is None:continue
            take=scene.by_project.get(project.resolve())
            if take is not None and id(take) not in seen:
                seen.add(id(take));take.transform[:3,3]+=new-value
        linked=False
        for character in getattr(window,'characters',[]):
            for take in character.get('takes',[]):
                if take.get('source') and Path(take['source']).resolve()==project.resolve():
                    transform=np.array(take.get('transform') if take.get('transform') is not None else np.eye(4),float)
                    transform[:3,3]+=new-value;take['transform']=transform.tolist();linked=True
        if linked:window.save_characters()
        view.single_transform=matrix(project)
        window.request_people_scene(force=True)
        if window.preview_data is not None:window.apply_preview()
        view.update()
        window.status.setText('Character position reset.' if reset else 'Selected take centered at this frame; the same constant shift applies to preview and exports.')
    except (OSError,ValueError,KeyError) as exc:window.status.setText('Cannot recenter character: '+str(exc))
