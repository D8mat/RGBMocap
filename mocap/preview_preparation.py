"""Plan explicit preview recovery without changing the selected motion result."""
import json
from pathlib import Path
from .motion_tools import RESULTS,base_result,result_is_current,result_sources
from .mixamo_data import current
from .result_artifacts import artifacts_current
from .flow_placement import needs_update


def surface_current(project,kind):
    from .mesh_data import MESH_VERSION
    try:
        for result in result_sources(kind):
            if not (project/f'{result}_joints.npz').is_file():return False
            info=json.loads((project/f'{result}_mesh.json').read_text(encoding='utf8'))
            name=info['frames_file']
            if info.get('version')!=MESH_VERSION or Path(name).name!=name or not (project/name).is_file():return False
            if not artifacts_current(project,result):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def plan(projects,kind,model):
    jobs=[];ready=0;unavailable=0
    for project in dict.fromkeys(map(Path,projects)):
        if not (project/RESULTS[base_result(kind)][0]).is_file():
            unavailable+=1;continue
        placement_old=model=='flowhmr' and needs_update(project)
        if placement_old:
            jobs.append(('rig',['--project',str(project),'--result',kind]));continue
        if not (project/RESULTS[kind][0]).is_file() or not result_is_current(project,kind):
            unavailable+=1;continue
        if not current(project,kind) or not surface_current(project,kind):
            jobs.append(('rig',['--project',str(project),'--result',kind]))
        else:ready+=1
    return jobs,ready,unavailable


def reload_preview(window):
    window.preview_signature=None
    window.request_people_scene(force=True)
    tools=getattr(window,'camera_tools',None)
    if tools is not None:tools.key=None;tools.sync()
    panel=getattr(window,'mamma_video_pair',None)
    if hasattr(panel,'overlay'):panel.overlay.clear();panel.repaint_videos()
    window.refresh_results()
