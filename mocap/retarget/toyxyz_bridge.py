"""Prepare an immutable BVH input for the optional Blender retargeter."""
import json,shutil
from pathlib import Path
from .source import stamp


def prepare(project,result,folder,options):
    from mocap.mixamo_data import current
    from mocap.motion_tools import result_is_current
    project=Path(project)
    if not result_is_current(project,result) or not current(project,result):
        raise ValueError('Toyxyz Retarget needs a current source skeleton. Prepare Preview first. Native77 sources currently use the original retargeter.')
    metadata=project/(result+'_mixamo.json')
    info=json.loads(metadata.read_text(encoding='utf-8'))
    name=info['bvh_file']
    if Path(name).name!=name:raise ValueError('Invalid source BVH path')
    from mocap.project_storage import resolve_reference
    inputs=[metadata,project/name,project/'project.json']
    inputs.extend(resolve_reference(project,p) for p in info.get('sources',{}) if resolve_reference(project,p).is_file())
    inputs.extend(project/p for p in info.get('outputs',{}) if Path(p).name==p)
    before=stamp(inputs)
    shutil.copy2(project/name,folder/'toyxyz-source.bvh')
    (folder/'toyxyz-options.json').write_text(json.dumps(dict(options=options,fps=info['fps'],frames=info['frames'])),encoding='utf-8')
    return dict(frames=info['frames'],fps=info['fps'],source_stats=before)
