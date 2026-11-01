"""MAMMA actors already share one metric world; never re-estimate placement."""
import json
from pathlib import Path
from .storage import atomic_json
from .people_store import load_people,media_receipt,safe_child
from .scene_camera import read_camera,VERSION,take_projects


def prepare_cameras(root):
    root=Path(root);state=load_people(root)
    rows=take_projects(root,state)
    if not rows:raise ValueError('MAMMA shared scene has no source takes.')
    for _,project in rows:
        if read_camera(project) is None:raise ValueError('MAMMA calibrated camera/body receipt changed. Prepare its source result again.')
    person,project=rows[0];camera=read_camera(project)
    target=root/'people_scene.json'
    atomic_json(target,dict(version=VERSION,input=media_receipt(root),anchor=project.relative_to(root).as_posix(),
        anchor_id=person,world_to_camera=camera['world_to_camera'],sources=camera['sources'],floor=0.,K=camera['K'],camera_mode='static',scale=1.))
    return target
