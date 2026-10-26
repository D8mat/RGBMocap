"""One project, independent model result directories, and small UI metadata."""
from pathlib import Path
import json
import os
from .storage import atomic_json
from .motion_tools import require_supported_tool, require_supported_project

VERSION = 2


def read(root):
    value=json.loads((Path(root)/'project.json').read_text(encoding='utf-8'))
    if value.get('workspace_version')!=VERSION:
        raise ValueError('This is not a model-folder project. Create a new project from the video.')
    require_supported_tool(value['active_model'])
    for model in value.get('models',{}):require_supported_tool(model)
    return value


def is_workspace(project):
    try:
        return json.loads((Path(project)/'project.json').read_text(encoding='utf-8')).get('workspace_version')==VERSION
    except (OSError,ValueError):return False


def root_for(project,state=None):
    project=Path(project).resolve()
    state=state if state is not None else json.loads((project/'project.json').read_text(encoding='utf-8'))
    if state.get('workspace_version')==VERSION:
        read(project)
        return project
    relative=state.get('workspace_root')
    if relative is None:return None
    root=(project/relative).resolve()
    if root==project or not project.is_relative_to(root):raise ValueError('Invalid workspace location')
    read(root)
    return root


def child(root,relative,model=None):
    root=Path(root).resolve();path=(root/relative).resolve()
    boundary=root/model if model else root
    if path==root or not path.is_relative_to(boundary):raise ValueError('Invalid model project location')
    return path


def model_project(root,model,state=None):
    require_supported_tool(model)
    state=state if state is not None else read(root)
    entry=state.get('models',{}).get(model,{})
    path=child(root,entry.get('active',model),model)
    if (path/'project.json').is_file() and require_supported_project(path).key!=model:
        raise ValueError('Saved model identity does not match its folder. Its files were preserved.')
    return path


def range_key(start,end):
    import hashlib
    return hashlib.sha256(f'{start:.6f}|{end:.6f}|working-v1'.encode()).hexdigest()[:14]


def create(root,source,info,model):
    require_supported_tool(model)
    root=Path(root).resolve()
    if (root/'project.json').exists():
        read(root)
        return root
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'project.json',dict(workspace_version=VERSION,name=root.name,
        source=info,media=None,time_offset=0.,active_model=model,models={},view={},origins=[]))
    return root


def set_active(root,model,project=None,view=None):
    state=read(root);require_supported_tool(model)
    state['active_model']=model
    entry=state.setdefault('models',{}).setdefault(model,{})
    if project is not None:
        project=Path(project).resolve()
        child(root,project.relative_to(Path(root).resolve()),model)
        entry['active']=project.relative_to(Path(root).resolve()).as_posix()
        state['reference']=entry['active']
    if view is not None:
        entry['view']=view
        state['view']=view
    atomic_json(Path(root)/'project.json',state)
    return state


def media_path(root,state=None):
    root=Path(root).resolve();state=state if state is not None else read(root)
    return child(root,state['media']) if state.get('media') else Path(state['source']['path'])


def inherit(parent,target,state):
    root=root_for(parent,state)
    if root is not None:state['workspace_root']=os.path.relpath(root,target)
    return state
