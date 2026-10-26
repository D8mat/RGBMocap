"""Relocatable project paths. Original source/receipt provenance stays intact."""
from pathlib import Path
import json


def manifest(project):
    return json.loads((Path(project) / 'project.json').read_text(encoding='utf-8'))


def storage_root(project, state=None):
    project = Path(project).resolve()
    state=state if state is not None else manifest(project)
    from .project_workspace import root_for
    workspace=root_for(project,state)
    if workspace is not None:return workspace
    record = state.get('storage')
    if not record:
        return None
    root = (project / record['root']).resolve()
    if not project.is_relative_to(root):
        raise ValueError('Invalid saved project root')
    return root


def source_video(project, state=None):
    state = state if state is not None else manifest(project)
    from .project_workspace import root_for, media_path
    workspace=root_for(project,state)
    if workspace is not None:return media_path(workspace)
    root = storage_root(project, state)
    if root is None:
        return Path(state['source']['path'])
    path = (root / state['storage']['media']).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Invalid saved project media path')
    return path


def resolve_reference(project, name):
    """Rebase only recorded project-internal paths, never model installations."""
    project = Path(project)
    path = Path(name)
    if not path.is_absolute():
        return project / path
    state = manifest(project)
    root = storage_root(project, state)
    if root is not None:
        from .project_workspace import is_workspace, read, child
        if is_workspace(root):
            for origin in read(root).get('origins',[]):
                try:
                    relative=path.relative_to(Path(origin['path']))
                except ValueError:continue
                return child(root,Path(origin.get('relative','.'))/relative)
            return path
        for original in state['storage'].get('origins', []):
            try:
                return root / path.relative_to(Path(original))
            except ValueError:
                pass
    return path


def project_to_open(project):
    """A named workspace reopens its saved active actor/take."""
    project = Path(project).resolve()
    path = project / 'workspace_state.json'
    if not path.is_file():
        return project, None
    state = json.loads(path.read_text(encoding='utf-8'))
    active = (project / state.get('active', '.')).resolve()
    if not active.is_relative_to(project) or not (active / 'project.json').is_file():
        raise ValueError('The saved active project is missing or outside this folder.')
    return active, state


def inherit_storage(parent, child, state):
    """A new person/take keeps using its enclosing saved workspace."""
    if state.get('workspace_root'):
        from .project_workspace import inherit
        return inherit(parent,child,state)
    if state.get('storage'):
        root = storage_root(parent, state)
        import os
        state['storage'] = {**state['storage'], 'root': os.path.relpath(root, child)}
    return state
