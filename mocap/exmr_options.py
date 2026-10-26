"""Lightweight ExMR settings; safe to import in the editor process."""
import json

DEFAULTS = {'static_camera': True, 'smoothing': 'global'}


def options(value=None):
    value = {**DEFAULTS, **(value or {})}
    if type(value['static_camera']) is not bool or value['smoothing'] not in ('global', 'none'):
        raise ValueError('Invalid ExMR settings')
    return {key: value[key] for key in DEFAULTS}


def saved(project, state=None):
    state = state or json.loads((project/'project.json').read_text(encoding='utf-8'))
    return options(state.get('exmr_options'))
