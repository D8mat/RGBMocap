"""A small PEAR support profile over the existing read-only shared runtime."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
from .paths import ROOT

@lru_cache(maxsize=1)
def identity():
    names=['installer/pear-sources.json','installer/pear_adapters.py','tools/setup_pear.py','mocap/runtime_registry.py']
    return hashlib.sha256(b''.join((ROOT/name).read_bytes() for name in names)).hexdigest()

def base_identity():
    data=json.loads((ROOT/'SETUP_RECEIPT.json').read_text(encoding='utf8'))
    value=data['group_ids']['shared']
    if not isinstance(value,str) or not value:raise ValueError('Shared runtime receipt is unavailable')
    return value


def configuration():
    data=json.loads((ROOT/'.setup/pear-runtime.json').read_text(encoding='utf8'))
    folder=(ROOT/data['folder']).resolve()
    if not folder.is_relative_to((ROOT/'runtime_pear').resolve()) or data.get('identity')!=identity() or data.get('base_identity')!=base_identity():raise ValueError('PEAR support profile needs installation')
    if not isinstance(data.get('files'),dict) or not {'engine/inference_images.py','packages/roma/__init__.py','packages/pytorch3d/transforms/rotation_conversions.py'}<=data['files'].keys():raise ValueError('Incomplete PEAR support inventory')
    for name,stamp in data['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder):raise ValueError('Invalid PEAR support path')
        st=p.stat()
        if [st.st_size,st.st_mtime_ns]!=stamp:raise ValueError('PEAR support files changed; rerun install.cmd')
    return folder

def engine_folder():return configuration()/'engine'

def status():
    try:configuration();return True,'PEAR support files ready · shared runtime'
    except (OSError,ValueError,KeyError,TypeError):return False,'Run install.cmd to prepare PEAR support, then prepare models in Model Manager.'

def activate():
    import sys,os
    folder=configuration()
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(folder/'packages'),str(folder/'engine')]
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
    return folder/'engine'
