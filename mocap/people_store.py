"""Small, relocatable person manifests; no inference imports on the GUI thread."""
from __future__ import annotations
import colorsys
import json
import secrets
from pathlib import Path
from .storage import atomic_json

VERSION = 1
DEFAULT_COLOR = '#b5b5b5'


def read_people_json(path):
    """A Windows reader must allow the GUI's atomic manifest replacement."""
    import os
    if os.name!='nt':return json.loads(path.read_text(encoding='utf-8'))
    import ctypes,msvcrt
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    handle=create(str(path),0x80000000,0x1|0x2|0x4,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except Exception:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(handle);raise
    with os.fdopen(descriptor,'r',encoding='utf-8') as source:return json.load(source)


def person_color(number):
    if number == 1:
        return DEFAULT_COLOR
    hue = secrets.randbelow(10000) / 10000
    return '#' + ''.join(f'{round(c*255):02x}' for c in colorsys.hsv_to_rgb(hue, .52, .86))


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Invalid person project path')
    return path


def media_receipt(root):
    stat = (Path(root) / 'input.mp4').stat()
    return [stat.st_size, stat.st_mtime_ns]


def load_people(root):
    root = Path(root)
    path = root / 'people.json'
    if not path.is_file():
        return None
    state = read_people_json(path)
    if state.get('version') != VERSION or state.get('input') != media_receipt(root):
        raise ValueError('Person detection is out of date. Run Find person again.')
    if not isinstance(state.get('people'), list) or len(state['people']) > 256:
        raise ValueError('Invalid person list')
    ids = set()
    for person in state['people']:
        ident = person['id']
        if not isinstance(ident, int) or ident < 1 or ident in ids:
            raise ValueError('Invalid person identity')
        ids.add(ident)
        color = person['color']
        if len(color) != 7 or color[0] != '#':
            raise ValueError('Invalid person color')
        int(color[1:], 16)
        safe_child(root, person['preview'])
        for take in person.get('takes', []):
            safe_child(root, take['project'])
    return state


def save_people(root, state):
    atomic_json(Path(root) / 'people.json', state)


def root_for(project, manifest=None):
    project = Path(project).resolve()
    if manifest is None:
        manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    relative = manifest.get('people_root')
    if relative is None:
        return project
    root = (project / relative).resolve()
    # Actor projects must remain descendants of their group after relocation.
    if root == project or not project.is_relative_to(root):
        raise ValueError('Invalid person group')
    return root


def visible_runs(indices, fps, first, end, max_gap_seconds=.5):
    """Split long absences instead of fabricating full-body motion across them."""
    gap = max(1, round(float(fps) * max_gap_seconds))
    start = previous = None
    runs = []
    count = 0
    for raw in indices:
        index = int(raw)
        if not first <= index < end:
            continue
        if previous is not None and index - previous > gap + 1:
            if previous + 1 - start >= 4 and count >= 3:
                runs.append((start, previous + 1))
            start = None
            count = 0
        if start is None:
            start = index
        previous = index
        count += 1
    if start is not None and previous + 1 - start >= 4 and count >= 3:
        runs.append((start, previous + 1))
    return runs
