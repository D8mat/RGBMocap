"""Validate the isolated optional face overlay before importing native modules."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
folder = root / 'runtime_face'
lock = json.loads((root / 'installer/face-runtime-lock.json').read_text(encoding='utf-8'))
if Path(sys.executable).resolve() != (root / 'runtime/python.exe').resolve():
    raise RuntimeError('Face capture requires the managed shared Python.')
if '.'.join(map(str, sys.version_info[:3])) != lock['python']:
    raise RuntimeError('Unexpected face Python version. Run Setup_Face.cmd.')
receipt = json.loads((folder / 'RGB_MOCAP_FACE.json').read_text(encoding='utf-8'))
if receipt.get('packages') != lock['packages']:
    raise RuntimeError('Face runtime is stale. Run Setup_Face.cmd.')
sys.path[:0] = [str(folder), str(root)]
for name in ('numpy', 'cv2', 'mediapipe'):
    module = importlib.util.find_spec(name)
    if module is None or not Path(module.origin).resolve().is_relative_to(folder.resolve()):
        raise RuntimeError('Unexpected face package path: ' + name)
for spec in lock['packages']:
    name, version = spec.split('==')
    if importlib.metadata.version(name) != version:
        raise RuntimeError('Unexpected face dependency: ' + name)
sys.argv = ['face_worker_boot', 'face', *sys.argv[1:]]
from mocap.face_extract import main
raise SystemExit(main())
