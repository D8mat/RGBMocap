"""Optional face process, independent of the locked base runtime registry."""
from .paths import ROOT


def command(args):
    python = ROOT / 'runtime' / 'python.exe'
    if not python.is_file() or not (ROOT / 'runtime_face' / 'RGB_MOCAP_FACE.json').is_file():
        raise RuntimeError('Face capture is missing. Run Setup_Face.cmd.')
    return [str(python), '-I', '-X', 'utf8', '-B', str(ROOT / 'face_worker_boot.py'), *map(str, args)]
