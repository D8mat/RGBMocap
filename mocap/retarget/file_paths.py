"""Filesystem paths for the bundled Blender Python runtime on Windows."""
import os


def io_path(path):
    """Use extended paths for Python I/O, without changing stored provenance."""
    value = os.path.abspath(os.fspath(path))
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return value
    if value.startswith('\\\\'):
        return '\\\\?\\UNC\\' + value[2:]
    return '\\\\?\\' + value


def save_blend(path):
    """Save Blender files despite its native long-path rename limitation."""
    import bpy
    value = os.path.abspath(os.fspath(path))
    if os.name != 'nt' or len(value) < 240:
        bpy.ops.wm.save_as_mainfile(filepath=value)
        return
    import shutil
    import tempfile
    import uuid
    # Blender can read an extended path but its save/backup rename cannot.
    # Keep packed resources and existing relative references unchanged, then
    # publish through a sibling temporary file (also safe across volumes).
    pending = io_path(value + '.' + uuid.uuid4().hex + '.partial')
    try:
        with tempfile.TemporaryDirectory(prefix='rgbmocap-blend-') as stage:
            saved = os.path.join(stage, 'animation.blend')
            bpy.ops.wm.save_as_mainfile(filepath=saved, relative_remap=False)
            shutil.copyfile(saved, pending)
            os.replace(pending, io_path(value))
    finally:
        if os.path.exists(pending):
            os.unlink(pending)
