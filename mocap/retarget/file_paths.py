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
