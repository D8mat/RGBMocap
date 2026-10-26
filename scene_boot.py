"""Isolated CPU metadata helper for the portable editor."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mocap.runtime_limits import configure_preview_cpu
configure_preview_cpu()
runpy.run_module('mocap.scene_camera',run_name='__main__')
