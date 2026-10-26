"""Set up the standalone editor before any data paths or workers are imported."""
from __future__ import annotations
import os
import sys
from pathlib import Path

def configure_portable_environment() -> Path | None:
    """Confine a frozen build to its folder; leave source/developer launches alone."""
    if not getattr(sys, "frozen", False):
        return None
    root = Path(sys.executable).resolve().parent
    os.environ.pop("PYTHONHOME", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.update(
        RGB_MOCAP_DATA=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(root / "cache" / "huggingface"),
        TORCH_HOME=str(root / "cache" / "torch"),
        XDG_CACHE_HOME=str(root / "cache"),
    )
    sys.dont_write_bytecode = True
    return root

