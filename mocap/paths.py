from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("RGB_MOCAP_DATA", ROOT)).resolve()
MODELS = HOME / "models"
PROJECTS = HOME / "projects"
UPSTREAM = ROOT / "third_party"
COMPAT = ROOT / "native_compat"


def worker_command(stage: str, *args: str) -> list[str]:
    from .runtime_registry import command
    return command(ROOT, stage, args, frozen=bool(getattr(sys, "frozen", False)))
