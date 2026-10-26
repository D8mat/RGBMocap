"""Project paths and import setup for PVA-Net inference."""

from __future__ import annotations

import sys
from pathlib import Path

PVA_NET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PVA_NET_ROOT.parent

for path in (PVA_NET_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

__all__ = ["PROJECT_ROOT", "PVA_NET_ROOT"]
