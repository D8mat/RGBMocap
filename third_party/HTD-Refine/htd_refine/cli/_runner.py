"""Run project scripts through installed console entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


def run_script(script_name: str) -> None:
    script_path = SCRIPTS_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"HTD-Refine script not found: {script_path}")
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    namespace = runpy.run_path(str(script_path), run_name=f"htd_refine.cli.{script_path.stem.replace('-', '_')}")
    main = namespace.get("main")
    if main is None:
        raise RuntimeError(f"HTD-Refine script has no main(): {script_path}")
    main()
