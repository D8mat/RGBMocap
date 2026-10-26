"""Installable helpers for HTD-Refine."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

__version__ = "0.2.0"

__all__ = ["PROJECT_ROOT", "__version__"]
