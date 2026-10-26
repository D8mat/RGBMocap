"""Compatibility module for serialized PVA-Net model checkpoints."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


__path__ = [str(_PACKAGE_ROOT)]


def __getattr__(name: str):
    if name == "PVANet":
        from pvanet.network import PVANet

        return PVANet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PVANet"]
