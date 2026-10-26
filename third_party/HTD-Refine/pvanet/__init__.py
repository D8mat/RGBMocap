"""Installable PVA-Net inference package for HTD-Refine."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "PVANet":
        from pvanet.network import PVANet

        return PVANet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PVANet"]
