"""Small network-layer helpers used by PVA-Net."""

from __future__ import annotations

import torch.nn as nn


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters of a module and return the module."""
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


__all__ = ["zero_module"]
