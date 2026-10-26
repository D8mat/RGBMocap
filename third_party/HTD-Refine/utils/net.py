"""Small tensor utilities for HTD-Refine demos."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def to_device(data, device: str | torch.device):
    """Recursively move tensors in common Python containers to a torch device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [to_device(value, device) for value in data]
    return data


def moving_average_smooth(x: torch.Tensor, window_size: int = 5, dim: int = -1) -> torch.Tensor:
    kernel = torch.ones(window_size).float() / window_size
    kernel = kernel[None, None].to(x)
    radius = kernel.size(-1) // 2
    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")
    x = F.pad(x[None], (radius, radius, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel)
    x = x.squeeze(1).reshape(*x_shape, -1)
    return x.transpose(-1, dim)


__all__ = ["moving_average_smooth", "to_device"]
