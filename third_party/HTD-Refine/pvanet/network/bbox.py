"""Bounding-box helpers used by PVA-Net inference."""

from __future__ import annotations

import numpy as np
import torch


def convert_bbx_xyxy_to_bbx_xywh(bbx_xyxy):
    """Convert left/top/right/bottom boxes to left/top/width/height boxes."""
    width = bbx_xyxy[..., 2] - bbx_xyxy[..., 0]
    height = bbx_xyxy[..., 3] - bbx_xyxy[..., 1]
    if isinstance(bbx_xyxy, torch.Tensor):
        return torch.stack([bbx_xyxy[..., 0], bbx_xyxy[..., 1], width, height], dim=-1)
    return np.stack([bbx_xyxy[..., 0], bbx_xyxy[..., 1], width, height], axis=-1)


__all__ = ["convert_bbx_xyxy_to_bbx_xywh"]
