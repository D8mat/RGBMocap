"""Spatial position encoding helpers for PVA-Net."""

from __future__ import annotations

import torch


def get_spatial_cliff_pe(bbx_xyxy: torch.Tensor, K_fullimg: torch.Tensor, out_size=(16, 12)) -> torch.Tensor:
    """Return CLIFF-style image-plane coordinates normalized by focal length."""
    H_out, W_out = out_size
    fl = K_fullimg[..., 0, 0][..., None, None, None]
    icx = K_fullimg[..., 0, 2][..., None, None]
    icy = K_fullimg[..., 1, 2][..., None, None]

    x1 = bbx_xyxy[..., 0].unsqueeze(-1)
    y1 = bbx_xyxy[..., 1].unsqueeze(-1)
    x2 = bbx_xyxy[..., 2].unsqueeze(-1)
    y2 = bbx_xyxy[..., 3].unsqueeze(-1)

    grid_x_rel = torch.linspace(0, 1, steps=W_out, device=bbx_xyxy.device)
    grid_y_rel = torch.linspace(0, 1, steps=H_out, device=bbx_xyxy.device)
    grid_x = x1 + (x2 - x1) * grid_x_rel
    grid_y = y1 + (y2 - y1) * grid_y_rel

    prefix = list(grid_x.shape)[:-1]
    grid_x = grid_x.unsqueeze(-2).expand(*prefix, H_out, W_out)
    grid_y = grid_y.unsqueeze(-1).expand(*prefix, H_out, W_out)
    return torch.stack((grid_x - icx, grid_y - icy), dim=-1) / fl


__all__ = ["get_spatial_cliff_pe"]
