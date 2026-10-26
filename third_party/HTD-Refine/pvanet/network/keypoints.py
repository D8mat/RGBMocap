"""Heatmap keypoint decoding helpers for PVA-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_opencv_1d_gaussian_kernel(kernel_size: int, sigma: float = 0) -> torch.Tensor:
    """Return the OpenCV-compatible 1D Gaussian kernel used by DARK refinement."""
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if sigma <= 0 and kernel_size <= 9:
        presets = {
            1: [1.0],
            3: [0.25, 0.5, 0.25],
            5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
            7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
            9: [4 / 256, 13 / 256, 30 / 256, 51 / 256, 60 / 256, 51 / 256, 30 / 256, 13 / 256, 4 / 256],
        }
        return torch.tensor(presets[kernel_size]).unsqueeze(-1)

    if sigma <= 0:
        sigma = 0.15 * kernel_size + 0.35
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).unsqueeze(-1)


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply depthwise 2D Gaussian blur to the last two dimensions."""
    kernel1d = get_opencv_1d_gaussian_kernel(kernel_size=kernel_size)
    kernel2d = (kernel1d @ kernel1d.transpose(-1, -2))[None, None].to(x)
    x_shape = x.shape[:-2]
    H, W = x.shape[-2:]
    x = x.reshape(-1, 1, H, W)
    x = F.conv2d(x, kernel2d, padding=kernel_size // 2)
    return x.reshape(*x_shape, H, W)


def get_max_score_kpts_from_heatmaps(heatmaps: torch.Tensor, thr: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return max-confidence heatmap locations and scores."""
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    scores = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = torch.tile(idx, (1, 1, 2)).float()
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = torch.where(torch.tile(scores, (1, 1, 2)) > thr, preds, -1)
    return preds, scores


def refine_kpt_DARK_v2(coords: torch.Tensor, heatmaps: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """Refine argmax keypoints with DARK second-order heatmap offsets."""
    heatmaps = gaussian_blur2d(heatmaps.clone(), kernel)
    heatmaps = torch.clamp(heatmaps, 0.001, 50).log()
    B, K, H, W = heatmaps.shape

    heatmaps_pad = F.pad(heatmaps, (1, 1, 1, 1), mode="replicate").flatten()
    rounded = torch.round(coords).long()
    x = rounded[..., 0].clamp(0, W - 1)
    y = rounded[..., 1].clamp(0, H - 1)
    index = x + 1 + (y + 1) * (W + 2)
    index += (W + 2) * (H + 2) * torch.arange(0, B * K, device=coords.device).view(-1, K)
    index = index.long().view(-1, 1)

    center = heatmaps_pad[index]
    right = heatmaps_pad[index + 1]
    down = heatmaps_pad[index + (W + 2)]
    right_down = heatmaps_pad[index + (W + 3)]
    left_up = heatmaps_pad[index - (W + 3)]
    right_up = heatmaps_pad[index + 1 - (W + 2)]
    left_down = heatmaps_pad[index - 1 + (W + 2)]

    dx = 0.5 * (right - heatmaps_pad[index - 1])
    dy = 0.5 * (down - heatmaps_pad[index - (W + 2)])
    dxx = right - 2 * center + heatmaps_pad[index - 1]
    dyy = down - 2 * center + heatmaps_pad[index - (W + 2)]
    dxy = 0.25 * (right_down + left_up - right_up - left_down)

    hessian = torch.stack([dxx, dxy, dxy, dyy], dim=1).view(B, K, 2, 2)
    hessian = hessian + 1e-3 * torch.eye(2, device=coords.device)
    hessian_inv = torch.linalg.inv(hessian)
    derivative = torch.cat([dx, dy], dim=1).view(B, K, 2)
    coords -= torch.einsum("bkmn,bkn->bkm", hessian_inv, derivative)
    return coords


__all__ = ["get_max_score_kpts_from_heatmaps", "refine_kpt_DARK_v2"]
