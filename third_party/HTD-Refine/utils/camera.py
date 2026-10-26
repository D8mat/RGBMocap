"""Camera helpers used by HTD-Refine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

INTRINSIC_KEYS = ("K_fullimg", "camera_intrinsics", "intrinsics", "K")


def estimate_focal_length(img_w: int | float, img_h: int | float) -> float:
    return float((img_w**2 + img_h**2) ** 0.5)


def estimate_K(img_w: int | float, img_h: int | float) -> torch.Tensor:
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def normalize_camera_intrinsics(
    value: torch.Tensor | np.ndarray,
    length: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """Broadcast camera intrinsics to contiguous `[B,L,3,3]` tensors."""
    if length <= 0 or batch_size <= 0:
        raise ValueError(f"length and batch_size must be positive, got {length=} {batch_size=}")

    K = torch.as_tensor(value).float()
    if K.ndim == 2 and K.shape == (3, 3):
        K = K[None, None]
    elif K.ndim == 3 and K.shape[-2:] == (3, 3):
        K = K[None]
    elif K.ndim != 4 or K.shape[-2:] != (3, 3):
        raise ValueError(
            f"Camera intrinsics must have shape [3,3], [L,3,3], or [B,L,3,3], got {tuple(K.shape)}"
        )

    if K.shape[0] == 1:
        K = K.expand(batch_size, -1, -1, -1)
    elif K.shape[0] != batch_size:
        raise ValueError(f"Camera intrinsics batch mismatch: {K.shape[0]} != {batch_size}")
    if K.shape[1] == 1:
        K = K.expand(-1, length, -1, -1)
    elif K.shape[1] < length:
        raise ValueError(f"Camera intrinsics are shorter than the input: {K.shape[1]} < {length}")
    else:
        K = K[:, :length]

    if not torch.isfinite(K).all():
        raise ValueError("Camera intrinsics contain non-finite values.")
    if not ((K[..., 0, 0] > 0).all() and (K[..., 1, 1] > 0).all()):
        raise ValueError("Camera focal lengths must be positive.")
    return K.contiguous()


def load_camera_intrinsics(
    path: str | Path,
    length: int,
    batch_size: int = 1,
) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {path}")
    if path.suffix.lower() in {".npy", ".npz"}:
        value = np.load(path, allow_pickle=True)
        if isinstance(value, np.lib.npyio.NpzFile):
            archive = value
            try:
                value = {key: archive[key] for key in archive.files}
            finally:
                archive.close()
        elif isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
    else:
        value = torch.load(path, map_location="cpu")

    if isinstance(value, dict):
        for key in INTRINSIC_KEYS:
            if key in value:
                value = value[key]
                break
        else:
            raise KeyError(f"Camera intrinsics file has none of {INTRINSIC_KEYS}: {path}")

    try:
        return normalize_camera_intrinsics(value, length=length, batch_size=batch_size)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{exc} File: {path}") from exc


def create_camera_sensor(width=None, height=None, f_fullframe=None):
    if width is None or height is None:
        width, height = (1200, 1600) if np.random.rand() < 0.5 else (1600, 1200)
    if f_fullframe is None:
        f_fullframe = np.random.choice([24, 26, 28, 30, 35, 40, 50, 60, 70])
    diag_fullframe = (24**2 + 36**2) ** 0.5
    diag_img = (width**2 + height**2) ** 0.5
    focal_length = diag_img / diag_fullframe * f_fullframe
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    return width, height, K


def get_bbx_xyxy_from_xys(bbx_xys: torch.Tensor) -> torch.Tensor:
    center_x, center_y, size = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    return torch.stack(
        [center_x - size / 2, center_y - size / 2, center_x + size / 2, center_y + size / 2],
        dim=-1,
    )


__all__ = [
    "INTRINSIC_KEYS",
    "create_camera_sensor",
    "estimate_K",
    "estimate_focal_length",
    "get_bbx_xyxy_from_xys",
    "load_camera_intrinsics",
    "normalize_camera_intrinsics",
]
