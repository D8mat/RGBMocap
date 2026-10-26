"""Geometry helpers required by the HTD-Refine optimizer."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def apply_T_on_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply homogeneous transforms to point tensors."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    extra_dims = (points.ndim - 1) - (T.ndim - 2)
    for _ in range(max(extra_dims, 0)):
        R = R.unsqueeze(-3)
        t = t.unsqueeze(-2)
    return (R @ points.unsqueeze(-1)).squeeze(-1) + t


def perspective_projection(points: torch.Tensor, K: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Project camera-space 3D points with a pinhole intrinsic matrix."""
    points2d = points[..., :2] / points[..., 2:].clamp(min=eps)
    R = K[..., :2, :2]
    t = K[..., :2, 2]
    extra_dims = (points2d.ndim - 1) - (K.ndim - 2)
    for _ in range(max(extra_dims, 0)):
        R = R.unsqueeze(-3)
        t = t.unsqueeze(-2)
    return (R @ points2d.unsqueeze(-1)).squeeze(-1) + t


def get_new_transl(T_w2c: torch.Tensor, transl_world: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    R_w2c = T_w2c[..., :3, :3]
    t_w2c = T_w2c[..., :3, 3]
    return torch.einsum("fij,fj->fi", R_w2c, transl_world + offset) + t_w2c - offset


def scale_camera_trajectory(
    T_w2c: torch.Tensor,
    scale: torch.Tensor | float,
    axes: tuple[int, ...] = (0, 2),
) -> torch.Tensor:
    """Scale camera-center displacement along selected world-coordinate axes."""
    T_c2w = torch.linalg.inv(T_w2c)
    if T_c2w.ndim != 3 or T_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected transform shape (L, 4, 4), got {tuple(T_c2w.shape)}")
    if T_c2w.shape[0] == 0:
        return T_w2c.clone()

    T_c2w_scaled = T_c2w.clone()
    delta = T_c2w[:, :3, 3] - T_c2w[:1, :3, 3]
    delta[:, list(axes)] = delta[:, list(axes)] * scale
    T_c2w_scaled[:, :3, 3] = T_c2w[:1, :3, 3] + delta
    return torch.linalg.inv(T_c2w_scaled)


def T_w2c_from_R_t_oneseq(R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    R_w2c = R_c2w.transpose(-2, -1)
    t_w2c = -torch.matmul(R_w2c, t_c2w.unsqueeze(-1)).squeeze(-1)

    transform = torch.zeros(R_c2w.shape[0], 4, 4, dtype=R_c2w.dtype, device=R_c2w.device)
    transform[:, :3, :3] = R_w2c
    transform[:, :3, 3] = t_w2c
    transform[:, 3, 3] = 1.0
    return transform


def solve_pnp(j2d: torch.Tensor, j3d: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = j3d.device
    j3d_np = np.ascontiguousarray(j3d.detach().cpu().numpy().astype(np.float64))
    j2d_np = np.ascontiguousarray(j2d.detach().cpu().numpy().astype(np.float64))
    K_np = np.ascontiguousarray(K.detach().cpu().numpy().astype(np.float64))
    dist_coeffs = np.zeros((8, 1), dtype=np.float64)

    transforms = []
    rvecs = []
    tvecs = []
    rvec_prev = None
    tvec_prev = None
    for i in range(j3d_np.shape[0]):
        if i == 0:
            _, rvec, tvec = cv2.solvePnP(
                objectPoints=j3d_np[i],
                imagePoints=j2d_np[i],
                cameraMatrix=K_np[i],
                distCoeffs=dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
        else:
            _, rvec, tvec = cv2.solvePnP(
                j3d_np[i],
                j2d_np[i],
                K_np[i],
                dist_coeffs,
                rvec=rvec_prev,
                tvec=tvec_prev,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        rvec_prev = rvec
        tvec_prev = tvec
        rotation = cv2.Rodrigues(rvec)[0]
        pose = np.concatenate([rotation, tvec], axis=-1)
        transform = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        transforms.append(torch.from_numpy(transform.astype(np.float32)).to(device))
        rvecs.append(torch.from_numpy(rvec.astype(np.float32)).to(device))
        tvecs.append(torch.from_numpy(tvec.astype(np.float32)).to(device))

    return torch.stack(transforms), torch.stack(rvecs), torch.stack(tvecs)


__all__ = [
    "apply_T_on_points",
    "perspective_projection",
    "T_w2c_from_R_t_oneseq",
    "get_new_transl",
    "scale_camera_trajectory",
    "solve_pnp",
]
