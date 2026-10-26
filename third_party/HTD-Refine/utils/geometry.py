"""Geometry helpers shared by HTD-Refine demos."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import einsum


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    bottom = torch.zeros((*R.shape[:-2], 1, 4), dtype=R.dtype, device=R.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat([torch.cat([R, t[..., None]], dim=-1), bottom], dim=-2)


def apply_T_on_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    extra_dims = (points.ndim - 1) - (T.ndim - 2)
    for _ in range(max(extra_dims, 0)):
        R = R.unsqueeze(-3)
        t = t.unsqueeze(-2)
    return (R @ points.unsqueeze(-1)).squeeze(-1) + t


def compute_root_facing_transform(joints: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Build a ground-aligned transform whose positive z-axis faces forward."""
    t = joints[:, 0, :].detach().clone()
    t[:, 1] = 0
    rl_xz = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]] + joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]
    identity_mask = rl_xz.pow(2).sum(-1) < 1e-4
    x_dir = torch.zeros_like(t)
    x_dir[:, [0, 2]] = F.normalize(rl_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R = torch.stack([x_dir, y_dir, z_dir], dim=-1)
    R[identity_mask] = torch.eye(3, dtype=R.dtype, device=R.device)
    if inverse:
        R_inv = R.transpose(1, 2)
        t_inv = -einsum(R, t, "b i j, b i -> b j")
        return transform_mat(R_inv, t_inv)
    return transform_mat(R, t)


def move_to_start_point_face_z(
    verts: torch.Tensor, J_regressor: torch.Tensor, y_offset: str = "min_first"
) -> torch.Tensor:
    verts = verts.clone()
    offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]
    if y_offset == "min_all":
        offset[1] = verts[:, :, [1]].min()
    elif y_offset == "min_first":
        offset[1] = verts[[0], :, [1]].min()
    else:
        raise NotImplementedError(f"unsupported y_offset {y_offset!r}")
    verts = verts - offset
    facing_transform = compute_root_facing_transform(
        einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"),
        inverse=True,
    )
    return apply_T_on_points(verts, facing_transform)


__all__ = ["apply_T_on_points", "compute_root_facing_transform", "move_to_start_point_face_z", "transform_mat"]
