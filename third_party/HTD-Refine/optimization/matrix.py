"""Minimal homogeneous-transform helpers for HTD-Refine IK."""

from __future__ import annotations

import torch


def get_rotation(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :3, :3]


def get_position(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :3, 3]


def get_mat_BtoA(matA: torch.Tensor, matB: torch.Tensor) -> torch.Tensor:
    """Return transform B expressed in coordinate frame A."""
    return torch.linalg.inv(matA) @ matB


def get_mat_BfromA(matA: torch.Tensor, matBtoA: torch.Tensor) -> torch.Tensor:
    """Compose world transform A with local transform B-to-A."""
    return matA @ matBtoA


def get_TRS(rot_mat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Build homogeneous transforms from rotations and translations."""
    mat = torch.eye(4, dtype=pos.dtype, device=pos.device).repeat(pos.shape[:-1] + (1, 1))
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    return mat


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp_min(eps).unsqueeze(-1)


def forward_kinematics(mat: torch.Tensor, parent: list[int]) -> torch.Tensor:
    """Forward kinematics over a joint axis at `-3`."""
    rotations = torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device).repeat(mat.shape[:-2] + (1, 1))
    for i in range(mat.shape[-3]):
        if parent[i] != -1:
            new_mat = get_mat_BfromA(rotations[..., parent[i], :, :], mat[..., i, :, :])
        else:
            new_mat = mat[..., i, :, :]
        rotations = torch.cat(
            (rotations[..., :i, :, :], new_mat[..., None, :, :], rotations[..., i + 1 :, :, :]), dim=-3
        )
    return rotations


__all__ = ["forward_kinematics", "get_TRS", "get_mat_BtoA", "get_position", "get_rotation", "normalize"]
