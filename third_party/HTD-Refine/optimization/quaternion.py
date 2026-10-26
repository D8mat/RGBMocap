"""Quaternion helpers for CCD IK."""

from __future__ import annotations

import torch


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)


def qinv(q: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    original_shape = q.shape
    terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qpow(q0: torch.Tensor, t, dtype=torch.float) -> torch.Tensor:
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., :1])
    close = ((theta0 <= 1e-9) & (theta0 >= -1e-9)).float()
    theta0 = (1 - close) * theta0 + close * 1e-9
    v0 = q0[..., 1:] / torch.sin(theta0)
    if isinstance(t, torch.Tensor):
        q = torch.zeros(t.shape + q0.shape, device=q0.device)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:
        q = torch.zeros(q0.shape, device=q0.device)
        theta = t * theta0
    q[..., :1] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta)
    return q.to(dtype)


def qslerp(q0: torch.Tensor, q1: torch.Tensor, t) -> torch.Tensor:
    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    return qmul(qpow(qmul(q1, qinv(q0)), t), q0)


def qbetween(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    v = torch.cross(v0, v1, dim=-1)
    w = torch.sqrt((v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(
        dim=-1, keepdim=True
    )
    y_vec = torch.zeros_like(v)
    y_vec[..., 1] = 1.0
    mask = torch.logical_and(v.norm(dim=-1) == 0, w.sum(dim=-1).abs() <= 1e-4)
    v[mask] = y_vec[mask]
    return qnormalize(torch.cat([w, v], dim=-1))


__all__ = ["qbetween", "qinv", "qmul", "qnormalize", "qrot", "qslerp"]
