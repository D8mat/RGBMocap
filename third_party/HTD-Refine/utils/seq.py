"""Sequence mask/interpolation helpers for demo preprocessing."""

from __future__ import annotations

import torch


def get_frame_id_list_from_mask(mask: torch.Tensor) -> list[torch.Tensor]:
    padded = torch.cat([torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)])
    diffs = torch.diff(padded.int())
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze()
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze()
    if starts.numel() == 0:
        return []
    if starts.numel() == 1:
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)
    return [torch.arange(start, end, device=mask.device) for start, end in zip(starts, ends)]


def rearrange_by_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    if M <= L or mask.sum() != L:
        raise ValueError(f"Cannot rearrange shape {tuple(x.shape)} by mask length {M} with sum {int(mask.sum())}")
    out = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    out[mask] = x
    return out


def frame_id_to_mask(frame_id: torch.Tensor, max_len: int) -> torch.Tensor:
    mask = torch.zeros(max_len, dtype=torch.bool, device=frame_id.device)
    mask[frame_id] = True
    return mask


def linear_interpolate_frame_ids(data: torch.Tensor, frame_id_list: list[torch.Tensor]) -> torch.Tensor:
    data = data.clone()
    for invalid_frame_ids in frame_id_list:
        if invalid_frame_ids[0] - 1 < 0:
            data[invalid_frame_ids] = data[invalid_frame_ids[-1] + 1].clone()
        elif invalid_frame_ids[-1] + 1 >= len(data):
            data[invalid_frame_ids] = data[invalid_frame_ids[0] - 1].clone()
        else:
            prev = data[invalid_frame_ids[0] - 1]
            next_value = data[invalid_frame_ids[-1] + 1]
            t = torch.linspace(0, 1, len(invalid_frame_ids) + 2, device=data.device, dtype=data.dtype)[1:-1]
            data[invalid_frame_ids] = t[:, None] * (next_value - prev)[None] + prev[None]
    return data


__all__ = ["frame_id_to_mask", "get_frame_id_list_from_mask", "linear_interpolate_frame_ids", "rearrange_by_mask"]
