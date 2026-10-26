"""Positional encodings used by PVA-Net decoders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_encoding(d_model: int, max_seq_len: int = 4096) -> torch.Tensor:
    """Return standard sinusoidal positional encoding with shape [L, D]."""
    position = torch.arange(max_seq_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(max_seq_len, d_model)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class SincosPE(nn.Module):
    """Add fixed sinusoidal positional encoding over sequence length."""

    def __init__(self, d_model: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.register_buffer("encoding", get_encoding(d_model, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        del B
        if C != self.d_model:
            raise ValueError(f"Expected channel dim {self.d_model}, got {C}")
        encoding = get_encoding(C, L).to(x) if L > self.max_seq_len else self.encoding[:L]
        return x + encoding


__all__ = ["SincosPE", "get_encoding"]
