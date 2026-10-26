"""PVA-Net temporal modeling and feature aggregation layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

import pvanet.paths

from .rope import EncoderRoPEBlock


class TemporalLayer(nn.Module):
    """Temporal RoPE module with eight temporal blocks and four spatial blocks interleaved between them by default."""

    def __init__(
        self,
        out_channels=1280,
        in_channels=1280,
        temporal_layer=8,
        spatial_layer=4,
        num_heads=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transformer_type = "rope_per_loc_alternating"
        self.temporal_blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    hidden_size=in_channels,
                    num_heads=num_heads,
                    rope_type="1d",
                    max_seq_len=512,
                    mlp_ratio=2.0,
                )
                for _ in range(temporal_layer)
            ]
        )
        self.spatial_blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    hidden_size=in_channels,
                    num_heads=num_heads,
                    rope_type="axial3d_learnable",
                    mlp_ratio=2.0,
                    max_time_len=8,
                    max_h_len=64,
                    max_w_len=64,
                )
                for _ in range(spatial_layer)
            ]
        )
        for block in self.spatial_blocks:
            nn.init.zeros_(block.gate_msa)
            nn.init.zeros_(block.gate_mlp)
        self.alt_schedule = self._build_alt_schedule(temporal_layer, spatial_layer)

    @staticmethod
    def _build_alt_schedule(n_temporal, n_spatial):
        """Return an interleaved execution order of temporal/spatial blocks."""
        if n_spatial <= 0:
            return [("t", i) for i in range(n_temporal)]
        schedule, si = [], 0
        stride = max(n_temporal // n_spatial, 1)
        for ti in range(n_temporal):
            schedule.append(("t", ti))
            if (ti + 1) % stride == 0 and si < n_spatial:
                schedule.append(("s", si))
                si += 1
        while si < n_spatial:
            schedule.append(("s", si))
            si += 1
        return schedule

    @staticmethod
    def _apply_temporal_block(x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        """Apply one per-location temporal RoPE block."""
        B, L, D, H, W = x.shape
        x = rearrange(x, "b l d h w -> (b h w) l d", b=B, h=H, w=W)
        x = block(x)
        return rearrange(x, "(b h w) l d -> b l d h w", b=B, h=H, w=W)

    @staticmethod
    def _apply_spatial_block(x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        """Apply one per-frame spatial RoPE block."""
        B, L, D, H, W = x.shape
        x = rearrange(x, "b l d h w -> (b l) (h w) d", b=B, l=L, h=H, w=W)
        x = block(x, time_len=1, h_len=H, w_len=W)
        return rearrange(x, "(b l) (h w) d -> b l d h w", b=B, l=L, h=H, w=W)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """Run alternating temporal/spatial RoPE blocks."""
        if x.ndim != 5:
            raise ValueError(f"TemporalLayer expects a 5-D tensor, got shape {tuple(x.shape)}")
        if x.shape[2] != self.in_channels:
            raise ValueError(f"TemporalLayer expects channel dim {self.in_channels}, got {x.shape[2]}")

        intermediates = [x] if return_intermediates else None
        schedule = getattr(
            self,
            "alt_schedule",
            self._build_alt_schedule(len(self.temporal_blocks), len(self.spatial_blocks)),
        )
        for kind, idx in schedule:
            if kind == "t":
                x = self._apply_temporal_block(x, self.temporal_blocks[idx])
            elif kind == "s":
                x = self._apply_spatial_block(x, self.spatial_blocks[idx])
                if intermediates is not None:
                    intermediates.append(x)
            else:
                raise ValueError(f"Invalid TemporalLayer schedule entry: {(kind, idx)!r}")

        if return_intermediates:
            return x, intermediates
        return x


class TemporalFeatureAggregation(nn.Module):
    """Gated aggregation over intermediate temporal features."""

    def __init__(self, num_reads: int = 5, channels: int = 1280):
        super().__init__()
        self.num_reads = num_reads
        self.channels = channels
        self.lns = nn.ModuleList([nn.LayerNorm(channels) for _ in range(num_reads)])
        self.logits = nn.Parameter(torch.zeros(num_reads))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_list, x_last):
        if len(x_list) != self.num_reads:
            raise ValueError(f"TemporalFeatureAggregation expected {self.num_reads} reads, got {len(x_list)}")
        alpha = self.logits.softmax(dim=0)
        agg = 0
        for i, xi in enumerate(x_list):
            xt = xi.permute(0, 1, 3, 4, 2)
            xn = self.lns[i](xt)
            xn = xn.permute(0, 1, 4, 2, 3)
            agg = agg + alpha[i] * xn
        return x_last + self.gate * agg


__all__ = ["TemporalLayer", "TemporalFeatureAggregation"]
