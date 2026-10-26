"""RoPE transformer blocks used by PVA-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


def apply_rotary_emb(cos: torch.Tensor, sin: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    rot_dim = cos.shape[-1]
    if rot_dim > tensor.shape[-1]:
        raise ValueError(f"feature dim {tensor.shape[-1]} < rotary dim {rot_dim}")
    if rot_dim == tensor.shape[-1]:
        return tensor * cos + rotate_half(tensor) * sin
    tensor_rot, tensor_pass = tensor[..., :rot_dim], tensor[..., rot_dim:]
    tensor_rot = tensor_rot * cos + rotate_half(tensor_rot) * sin
    return torch.cat([tensor_rot, tensor_pass], dim=-1)


def get_rope_encoding(d_model: int, max_seq_len: int = 4096) -> torch.Tensor:
    positions = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    angles = torch.einsum("i,j->ij", positions, freqs)
    return repeat(angles, "i j -> i (j r)", r=2)


def _init_axis_freqs(d_axis: int) -> torch.Tensor:
    return 1.0 / (10000 ** (torch.arange(0, d_axis, 2).float() / d_axis))


class ROPE(nn.Module):
    """1D rotary positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        angles = get_rope_encoding(d_model, max_seq_len)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def _expand(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        angles = get_rope_encoding(self.d_model, seq_len).to(device)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, d_model = x.shape[-2:]
        if d_model != self.d_model:
            raise ValueError(f"Expected rotary dim {self.d_model}, got {d_model}")
        if seq_len > self.max_seq_len:
            cos, sin = self._expand(seq_len, x.device, x.dtype)
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)
        return apply_rotary_emb(cos, sin, x)


class AxialROPE3DLearnable(nn.Module):
    """Learnable axial 3D RoPE over flattened [T, H, W] tokens."""

    def __init__(
        self,
        d_model: int,
        d_time: int | None = None,
        d_h: int | None = None,
        d_w: int | None = None,
        max_time_len: int = 512,
        max_h_len: int = 64,
        max_w_len: int = 64,
    ):
        super().__init__()
        if d_time is None:
            d_time = d_model // 2
            d_h = (d_model - d_time) // 2
            d_w = d_model - d_time - d_h
        assert d_h is not None and d_w is not None
        if d_time % 2 or d_h % 2 or d_w % 2:
            raise ValueError(f"axis dims must be even, got T={d_time}, H={d_h}, W={d_w}")
        if d_time + d_h + d_w != d_model:
            raise ValueError(f"axis split {d_time}+{d_h}+{d_w} != {d_model}")

        self.d_model = d_model
        self.d_time = d_time
        self.d_h = d_h
        self.d_w = d_w
        self.max_time_len = max_time_len
        self.max_h_len = max_h_len
        self.max_w_len = max_w_len
        self.freqs_t = nn.Parameter(_init_axis_freqs(d_time))
        self.freqs_h = nn.Parameter(_init_axis_freqs(d_h))
        self.freqs_w = nn.Parameter(_init_axis_freqs(d_w))
        self.register_buffer("pos_t", torch.arange(max_time_len).float(), persistent=False)
        self.register_buffer("pos_h", torch.arange(max_h_len).float(), persistent=False)
        self.register_buffer("pos_w", torch.arange(max_w_len).float(), persistent=False)

    @staticmethod
    def _axis_cos_sin(pos: torch.Tensor, freqs: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        angles = pos[:, None] * freqs[None, :]
        angles = repeat(angles, "i j -> i (j r)", r=2)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def rotate(self, x: torch.Tensor, time_len: int, h_len: int, w_len: int) -> torch.Tensor:
        B, NH, N, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected rotary dim {self.d_model}, got {D}")
        if N != time_len * h_len * w_len:
            raise ValueError(f"{N} != {time_len}*{h_len}*{w_len}")
        if time_len > self.max_time_len or h_len > self.max_h_len or w_len > self.max_w_len:
            raise ValueError(f"RoPE grid {(time_len, h_len, w_len)} exceeds configured maximum")

        x_t = x[..., : self.d_time].view(B, NH, time_len, h_len, w_len, self.d_time)
        x_h = x[..., self.d_time : self.d_time + self.d_h].view(B, NH, time_len, h_len, w_len, self.d_h)
        x_w = x[..., self.d_time + self.d_h :].view(B, NH, time_len, h_len, w_len, self.d_w)

        cos_t, sin_t = self._axis_cos_sin(self.pos_t[:time_len], self.freqs_t, x.dtype)
        cos_h, sin_h = self._axis_cos_sin(self.pos_h[:h_len], self.freqs_h, x.dtype)
        cos_w, sin_w = self._axis_cos_sin(self.pos_w[:w_len], self.freqs_w, x.dtype)
        x_t = x_t * cos_t[None, None, :, None, None, :] + rotate_half(x_t) * sin_t[None, None, :, None, None, :]
        x_h = x_h * cos_h[None, None, None, :, None, :] + rotate_half(x_h) * sin_h[None, None, None, :, None, :]
        x_w = x_w * cos_w[None, None, None, None, :, :] + rotate_half(x_w) * sin_w[None, None, None, None, :, :]
        return torch.cat(
            [
                x_t.view(B, NH, N, self.d_time),
                x_h.view(B, NH, N, self.d_h),
                x_w.view(B, NH, N, self.d_w),
            ],
            dim=-1,
        )


class RoPEAttention(nn.Module):
    """Multi-head self-attention with 1D or axial 3D RoPE."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_type: str = "1d",
        max_seq_len: int = 4096,
        max_time_len: int = 512,
        max_h_len: int = 64,
        max_w_len: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope_type = rope_type
        self.dropout_p = dropout
        if rope_type == "1d":
            self.rope = ROPE(self.head_dim, max_seq_len=max_seq_len)
        elif rope_type == "axial3d_learnable":
            self.rope = AxialROPE3DLearnable(
                self.head_dim,
                max_time_len=max_time_len,
                max_h_len=max_h_len,
                max_w_len=max_w_len,
            )
        else:
            raise ValueError(f"Unknown rope_type: {rope_type}")
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _apply_rope(self, xq, xk, time_len=None, h_len=None, w_len=None):
        if self.rope_type == "1d":
            return self.rope.rotate_queries_or_keys(xq), self.rope.rotate_queries_or_keys(xk)
        if time_len is None or h_len is None or w_len is None:
            raise ValueError("axial3d_learnable RoPE requires time_len, h_len and w_len")
        return self.rope.rotate(xq, time_len, h_len, w_len), self.rope.rotate(xk, time_len, h_len, w_len)

    @staticmethod
    def _build_sdpa_mask(attn_mask, key_padding_mask, B: int, L: int, device: torch.device):
        if attn_mask is None and key_padding_mask is None:
            return None
        blocked = torch.zeros(B, 1, L, L, dtype=torch.bool, device=device)
        if attn_mask is not None:
            blocked = blocked | attn_mask.to(torch.bool).view(1, 1, L, L)
        if key_padding_mask is not None:
            blocked = blocked | key_padding_mask.to(torch.bool).view(B, 1, 1, L)
        return ~blocked

    def forward(self, x, attn_mask=None, key_padding_mask=None, time_len=None, h_len=None, w_len=None):
        B, L, _ = x.shape
        xq = self.query(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.key(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        xv = self.value(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        xq, xk = self._apply_rope(xq, xk, time_len=time_len, h_len=h_len, w_len=w_len)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=self._build_sdpa_mask(attn_mask, key_padding_mask, B, L, xq.device),
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.proj(output.transpose(1, 2).reshape(B, L, -1))


class EncoderRoPEBlock(nn.Module):
    """Transformer encoder block with gated RoPE attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        rope_type: str = "1d",
        max_seq_len: int = 4096,
        max_time_len: int = 512,
        max_h_len: int = 64,
        max_w_len: int = 64,
        **block_kwargs,
    ):
        super().__init__()
        del block_kwargs
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(
            hidden_size,
            num_heads,
            dropout,
            rope_type=rope_type,
            max_seq_len=max_seq_len,
            max_time_len=max_time_len,
            max_h_len=max_h_len,
            max_w_len=max_w_len,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=dropout
        )
        self.gate_msa = nn.Parameter(torch.ones(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.ones(1, 1, hidden_size))

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None, time_len=None, h_len=None, w_len=None):
        x = x + self.gate_msa * self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            time_len=time_len,
            h_len=h_len,
            w_len=w_len,
        )
        return x + self.gate_mlp * self.mlp(self.norm2(x))


__all__ = ["AxialROPE3DLearnable", "EncoderRoPEBlock", "ROPE", "RoPEAttention"]
