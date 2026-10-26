"""PVA-Net visual encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.layers import DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from einops import rearrange

import pvanet.paths


class Block(nn.Module):
    """ViT encoder block matching the bundled ViTPose backbone."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, proj_drop=drop, qkv_bias=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTPoseEncoder(nn.Module):
    """ViTPose-derived encoder used by PVA-Net.

    The serialized model is loaded from `pvanet.pt`, which already contains these
    weights. The constructor builds the architecture without reading external
    weight files.
    """

    def __init__(
        self,
        img_size=(256, 192),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.3,
        last_norm=True,
        max_seq_len=4096,
    ):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.patch_embed.proj.padding = (2, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, drop_path=dpr[i]) for i in range(depth)])
        self.last_norm = nn.LayerNorm(embed_dim) if last_norm else nn.Identity()

    def forward(self, x, mask_keep_near=-1):
        """Encode `[B,L,3,256,192]` images into `[B,L,D,16,12]` features."""
        del mask_keep_near
        B, L, D, H, W = x.shape
        if H != 256 or W != 192:
            raise ValueError(f"ViTPoseEncoder expects 256x192 crops, got {H}x{W}")
        x = rearrange(x, "b l d h w -> (b l) d h w")
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        Hp, Wp = self.patch_embed.grid_size
        for block in self.blocks:
            x = block(x)
        x = self.last_norm(x)
        return rearrange(x, "(b l) (h w) d -> b l d h w", b=B, l=L, h=Hp, w=Wp).contiguous()


__all__ = ["Block", "ViTPoseEncoder"]
