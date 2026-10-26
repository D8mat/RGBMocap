"""PVA-Net keypoint, velocity, and acceleration decoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

import pvanet.paths

from .positional import SincosPE
from .rope import EncoderRoPEBlock


class KeypointsDecoder(nn.Module):
    def __init__(
        self,
        in_channels=768,
        out_channels=17,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        freeze=False,
        pred_type="coco17",
    ):
        super().__init__()
        self.pred_type = pred_type
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
        self.final_layer = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if freeze:
            for param in self.deconv_layers.parameters():
                param.requires_grad = False

            for param in self.final_layer.parameters():
                param.requires_grad = False

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Build the transposed-convolution keypoint head."""
        deconv_cfg = {
            4: (4, 1, 0),
            3: (3, 1, 1),
            2: (2, 0, 0),
        }

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = deconv_cfg[num_kernels[i]]
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x


class VelocityDecoder(nn.Module):
    def __init__(
        self,
        in_channels=768,
        use_abs_pe=False,
        abs_pe_max_seq_len=4096,
        use_tconv=False,
        tconv_kernel=5,
        use_dt_embed=False,
        dt_hidden=64,
        num_out_vecs=17,
    ):
        super().__init__()
        self.use_abs_pe = use_abs_pe
        self.use_dt_embed = use_dt_embed
        self.num_out_vecs = num_out_vecs
        self.out_channels = num_out_vecs * 3

        self.use_tconv = use_tconv
        if use_tconv:
            self.tconv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(tconv_kernel, 1, 1),
                padding=(tconv_kernel // 2, 0, 0),
                groups=in_channels,
            )
            with torch.no_grad():
                nn.init.zeros_(self.tconv.weight)
                self.tconv.weight[:, 0, tconv_kernel // 2, 0, 0] = 1.0
                nn.init.zeros_(self.tconv.bias)
        else:
            self.tconv = None

        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        temporal_blocks_final_L = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    hidden_size=512,
                    num_heads=8,
                )
                for _ in range(8)
            ]
        )
        self.temporal_blocks_final_L = temporal_blocks_final_L
        if self.use_abs_pe:
            self.abs_pe = SincosPE(d_model=512, max_seq_len=abs_pe_max_seq_len)

        if self.use_dt_embed:
            self.dt_mlp = nn.Sequential(
                nn.Linear(1, dt_hidden),
                nn.SiLU(),
                nn.Linear(dt_hidden, 512),
            )
            with torch.no_grad():
                nn.init.zeros_(self.dt_mlp[2].weight)
                nn.init.zeros_(self.dt_mlp[2].bias)
        else:
            self.dt_mlp = None

        self.final_layer = nn.Linear(in_features=512, out_features=self.out_channels)

    def forward(self, x, dt=None):
        B, L = x.shape[:2]
        if self.tconv is not None:
            x = self.tconv(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = rearrange(x, "b l c h w -> (b l) c h w")

        x = self.conv_layers_1(x)
        x = self.conv_layers_2(x)
        x = self.pool(x)

        x = x.squeeze(-1).squeeze(-1)

        x = rearrange(x, "(b l) c-> b l c", b=B, l=L)
        if self.use_abs_pe:
            x = self.abs_pe(x)
        if self.dt_mlp is not None and dt is not None:
            dt_t = dt.to(x.dtype).view(B, 1)
            dt_e = self.dt_mlp(dt_t).unsqueeze(1)
            x = x + dt_e
        for block in self.temporal_blocks_final_L:
            x = block(x)

        x = self.final_layer(x)
        x = rearrange(x, "b l (j t) -> b l j t", b=B, l=L, j=self.num_out_vecs, t=3)

        return x


class AccelerationDecoder(nn.Module):
    def __init__(
        self,
        in_channels=768,
        out_channels=17 * 3,
        use_abs_pe=False,
        abs_pe_max_seq_len=4096,
        use_tconv=False,
        tconv_kernel=5,
        use_dt_embed=False,
        dt_hidden=64,
        num_out_vecs=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_out_vecs = num_out_vecs
        self.use_abs_pe = use_abs_pe
        self.use_dt_embed = use_dt_embed

        self.use_tconv = use_tconv
        if use_tconv:
            self.tconv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(tconv_kernel, 1, 1),
                padding=(tconv_kernel // 2, 0, 0),
                groups=in_channels,
            )
            with torch.no_grad():
                nn.init.zeros_(self.tconv.weight)
                self.tconv.weight[:, 0, tconv_kernel // 2, 0, 0] = 1.0
                nn.init.zeros_(self.tconv.bias)
        else:
            self.tconv = None

        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        temporal_blocks_final_L = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    hidden_size=512,
                    num_heads=8,
                )
                for _ in range(8)
            ]
        )
        self.temporal_blocks_final_L = temporal_blocks_final_L
        if self.use_abs_pe:
            self.abs_pe = SincosPE(d_model=512, max_seq_len=abs_pe_max_seq_len)

        if self.use_dt_embed:
            self.dt_mlp = nn.Sequential(
                nn.Linear(1, dt_hidden),
                nn.SiLU(),
                nn.Linear(dt_hidden, 512),
            )
            with torch.no_grad():
                nn.init.zeros_(self.dt_mlp[2].weight)
                nn.init.zeros_(self.dt_mlp[2].bias)
        else:
            self.dt_mlp = None

        self.final_layer = nn.Linear(in_features=512, out_features=out_channels)

    def forward(self, x, dt=None):
        B, L = x.shape[:2]
        if self.tconv is not None:
            x = self.tconv(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = rearrange(x, "b l c h w -> (b l) c h w")

        x = self.conv_layers_1(x)
        x = self.conv_layers_2(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)

        x = rearrange(x, "(b l) c-> b l c", b=B, l=L)
        if self.use_abs_pe:
            x = self.abs_pe(x)
        if self.dt_mlp is not None and dt is not None:
            dt_t = dt.to(x.dtype).view(B, 1)
            dt_e = self.dt_mlp(dt_t).unsqueeze(1)
            x = x + dt_e
        for block in self.temporal_blocks_final_L:
            x = block(x)

        x = self.final_layer(x)
        if self.num_out_vecs is not None:
            x = rearrange(x, "b l (j t) -> b l j t", b=B, l=L, j=self.num_out_vecs, t=3)
        elif self.out_channels == 17 * 3:
            x = rearrange(x, "b l (j t) -> b l j t", b=B, l=L, j=17, t=3)
        else:
            x = x

        return x


__all__ = ["KeypointsDecoder", "VelocityDecoder", "AccelerationDecoder"]
