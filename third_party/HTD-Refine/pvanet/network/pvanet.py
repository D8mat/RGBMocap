"""Tensor-only PVA-Net inference network."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

import pvanet.paths

from .bbox import convert_bbx_xyxy_to_bbx_xywh
from .decoders import AccelerationDecoder, KeypointsDecoder, VelocityDecoder
from .encoders import ViTPoseEncoder
from .keypoints import get_max_score_kpts_from_heatmaps, refine_kpt_DARK_v2
from .layers import zero_module
from .spatial import get_spatial_cliff_pe
from .temporal import TemporalFeatureAggregation, TemporalLayer


class PVANet(nn.Module):
    """PVA-Net tensor inference network."""

    def __init__(self) -> None:
        super().__init__()

        self.vitpose_encoder = ViTPoseEncoder(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            drop_path_rate=0.55,
        )
        self.spatial_position_projection = nn.Sequential(
            zero_module(nn.Linear(2, 1280)),
            nn.Dropout(0.1),
        )
        self.temporal_layer = TemporalLayer(
            out_channels=1280,
            in_channels=1280,
            temporal_layer=8,
            spatial_layer=4,
            num_heads=8,
        )
        self.temporal_feature_aggregation = TemporalFeatureAggregation(num_reads=5, channels=1280)
        self.keypoints_decoder = KeypointsDecoder(out_channels=17, in_channels=1280, pred_type="coco17_h")
        self.velocity_decoder = VelocityDecoder(
            in_channels=1280,
            use_abs_pe=True,
            use_tconv=True,
            tconv_kernel=5,
        )
        self.acceleration_decoder = AccelerationDecoder(
            out_channels=17 * 3,
            in_channels=1280,
            use_abs_pe=True,
            use_tconv=True,
            tconv_kernel=5,
        )

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]), persistent=False)
        self.register_buffer("velocity_mean", torch.zeros(3))
        self.register_buffer("velocity_std", torch.ones(3))
        self.register_buffer("acceleration_mean", torch.zeros(17, 3))
        self.register_buffer("acceleration_std", torch.ones(17, 3))

        for parameter in self.vitpose_encoder.parameters():
            parameter.requires_grad = False

    def denormalize_motion(
        self,
        normalized_velocities: torch.Tensor,
        normalized_accelerations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert standardized motion-head outputs to camera-space SI units."""
        velocity_mean = self.velocity_mean.to(dtype=normalized_velocities.dtype)
        velocity_std = self.velocity_std.to(dtype=normalized_velocities.dtype)
        acceleration_mean = self.acceleration_mean.to(dtype=normalized_accelerations.dtype)
        acceleration_std = self.acceleration_std.to(dtype=normalized_accelerations.dtype)
        camera_velocities = normalized_velocities * velocity_std + velocity_mean
        camera_accelerations = normalized_accelerations * acceleration_std + acceleration_mean
        return camera_velocities, camera_accelerations

    def validate_forward_inputs(
        self,
        images: torch.Tensor,
        bounding_boxes_xyxy: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> None:
        """Fail early when sequence-level tensors do not share `[B,L]`."""
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B,L,3,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"images channel dimension must be 3, got shape {tuple(images.shape)}")
        B, L = images.shape[:2]
        if bounding_boxes_xyxy.ndim != 3 or tuple(bounding_boxes_xyxy.shape[:2]) != (B, L):
            raise ValueError(
                "bounding_boxes_xyxy must have shape [B,L,4] with the same B/L as images, "
                f"got images={tuple(images.shape)} boxes={tuple(bounding_boxes_xyxy.shape)}"
            )
        if bounding_boxes_xyxy.shape[-1] != 4:
            raise ValueError(f"bounding_boxes_xyxy last dimension must be 4, got {tuple(bounding_boxes_xyxy.shape)}")
        if camera_intrinsics.ndim != 4 or tuple(camera_intrinsics.shape[:2]) != (B, L):
            raise ValueError(
                "camera_intrinsics must have shape [B,L,3,3] with the same B/L as images, "
                f"got images={tuple(images.shape)} K={tuple(camera_intrinsics.shape)}"
            )
        if tuple(camera_intrinsics.shape[-2:]) != (3, 3):
            raise ValueError(f"camera_intrinsics last dimensions must be [3,3], got {tuple(camera_intrinsics.shape)}")

    def encode_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """ViTPose encoder: `[B,L,3,256,192] -> [B,L,1280,16,12]`."""
        normalized_images = (images - self.image_mean[None, None, :, None, None]) / self.image_std[
            None, None, :, None, None
        ]
        visual_features = self.vitpose_encoder(normalized_images)
        return visual_features

    def apply_spatial_position_encoding(
        self,
        visual_features: torch.Tensor,
        bounding_boxes_xyxy: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Add CLIFF-style 2D spatial encoding before temporal attention."""
        B, L, _, H, W = visual_features.shape
        spatial_position = get_spatial_cliff_pe(bounding_boxes_xyxy[:B, :L], camera_intrinsics[:B, :L], out_size=(H, W))
        projected_spatial_position = self.spatial_position_projection(spatial_position)
        projected_spatial_position = rearrange(
            projected_spatial_position,
            "b l h w c -> b l c h w",
            b=B,
            l=L,
            h=H,
            w=W,
        )
        return visual_features + projected_spatial_position

    def forward_network(
        self,
        images: torch.Tensor,
        bounding_boxes_xyxy: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return heatmaps and de-standardized motion outputs."""
        self.validate_forward_inputs(images, bounding_boxes_xyxy, camera_intrinsics)
        visual_features = self.encode_visual_features(images)
        visual_features = self.apply_spatial_position_encoding(
            visual_features,
            bounding_boxes_xyxy,
            camera_intrinsics,
        )

        temporal_features, temporal_intermediate_features = self.temporal_layer(
            visual_features,
            return_intermediates=True,
        )
        temporal_motion_features = self.temporal_feature_aggregation(
            temporal_intermediate_features,
            temporal_features,
        )

        normalized_velocity_sequence = self.velocity_decoder(temporal_motion_features)
        normalized_acceleration_sequence = self.acceleration_decoder(temporal_motion_features)
        camera_velocities, camera_accelerations = self.denormalize_motion(
            normalized_velocity_sequence[:, :-1],
            normalized_acceleration_sequence[:, :-2],
        )

        B, _, C, _, _ = temporal_features.shape
        heatmap_features = rearrange(temporal_features[:, :, :C], "b l c h w -> (b l) c h w", b=B)
        heatmaps = self.keypoints_decoder(heatmap_features)
        heatmaps = rearrange(heatmaps, "(b l) j h w -> b l j h w", b=B)
        return heatmaps, camera_velocities, camera_accelerations

    @staticmethod
    def decode_keypoints(
        heatmaps: torch.Tensor,
        bounding_boxes_xyxy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode heatmaps to bbox-normalized and full-image 2D keypoints."""
        B, L, J, H, W = heatmaps.shape
        flat_heatmaps = rearrange(heatmaps, "b l j h w -> (b l) j h w")
        keypoint_coords, keypoint_scores = get_max_score_kpts_from_heatmaps(flat_heatmaps)
        keypoint_coords = refine_kpt_DARK_v2(keypoint_coords, flat_heatmaps, kernel=11)
        keypoints_bbox01 = keypoint_coords / torch.tensor([W - 1, H - 1], device=heatmaps.device, dtype=heatmaps.dtype)
        keypoints_bbox01 = rearrange(keypoints_bbox01, "(b l) j xy -> b l j xy", b=B, l=L)
        keypoint_confidence = rearrange(keypoint_scores, "(b l) j one -> b l j one", b=B, l=L)

        bounding_boxes_xywh = convert_bbx_xyxy_to_bbx_xywh(bounding_boxes_xyxy)
        image_keypoints = keypoints_bbox01 * bounding_boxes_xywh[:, :, None, 2:] + bounding_boxes_xywh[:, :, None, :2]
        return image_keypoints, keypoints_bbox01, keypoint_confidence

    def forward(
        self,
        images: torch.Tensor,
        bounding_boxes_xyxy: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict full-image keypoints and camera-space motion targets."""
        heatmaps, camera_velocities, camera_accelerations = self.forward_network(
            images,
            bounding_boxes_xyxy,
            camera_intrinsics,
        )
        image_keypoints, keypoints_bbox01, keypoint_confidence = self.decode_keypoints(
            heatmaps,
            bounding_boxes_xyxy,
        )
        return (
            image_keypoints,
            keypoints_bbox01,
            keypoint_confidence,
            camera_velocities,
            camera_accelerations,
            heatmaps,
        )
