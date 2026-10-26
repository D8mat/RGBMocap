"""HTD-Refine optimization configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmplLossWeightsConfig:
    vel_3d_mse: float = 1.0
    accel_3d_mse: float = 0.1
    kp2d_mse: float = 1.0
    jitter: float = 1.0e4
    reg: float = 1.0e4


@dataclass(frozen=True)
class SmplStageConfig:
    fit_keys: tuple[str, ...]
    max_steps: int
    lr: float
    weights: SmplLossWeightsConfig
    milestones: tuple[int, ...]
    warmup: int = 10
    gamma: float = 0.1
    vel_joint_weight_scale: float = 1.0
    accel_joint_weight_scale: float = 1.0


@dataclass(frozen=True)
class ScaleCalibrationConfig:
    max_steps: int = 1500
    lr: float = 1.0e-4
    milestones: tuple[int, ...] = (1000, 1200, 1400)
    warmup: int = 10
    gamma: float = 0.5
    init_scale: float = 1.0
    clamp: tuple[float, float] = (1.0, 1.05)
    vel_mse: float = 1.0
    accel_mse: float = 10.0
    drop_joints: tuple[int, ...] = (9, 10, 15, 16)


@dataclass(frozen=True)
class PostprocessConfig:
    joint_ids: tuple[int, ...] = (15, 16, 9, 10)
    vel_thr: float = 0.1
    displacement_thr: float = 0.01


@dataclass(frozen=True)
class OptimizationConfig:
    frame_rate: float
    camera_alignment: SmplStageConfig
    scale_calibration: ScaleCalibrationConfig
    motion_refinement: SmplStageConfig
    postprocess: PostprocessConfig


OPTIMIZATION_CONFIG = OptimizationConfig(
    frame_rate=30.0,
    camera_alignment=SmplStageConfig(
        fit_keys=("transl", "global_orient"),
        max_steps=500,
        lr=1.0e-3,
        weights=SmplLossWeightsConfig(),
        milestones=(333, 400, 466),
    ),
    scale_calibration=ScaleCalibrationConfig(),
    motion_refinement=SmplStageConfig(
        fit_keys=("transl", "global_orient", "body_pose"),
        max_steps=1500,
        lr=1.0e-2,
        weights=SmplLossWeightsConfig(),
        milestones=(1000,),
        vel_joint_weight_scale=0.5,
        accel_joint_weight_scale=0.15,
    ),
    postprocess=PostprocessConfig(),
)


__all__ = [
    "OPTIMIZATION_CONFIG",
    "OptimizationConfig",
    "PostprocessConfig",
    "ScaleCalibrationConfig",
    "SmplLossWeightsConfig",
    "SmplStageConfig",
]
