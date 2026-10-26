"""Optimization loops for HTD-Refine motion refinement."""

from __future__ import annotations

import torch
from tqdm import tqdm

from .config import ScaleCalibrationConfig, SmplStageConfig
from .geometry import scale_camera_trajectory
from .losses import ScaleCalibrationLoss, ScaleCalibrationWeights, SMPLRefitLoss, SMPLRefitWeights
from .scheduler import WarmupMultiStepLR


def optimize_smpl(
    smpl_params: dict[str, torch.Tensor],
    tgt_vel_3d: torch.Tensor,
    tgt_accel_3d: torch.Tensor,
    tgt_kp2d: torch.Tensor,
    model_name: str,
    K: torch.Tensor,
    T_w2c: torch.Tensor,
    weights: SMPLRefitWeights,
    stage: SmplStageConfig,
    frame_rate: float,
    mask: torch.Tensor | None = None,
    tgt_conf: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = tgt_kp2d.device
    params_dict = {
        k: v.clone().detach().to(device).contiguous().requires_grad_(False)
        for k, v in smpl_params.items()
    }
    for key in stage.fit_keys:
        params_dict[key] = params_dict[key].requires_grad_(True)

    optimizer = torch.optim.Adam(params_dict.values(), lr=stage.lr)
    scheduler = WarmupMultiStepLR(optimizer, milestones=list(stage.milestones), warmup=stage.warmup, gamma=stage.gamma)
    loss_fn = SMPLRefitLoss(
        src_kp2d=tgt_kp2d,
        tgt_vel_3d=tgt_vel_3d,
        tgt_accel_3d=tgt_accel_3d,
        body_pose=smpl_params["body_pose"].clone().detach().to(device).contiguous().requires_grad_(False),
        global_orient=smpl_params["global_orient"].clone().detach().to(device).contiguous().requires_grad_(False),
        transl=smpl_params["transl"].clone().detach().to(device).contiguous().requires_grad_(False),
        model_name=model_name,
        K_fullimg=K,
        T_w2c=T_w2c,
        weights=weights,
        stage=stage,
        frame_rate=frame_rate,
        mask=mask,
        tgt_conf=tgt_conf,
    )
    closure = loss_fn.create_closure(optimizer, params_dict)
    pbar = tqdm(range(stage.max_steps))
    for _ in pbar:
        optimizer.zero_grad()
        optimizer.step(closure).item()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        message = "".join(f"{key}={value:.1f}, " for key, value in loss_fn.log_dict.items())
        pbar.set_description(f"{message}lr={current_lr:.2e}")
    return params_dict


def optimize_camera_scale(
    tgt_vel: torch.Tensor,
    tgt_accel: torch.Tensor,
    j3d_cam: torch.Tensor,
    T_w2c: torch.Tensor,
    weights: ScaleCalibrationWeights,
    config: ScaleCalibrationConfig,
    frame_rate: float,
    mask: torch.Tensor | None = None,
    tgt_conf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"opt cam with lr {config.lr}")
    device = j3d_cam.device
    params_dict = {"scale": torch.tensor(config.init_scale, device=device).requires_grad_(True)}
    optimizer = torch.optim.Adam(params_dict.values(), lr=config.lr)
    scheduler = WarmupMultiStepLR(
        optimizer, milestones=list(config.milestones), warmup=config.warmup, gamma=config.gamma
    )
    loss_fn = ScaleCalibrationLoss(
        tgt_vel=tgt_vel,
        tgt_accel=tgt_accel,
        j3d_cam=j3d_cam,
        T_w2c=T_w2c,
        weights=weights,
        config=config,
        frame_rate=frame_rate,
        mask=mask,
        tgt_conf=tgt_conf,
    )
    closure = loss_fn.create_closure(optimizer, params_dict)
    pbar = tqdm(range(config.max_steps))
    for _ in pbar:
        optimizer.zero_grad()
        optimizer.step(closure).item()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        parts = []
        for key, value in loss_fn.log_dict.items():
            parts.append(f"{key}={float(value):.4f}" if key == "scale" else f"{key}={value:.1f}")
        pbar.set_description(", ".join(parts) + f", lr={current_lr:.2e}")

    scale = params_dict["scale"].clamp(*config.clamp)
    T_w2c = scale_camera_trajectory(T_w2c, scale, axes=(0, 1, 2))
    return T_w2c.detach().to(device).requires_grad_(False), scale


__all__ = ["optimize_camera_scale", "optimize_smpl"]
