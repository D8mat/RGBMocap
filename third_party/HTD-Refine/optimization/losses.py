"""Loss modules for HTD-Refine motion optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .body_model import create_body_model
from .config import ScaleCalibrationConfig, SmplStageConfig
from .geometry import apply_T_on_points, perspective_projection, scale_camera_trajectory

COCO17_JOINT_NAMES = (
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)

VEL_P999_MAG_THRESHOLDS = (
    2.9097,
    2.9159,
    2.9111,
    2.8630,
    2.8225,
    2.8298,
    2.8383,
    4.0343,
    4.1615,
    5.7119,
    6.0373,
    2.7371,
    2.7457,
    3.9189,
    4.2014,
    5.8695,
    6.4464,
)
ACCEL_P9999_MAG_THRESHOLDS = (
    76.2970,
    75.3610,
    76.1500,
    73.3050,
    73.9110,
    64.3060,
    64.8900,
    86.8350,
    97.0060,
    115.8650,
    150.0760,
    61.1690,
    58.9190,
    84.2080,
    85.6460,
    126.3640,
    150.7630,
)


@dataclass
class SMPLRefitWeights:
    vel_3d_mse: float = 1.0
    accel_3d_mse: float = 0.1
    kp2d_mse: float = 1.0
    jitter: float = 1.0e4
    reg: float = 1.0e4


@dataclass
class ScaleCalibrationWeights:
    vel_mse: float = 1.0
    accel_mse: float = 10.0


def make_smpl_weights(stage: SmplStageConfig) -> SMPLRefitWeights:
    return SMPLRefitWeights(**stage.weights.__dict__)


def make_scale_weights(config: ScaleCalibrationConfig) -> ScaleCalibrationWeights:
    return ScaleCalibrationWeights(vel_mse=config.vel_mse, accel_mse=config.accel_mse)


def map_conf(conf: torch.Tensor, mu: float = 0.60, k: float = 20.0) -> torch.Tensor:
    return torch.sigmoid(k * (conf - mu))


def build_conf_tensors(
    tgt_conf: torch.Tensor | None, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if tgt_conf is None:
        return None, None, None
    conf = tgt_conf.float().to(device)
    if conf.ndim == 2:
        conf = conf.unsqueeze(-1)
    elif conf.ndim == 3 and conf.shape[-1] != 1:
        raise ValueError(f"expected confidence shape (L, 17) or (L, 17, 1), got {tuple(conf.shape)}")
    mapped = map_conf(conf)
    vel_conf = torch.minimum(mapped[:-1], mapped[1:])
    accel_conf = torch.minimum(torch.minimum(mapped[:-2], mapped[1:-1]), mapped[2:])
    return mapped, vel_conf, accel_conf


def filter_outliers_std(values: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    mean = torch.mean(values)
    std = torch.std(values)
    return torch.where(
        (values >= mean - sigma * std) & (values <= mean + sigma * std),
        values,
        torch.zeros_like(values),
    )


def make_per_joint_weights(device: torch.device, *, fit_body_pose: bool = False) -> dict[str, torch.Tensor]:
    weight_2d = torch.ones(17, device=device)
    if fit_body_pose:

        weights_joints = torch.tensor(
            [
                200.0,
                200.0,
                200.0,
                200.0,
                200.0,
                1000.0,
                1000.0,
                500.0,
                500.0,
                250.0,
                250.0,
                1000.0,
                1000.0,
                500.0,
                500.0,
                250.0,
                250.0,
            ],
            device=device,
            dtype=torch.float32,
        )
    else:
        weights_joints = torch.ones(17, device=device) * 100.0
        weights_joints[[9, 10, 15, 16]] *= 10.0
    weights_jitter = torch.ones(17, device=device)
    return {"weight_2d": weight_2d, "weights_joints": weights_joints, "weights_jitter": weights_jitter}


def make_refinement_velacc_thresholds(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "vel": torch.tensor(VEL_P999_MAG_THRESHOLDS, device=device, dtype=dtype),
        "accel": torch.tensor(ACCEL_P9999_MAG_THRESHOLDS, device=device, dtype=dtype),
    }


def make_velacc_valid_mask(residual: torch.Tensor, kind: str, thresholds: dict[str, torch.Tensor]) -> torch.Tensor:
    if residual.shape[-2:] != (17, 3):
        raise ValueError(f"expected residual shape (..., 17, 3), got {tuple(residual.shape)}")
    threshold = thresholds[kind].to(device=residual.device, dtype=residual.dtype)
    view_shape = (1,) * (residual.ndim - 2) + (17,)
    return residual.norm(dim=-1) <= threshold.view(view_shape)


def masked_mean_by_valid_count(loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=loss.device, dtype=loss.dtype).expand_as(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class SMPLRefitLoss(torch.nn.Module):
    KP2D_REF_W = 1280
    KP2D_REF_H = 960

    def __init__(
        self,
        src_kp2d: torch.Tensor,
        tgt_vel_3d: torch.Tensor,
        tgt_accel_3d: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
        model_name: str,
        K_fullimg: torch.Tensor,
        T_w2c: torch.Tensor,
        weights: SMPLRefitWeights,
        stage: SmplStageConfig,
        frame_rate: float,
        mask: torch.Tensor | None = None,
        tgt_conf: torch.Tensor | None = None,
    ):
        super().__init__()
        device = src_kp2d.device
        self.src_kp2d = src_kp2d.float().to(device)
        self.tgt_vel_3d = tgt_vel_3d.float().to(device)
        self.tgt_accel_3d = tgt_accel_3d.float().to(device)
        self.K_fullimg = K_fullimg.float().to(device)
        self.T_w2c = T_w2c.float().to(device)
        self.body_pose_ref = body_pose.float().to(device)
        self.global_orient_ref = global_orient.float().to(device)
        self.transl_ref = transl.float().to(device)
        self.weights = weights
        self.stage = stage
        self.frame_dt = 1.0 / float(frame_rate)
        self.mask = (
            torch.ones(self.src_kp2d.shape[0], dtype=torch.bool, device=device)
            if mask is None
            else mask.bool().to(device)
        )
        self.tgt_conf, self.vel_conf, self.accel_conf = build_conf_tensors(tgt_conf, device)

        if model_name == "smplx":
            self.smpl_coco17 = create_body_model("smplx_coco17").to(device)
            num_body_joints = 21
        elif model_name == "smpl":
            self.smpl_coco17 = create_body_model("smpl_coco17").to(device)
            num_body_joints = 23
        else:
            raise ValueError(f"unsupported model {model_name!r}")

        bp_joint_weight = torch.ones(num_body_joints, device=device)
        self.bp_joint_weight = bp_joint_weight
        self.num_body_joints = num_body_joints
        self.reg_subweights = {"bp": 10.0, "go": 1.0, "transl": 0.1}

        img_w = self.K_fullimg[0, 0, 2].item() * 2.0
        img_h = self.K_fullimg[0, 1, 2].item() * 2.0
        self.kp2d_res_weight = (self.KP2D_REF_W * self.KP2D_REF_H) / (img_w * img_h)

        fit_body_pose = "body_pose" in stage.fit_keys
        joint_weights = make_per_joint_weights(device, fit_body_pose=fit_body_pose)
        self.weight_2d = joint_weights["weight_2d"]
        self.vel_weights_joints = joint_weights["weights_joints"] * stage.vel_joint_weight_scale
        self.accel_weights_joints = joint_weights["weights_joints"] * stage.accel_joint_weight_scale
        self.weights_jitter = joint_weights["weights_jitter"]
        self.velacc_thresholds = (
            make_refinement_velacc_thresholds(device=device, dtype=torch.float32) if fit_body_pose else None
        )

    def compute_jitter_loss(self, joints: torch.Tensor) -> torch.Tensor:
        jerk = joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]
        jerk = (
            jerk
            * self.mask[:-3, None, None]
            * self.mask[1:-2, None, None]
            * self.mask[2:-1, None, None]
            * self.mask[3:, None, None]
        )
        jerk_norm = torch.norm(jerk, dim=-1)
        return (jerk_norm * self.weights_jitter[None, :]).mean()

    def compute_vector_loss(
        self, pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor | None, kind: str
    ) -> torch.Tensor:
        joint_weights = self.vel_weights_joints if kind == "vel" else self.accel_weights_joints
        residual = pred - target
        valid_mask = None
        if self.velacc_thresholds is not None:
            valid_mask = make_velacc_valid_mask(residual, kind, self.velacc_thresholds)

        loss = (residual**2).sum(dim=-1) * joint_weights[None, :]
        if kind == "vel":
            loss = loss * (self.mask[:-1] & self.mask[1:])[:, None]
        else:
            loss = loss * (self.mask[:-2] & self.mask[1:-1] & self.mask[2:])[:, None]
        if conf is not None:
            loss = loss * conf.squeeze(-1)
        return masked_mean_by_valid_count(loss, valid_mask) if valid_mask is not None else loss.mean()

    def kp2d_loss(self, kp2d_proj: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(kp2d_proj, self.src_kp2d, reduction="none")
        conf = self.tgt_conf if self.tgt_conf is not None else 1.0
        return (
            loss
            * conf
            * self.mask[:, None, None]
            * self.weight_2d[None, :, None]
            * self.kp2d_res_weight
        ).mean()

    def create_closure(self, optimizer: torch.optim.Optimizer, param_dict: dict[str, torch.Tensor]):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            smpl_params = param_dict
            joints = self.smpl_coco17(**smpl_params)
            joints_cam = apply_T_on_points(joints, self.T_w2c)
            joints_cam_next = apply_T_on_points(joints[1:], self.T_w2c[:-1])
            joints_cam_next_next = apply_T_on_points(joints[2:], self.T_w2c[:-2])
            vel_3d = (joints_cam_next - joints_cam[:-1]) / self.frame_dt
            accel_3d = (
                (joints_cam_next_next - 2 * joints_cam_next[:-1] + joints_cam[:-2])
                / self.frame_dt
                / self.frame_dt
            )
            kp2d_proj = perspective_projection(joints_cam, self.K_fullimg)

            loss_dict = {
                "vel_3d_mse": self.compute_vector_loss(vel_3d, self.tgt_vel_3d, self.vel_conf, "vel"),
                "accel_3d_mse": self.compute_vector_loss(accel_3d, self.tgt_accel_3d, self.accel_conf, "accel"),
                "kp2d_mse": self.kp2d_loss(kp2d_proj),
                "jitter": self.compute_jitter_loss(joints),
                "reg_go": torch.mean((smpl_params["global_orient"] - self.global_orient_ref) ** 2),
                "reg_transl": torch.mean((smpl_params["transl"] - self.transl_ref) ** 2),
            }
            bp_diff_sq = (smpl_params["body_pose"] - self.body_pose_ref) ** 2
            bp_diff_sq = bp_diff_sq.reshape(bp_diff_sq.shape[0], self.num_body_joints, 3)
            loss_dict["reg_bp"] = (bp_diff_sq * self.bp_joint_weight[None, :, None]).mean()
            loss_dict["reg"] = (
                self.reg_subweights["bp"] * loss_dict["reg_bp"]
                + self.reg_subweights["go"] * loss_dict["reg_go"]
                + self.reg_subweights["transl"] * loss_dict["reg_transl"]
            )

            loss = sum(v * loss_dict[k] for k, v in self.weights.__dict__.items() if v > 0)
            loss.backward()
            with torch.no_grad():
                self.log_dict = {k: v * loss_dict[k].detach() for k, v in self.weights.__dict__.items()}
                self.log_dict["reg_bp"] = self.weights.reg * self.reg_subweights["bp"] * loss_dict["reg_bp"].detach()
                self.log_dict["reg_go"] = self.weights.reg * self.reg_subweights["go"] * loss_dict["reg_go"].detach()
                self.log_dict["reg_transl"] = (
                    self.weights.reg * self.reg_subweights["transl"] * loss_dict["reg_transl"].detach()
                )
            return loss

        return closure


class ScaleCalibrationLoss(torch.nn.Module):
    def __init__(
        self,
        tgt_vel: torch.Tensor,
        tgt_accel: torch.Tensor,
        j3d_cam: torch.Tensor,
        T_w2c: torch.Tensor,
        weights: ScaleCalibrationWeights,
        config: ScaleCalibrationConfig,
        frame_rate: float,
        mask: torch.Tensor | None = None,
        tgt_conf: torch.Tensor | None = None,
    ):
        super().__init__()
        device = j3d_cam.device
        self.tgt_vel = tgt_vel.detach().float().to(device)
        self.tgt_accel = tgt_accel.detach().float().to(device)
        self.j3d_cam = j3d_cam.detach().float().to(device)
        self.T_w2c = T_w2c.detach().float().to(device)
        self.weights = weights
        self.mask = (
            torch.ones(self.j3d_cam.shape[0], dtype=torch.bool, device=device)
            if mask is None
            else mask.bool().to(device)
        )
        self.config = config
        self.frame_dt = 1.0 / float(frame_rate)
        self.weights_joints = make_per_joint_weights(device)["weights_joints"].clone()
        self.weights_joints[list(config.drop_joints)] = 0.0
        _, vel_conf, accel_conf = build_conf_tensors(tgt_conf, device)
        self.vel_conf = vel_conf.squeeze(-1) if vel_conf is not None else None
        self.accel_conf = accel_conf.squeeze(-1) if accel_conf is not None else None

    def create_closure(self, optimizer: torch.optim.Optimizer, param_dict: dict[str, torch.Tensor]):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scale = param_dict["scale"]
            T_w2c = scale_camera_trajectory(self.T_w2c, scale, axes=(0, 1, 2))
            j3d_w = apply_T_on_points(self.j3d_cam, torch.inverse(T_w2c))
            vel_w_3d = j3d_w[1:] - j3d_w[:-1]
            vel_w = vel_w_3d.norm(dim=-1) / self.frame_dt
            accel_w = (vel_w_3d[1:] - vel_w_3d[:-1]).norm(dim=-1) / self.frame_dt / self.frame_dt

            loss_vel = F.mse_loss(self.tgt_vel, vel_w, reduction="none")
            loss_accel = F.mse_loss(self.tgt_accel, accel_w, reduction="none")
            loss_vel = filter_outliers_std(loss_vel)
            loss_accel = filter_outliers_std(loss_accel)
            loss_vel = loss_vel * self.weights_joints[None, :]
            loss_accel = loss_accel * self.weights_joints[None, :]
            loss_vel = loss_vel * (self.mask[:-1] & self.mask[1:])[:, None]
            loss_accel = loss_accel * (self.mask[:-2] & self.mask[1:-1] & self.mask[2:])[:, None]
            if self.vel_conf is not None:
                loss_vel = loss_vel * self.vel_conf
            if self.accel_conf is not None:
                loss_accel = loss_accel * self.accel_conf

            loss_dict = {"vel_mse": loss_vel.mean(), "accel_mse": loss_accel.mean()}
            loss = sum(v * loss_dict[k] for k, v in self.weights.__dict__.items() if v > 0)
            loss.backward()
            with torch.no_grad():
                self.log_dict = {k: v * loss_dict[k].detach() for k, v in self.weights.__dict__.items()}
                self.log_dict["scale"] = scale.detach()
            return loss

        return closure


__all__ = [
    "COCO17_JOINT_NAMES",
    "SMPLRefitLoss",
    "SMPLRefitWeights",
    "ScaleCalibrationLoss",
    "ScaleCalibrationWeights",
    "build_conf_tensors",
    "make_scale_weights",
    "make_smpl_weights",
]
