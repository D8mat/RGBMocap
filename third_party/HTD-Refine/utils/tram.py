"""Load TRAM sequence folders into the optimization input format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from optimization.geometry import T_w2c_from_R_t_oneseq

from .gvhmr import normalize_smpl_params, to_tensor


def _load_npy_item(path: Path) -> Any:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    return obj


def select_hps_file(hps_dir: str | Path) -> Path:
    hps_dir = Path(hps_dir)
    files = sorted(hps_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No TRAM hps files found in {hps_dir}")
    selected = max(files, key=lambda p: len(_load_npy_item(p).get("frame", [])))
    if len(files) > 1:
        print(
            f"[TRAM] found {len(files)} tracks; selecting the longest track {selected.name}. "
            "HTD-Refine assumes that PVA-Net and TRAM track the same primary person."
        )
    return selected


def load_tram_result(seq_dir: str | Path) -> dict[str, Any]:
    """Return `{pred_smpl_params_incam, T_w2c, K_fullimg, frames}` from a TRAM sequence folder."""
    seq_dir = Path(seq_dir)
    if not seq_dir.exists():
        raise FileNotFoundError(f"TRAM sequence folder not found: {seq_dir}")

    hps_file = select_hps_file(seq_dir / "hps")
    pred = _load_npy_item(hps_file)
    pred_rotmat = to_tensor(pred["pred_rotmat"])
    pred_axis = matrix_to_axis_angle(pred_rotmat).reshape(pred_rotmat.shape[0], -1)
    pred_shape = to_tensor(pred["pred_shape"])
    pred_shape = pred_shape.mean(dim=0, keepdim=True).repeat(pred_shape.shape[0], 1)
    pred_trans = to_tensor(pred["pred_trans"]).reshape(pred_axis.shape[0], -1, 3)[:, 0]
    frames = to_tensor(pred.get("frame", torch.arange(pred_axis.shape[0])), dtype=torch.long).long()

    smpl_params = normalize_smpl_params(
        {
            "global_orient": pred_axis[:, :3],
            "body_pose": pred_axis[:, 3:],
            "betas": pred_shape,
            "transl": pred_trans,
        }
    )

    camera_path = seq_dir / "camera.npy"
    if camera_path.exists():
        camera = _load_npy_item(camera_path)
        world_cam_R = to_tensor(camera["world_cam_R"])
        world_cam_T = to_tensor(camera["world_cam_T"])
        T_w2c = T_w2c_from_R_t_oneseq(world_cam_R, world_cam_T).float()
        valid = (frames >= 0) & (frames < T_w2c.shape[0])
        frames = frames[valid]
        smpl_params = {key: value[valid] for key, value in smpl_params.items()}
        T_w2c = T_w2c[frames]
        img_focal = float(np.asarray(camera.get("img_focal", camera.get("spec_focal", 0.0))))
        img_center = camera.get("img_center")
        if img_center is not None and img_focal > 0:
            center = to_tensor(img_center).float()
            K_fullimg = torch.eye(3).repeat(T_w2c.shape[0], 1, 1)
            K_fullimg[:, 0, 0] = img_focal
            K_fullimg[:, 1, 1] = img_focal
            K_fullimg[:, 0, 2] = center[0]
            K_fullimg[:, 1, 2] = center[1]
        else:
            K_fullimg = None
    else:
        T_w2c = torch.eye(4).repeat(pred_axis.shape[0], 1, 1)
        K_fullimg = None

    return {
        "source": "tram",
        "pred_smpl_params_global": None,
        "pred_smpl_params_incam": smpl_params,
        "T_w2c": T_w2c,
        "K_fullimg": K_fullimg,
        "frames": frames,
        "meta": {"seq_dir": str(seq_dir), "hps_file": str(hps_file)},
    }


__all__ = ["load_tram_result", "select_hps_file"]
