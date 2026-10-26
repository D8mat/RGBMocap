"""Load GVHMR demo results into the optimization input format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import numpy as np

SMPL_KEYS = ("global_orient", "body_pose", "betas", "transl")


def to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype)


def load_torch_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GVHMR result not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"GVHMR result must be a dict, got {type(data)!r}: {path}")
    return data


def normalize_smpl_params(params: dict[str, Any] | None) -> dict[str, torch.Tensor] | None:
    if params is None:
        return None
    missing = [key for key in SMPL_KEYS if key not in params]
    if missing:
        raise KeyError(f"Missing SMPL params {missing}; available keys={list(params.keys())}")
    return {key: to_tensor(params[key]).float() for key in SMPL_KEYS}


def load_gvhmr_result(path: str | Path) -> dict[str, Any]:
    """Return GVHMR motion normalized for HTD-Refine optimization."""
    data = load_torch_dict(path)
    global_params = (
        data.get("smpl_params_global") or data.get("smplx_params_global") or data.get("pred_smpl_params_global")
    )
    incam_params = data.get("smpl_params_incam") or data.get("smplx_params_incam") or data.get("pred_smpl_params_incam")
    if incam_params is None and isinstance(data.get("net_outputs"), dict):
        incam_params = data["net_outputs"].get("pred_smpl_params_incam")
    if global_params is None and isinstance(data.get("net_outputs"), dict):
        global_params = data["net_outputs"].get("pred_smpl_params_global")

    if global_params is None and incam_params is None:
        raise KeyError("Cannot find GVHMR global or incam SMPL params.")

    global_params = normalize_smpl_params(global_params)
    incam_params = normalize_smpl_params(incam_params)

    k_fullimg = None
    for key in ("K_fullimg", "K", "intrinsics"):
        if key in data:
            k_fullimg = data[key]
            break
    t_w2c = None
    for key in ("T_w2c", "pred_T_w2c"):
        if key in data:
            t_w2c = to_tensor(data[key]).float()
            break
    out = {
        "source": "gvhmr",
        "pred_smpl_params_global": global_params,
        "pred_smpl_params_incam": incam_params,
        "K_fullimg": None if k_fullimg is None else to_tensor(k_fullimg).float(),
        "T_w2c": t_w2c,
        "meta": {"path": str(path), "keys": sorted(map(str, data.keys()))},
    }
    return out


def motion_length(motion: dict[str, Any]) -> int:
    params = motion.get("pred_smpl_params_global")
    if params is None:
        params = motion.get("pred_smpl_params_incam")
    if not params or "global_orient" not in params:
        raise KeyError("Motion has no global_orient length source.")
    return int(params["global_orient"].shape[0])


__all__ = [
    "SMPL_KEYS",
    "load_gvhmr_result",
    "motion_length",
    "normalize_smpl_params",
    "to_tensor",
]
