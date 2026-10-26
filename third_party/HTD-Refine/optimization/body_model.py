"""Construct the SMPL and SMPL-X variants used by HTD-Refine."""

from __future__ import annotations

from pathlib import Path

from utils.body_model import BodyModelSMPLH, BodyModelSMPLX
from utils.body_model_lite.smpl_lite import SmplLite, SmplLiteCoco17
from utils.body_model_lite.smplx_lite import SmplxLiteCoco17, SmplxLiteV437Coco17

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BODY_MODELS_PATH = PROJECT_ROOT / "inputs/checkpoints/body_models"


def create_body_model(model_type: str, **kwargs):
    """Construct one of the explicitly supported body-model variants."""
    if model_type == "smplx_full_pose":
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "use_pca": False,
            "flat_hand_mean": True,
        }
        bm_kwargs.update(kwargs)
        return BodyModelSMPLX(model_path=BODY_MODELS_PATH, **bm_kwargs)
    if model_type == "smplx_pca":
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "num_betas": 10,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        return BodyModelSMPLX(model_path=BODY_MODELS_PATH, **bm_kwargs)
    if model_type == "smplx_coco17":
        return SmplxLiteCoco17(gender=kwargs.get("gender", "neutral"))
    if model_type == "smplx_v437_coco17":
        return SmplxLiteV437Coco17()
    if model_type == "smpl_coco17":
        return SmplLiteCoco17(gender=kwargs.get("gender", "neutral"))
    if model_type == "smpl_lite":
        return SmplLite(gender=kwargs.get("gender", "neutral"))
    if model_type == "smpl":
        bm_kwargs = {
            "model_path": BODY_MODELS_PATH,
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        return BodyModelSMPLH(**bm_kwargs)
    raise NotImplementedError(f"unsupported body model type {model_type!r}")


__all__ = ["BODY_MODELS_PATH", "PROJECT_ROOT", "create_body_model"]
