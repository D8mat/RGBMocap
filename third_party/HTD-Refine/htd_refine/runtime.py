"""Shared runtime helpers for the installed HTD-Refine demo commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PVA_NET_ROOT = PROJECT_ROOT / "pvanet"
DEFAULT_MODEL_PATH = "inputs/checkpoints/pvanet.pt"

for path in (PVA_NET_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from pvanet.model_io import _torch_load as torch_load
from pvanet.model_io import load_full_model
from pvanet.model_io import project_path, resolve_existing_project_path


def resolve_device(value: str | torch.device) -> torch.device:
    """Validate and return the torch device requested by a CLI user."""
    device = torch.device(value)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested, but CUDA is unavailable.")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s).")
    return torch.device("cuda", index)


def to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(child) for child in value]
    return value


__all__ = [
    "DEFAULT_MODEL_PATH",
    "PVA_NET_ROOT",
    "PROJECT_ROOT",
    "load_full_model",
    "project_path",
    "resolve_device",
    "resolve_existing_project_path",
    "to_cpu_tree",
    "torch_load",
]
