"""Model serialization helpers for PVA-Net checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .paths import PROJECT_ROOT


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_existing_project_path(path: str | Path) -> Path:
    candidate = project_path(path)
    if candidate.exists():
        return candidate
    if candidate.suffix in {".pt", ".pth"}:
        alt_suffix = ".pth" if candidate.suffix == ".pt" else ".pt"
        alternative = candidate.with_suffix(alt_suffix)
        if alternative.exists():
            return alternative
    return candidate


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_full_model(model_path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    resolved_model = resolve_existing_project_path(model_path)
    model = _torch_load(resolved_model, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected a serialized torch.nn.Module in {resolved_model}, got {type(model)!r}")
    return model.to(device).eval()


__all__ = ["_torch_load", "load_full_model", "project_path", "resolve_existing_project_path"]
