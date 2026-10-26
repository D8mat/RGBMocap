"""Preserve the world/camera gauge of a paired motion prediction."""
from __future__ import annotations

import torch
from pytorch3d.transforms import axis_angle_to_matrix


def static_camera_from_motion(incam, world, pelvis_offset):
    """Match GVHMR's static-camera reference, including the SMPL pelvis pivot.

    SMPL translation is not the pelvis position. Transform translation + the
    shape-dependent rest pelvis, then remove that same pivot in camera space.
    Use the first pair just as GVHMR's static-camera postprocess does; averaging
    per-frame predictions would introduce a different world/camera gauge.
    """
    if incam is None or world is None:
        raise ValueError('Camera alignment requires both GVHMR world and camera-space motion. Reconstruct GVHMR first.')
    length = len(world['global_orient'])
    if not length:
        raise ValueError('Cannot align an empty motion')
    for params in (incam, world):
        for key in ('global_orient', 'transl'):
            value = params[key]
            if value.shape != (length, 3) or not torch.isfinite(value).all():
                raise ValueError('Camera alignment requires finite, synchronized root motion')
    if pelvis_offset.shape != (3,) or not torch.isfinite(pelvis_offset).all():
        raise ValueError('Invalid SMPL pelvis offset')
    if not torch.allclose(incam['betas'][0], world['betas'][0], atol=1e-5, rtol=1e-5):
        raise ValueError('World and camera motion must use the same actor shape')
    rotation = (axis_angle_to_matrix(incam['global_orient'][:1]) @
                axis_angle_to_matrix(world['global_orient'][:1]).transpose(-1, -2))
    camera_root = incam['transl'][:1] + pelvis_offset
    world_root = world['transl'][:1] + pelvis_offset
    translation = camera_root - (rotation @ world_root[..., None]).squeeze(-1)
    transform = torch.eye(4, dtype=rotation.dtype, device=rotation.device)[None].repeat(length, 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform
