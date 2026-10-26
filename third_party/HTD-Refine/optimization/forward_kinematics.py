"""Forward kinematics used by the HTD-Refine IK postprocess."""

from __future__ import annotations

import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix

from . import matrix
from .body_model import create_body_model


class BodyForwardKinematics(nn.Module):
    """Compute the first 22 SMPL-X joints and their transforms."""

    def __init__(self):
        super().__init__()
        self.smplx_model = create_body_model("smplx_v437_coco17")
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, persistent=False)
        self.parents = parents.tolist()

    def forward(self, body_pose, betas, global_orient=None, transl=None, return_transforms=False):
        """Forward-kinematics over the first 22 body joints."""
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        if transl is not None:
            local_skeleton[..., 0, :] += transl

        local_mat = matrix.get_TRS(rotmat, local_skeleton)
        fk_mat = matrix.forward_kinematics(local_mat, self.parents)
        joints = matrix.get_position(fk_mat)
        if return_transforms:
            return joints, local_mat, fk_mat
        return joints


__all__ = ["BodyForwardKinematics"]
