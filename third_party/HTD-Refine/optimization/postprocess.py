"""IK postprocess for HTD-Refine optimized SMPL parameters."""

from __future__ import annotations

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torch.cuda.amp import autocast

from . import matrix
from .forward_kinematics import BodyForwardKinematics
from .ik import CCD_IK


@autocast(enabled=False)
def apply_ik_postprocess(outputs, forward_kinematics: BodyForwardKinematics, displacement_thr=None):
    """Apply distance-gated CCD IK to ankle and wrist endpoints."""
    static_conf = outputs["static_conf_logits"]
    post_w_j3d, local_mat, post_w_mat = forward_kinematics(
        **outputs["pred_smpl_params_global"],
        return_transforms=True,
    )

    joint_ids = [7, 8, 20, 21]
    if static_conf.shape[2] != len(joint_ids):
        raise ValueError(f"Expected {len(joint_ids)} IK static confidences, got {static_conf.shape[2]}")
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    def _rational_weight(x, threshold, k=2.0):
        if threshold <= 0:
            return torch.zeros_like(x)
        t = (x / threshold).clamp(min=0.0, max=1.0)
        return (1.0 - t) / (1.0 + k * t)

    alpha_l_leg = alpha_r_leg = alpha_l_hand = alpha_r_hand = None
    if displacement_thr is not None:
        chain_dist = (post_target_j3d[0, :, joint_ids] - post_w_j3d[0, :, joint_ids]).norm(dim=-1)
        alpha = _rational_weight(chain_dist, displacement_thr).clamp(min=0.0, max=1.0)
        alpha_l_leg = alpha[:, 0]
        alpha_r_leg = alpha[:, 1]
        alpha_l_hand = alpha[:, 2]
        alpha_r_hand = alpha[:, 3]

    global_rot = matrix.get_rotation(post_w_mat)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    left_leg_chain = [0, 1, 4, 7]
    right_leg_chain = [0, 2, 5, 8]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )
        chain_local_mat = solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :3, :3] = chain_rotmat[:, :, 1:]
        return local_mat

    def ik_with_soft_blend(local_mat, target_pos, target_rot, target_ind, chain, alpha):
        if alpha is None:
            return ik(local_mat, target_pos, target_rot, target_ind, chain)
        pre = local_mat.clone()
        local_mat = ik(local_mat, target_pos, target_rot, target_ind, chain)
        a = alpha.to(local_mat.device).to(local_mat.dtype).view(1, -1, 1)
        for joint_idx in chain[1:]:
            R_pre = pre[:, :, joint_idx, :3, :3]
            R_ik = local_mat[:, :, joint_idx, :3, :3]
            R_delta = R_pre.transpose(-1, -2) @ R_ik
            aa_delta = matrix_to_axis_angle(R_delta)
            local_mat[:, :, joint_idx, :3, :3] = R_pre @ axis_angle_to_matrix(aa_delta * a)
        return local_mat

    local_mat = ik_with_soft_blend(
        local_mat, post_target_j3d[:, :, [7]], global_rot[:, :, [7]], [3], left_leg_chain, alpha_l_leg
    )
    local_mat = ik_with_soft_blend(
        local_mat, post_target_j3d[:, :, [8]], global_rot[:, :, [8]], [3], right_leg_chain, alpha_r_leg
    )
    local_mat = ik_with_soft_blend(
        local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain, alpha_l_hand
    )
    local_mat = ik_with_soft_blend(
        local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain, alpha_r_hand
    )

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))
    return body_pose.flatten(2)


__all__ = ["apply_ik_postprocess"]
