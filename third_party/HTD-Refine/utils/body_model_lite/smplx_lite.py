import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from einops import einsum, rearrange

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .utils import Struct, VERTEX_IDS, batch_rigid_transform, to_np, to_tensor
from .joint_names import JOINT_NAMES as SKELETON_NAMES
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix

MODEL_PATH = PROJECT_ROOT / "inputs/checkpoints/body_models/smplx"
SMPLX2SMPL_PATH = PROJECT_ROOT / "utils/body_model/smplx2smpl_sparse.pt"
SMPL2COCO17_PATH = PROJECT_ROOT / "utils/body_model/smpl_coco17_J_regressor.pt"
SMPLX_VERTS437_PATH = PROJECT_ROOT / "utils/body_model/smplx_verts437.pt"
SMPLN_J_REGRESSOR_PATH = PROJECT_ROOT / "utils/body_model/smpl_neutral_J_regressor.pt"


class SmplxLite(nn.Module):
    def __init__(
        self,
        model_path=MODEL_PATH,
        gender="neutral",
        num_betas=10,
        flat_hand_mean=False,
    ):
        super().__init__()

        model_path = Path(model_path)
        if model_path.is_dir():
            smplx_path = Path(model_path) / f"SMPLX_{gender.upper()}.npz"
        else:
            smplx_path = model_path
        assert smplx_path.exists()
        data_struct = Struct(**np.load(smplx_path, allow_pickle=True))
        self.faces = data_struct.f

        self.register_smpl_buffers(data_struct, num_betas)
        self.register_smplx_buffers(data_struct, flat_hand_mean)
        self.register_fast_skeleton_computing_buffers()

        self.register_buffer("identity", torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), False)

    def register_smpl_buffers(self, data_struct, num_betas):

        shapedirs = to_tensor(to_np(data_struct.shapedirs[:, :, :num_betas])).float()
        self.register_buffer("shapedirs", shapedirs, False)

        v_template = to_tensor(to_np(data_struct.v_template)).float()
        self.register_buffer("v_template", v_template, False)

        J_regressor = to_tensor(to_np(data_struct.J_regressor)).float()
        self.register_buffer("J_regressor", J_regressor, False)

        posedirs = to_tensor(to_np(data_struct.posedirs)).float()
        posedirs = rearrange(posedirs, "v c n -> n v c")
        self.register_buffer("posedirs", posedirs, False)

        lbs_weights = to_tensor(to_np(data_struct.weights)).float()
        self.register_buffer("lbs_weights", lbs_weights, False)

        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents, False)

    def register_smplx_buffers(self, data_struct, flat_hand_mean):

        expr_dirs = to_tensor(to_np(data_struct.shapedirs[:, :, 300:310])).float()
        self.register_buffer("expr_dirs", expr_dirs, False)

        if flat_hand_mean:
            left_hand_mean = torch.zeros(45)
            right_hand_mean = torch.zeros(45)
        else:
            left_hand_mean = to_tensor(data_struct.hands_meanl).float()
            right_hand_mean = to_tensor(data_struct.hands_meanr).float()
        self.register_buffer("left_hand_mean", left_hand_mean, False)
        self.register_buffer("right_hand_mean", right_hand_mean, False)

        default_aa_jaw = torch.zeros(3)
        self.register_buffer("default_aa_jaw", default_aa_jaw, False)
        default_aa_leye = torch.zeros(3)
        self.register_buffer("default_aa_leye", default_aa_leye, False)
        default_aa_reye = torch.zeros(3)
        self.register_buffer("default_aa_reye", default_aa_reye, False)
        self.register_buffer("default_aa_left_hand", left_hand_mean, False)
        self.register_buffer("default_aa_right_hand", right_hand_mean, False)

    def register_fast_skeleton_computing_buffers(self):

        J_template = self.J_regressor @ self.v_template
        J_shapedirs = torch.einsum("jv, vcd -> jcd", self.J_regressor, self.shapedirs)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

    def get_skeleton(self, betas):
        return self.J_template + torch.einsum("...k, jck -> ...jc", betas, self.J_shapedirs)

    def fk_skeleton(
        self,
        betas,
        body_pose,
        global_orient,
        transl=None,
        left_hand_pose=None,
        right_hand_pose=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        rotation_type="aa",
    ):
        assert betas.shape[:-1] == body_pose.shape[:-1]

        rot_mats = self.get_rotmats(
            body_pose, global_orient, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose, rotation_type
        )

        skeleton = self.get_skeleton(betas)
        skeleton, A = batch_rigid_transform(rot_mats, skeleton, self.parents)

        if transl is not None:
            skeleton = skeleton + transl[..., None, :]

        return skeleton

    def get_rotmats(
        self,
        body_pose,
        global_orient,
        jaw_pose,
        leye_pose,
        reye_pose,
        left_hand_pose,
        right_hand_pose,
        rotation_type="aa",
    ):
        BL = body_pose.shape[:-1]
        if rotation_type == "aa":
            if jaw_pose is None:
                jaw_pose = self.default_aa_jaw.expand(*BL, -1)
            if leye_pose is None:
                leye_pose = self.default_aa_leye.expand(*BL, -1)
            if reye_pose is None:
                reye_pose = self.default_aa_reye.expand(*BL, -1)

            if left_hand_pose is None:
                left_hand_pose = self.default_aa_left_hand.expand(*BL, -1)
            else:
                left_hand_pose = left_hand_pose + self.left_hand_mean
            if right_hand_pose is None:
                right_hand_pose = self.default_aa_right_hand.expand(*BL, -1)
            else:
                right_hand_pose = right_hand_pose + self.right_hand_mean

            rot_mats = axis_angle_to_matrix(
                torch.cat(
                    [
                        global_orient,
                        body_pose,
                        jaw_pose,
                        leye_pose,
                        reye_pose,
                        left_hand_pose,
                        right_hand_pose,
                    ],
                    dim=-1,
                ).reshape(*BL, 55, 3)
            )
        else:
            assert rotation_type == "r6d"
            raise NotImplementedError("R6D is not implemented yet, hand mean should be carefully handled")
            other_default_pose = axis_angle_to_matrix(other_default_pose.view(33, 3))
            part_full_pose = torch.cat([global_orient, body_pose], dim=-1)
            rot_mats = rotation_6d_to_matrix(part_full_pose.view(*part_full_pose.shape[:-1], 22, 6))
            other_default_pose = other_default_pose.expand(*rot_mats.shape[:-3], -1, -1, -1)
            rot_mats = torch.cat([rot_mats, other_default_pose], dim=-3)
            del part_full_pose, other_default_pose

        return rot_mats

    def forward(
        self,
        body_pose,
        betas,
        global_orient,
        transl=None,
        left_hand_pose=None,
        right_hand_pose=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        rotation_type="aa",
    ):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
            left_hand_pose: (B, L, 45)
            right_hand_pose: (B, L, 45)
        Returns:
            vertices: (B, L, V, 3)
            skeleton: (B, L, J, 3)
        """
        assert betas.shape[:-1] == body_pose.shape[:-1]

        BL = body_pose.shape[:-1]
        rot_mats = self.get_rotmats(
            body_pose, global_orient, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose, rotation_type
        )

        skeleton = self.get_skeleton(betas)
        skeleton, A = batch_rigid_transform(rot_mats, skeleton, self.parents)

        pose_feature = (rot_mats[..., 1:, :, :] - self.identity).view(*BL, -1)
        verts = (
            self.v_template
            + einsum(betas, self.shapedirs, "... k, v c k -> ... v c")
            + einsum(pose_feature, self.posedirs, "... k, k v c -> ... v c")
        )
        del pose_feature, rot_mats

        lbs_R = torch.einsum("vj, ...jcd -> ...vcd", self.lbs_weights, A[..., :3, :3])
        lbs_t = torch.einsum("vj, ...jc -> ...vc", self.lbs_weights, A[..., :3, 3])
        if transl is not None:
            lbs_t = lbs_t + transl[..., None, :]
            skeleton = skeleton + transl[..., None, :]
        verts = torch.einsum("...vcd, ...vd -> ...vc", lbs_R, verts) + lbs_t

        return verts, skeleton


class SmplxLiteSmplN24(SmplxLite):
    def __init__(self, **kwargs):
        """Configure the model to output the 24 neutral-SMPL joints."""
        super().__init__(**kwargs)

        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=True)
        smpl2joints = torch.load(SMPLN_J_REGRESSOR_PATH, weights_only=True)
        smplx2joints = torch.matmul(smpl2joints, smplx2smpl.to_dense())

        jids, smplx_vids = torch.where(smplx2joints != 0)
        smplx2joints_interested = torch.zeros([len(smplx_vids), smplx2joints.size(0)])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            smplx2joints_interested[idx, jid] = smplx2joints[jid, smplx_vid]
        self.register_buffer("smplx2joints_interested", smplx2joints_interested, False)

        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, **kwargs):
        """Returns: joints (*, J, 3). (B, L) or  (B,) are both supported."""

        verts, skeleton = super().forward(**kwargs)
        joints = einsum(self.smplx2joints_interested, verts, "v j, ... v c -> ... j c")
        return joints


class SmplxLiteCoco17(SmplxLite):
    def __init__(self, **kwargs):
        """Configure fast COCO17-joint output without full vertices."""
        super().__init__(**kwargs)

        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=True)
        coco17_regressor = torch.load(SMPL2COCO17_PATH, weights_only=True)
        smplx2coco17 = torch.matmul(coco17_regressor, smplx2smpl.to_dense())

        jids, smplx_vids = torch.where(smplx2coco17 != 0)
        smplx2coco17_interestd = torch.zeros([len(smplx_vids), 17])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            smplx2coco17_interestd[idx, jid] = smplx2coco17[jid, smplx_vid]
        self.register_buffer("smplx2coco17_interestd", smplx2coco17_interestd, False)

        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, **kwargs):
        """Returns: joints (*, 17, 3). (B, L) or  (B,) are both supported."""

        verts, skeleton = super().forward(**kwargs)
        joints = einsum(self.smplx2coco17_interestd, verts, "v j, ... v c -> ... j c")
        return joints


class SmplxLiteV437Coco17(SmplxLite):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=True)
        coco17_regressor = torch.load(SMPL2COCO17_PATH, weights_only=True)
        smplx2coco17 = torch.matmul(coco17_regressor, smplx2smpl.to_dense())

        jids, smplx_vids = torch.where(smplx2coco17 != 0)
        smplx2coco17_interestd = torch.zeros([len(smplx_vids), 17])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            smplx2coco17_interestd[idx, jid] = smplx2coco17[jid, smplx_vid]
        self.register_buffer("smplx2coco17_interestd", smplx2coco17_interestd, False)
        assert len(smplx_vids) == 132

        smplx_vids2 = torch.load(SMPLX_VERTS437_PATH, weights_only=True)
        smplx_vids = torch.cat([smplx_vids, smplx_vids2])

        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, **kwargs):
        """
        Returns:
            verts_437: (*, 437, 3)
            joints (*, 17, 3). (B, L) or  (B,) are both supported.
        """

        verts, skeleton = super().forward(**kwargs)

        verts_437 = verts[..., 132:, :].clone()
        joints = einsum(self.smplx2coco17_interestd, verts[..., :132, :], "v j, ... v c -> ... j c")
        return verts_437, joints


class SmplxLiteV437Coco65(SmplxLite):
    def __init__(self, **kwargs):
        """
        Regressor: COCO17
        Vertex Points: Foot6, Lhands and Rhands end-points
        SmplX Skeleton: Foot6, Lhands21, Rhands21
        """
        super().__init__(**kwargs)

        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=True)
        coco17_regressor = torch.load(SMPL2COCO17_PATH, weights_only=True)
        smplx2coco17 = torch.matmul(coco17_regressor, smplx2smpl.to_dense())

        jids, smplx_vids_coco17 = torch.where(smplx2coco17 != 0)
        smplx2coco17_interestd = torch.zeros([len(smplx_vids_coco17), 17])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids_coco17)):
            smplx2coco17_interestd[idx, jid] = smplx2coco17[jid, smplx_vid]
        self.register_buffer("smplx2coco17_interestd", smplx2coco17_interestd, False)
        assert len(smplx_vids_coco17) == 132

        smplx_vids_foot = torch.tensor(
            [VERTEX_IDS["smplx"][n] for n in ["LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]]
        )

        smplx_vids_hand_tips = torch.tensor(
            [VERTEX_IDS["smplx"][f"l{n}"] for n in ["thumb", "index", "middle", "ring", "pinky"]]
            + [VERTEX_IDS["smplx"][f"r{n}"] for n in ["thumb", "index", "middle", "ring", "pinky"]]
        )
        skeleton_names_to_ids = {n: i for i, n in enumerate(SKELETON_NAMES[:55])}
        for lr in ["left", "right"]:
            for n in ["thumb4", "index4", "middle4", "ring4", "pinky4"]:
                skeleton_names_to_ids[f"{lr}_{n}"] = len(skeleton_names_to_ids)

        hand_joint_ids = []
        for lr in ["left", "right"]:
            hand_joint_ids.append(skeleton_names_to_ids[f"{lr}_wrist"])
            for n in ["thumb", "index", "middle", "ring", "pinky"]:
                hand_joint_ids.append(skeleton_names_to_ids[f"{lr}_{n}1"])
                hand_joint_ids.append(skeleton_names_to_ids[f"{lr}_{n}2"])
                hand_joint_ids.append(skeleton_names_to_ids[f"{lr}_{n}3"])
                hand_joint_ids.append(skeleton_names_to_ids[f"{lr}_{n}4"])
        self.hand_joint_ids = hand_joint_ids

        smplx_v437_vids = torch.load(SMPLX_VERTS437_PATH, weights_only=True)

        smplx_vids = torch.cat([smplx_vids_coco17, smplx_vids_foot, smplx_vids_hand_tips, smplx_v437_vids])

        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, **kwargs):
        """
        Returns:
            verts_437: (*, 437, 3)
            joints (*, 65, 3). (B, L) or  (B,) are both supported.
        """

        verts, skeleton = super().forward(**kwargs)
        assert skeleton.shape[-2] == 55

        verts_coco17 = verts[..., :132, :]
        joints_foot = verts[..., 132:138, :]
        joints_hand_tips = verts[..., 138:148, :]
        verts_437 = verts[..., -437:, :]
        assert verts.shape[-2] == 148 + 437

        joints_coco17 = einsum(self.smplx2coco17_interestd, verts_coco17, "v j, ... v c -> ... j c")
        skeleton_extended = torch.cat([skeleton, joints_hand_tips], dim=-2)

        joints_lrhand = skeleton_extended[..., self.hand_joint_ids, :]
        joints_65 = torch.cat([joints_coco17, joints_foot, joints_lrhand], dim=-2)

        return verts_437, joints_65
