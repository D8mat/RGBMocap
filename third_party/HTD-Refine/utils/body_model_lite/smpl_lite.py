import torch
import torch.nn as nn
import pickle
from pathlib import Path
from einops import einsum, rearrange

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .utils import Struct, batch_rigid_transform, to_np, to_tensor
from pytorch3d.transforms import axis_angle_to_matrix

MODEL_PATH = PROJECT_ROOT / "inputs/checkpoints/body_models/smpl"
SMPL2COCO17_PATH = PROJECT_ROOT / "utils/body_model/smpl_coco17_J_regressor.pt"


class SmplLite(nn.Module):
    def __init__(
        self,
        model_path=MODEL_PATH,
        gender="neutral",
        num_betas=10,
    ):
        super().__init__()

        model_path = Path(model_path)
        if model_path.is_dir():
            smpl_path = Path(model_path) / f"SMPL_{gender.upper()}.pkl"
        else:
            smpl_path = model_path
        assert smpl_path.exists()
        with open(smpl_path, "rb") as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))
        self.faces = data_struct.f

        self.register_smpl_buffers(data_struct, num_betas)
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

    def register_fast_skeleton_computing_buffers(self):

        J_template = self.J_regressor @ self.v_template
        J_shapedirs = torch.einsum("jv, vcd -> jcd", self.J_regressor, self.shapedirs)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

    def get_skeleton(self, betas):
        return self.J_template + torch.einsum("...k, jck -> ...jc", betas, self.J_shapedirs)

    def fk_skeleton(self, body_pose, betas, global_orient, transl):
        """
        Args:
            body_pose: (B, L, 69)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
        """
        assert betas.shape[:-1] == body_pose.shape[:-1]

        BL = body_pose.shape[:-1]
        rot_mats = axis_angle_to_matrix(
            torch.cat(
                [
                    global_orient,
                    body_pose,
                ],
                dim=-1,
            ).view(*BL, 24, 3)
        )

        skeleton = self.get_skeleton(betas)
        skeleton, A = batch_rigid_transform(rot_mats, skeleton, self.parents)

        if transl is not None:
            skeleton = skeleton + transl[..., None, :]

        return skeleton

    def forward(self, body_pose, betas, global_orient, transl):
        """
        Args:
            body_pose: (B, L, 69)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
        Returns:
            vertices: (B, L, V, 3)
            skeleton: (B, L, J, 3)
        """
        assert betas.shape[:-1] == body_pose.shape[:-1]

        BL = body_pose.shape[:-1]
        rot_mats = axis_angle_to_matrix(
            torch.cat(
                [
                    global_orient,
                    body_pose,
                ],
                dim=-1,
            ).view(*BL, 24, 3)
        )

        skeleton = self.get_skeleton(betas)
        skeleton, A = batch_rigid_transform(rot_mats, skeleton, self.parents)

        pose_feature = (rot_mats[..., 1:, :, :] - self.identity).view(*BL, -1)
        verts = (
            self.v_template
            + torch.einsum("...k, vck -> ...vc", betas, self.shapedirs)
            + torch.einsum("...k, kvc -> ...vc", pose_feature, self.posedirs)
        )
        del pose_feature, rot_mats

        lbs_R = torch.einsum("vj, ...jcd -> ...vcd", self.lbs_weights, A[..., :3, :3])
        lbs_t = torch.einsum("vj, ...jc -> ...vc", self.lbs_weights, A[..., :3, 3])
        if transl is not None:
            lbs_t = lbs_t + transl[..., None, :]
            skeleton = skeleton + transl[..., None, :]
        verts = torch.einsum("...vcd, ...vd -> ...vc", lbs_R, verts) + lbs_t

        return verts, skeleton


class SmplLiteJ24(SmplLite):
    """
    Note that J24 is not exactly identical to the original SMPL posed skeleton,
    as SMPL uses posed shape to modify the vertices, and J24 is computed from the vertices from LBS.
    To get the exact posed skeleton, use fk_skeleton() instead (Fast).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        smpl2j24 = self.J_regressor

        jids, smplx_vids = torch.where(smpl2j24 != 0)
        interestd = torch.zeros([len(smplx_vids), 24])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            interestd[idx, jid] = smpl2j24[jid, smplx_vid]
        self.register_buffer("interestd", interestd, False)

        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, body_pose, betas, global_orient, transl):
        """Returns: joints (*, J, 3). (B, L) or (B,) are both supported."""

        verts, skeleton = super().forward(body_pose, betas, global_orient, transl)

        J24 = torch.einsum("vj, ...vc -> ...jc", self.interestd, verts)
        return J24


class SmplLiteCoco17(SmplLite):
    def __init__(self, **kwargs):
        """Configure fast COCO17-joint output without full vertices."""
        super().__init__(**kwargs)

        smpl2coco17 = torch.load(SMPL2COCO17_PATH, weights_only=True)

        jids, smplx_vids = torch.where(smpl2coco17 != 0)
        smpl2coco17_interestd = torch.zeros([len(smplx_vids), 17])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            smpl2coco17_interestd[idx, jid] = smpl2coco17[jid, smplx_vid]
        self.register_buffer("smpl2coco17_interestd", smpl2coco17_interestd, False)

        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, **kwargs):
        """Returns: joints (*, 17, 3). (B, L) or  (B,) are both supported."""

        verts, skeleton = super().forward(**kwargs)
        joints = einsum(self.smpl2coco17_interestd, verts, "v j, ... v c -> ... j c")
        return joints
