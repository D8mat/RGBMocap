import torch
import torch.nn.functional as F
import numpy as np


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array, dtype=torch.float32):
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


VERTEX_IDS = {
    "smplh": {
        "nose": 332,
        "reye": 6260,
        "leye": 2800,
        "rear": 4071,
        "lear": 583,
        "rthumb": 6191,
        "rindex": 5782,
        "rmiddle": 5905,
        "rring": 6016,
        "rpinky": 6133,
        "lthumb": 2746,
        "lindex": 2319,
        "lmiddle": 2445,
        "lring": 2556,
        "lpinky": 2673,
        "LBigToe": 3216,
        "LSmallToe": 3226,
        "LHeel": 3387,
        "RBigToe": 6617,
        "RSmallToe": 6624,
        "RHeel": 6787,
    },
    "smplx": {
        "nose": 9120,
        "reye": 9929,
        "leye": 9448,
        "rear": 616,
        "lear": 6,
        "rthumb": 8079,
        "rindex": 7669,
        "rmiddle": 7794,
        "rring": 7905,
        "rpinky": 8022,
        "lthumb": 5361,
        "lindex": 4933,
        "lmiddle": 5058,
        "lring": 5169,
        "lpinky": 5286,
        "LBigToe": 5770,
        "LSmallToe": 5780,
        "LHeel": 8846,
        "RBigToe": 8463,
        "RSmallToe": 8474,
        "RHeel": 8635,
    },
    "mano": {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671},
}


def batch_rigid_transform(rot_mats, joints, parents):
    """Apply a batched kinematic chain and return posed joints and transforms."""

    root_joint = joints[..., :1, :]
    child_joints = joints[..., 1:, :]
    parent_joints = joints[..., parents[1:], :]
    rel_joints = torch.cat([root_joint, child_joints - parent_joints], dim=-2)

    top_part = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)

    shape = top_part.shape
    dtype = top_part.dtype
    device = top_part.device

    transforms_mat = torch.zeros((*shape[:-2], 4, 4), dtype=dtype, device=device)

    transforms_mat[..., :3, :3] = rot_mats

    transforms_mat[..., :3, 3:4] = rel_joints[..., None]

    transforms_mat[..., 3, 3] = 1

    transform_chain = [transforms_mat[..., 0, :, :]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[..., i, :, :])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=-3)

    posed_joints = transforms[..., :3, 3]

    correction = torch.matmul(transforms[..., :3, :3], joints[..., None]).squeeze(-1)

    corrected_transforms = transforms.clone()

    new_translation = transforms[..., :3, 3] - correction

    corrected_transforms = torch.cat(
        [torch.cat([transforms[..., :3, :3], new_translation[..., None]], dim=-1), transforms[..., 3:, :]], dim=-2
    )

    return posed_joints, corrected_transforms
