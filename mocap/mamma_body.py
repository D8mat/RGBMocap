"""Exact MAMMA body-data bridge. Never substitute the standard SMPL-X model.

The editor's GPU skin format has ten shape slots. For a constant 16-beta
MAMMA actor, fold the final six linear shape terms into a private template.
This is algebraically exact, not a shape approximation or a motion refit.
All derived body data stays in the user's project, under its original license.
"""
from pathlib import Path
import hashlib
import json
import os
import uuid
import numpy as np
from .storage import atomic_json
from .result_artifacts import file_stat

VERSION = 1
WORLD_ROTATION = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def body_path(project):
    """Resolve a project-private registered body; no fallback on bad metadata."""
    project = Path(project).resolve()
    state = json.loads((project/'project.json').read_text(encoding='utf-8'))
    if state.get('pipeline', {}).get('model') != 'mamma':
        from .models import BY_KEY
        return BY_KEY['smplx'].path
    entry = state.get('mamma_body')
    if not isinstance(entry, dict):
        raise ValueError('MAMMA actor body data is missing. Prepare its source result again.')
    name = entry.get('file', '')
    path = (project/name).resolve()
    if not name or not path.is_relative_to(project) or path == project:
        raise ValueError('Invalid MAMMA body data path.')
    if not path.is_file() or file_stat(path) != entry.get('stat'):
        raise ValueError('MAMMA actor body data changed or is missing; the result was preserved.')
    return path


def model_for(project):
    import smplx
    return smplx.SMPLX(str(body_path(project)), gender='neutral', num_pca_comps=12,
                      flat_hand_mean=False, batch_size=1).eval()


def make_actor_body(source, betas, project):
    source, project = Path(source), Path(project)
    source_stamp=file_stat(source)
    beta = np.asarray(betas, np.float64).reshape(-1)
    if beta.shape != (16,) or not np.isfinite(beta).all():
        raise ValueError('MAMMA requires sixteen constant finite shape coefficients.')
    # Registered model NPZs are arrays, never arbitrary pickled Python objects.
    required = ('shapedirs', 'v_template', 'f', 'J_regressor', 'posedirs',
                'kintree_table', 'weights', 'hands_componentsl', 'hands_componentsr',
                'hands_meanl', 'hands_meanr', 'lmk_faces_idx', 'lmk_bary_coords')
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in required}
        for key in ('dynamic_lmk_faces_idx', 'dynamic_lmk_bary_coords'):
            if key in archive.files:
                data[key] = archive[key]
    dirs, template = data['shapedirs'], data['v_template']
    if dirs.shape[:2] != (10475, 3) or dirs.shape[-1] < 310 or template.shape != (10475, 3):
        raise ValueError('Import the supported locked-head SMPL-X NPZ, including its expression directions.')
    if not np.isfinite(dirs).all() or not np.isfinite(template).all():
        raise ValueError('The registered body model contains non-finite data.')
    delta = np.einsum('vci,i->vc', dirs[..., 10:16].astype(np.float64), beta[10:])
    data['v_template'] = (template.astype(np.float64)+delta).astype(template.dtype)
    if file_stat(source)!=source_stamp:raise ValueError('The registered MAMMA body changed during conversion.')
    from .storage import sha256
    token = hashlib.sha256((sha256(Path(__file__))+str(VERSION)+str(source_stamp)+beta.tobytes().hex()).encode()).hexdigest()[:20]
    destination = project/'body'/('SMPLX_NEUTRAL-'+token+'.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt=destination.with_suffix('.receipt.json')
    existing=False
    try:existing=json.loads(receipt.read_text(encoding='utf-8')).get('stat')==file_stat(destination)
    except (OSError,ValueError,TypeError):pass
    if destination.exists() and not existing:
        # Never certify or overwrite an unverified derived model. An old
        # renderer may still own arrays from its immutable generation.
        destination=destination.with_name(destination.stem+'-'+uuid.uuid4().hex[:12]+'.npz')
        receipt=destination.with_suffix('.receipt.json')
    if not existing:
        temporary = destination.with_name('.'+uuid.uuid4().hex+'.npz')
        try:
            np.savez_compressed(temporary, **data)
            os.replace(temporary, destination)
            atomic_json(receipt,dict(version=VERSION,stat=file_stat(destination)))
        finally:
            temporary.unlink(missing_ok=True)
    expected = template.astype(np.float64)+np.einsum('vci,i->vc', dirs[..., :16].astype(np.float64), beta)
    actual = data['v_template'].astype(np.float64)+np.einsum('vci,i->vc', dirs[..., :10].astype(np.float64), beta[:10])
    error = float(np.abs(expected-actual).max())
    if error > 1e-6:
        raise ValueError('MAMMA body conversion did not preserve registered geometry.')
    return dict(version=VERSION, file=destination.relative_to(project).as_posix(),
                stat=file_stat(destination), source_stat=file_stat(source),
                original_betas=beta.tolist(), template_error_max_m=error,
                license='Derived registered SMPL-X data; do not redistribute.')


def convert_motion(pose, betas, translation, body):
    """Z-up MAMMA world to Y-up editor, around the shaped pelvis pivot."""
    import torch
    from scipy.spatial.transform import Rotation
    pose = np.asarray(pose, np.float32)
    trans = np.asarray(translation, np.float32)
    beta = np.asarray(betas, np.float32).reshape(1, 16)
    n = len(pose)
    if pose.shape != (n, 165) or trans.shape != (n, 3) or n < 1:
        raise ValueError('Invalid MAMMA frame/pose/translation dimensions.')
    if not all(np.isfinite(a).all() for a in (pose, trans, beta)):
        raise ValueError('MAMMA motion contains non-finite data.')
    shaped = body.v_template.detach().numpy()+np.einsum('vci,i->vc', body.shapedirs.detach().numpy(), beta[0, :10])
    pelvis = body.J_regressor.detach().numpy()[0]@shaped
    root = Rotation.from_matrix(WORLD_ROTATION@Rotation.from_rotvec(pose[:, :3]).as_matrix()).as_rotvec().astype(np.float32)
    moved = ((trans+pelvis)@WORLD_ROTATION.T-pelvis).astype(np.float32)
    # MAMMA's cached parameters already include the body's non-flat hand mean
    # convention. Do not subtract/re-add it as though these were ExMR angles.
    params = dict(global_orient=torch.from_numpy(root), body_pose=torch.from_numpy(pose[:, 3:66].copy()),
                  jaw_pose=torch.from_numpy(pose[:, 66:69].copy()), leye_pose=torch.from_numpy(pose[:, 69:72].copy()),
                  reye_pose=torch.from_numpy(pose[:, 72:75].copy()), left_hand_pose=torch.from_numpy(pose[:, 75:120].copy()),
                  right_hand_pose=torch.from_numpy(pose[:, 120:165].copy()), transl=torch.from_numpy(moved),
                  betas=torch.from_numpy(np.repeat(beta[:, :10], n, axis=0)), expression=torch.zeros(n, 10))
    return params


def contact_probabilities(body, betas, downsample, floor_contact):
    """Map MAMMA surface support to the existing four sole-region inputs."""
    weights = body.lbs_weights.detach().numpy()
    shaped = body.v_template.detach().numpy()+np.einsum('vci,i->vc', body.shapedirs.detach().numpy(), np.asarray(betas)[:10])
    D = np.asarray(downsample)
    probabilities = np.asarray(floor_contact, np.float32)
    if D.shape != (512, 10475) or probabilities.ndim != 2 or probabilities.shape[1] != 512:
        raise ValueError('Invalid MAMMA contact topology.')
    if not np.isfinite(D).all() or not np.isfinite(probabilities).all():
        raise ValueError('Non-finite MAMMA contact data.')
    points, skin = D@shaped, D@weights
    result = []
    for ankle, toe in ((7, 10), (8, 11)):
        ids = np.flatnonzero(skin[:, ankle]+skin[:, toe] > .6)
        physical = np.flatnonzero(weights[:, ankle]+weights[:, toe] > .6)
        middle = np.median(shaped[physical, 2])
        for front in (False, True):
            region = ids[(points[ids, 2] >= middle) if front else (points[ids, 2] <= middle)]
            if not len(region):
                raise ValueError('MAMMA sole landmark region is empty.')
            result.append(probabilities[:, region].max(1))
    return np.clip(np.stack(result, axis=1), 0, 1)
