"""Full finger rotations without PCA loss; legacy body evaluation stays unchanged."""
import numpy as np


def evaluate(body, parameters):
    if not any(k in parameters and parameters[k].shape[-1]==45 for k in ('left_hand_pose','right_hand_pose')):
        return body(**parameters)
    import torch
    bm=getattr(body,'bm',body);kwargs=dict(parameters)
    count=len(parameters['body_pose'])
    for side in ('left','right'):
        name=side+'_hand_pose';hand=kwargs.get(name)
        if hand is None:hand=torch.zeros(count,45,dtype=bm.v_template.dtype,device=bm.v_template.device)
        elif hand.shape[-1]!=45:hand=hand@getattr(bm,side+'_hand_components')[:hand.shape[-1]]
        kwargs[name]=hand
    for k,size in [('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]:
        kwargs.setdefault(k,torch.zeros(count,size,dtype=bm.v_template.dtype,device=bm.v_template.device))
    pca=bm.use_pca;dim=getattr(body,'hand_pose_dim',None)
    try:
        bm.use_pca=False
        if dim is not None:body.hand_pose_dim=45
        return body(**kwargs)
    finally:
        bm.use_pca=pca
        if dim is not None:body.hand_pose_dim=dim


def apply_fingers(local,params,body,binding,first,end):
    """Transport each outgoing finger rotation from its fixed neutral frame."""
    if not any(k in params and params[k].shape[-1]==45 for k in ('left_hand_pose','right_hand_pose')):return
    from scipy.spatial.transform import Rotation
    from .rig_mesh import JOINT_SOURCE
    bm=getattr(body,'bm',body)
    for side in range(2):
        field=('left','right')[side]+'_hand_pose'
        value=params[field][first:end].detach().cpu().numpy()
        mean=getattr(bm,('left','right')[side]+'_hand_mean').detach().cpu().numpy()
        rotations=Rotation.from_rotvec((value+mean).reshape(-1,3)).as_matrix().reshape(end-first,15,3,3)
        for f,source in enumerate((37,25,28,34,31)):
            for segment in range(3):
                sj=source+side*15+segment;target=22+side*15+f*3+segment
                parent_source=20+side if segment==0 else sj-1
                local[:,target]=binding['frames'][parent_source]@rotations[:,sj-(25+side*15)]@binding['frames'][sj].T
