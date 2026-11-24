"""Virtual A-to-T reference calibration; never changes skin bind or source motion."""
from mathutils import Matrix


def reference(rig, matrices, roles, pose):
    if pose not in ('REST','T','A'):
        raise ValueError('Unknown target reference pose: '+str(pose))
    result={name:value.copy() for name,value in matrices.items()}
    if pose!='A':return result
    def head(role):return matrices[roles[role]].translation
    up=head('Head')-head('Hips')
    if up.length<1e-6:raise ValueError('Cannot determine target up axis from Hips/Head mapping')
    up.normalize()
    lateral=head('LeftArm')-head('RightArm')
    lateral-=up*lateral.dot(up)
    if lateral.length<1e-6:raise ValueError('Cannot determine target shoulder axis')
    lateral.normalize()
    for side,sign in (('Left',1),('Right',-1)):
        arm=rig.data.bones[roles[side+'Arm']]
        pivot=head(side+'Arm')
        direction=head(side+'Hand')-pivot
        if direction.length<1e-6:raise ValueError('Cannot calibrate zero-length arm: '+side)
        turn=direction.normalized().rotation_difference(lateral*sign).to_matrix().to_4x4()
        transform=Matrix.Translation(pivot) @ turn @ Matrix.Translation(-pivot)
        for bone in (arm,*arm.children_recursive):
            result[bone.name]=transform @ matrices[bone.name]
    return result


def configure(profile, pose):
    from toyxyz_live_retarget.mapping import mapped_names
    from toyxyz_live_retarget.retarget import reference_matrices,save_reference
    roles,_,_=mapped_names(profile)
    matrices=reference(profile.rig,reference_matrices(profile,use_saved=False),roles,pose)
    if pose=='A':save_reference(profile,profile.rig,matrices)
