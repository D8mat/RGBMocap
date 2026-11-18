"""Recognize non-Mixamo humanoids without renaming or rebuilding their bones."""
import sys
from pathlib import Path
import bpy
from .mapping import NAMES


def recognize(arm):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/toyxyz_retargeter'))
    import toyxyz_live_retarget as addon
    from toyxyz_live_retarget import actors,mapping
    from toyxyz_live_retarget.mapping_ui import export_profile
    addon.register()
    try:
        settings=actors.append(bpy.context.scene).settings
        profile=settings.target;profile.rig=arm
        mapping.build_mapping(profile)
        fixed,spine,neck=mapping.mapped_names(profile)
        if len(spine)<3 or not neck:
            raise ValueError('This import needs at least three distinct spine bones and a neck bone. This rig requires a custom import profile.')
        aliases={'LeftUpLeg':'LeftLeg','RightUpLeg':'RightLeg','LeftLeg':'LeftShin','RightLeg':'RightShin'}
        body={name:fixed.get(aliases.get(name,name),'') for name in NAMES}
        body.update(Spine=spine[0],Spine1=spine[(len(spine)-1)//2],Spine2=spine[-1],Neck=neck[0])
        if any(not body[n] for n in NAMES) or len(set(body.values()))!=22:
            raise ValueError('Humanoid import needs distinct body, shoulder and toe bones. Review the source rig.')
        semantic=dict(body)
        for side in ('Left','Right'):
            for digit in ('Thumb','Index','Middle','Ring','Pinky'):
                for i in range(1,4):
                    name=mapping.bone_for_role(profile,side+digit+str(i))
                    if name:semantic[side+'Hand'+digit+str(i)]=name
        return semantic,export_profile(profile)
    finally:
        bpy.context.scene.toyxyz_retarget_actors.clear()
        addon.unregister()
