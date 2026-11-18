"""Portable explicit mapping overrides, independent of Blender and Qt."""
SIDES=('Left','Right')
MAIN=('Hips','Head',*(s+p for s in SIDES for p in ('Shoulder','Arm','ForeArm','Hand','Leg','Shin','Foot','ToeBase')))
FINGERS=tuple(s+d+str(i) for s in SIDES for d in ('Thumb','Index','Middle','Ring','Pinky') for i in range(1,5))
TOES=tuple(s+d+str(i) for s in SIDES for d in ('BigToe','SecondToe','MiddleToe','FourthToe','LittleToe') for i in range(1,4))
ROLES=MAIN+FINGERS+TOES
FORMAT='rgbmocap.toyxyz_mapping'


def validate(value,available=None):
    if not isinstance(value,dict) or value.get('version')!=1 or value.get('format') not in (FORMAT,'toyxyz_live_retarget.bone_mapping'):
        raise ValueError('Choose an RGB MoCap or Toyxyz bone-mapping JSON (version 1).')
    result=dict(format=FORMAT,version=1)
    for side in ('source','target'):
        entry=value.get(side)
        if not isinstance(entry,dict) or not isinstance(entry.get('bones'),dict):raise ValueError('Missing '+side+' bone mapping.')
        bones=entry['bones']
        if set(bones)-set(ROLES):raise ValueError('Unknown bone role in '+side+' mapping.')
        out=dict(bones=dict(bones))
        for chain,minimum in [('spine',1),('neck',0)]:
            names=entry.get(chain)
            if names is not None and (not isinstance(names,list) or not minimum<=len(names)<=12):raise ValueError('Invalid '+side+' '+chain+' count.')
            out[chain]=None if names is None else list(names)
        names=list(bones.values())+sum((v for k,v in out.items() if k!='bones' and v is not None),[])
        if any(not isinstance(n,str) or '\0' in n or len(n)>512 for n in names):raise ValueError('Invalid bone name.')
        used=[n for n in names if n]
        if len(used)!=len(set(used)):raise ValueError(side.title()+': a bone is assigned to multiple roles.')
        if available is not None:
            missing=set(used)-set(available[side])
            if missing:raise ValueError(side.title()+' bones not found: '+', '.join(sorted(missing)))
        result[side]=out
    return result


def apply(settings,data,mapping):
    data=validate(data,{side:[b.name for b in getattr(settings,side).rig.data.bones] for side in ('source','target')})
    for side in ('source','target'):
        profile=getattr(settings,side);entry=data[side]
        for role,name in entry['bones'].items():setattr(profile,mapping.role_property(role),name)
        for chain in ('spine','neck'):
            names=entry[chain]
            if names is None:continue
            setattr(profile,chain+'_count',len(names));items=getattr(profile,chain+'_bones');mapping.resize(items,len(names))
            for item,name in zip(items,names):item.bone=name
        mapping.mapped_names(profile)
