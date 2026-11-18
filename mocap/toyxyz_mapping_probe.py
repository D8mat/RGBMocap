"""Read automatic mappings using the same Blender/add-on matching as the bake."""
import json,sys
from pathlib import Path


def main():
    import bpy
    root,rest,bvh,output=sys.argv[sys.argv.index('--')+1:]
    sys.path.insert(0,str(Path(root)/'third_party/toyxyz_retargeter'))
    import toyxyz_live_retarget as addon
    from toyxyz_live_retarget import actors,mapping
    from toyxyz_live_retarget.mapping_ui import export_profile
    bpy.ops.wm.open_mainfile(filepath=rest,load_ui=False)
    target=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    # Matching only needs the hierarchy. Avoid loading a complete long action.
    short=Path(output).with_suffix('.bvh')
    with open(bvh,encoding='utf-8-sig') as src,short.open('w',encoding='utf-8') as dst:
        for line in src:
            dst.write(line)
            if line.strip()=='MOTION':break
        else:raise ValueError('Missing BVH motion header.')
        next(src);dst.write('Frames: 1\n');dst.write(next(src));dst.write(next(src))
    before=set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(filepath=str(short),axis_forward='-Z',axis_up='Y',global_scale=1.,frame_start=1,use_fps_scale=False,update_scene_fps=False)
    source=next(o for o in bpy.context.scene.objects if o not in before and o.type=='ARMATURE')
    addon.register()
    try:
        settings=actors.append(bpy.context.scene).settings
        settings.source.rig=source;settings.target.rig=target
        mapping.build_mapping(settings.source);mapping.build_mapping(settings.target)
        result={side:export_profile(getattr(settings,side)) for side in ('source','target')}
        Path(output).write_text(json.dumps(result),encoding='utf-8')
    finally:addon.unregister()


if __name__=='__main__':main()
