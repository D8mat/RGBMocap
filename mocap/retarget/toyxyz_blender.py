"""Run the separately licensed Toyxyz add-on in the isolated Blender worker."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import bpy
from .file_paths import io_path,save_blend


def bake(out,kind):
    from .blender_io import report
    package=Path(__file__).resolve().parents[2]/'third_party/toyxyz_retargeter'
    sys.path.insert(0,str(package))
    import toyxyz_live_retarget as addon
    from toyxyz_live_retarget import actors,mapping,runtime,bake as action_bake
    data=json.loads((out/'toyxyz-options.json').read_text(encoding='utf-8'))
    options=data['options'];n=int(data['frames']);fps=float(data['fps'])
    bpy.ops.wm.open_mainfile(filepath=io_path(out/f'{kind}-rest.blend'),load_ui=False)
    target=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    before=set(bpy.context.scene.objects)
    bpy.ops.import_anim.bvh(filepath=io_path(out/'toyxyz-source.bvh'),axis_forward='-Z',axis_up='Y',
                            global_scale=1.,frame_start=1,use_fps_scale=False,update_scene_fps=False)
    source=next(o for o in bpy.context.scene.objects if o not in before and o.type=='ARMATURE')
    scene=bpy.context.scene;scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
    scene.frame_start=1;scene.frame_end=n;scene.frame_set(1)
    addon.register()
    try:
        row=actors.append(scene);settings=row.settings
        settings.source.rig=source;settings.target.rig=target
        mapping.build_mapping(settings.source);mapping.build_mapping(settings.target)
        if options.get('puppet_reference'):
            if bpy.ops.toyxyz_retarget.puppet_v5_mapping()!={'FINISHED'}:raise ValueError('Puppet V5 mapping could not be applied.')
        if options.get('mapping') is not None:
            from mocap.toyxyz_mapping import apply as apply_mapping
            apply_mapping(settings,options['mapping'],mapping)
        for name in ('solver_strength','hip_offset','floor_contact','hand_position_reach','foot_position_reach'):
            if name in options:setattr(settings,name,options[name])
        if options.get('travel') in {'PROPORTIONAL','SOURCE','IN_PLACE'}:settings.travel=options['travel']
        from .target_reference import configure
        configure(settings.target,options.get('target_pose','REST'))
        report('Toyxyz: target reference '+options.get('target_pose','REST'))
        report('Toyxyz: connecting mapped source and target')
        active=runtime.start(bpy.context,key=row.key)
        keyed=sorted(active.target.mapped.values())
        settings_bake=SimpleNamespace(existing_keys=False,step=1,interpolation='LINEAR',handles='AUTO_CLAMPED',keep_action=True,mute_constraints=True)
        last=[-1]
        def progress(value):
            step=int(value)//10
            if step!=last[0]:report(f'Toyxyz: baking and verifying {min(100,int(value))}%');last[0]=step
        action_bake.bake_action(bpy.context,active,settings_bake,1,n,'RGB MoCap Toyxyz',(),progress)
        settings.source.rig=None;settings.target.rig=None
    finally:addon.unregister()
    bpy.data.objects.remove(source,do_unlink=True)
    scene.frame_set(1);save_blend(out/'animation.blend')
    return target,keyed
