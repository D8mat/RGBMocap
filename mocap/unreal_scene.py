"""Blender-only FBX skin builder; centimetres, one root, native deformation bones."""
import sys,json,time
started=time.perf_counter();timings={}
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector

folder=Path(sys.argv[sys.argv.index('--')+1]);job=json.loads((folder/'unreal-job.json').read_text())
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=.01
scene.render.fps=round(job['fps']);scene.render.fps_base=round(job['fps'])/job['fps']
C=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.unreal_ground import offsets
with np.load(folder/'skin.npz') as z:
    native_vertices=z['vertices'];faces=z['faces'];weights=z['weights'];native_joints=z['joints'];parents=z['parents']
frames=np.load(folder/'motion.npy',mmap_mode='r');offset=int(job['prepend_tpose'])
tpose=np.load(folder/'tpose.npy') if offset else None
ground=offsets(native_vertices,native_joints,weights,frames[0],tpose,float(job.get('z_offset_cm',0.)),bool(job.get('ground_feet',True)))
REST_SHIFT=np.array([0.,0.,ground['rest']])
vertices=native_vertices@C.T*100+REST_SHIFT;joints=native_joints@C.T*100+REST_SHIFT
ROOT_SHIFT=np.array([0.,0.,float(job.get('z_offset_cm',0.))]) if job.get('ground_feet',True) else REST_SHIFT
print('GROUND '+json.dumps(ground),flush=True)
heading=Matrix.Identity(4)
if job.get('align_heading',False):
    from mocap.unreal_smplh import tpose as make_tpose
    import math
    neutral=tpose if tpose is not None else make_tpose(native_joints,parents,frames[0])
    rest=C@(native_joints[16]-native_joints[17])
    posed=C@(neutral[16,:3,3]-neutral[17,:3,3])
    if np.linalg.norm(posed[:2])<1e-6:raise ValueError('Cannot align heading: initial shoulders are vertical')
    yaw=math.atan2(rest[1],rest[0])-math.atan2(posed[1],posed[0])
    pivot=Vector(C@frames[0,0,:3,3]*100);pivot.z=0
    heading=Matrix.Translation(pivot)@Matrix.Rotation(yaw,4,'Z')@Matrix.Translation(-pivot)
    print('HEADING degrees='+str(math.degrees(yaw)),flush=True)
arm=bpy.data.armatures.new('RGBMoCapSkeleton');rig=bpy.data.objects.new('RGBMoCapRig',arm);scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
root=arm.edit_bones.new('root');root.head=ROOT_SHIFT;root.tail=ROOT_SHIFT+[0,0,10];root.use_deform=False
for i,name in enumerate(job['names']):
    bone=arm.edit_bones.new(name);bone.head=joints[i]
    children=np.flatnonzero(parents==i);tail=joints[children[0]] if len(children) else joints[i]+[0,0,3]
    if np.linalg.norm(tail-joints[i])<.01:tail=joints[i]+[0,0,3]
    bone.tail=tail
for i,name in enumerate(job['names']):arm.edit_bones[name].parent=root if parents[i]<0 else arm.edit_bones[job['names'][parents[i]]]
bpy.ops.object.mode_set(mode='OBJECT')
mesh=bpy.data.meshes.new('BodyMesh');mesh.from_pydata(vertices.tolist(),[],faces.tolist());mesh.update()
obj=bpy.data.objects.new('Body',mesh);scene.collection.objects.link(obj);obj.parent=rig
for face in mesh.polygons:face.use_smooth=True
for i,name in enumerate(job['names']):
    group=obj.vertex_groups.new(name=name)
    for v in np.flatnonzero(weights[:,i]>0):group.add([int(v)],float(weights[v,i]),'REPLACE')
mod=obj.modifiers.new('Skin','ARMATURE');mod.object=rig
mat=bpy.data.materials.new('Body');mat.diffuse_color=(.5,.55,.6,1);mesh.materials.append(mat)
scene.frame_start=0;scene.frame_end=len(frames)-1+offset
basis=[rig.data.bones[n].matrix_local.to_3x3() for n in job['names']]
for pb in rig.pose.bones:pb.rotation_mode='QUATERNION'
import itertools
sequence=itertools.chain([tpose] if offset else [],frames)
count=len(frames)+offset
if job.get('preview_only'):count=min(count,max(2,round(job['fps']*3)+offset))
scene.frame_end=count-1
samples=np.empty((len(job['names']),count,10),dtype=np.float64)
rest=[rig.data.bones[n].matrix_local.copy() for n in job['names']]
root_rest=rig.data.bones['root'].matrix_local.copy()
timings['skin_and_rig_seconds']=time.perf_counter()-started
bake_started=time.perf_counter()
for f,world in enumerate(itertools.islice(sequence,count)):
    SHIFT=np.array([0.,0.,ground['tpose' if offset and f==0 else 'motion']])
    desired=[]
    for i,name in enumerate(job['names']):
        matrix=Matrix((C@world[i,:3,:3]@C.T).tolist()).to_4x4();matrix.translation=Vector(C@world[i,:3,3]*100+SHIFT)
        desired.append(heading@matrix@basis[i].to_4x4())
    for i,name in enumerate(job['names']):
        p=int(parents[i])
        local=rig.data.bones[name].convert_local_to_pose(desired[i],rest[i],
            parent_matrix=desired[p] if p>=0 else root_rest,
            parent_matrix_local=rest[p] if p>=0 else root_rest,invert=True)
        loc,quat,scale=local.decompose()
        if f and np.dot(samples[i,f-1,3:7],quat)<0:quat.negate()
        samples[i,f]=[*loc,*quat,*scale]
    if f%100==0:print('@@BLEND@@'+json.dumps(dict(current=f,total=count,message='Baking Unreal skeleton')),flush=True)
rig.animation_data_create();action=bpy.data.actions.new('Motion');rig.animation_data.action=action
indices=np.arange(count,dtype=np.float64)
for i,name in enumerate(job['names']):
    pb=rig.pose.bones[name]
    for prop,start,width in (('location',0,3),('rotation_quaternion',3,4),('scale',7,3)):
        for axis in range(width):
            curve=action.fcurves.new(pb.path_from_id(prop),index=axis,action_group=name)
            curve.keyframe_points.add(count)
            curve.keyframe_points.foreach_set('co',np.column_stack((indices,samples[i,:,start+axis])).ravel())
            for k in curve.keyframe_points:k.interpolation='LINEAR'
            curve.update()
timings['animation_seconds']=time.perf_counter()-bake_started
if hasattr(action,'slots') and len(action.slots):rig.animation_data.action_slot=action.slots[0]
rig.animation_data.action.name='Motion'
scene.frame_set(0);rig.select_set(True);obj.select_set(True);bpy.context.view_layer.objects.active=rig
if job.get('preview_only'):
    bpy.context.view_layer.update()
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());bound=[evaluated.matrix_world@Vector(p) for p in evaluated.bound_box]
    center=sum(bound,Vector())/8;extent=max((p-center).length for p in bound)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active;space.clip_end=100000;space.region_3d.view_location=center;space.region_3d.view_distance=max(100,extent*3)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/job['target']))
    timings['total_seconds']=time.perf_counter()-started
    (folder/'timings.json').write_text(json.dumps(timings),encoding='utf8')
    raise SystemExit(0)
export_started=time.perf_counter()
bpy.ops.export_scene.fbx(filepath=str(folder/job['target']),use_selection=True,object_types={'ARMATURE','MESH'},global_scale=1,apply_unit_scale=True,apply_scale_options='FBX_SCALE_NONE',axis_forward='-Y',axis_up='Z',add_leaf_bones=False,use_armature_deform_only=False,armature_nodetype='NULL',bake_anim=True,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False,bake_anim_force_startend_keying=True,bake_anim_simplify_factor=0,mesh_smooth_type='FACE',use_mesh_modifiers=True)

timings['fbx_seconds']=time.perf_counter()-export_started
timings['total_seconds']=time.perf_counter()-started
(folder/'timings.json').write_text(json.dumps(timings),encoding='utf8')
