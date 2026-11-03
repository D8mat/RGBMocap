"""Run only inside Blender: metric point cloud, recorded rigs and aligned AI."""
import sys,json,math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector,Matrix,Quaternion

C=Matrix(((1,0,0),(0,0,-1),(0,1,0)))

def point(v):return C@Vector(v)

def make_rig(name,names,parents,positions):
    arm=bpy.data.armatures.new(name);obj=bpy.data.objects.new(name,arm);bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active=obj;obj.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
    for i,n in enumerate(names):
        bone=arm.edit_bones.new(n);bone.head=point(positions[i])
        children=[j for j,p in enumerate(parents) if p==i]
        tail=point(positions[children[0]]) if children else bone.head+Vector((0,0,.06))
        if (tail-bone.head).length<.005:tail=bone.head+Vector((0,0,.06))
        bone.tail=tail
    for i,p in enumerate(parents):
        if p>=0:arm.edit_bones[names[i]].parent=arm.edit_bones[names[p]]
    bpy.ops.object.mode_set(mode='OBJECT');obj.show_in_front=True
    for pb in obj.pose.bones:pb.rotation_mode='QUATERNION'
    obj.select_set(False);return obj

def pose(obj,names,positions,frame,matrices=None):
    # Parent-first traversal is necessary: Kinect's SpineShoulder index is 20.
    ordered=[]
    def visit(b):
        ordered.append(b)
        for child in b.children:visit(child)
    for b in obj.pose.bones:
        if b.parent is None:visit(b)
    for pb in ordered:
        i=names.index(pb.name);head=point(positions[i]);children=pb.children
        if matrices is not None:
            m=C.to_4x4()@Matrix(matrices[i].tolist())
        else:
            tail=point(positions[names.index(children[0].name)]) if children else head+pb.bone.matrix_local.to_3x3().col[1]*pb.bone.length
            d=tail-head
            if d.length<.005:d=pb.bone.matrix_local.to_3x3().col[1]*pb.bone.length
            m=d.to_track_quat('Y','Z').to_matrix().to_4x4();m.translation=head
            m=m@Matrix.Diagonal((1,d.length/pb.bone.length,1,1))
        pb.matrix=m;bpy.context.view_layer.update()
        for path in ('location','rotation_quaternion','scale'):pb.keyframe_insert(path,frame=frame)

def bvh(path,names,parents,samples,fps,orientations):
    # Six channels per joint preserve measured positions; no invented fixed lengths.
    rest=np.asarray(samples[0]);order=[];lines=[]
    def joint(i,depth):
        pad='  '*depth;order.append(i);p=parents[i];off=np.zeros(3) if p<0 else rest[i]-rest[p]
        lines.extend([pad+('ROOT ' if p<0 else 'JOINT ')+names[i],pad+'{',pad+'  OFFSET '+' '.join('%.6f'%v for v in off),pad+'  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation'])
        children=[j for j,p in enumerate(parents) if p==i]
        for j in children:joint(j,depth+1)
        if not children:lines.extend([pad+'  End Site',pad+'  {',pad+'    OFFSET 0 0.03 0',pad+'  }'])
        lines.append(pad+'}')
    lines.append('HIERARCHY');joint(0,0);lines.extend(['MOTION','Frames: '+str(len(samples)),'Frame Time: %.9f'%(1/fps)])
    with path.open('w',encoding='utf8') as f:
        f.write('\n'.join(lines)+'\n')
        for xyz,quats in zip(samples,orientations):
            rotations=[]
            for q in quats:
                quat=Quaternion((q[3],q[0],q[1],q[2]))
                if quat.magnitude<1e-6:quat=Quaternion()
                quat.normalize();rotations.append(quat.to_matrix())
            values=[]
            for i in order:
                p=parents[i];offset=np.zeros(3) if p<0 else rest[i]-rest[p]
                delta=Vector(xyz[i]) if p<0 else rotations[p].transposed()@Vector(xyz[i]-xyz[p])
                local=rotations[i] if p<0 else rotations[p].transposed()@rotations[i]
                angles=local.to_euler('YXZ')
                values.extend([*delta,math.degrees(angles.z),math.degrees(angles.x),math.degrees(angles.y)])
            f.write(' '.join('%.6f'%v for v in values)+'\n')

def main():
    folder=Path(sys.argv[sys.argv.index('--')+1]);spec=json.loads((folder/'scene.json').read_text(encoding='utf8'))
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.fps=spec['fps'];scene.frame_start=1;scene.frame_end=spec['frames'];scene.unit_settings.system='METRIC'
    with np.load(folder/'environment.npz') as data:xyz=data['xyz'];rgb=data['rgb']
    mesh=bpy.data.meshes.new('Depth point cloud');mesh.from_pydata([point(v) for v in xyz],[],[]);mesh.update()
    cloud=bpy.data.objects.new('Kinect depth snapshot',mesh);bpy.context.collection.objects.link(cloud)
    colors=mesh.color_attributes.new(name='Color',type='FLOAT_COLOR',domain='POINT')
    for i,c in enumerate(rgb):colors.data[i].color=(*(c/255),1)
    material=bpy.data.materials.new('Recorded RGB');material.use_nodes=True;nodes=material.node_tree.nodes;attr=nodes.new('ShaderNodeAttribute');attr.attribute_name='Color'
    material.node_tree.links.new(attr.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color']);cloud.data.materials.append(material)
    # Renderable points, not a fictitious reconstructed surface.
    group=bpy.data.node_groups.new('Depth points','GeometryNodeTree');group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry');group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes=group.nodes;inp=nodes.new('NodeGroupInput');out=nodes.new('NodeGroupOutput');points=nodes.new('GeometryNodeMeshToPoints');points.inputs['Radius'].default_value=.008;mat=nodes.new('GeometryNodeSetMaterial');mat.inputs['Material'].default_value=material
    group.links.new(inp.outputs['Geometry'],points.inputs['Mesh']);group.links.new(points.outputs['Points'],mat.inputs['Geometry']);group.links.new(mat.outputs['Geometry'],out.inputs['Geometry']);cloud.modifiers.new('Render point cloud','NODES').node_group=group
    for key,rows in spec['bodies'].items():
        rows=[r for r in rows if min(r['states'][:21])>0]
        if not rows:continue
        obj=make_rig('Kinect_'+key,spec['names'],spec['parents'],rows[0]['joints']);obj['tracking_id']=key
        valid={row['frame']:row for row in rows}
        for f in range(1,spec['frames']+1):
            row=valid.get(f);obj.hide_viewport=row is None;obj.hide_render=row is None
            obj.keyframe_insert('hide_viewport',frame=f);obj.keyframe_insert('hide_render',frame=f)
            if row:pose(obj,spec['names'],row['joints'],f)
        # Separate BVH takes prevent interpolation across tracking loss.
        takes=[]
        for row in rows:
            if not takes or row['frame']!=takes[-1][-1]['frame']+1:takes.append([])
            takes[-1].append(row)
        for i,take in enumerate(takes):
            bvh(folder/('Kinect_'+key+'_take_%d_frame_%d.bvh'%(i+1,take[0]['frame'])),spec['names'],spec['parents'],np.array([r['joints'] for r in take]),spec['fps'],[r['quaternions'] for r in take])
    ai=spec.get('ai')
    if ai:
        frames=np.load(folder/'ai_frames.npy',mmap_mode='r');valid=np.flatnonzero(ai['valid'])
        obj=make_rig('AI_depth_anchored',ai['names'],ai['parents'],frames[valid[0],:,:3,3]);obj['alignment_rms_m']=ai['calibration_rms_m']
        for i,ok in enumerate(ai['valid']):
            f=1+(ai['offset']+i/ai['fps'])*spec['fps'];obj.hide_viewport=not ok;obj.hide_render=not ok
            obj.keyframe_insert('hide_viewport',frame=f);obj.keyframe_insert('hide_render',frame=f)
            if ok:pose(obj,ai['names'],frames[i,:,:3,3],f,frames[i])
    camera=bpy.data.cameras.new('Kinect RGB reference');cam=bpy.data.objects.new('Kinect RGB reference',camera);bpy.context.collection.objects.link(cam)
    cam.matrix_world=Matrix(((-1,0,0,0),(0,0,1,0),(0,1,0,0),(0,0,0,1)));camera.lens=25;cam['note']='Camera location is exact; RGB focal length is approximate. Use point-cloud registration for depth alignment.'
    camera.show_background_images=True;bg=camera.background_images.new();bg.source='MOVIE_CLIP';bg.clip=bpy.data.movieclips.load(str(folder/'video.mp4'));bg.clip.filepath='//video.mp4';scene.camera=cam
    scene.render.resolution_x=1920;scene.render.resolution_y=1080;scene.render.resolution_percentage=100
    bpy.ops.object.light_add(type='AREA',location=(0,-2,4));bpy.context.object.data.energy=800
    scene.frame_set(1)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in scene.objects:
        if ob.type=='ARMATURE':ob.hide_viewport=False;ob.hide_set(False);ob.select_set(True)
    if any(ob.type=='ARMATURE' for ob in scene.objects):bpy.ops.export_scene.fbx(filepath=str(folder/'Kinect Motion.fbx'),use_selection=True,object_types={'ARMATURE'},add_leaf_bones=False,bake_anim=True,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'Kinect Scene.blend'))

if __name__=='__main__':main()
