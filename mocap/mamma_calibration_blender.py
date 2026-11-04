"""Standalone Blender camera editor. Original calibration stays immutable."""
import bpy,json,sys,time
from pathlib import Path
from mathutils import Matrix,Vector
import numpy as np


def folder_path():
    if '--' in sys.argv:return Path(sys.argv[sys.argv.index('--')+1])
    if bpy.data.filepath:return Path(bpy.data.filepath).parent
    raise ValueError('Open the saved camera editor project first')

FOLDER=folder_path();CV=Matrix.Diagonal((1,-1,-1,1))
DATA=json.loads((FOLDER/'calibration.original.json').read_text(encoding='utf8'))


def edited_data():
    value=json.loads(json.dumps(DATA));cameras={}
    objects=[o for o in bpy.context.scene.objects if o.type=='CAMERA' and 'calibration_id' in o]
    if len(objects)!=len(DATA['cameras']):raise ValueError('Keep exactly one camera for every original camera ID')
    for obj in objects:
        name=obj['calibration_id']
        if name not in DATA['cameras'] or name in cameras:raise ValueError('Unknown or duplicate camera ID')
        if obj.data.type!='PERSP':raise ValueError('Use perspective cameras')
        matrix=np.asarray(obj.matrix_world);r=matrix[:3,:3]
        if not np.isfinite(matrix).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or abs(np.linalg.det(r)-1)>1e-5:raise ValueError('Camera scale must stay 1 with no shear or reflection')
        c=json.loads(json.dumps(DATA['cameras'][name]));w,h=c['image_size'];d=obj.data
        if d.sensor_fit!='HORIZONTAL' or d.sensor_width<=0:raise ValueError('Keep horizontal sensor fit and a positive sensor width')
        aspect=float(obj.get('calibration_pixel_aspect_y',1))
        if not np.isfinite(aspect) or aspect<=0:raise ValueError('Pixel aspect must be positive')
        fx=d.lens*w/d.sensor_width;fy=fx/aspect;cx=w/2-d.shift_x*w;cy=h/2+d.shift_y*w/aspect
        c['intrinsic_matrix']=[[fx,0,cx],[0,fy,cy],[0,0,1]]
        c['extrinsics_matrix']=[list(row) for row in (CV@obj.matrix_world.inverted())][:3]
        cameras[name]=c
    value['cameras']=cameras;value['source']='Manually edited Blender cameras; requires a new MAMMA reconstruction'
    return value


class ExportCalibration(bpy.types.Operator):
    bl_idname='rgbmocap.export_calibration';bl_label='Export edited calibration'
    def execute(self,context):
        try:
            data=edited_data();target=FOLDER/'calibration.edited.json'
            if target.exists():(FOLDER/f'calibration.edited.backup-{time.time_ns()}.json').write_bytes(target.read_bytes())
            temp=target.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');temp.replace(target)
            self.report({'INFO'},'Saved calibration.edited.json; import it in RGB MoCap')
            return {'FINISHED'}
        except (ValueError,OSError,KeyError) as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


class CalibrationPanel(bpy.types.Panel):
    bl_label='Camera calibration';bl_idname='VIEW3D_PT_rgbmocap_calibration';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='RGB MoCap'
    def draw(self,context):
        layout=self.layout;layout.label(text='Metres, Z up. Move/rotate cameras.')
        obj=context.object
        if obj and obj.type=='CAMERA' and 'calibration_id' in obj:
            layout.label(text='ID: '+obj['calibration_id'])
            layout.prop(obj.data,'lens');layout.prop(obj.data,'shift_x');layout.prop(obj.data,'shift_y')
            layout.prop(obj,'["calibration_pixel_aspect_y"]',text='fx / fy')
            layout.label(text='Resolution: '+str(DATA['cameras'][obj['calibration_id']]['image_size']))
        layout.operator('rgbmocap.export_calibration')
        layout.label(text='Import JSON, then reconstruct again.')


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
    scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    for name,c in DATA['cameras'].items():
        data=bpy.data.cameras.new(name);obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
        ext=Matrix(c['extrinsics_matrix']+[[0,0,0,1]]);obj.matrix_world=ext.inverted()@CV;obj['calibration_id']=name
        k=c['intrinsic_matrix'];w,h=c['image_size'];aspect=k[0][0]/k[1][1]
        data.sensor_fit='HORIZONTAL';data.sensor_width=36;data.lens=k[0][0]*36/w
        data.shift_x=(w/2-k[0][2])/w;data.shift_y=(k[1][2]-h/2)*aspect/w
        obj['calibration_pixel_aspect_y']=aspect;obj.show_name=True;data.display_size=.3;data.clip_end=10000
        if scene.camera is None:scene.camera=obj
    origin=bpy.data.objects.new('World origin - metres',None);origin.empty_display_type='PLAIN_AXES';origin.empty_display_size=.5;scene.collection.objects.link(origin)
    points=[o.location for o in scene.objects if o.type=='CAMERA'];center=sum(points,Vector())/max(1,len(points));extent=max((p-center).length for p in points)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.region_3d.view_location=center;area.spaces.active.region_3d.view_distance=max(4,extent*3)
    text=bpy.data.texts.load(str(FOLDER/'camera_editor.py'));text.use_module=False
    bpy.ops.wm.save_as_mainfile(filepath=str(FOLDER/'camera_editor.blend'))


if not bpy.data.filepath or Path(bpy.data.filepath).resolve()!= (FOLDER/'camera_editor.blend').resolve():
    if (FOLDER/'camera_editor.blend').exists():bpy.ops.wm.open_mainfile(filepath=str(FOLDER/'camera_editor.blend'))
    else:build_scene()
for cls in (ExportCalibration,CalibrationPanel):
    old=getattr(bpy.types,cls.__name__,None)
    if old:
        try:bpy.utils.unregister_class(old)
        except RuntimeError:pass
    bpy.utils.register_class(cls)
