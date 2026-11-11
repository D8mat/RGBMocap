"""Local ARKit stream receiver for explicitly selected facial meshes."""
bl_info={'name':'RGB MoCap Face Stream','author':'RGB MoCap','version':(1,0,0),'blender':(4,5,0),'category':'Animation'}
import json
import math
import socket
import time
import bpy
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ImportHelper

ARKIT = ('eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut')
PORT=49169


class FaceReceiver:
    def __init__(self):
        self.sock=None;self.scene=None;self.layer=None;self.targets=[];self.original=[]
        self.status='Disconnected';self.session=None;self.sequence=-1;self.last=0.;self.timed_out=False

    def close(self):
        if self.sock:self.sock.close()
        self.sock=None
        for key,value in self.original:
            try:key.value=value
            except (ReferenceError,RuntimeError):pass
        self.targets=[];self.original=[];self.scene=None;self.layer=None
        self.session=None;self.sequence=-1;self.status='Disconnected'

    def connect(self,context):
        self.close()
        scene=context.scene
        for row in scene.rgbmocap_face_targets:
            obj=row.object
            if obj is None or obj.name not in context.view_layer.objects:continue
            keys=getattr(getattr(obj,'data',None),'shape_keys',None)
            if keys is None:continue
            if keys.library is not None:raise ValueError('Use local editable face meshes.')
            mapping=[(i,keys.key_blocks.get(name)) for i,name in enumerate(ARKIT) if keys.key_blocks.get(name)]
            if mapping:self.targets.append((obj,mapping))
        if not self.targets:raise ValueError('Import the face FBX or bind selected mesh objects first.')
        seen=set()
        for obj,mapping in self.targets:
            for i,key in mapping:
                address=key.as_pointer()
                if address not in seen:self.original.append((key,key.value));seen.add(address)
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:sock.bind(('127.0.0.1',scene.rgbmocap_face_port));sock.setblocking(False)
        except OSError:sock.close();self.close();raise
        self.sock=sock;self.scene=scene;self.layer=context.view_layer.name
        self.last=time.monotonic();self.timed_out=False;self.status='Waiting for RGB MoCap face stream'

    def validate(self,data):
        if not isinstance(data,dict) or data.get('version')!=1:raise ValueError('Unsupported face packet')
        values=data.get('values')
        if not isinstance(values,list) or len(values)!=52:raise ValueError('Invalid face channels')
        if any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Invalid face values')
        if type(data.get('sequence')) is not int or data['sequence']<0:raise ValueError('Invalid sequence')
        if type(data.get('frame')) is not int or data['frame']<0:raise ValueError('Invalid frame')
        if not isinstance(data.get('session'),str) or len(data['session'])!=32:raise ValueError('Invalid session')
        if not isinstance(data.get('label'),str) or len(data['label'])>160:raise ValueError('Invalid label')
        if type(data.get('valid')) is not bool:raise ValueError('Invalid validity')
        return values

    def apply(self,data):
        values=self.validate(data)
        if data['session']==self.session and data['sequence']<=self.sequence:return False
        self.session=data['session'];self.sequence=data['sequence'];self.last=time.monotonic();self.timed_out=False
        if not data['valid']:values=[0.]*52
        for obj,mapping in self.targets:
            for index,key in mapping:key.value=values[index]
        self.status=(f"{data['label']} · frame {data['frame']+1}" if data['valid'] else 'No valid face data · neutral')
        return True

    def tick(self):
        if self.sock is None:return None
        latest=None
        try:
            # Bounded latest-only draining. UDP never queues application history.
            for _ in range(64):
                try:raw,address=self.sock.recvfrom(8192)
                except BlockingIOError:break
                if address[0]!='127.0.0.1' or len(raw)>4096:continue
                try:
                    data=json.loads(raw);self.validate(data)
                    if latest is None or data['session']!=latest['session'] or data['sequence']>latest['sequence']:latest=data
                except (ValueError,TypeError,UnicodeError):continue
            context=bpy.context
            if context.scene!=self.scene or context.view_layer.name!=self.layer:
                self.status='Paused · return to the connected scene and view layer';return .033
            if context.mode!='OBJECT':self.status='Paused · switch to Object Mode';return .033
            if latest is not None:self.apply(latest)
            if time.monotonic()-self.last>1 and not self.timed_out:
                for obj,mapping in self.targets:
                    for _,key in mapping:key.value=0.
                self.timed_out=True;self.status='Stream timed out · neutral'
            for area in context.screen.areas if context.screen else ():
                if area.type=='VIEW_3D':area.tag_redraw()
        except (OSError,ReferenceError,RuntimeError,ValueError) as error:
            self.close();self.status='Stopped: '+str(error);return None
        return .033


receiver=FaceReceiver()


def stop():
    if bpy.app.timers.is_registered(tick):bpy.app.timers.unregister(tick)
    receiver.close()


def tick():return receiver.tick()


@persistent
def stop_on_change(_):stop()


def bind(context,objects):
    stop();rows=context.scene.rgbmocap_face_targets;rows.clear();matched=set()
    for obj in objects:
        keys=getattr(getattr(obj,'data',None),'shape_keys',None)
        if keys is None:continue
        names=set(keys.key_blocks.keys()).intersection(ARKIT)
        if names:row=rows.add();row.object=obj;matched.update(names)
    receiver.status=f'{len(rows)} meshes bound · {len(matched)}/52 channels'
    return len(rows)


class RGBFACE_Target(bpy.types.PropertyGroup):
    object:bpy.props.PointerProperty(type=bpy.types.Object)


class RGBFACE_OT_Import(bpy.types.Operator,ImportHelper):
    bl_idname='rgbmocap_face.import_fbx';bl_label='Import Face FBX...';bl_options={'REGISTER','UNDO'}
    filename_ext='.fbx'
    filter_glob:bpy.props.StringProperty(default='*.fbx',options={'HIDDEN'})
    def execute(self,context):
        before=set(bpy.data.objects)
        try:
            bpy.ops.import_scene.fbx(filepath=self.filepath,use_anim=False)
            count=bind(context,[o for o in bpy.data.objects if o not in before])
            if not count:self.report({'WARNING'},'Imported FBX has no matching ARKit shape keys.')
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


class RGBFACE_OT_Bind(bpy.types.Operator):
    bl_idname='rgbmocap_face.bind';bl_label='Bind Selected Meshes';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        objects=set(context.selected_objects)
        for obj in list(objects):objects.update(obj.children_recursive)
        if not bind(context,objects):self.report({'WARNING'},'Select the head, eyes and teeth with ARKit shape keys.')
        return {'FINISHED'}


class RGBFACE_OT_Connect(bpy.types.Operator):
    bl_idname='rgbmocap_face.connect';bl_label='Connect Face Stream'
    def execute(self,context):
        stop()
        try:receiver.connect(context);bpy.app.timers.register(tick,first_interval=.033)
        except (OSError,ValueError) as error:self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


class RGBFACE_OT_Disconnect(bpy.types.Operator):
    bl_idname='rgbmocap_face.disconnect';bl_label='Disconnect (Restore Values)'
    def execute(self,context):stop();return {'FINISHED'}


class RGBFACE_PT_Panel(bpy.types.Panel):
    bl_label='RGB MoCap Face Stream';bl_idname='RGBFACE_PT_panel';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='RGB MoCap'
    def draw(self,context):
        layout=self.layout;column=layout.column();column.enabled=receiver.sock is None
        column.operator('rgbmocap_face.import_fbx');column.operator('rgbmocap_face.bind')
        column.prop(context.scene,'rgbmocap_face_port')
        for row in context.scene.rgbmocap_face_targets:
            if row.object:column.label(text=row.object.name,icon='MESH_DATA')
        layout.operator('rgbmocap_face.disconnect' if receiver.sock else 'rgbmocap_face.connect')
        layout.label(text=receiver.status)
        layout.label(text='Source: selected face in RGB MoCap')


CLASSES=(RGBFACE_Target,RGBFACE_OT_Import,RGBFACE_OT_Bind,RGBFACE_OT_Connect,RGBFACE_OT_Disconnect,RGBFACE_PT_Panel)


def register():
    for cls in CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.rgbmocap_face_targets=bpy.props.CollectionProperty(type=RGBFACE_Target)
    bpy.types.Scene.rgbmocap_face_port=bpy.props.IntProperty(name='Face Port',default=PORT,min=1024,max=65535)
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):handlers.append(stop_on_change)


def unregister():
    stop()
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if stop_on_change in handlers:handlers.remove(stop_on_change)
    del bpy.types.Scene.rgbmocap_face_targets;del bpy.types.Scene.rgbmocap_face_port
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
