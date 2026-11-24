"""Non-destructive, per-take horizontal placement in application Y-up metres."""
import json
from pathlib import Path
import numpy as np
from .storage import atomic_json

FILE='character_position.json'


def offset(project):
    path=Path(project)/FILE
    if not path.exists():return np.zeros(3)
    value=np.asarray(json.loads(path.read_text(encoding='utf8'))['offset'],float)
    if value.shape!=(3,) or not np.isfinite(value).all() or value[1]!=0:
        raise ValueError('Invalid character placement')
    return value


def matrix(project):
    result=np.eye(4);result[:3,3]=offset(project);return result


def recenter_offset(previous,world_pelvis):
    pelvis=np.asarray(world_pelvis,float)
    if pelvis.shape!=(3,) or not np.isfinite(pelvis).all():raise ValueError('No finite pelvis position')
    result=np.asarray(previous,float).copy();result[[0,2]]-=pelvis[[0,2]];result[1]=0
    return result


def apply(window,reset=False,position=None):
    if window.busy or window.contact_saving:window.status.setText('Finish processing before changing character position.');return
    if window.selected_character() is not None and (position is not None or reset):
        return apply_imported(window, np.zeros(3) if reset else position)
    if window.project is None or window.model_choice.currentData()=='pear':
        window.status.setText('Select a reconstructed motion take first.');return
    if window.selected_character() is not None:
        window.status.setText('Select the source person to recenter its motion take.');return
    project=Path(window.project);view=window.motion_view
    try:
        value=offset(project)
        if position is not None:new=validate_position(position)
        elif reset:new=np.zeros(3)
        else:
            take=window.scene_data.by_project.get(project.resolve()) if window.scene_data else None
            if take is not None:
                frame=take.frame_at(view.scene_time)
                if frame is None:raise ValueError('Move the timeline inside the selected take')
                pelvis=view.scene_joint_points(take,frame)[0]
            else:
                if view.joints is None:raise ValueError('Prepare the selected motion preview first')
                pelvis=np.asarray(view.joints[min(view.index,len(view.joints)-1),0])+value
            new=recenter_offset(value,pelvis)
        atomic_json(project/FILE,dict(schema=1,units='metres',up='Y',offset=new.tolist()))
        # Update all already-loaded shared scene instances, then request fresh data.
        seen=set()
        for scene in [window.scene_data,*window.scene_cache.values()]:
            if scene is None:continue
            take=scene.by_project.get(project.resolve())
            if take is not None and id(take) not in seen:
                seen.add(id(take));take.transform[:3,3]+=new-value
        linked=False
        for character in getattr(window,'characters',[]):
            for take in character.get('takes',[]):
                if take.get('source') and Path(take['source']).resolve()==project.resolve():
                    transform=np.array(take.get('transform') if take.get('transform') is not None else np.eye(4),float)
                    transform[:3,3]+=new-value;take['transform']=transform.tolist();linked=True
        if linked:window.save_characters()
        view.single_transform=matrix(project)
        window.request_people_scene(force=True)
        if window.preview_data is not None:window.apply_preview()
        view.update()
        window._stream_frame=None
        window.status.setText('Character position reset.' if reset else ('Character moved; placement applies to preview and exports.' if position is not None else 'Selected take centered at this frame; the same constant shift applies to preview and exports.'))
        return True
    except (OSError,ValueError,KeyError) as exc:window.status.setText('Cannot position character: '+str(exc))


def validate_position(value):
    value=np.asarray(value,dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all() or value[1]!=0:
        raise ValueError('Position must be finite horizontal X/Z coordinates in metres')
    return value.copy()


def imported_transform(character,transform=None):
    result=np.eye(4) if transform is None else np.array(transform,dtype=float,copy=True)
    result[:3,3]+=validate_position(character.get('position_offset',[0,0,0]))
    return result


def apply_imported(window,position):
    character=window.selected_character()
    if character is None:return False
    try:
        new=validate_position(position)
        old=character.get('position_offset')
        character['position_offset']=new.tolist()
        try:window.save_characters()
        except Exception:
            if old is None:character.pop('position_offset',None)
            else:character['position_offset']=old
            raise
        window.motion_view.update()
        window.status.setText('Imported character moved; its source motion is unchanged. Placement applies to preview and exports.')
        return True
    except (OSError,ValueError,KeyError) as error:
        window.status.setText('Cannot position character: '+str(error))
        return False


def move_dialog(window):
    from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QDoubleSpinBox,QLabel,QGridLayout,QPushButton,QDialogButtonBox
    if window.busy or window.contact_saving:
        window.status.setText('Finish processing before changing character position.');return
    character=window.selected_character()
    if window.model_choice.currentData()=='pear' or (character is None and window.project is None):
        window.status.setText('Select a person or imported character first.');return
    try:value=validate_position(character.get('position_offset',[0,0,0])) if character else offset(window.project)
    except (OSError,ValueError,KeyError) as error:window.status.setText(str(error));return
    window.stop_playback()
    dialog=QDialog(window);dialog.setWindowTitle('Move Character');layout=QVBoxLayout(dialog)
    label=QLabel(('Move imported character independently.' if character else 'Move selected source take and its linked retargets.')+'\nWorld X/Z axes, metres. Height and animation are preserved.');label.setWordWrap(True);layout.addWidget(label)
    form=QFormLayout();layout.addLayout(form);fields=[]
    for title,index in (('X offset',0),('Z offset',2)):
        box=QDoubleSpinBox();box.setRange(-100000,100000);box.setDecimals(4);box.setSingleStep(.1);box.setSuffix(' m');box.setValue(float(value[index]));box.setKeyboardTracking(False);form.addRow(title,box);fields.append(box)
    step=QDoubleSpinBox();step.setRange(.001,100);step.setDecimals(3);step.setValue(.1);step.setSuffix(' m');form.addRow('Move step',step)
    note=QLabel('Apply or use an axis button to move. Close keeps applied changes.');note.setWordWrap(True);layout.addWidget(note)
    def commit():
        ok=apply(window,position=[fields[0].value(),0,fields[1].value()])
        note.setText(window.status.text());return ok
    grid=QGridLayout();layout.addLayout(grid)
    def nudge(axis,direction):
        field=fields[axis];previous=field.value();field.setValue(previous+direction*step.value())
        if not commit():field.setValue(previous)
    for text,axis,direction,row,col in (('−X',0,-1,1,0),('+X',0,1,1,2),('+Z',1,1,0,1),('−Z',1,-1,2,1)):
        button=QPushButton(text);button.clicked.connect(lambda checked=False,a=axis,d=direction:nudge(a,d));grid.addWidget(button,row,col)
    reset=QPushButton('Reset Position');layout.addWidget(reset)
    def clear():
        previous=[f.value() for f in fields]
        for f in fields:f.setValue(0)
        if not commit():
            for f,v in zip(fields,previous):f.setValue(v)
    reset.clicked.connect(clear)
    buttons=QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Close);buttons.button(QDialogButtonBox.Apply).clicked.connect(commit);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
    dialog.exec();dialog.deleteLater()
