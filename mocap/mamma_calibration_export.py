"""Export MAMMA fixed cameras in original video pixels without changing results."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
from .storage import atomic_json
from .mamma_inputs import read_calibration,match_calibration


def validate(data,entries=None):
    # Reuse the same parser as reconstruction, including rotation validation.
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'calibration.json';atomic_json(p,data);result=read_calibration(p)
    if entries is not None:match_calibration(result,entries)
    return result


def original_pixels(cameras,entries):
    output=copy.deepcopy(cameras)
    if set(output)!={e['id'] for e in entries}:raise ValueError('Camera names do not match the selected videos')
    for e in entries:
        c=output[e['id']];rw,rh=c['image_size'];w=e['metadata']['width'];h=e['metadata']['height']
        if min(rw,rh,w,h)<=0:raise ValueError('Invalid camera resolution')
        k=np.array(c['intrinsic_matrix'],float);k[0]*=w/rw;k[1]*=h/rh
        c.update(intrinsic_matrix=k.tolist(),image_size=[int(w),int(h)])
    value=dict(schema='rgbmocap-camera-calibration-v1',units='metres',world_up='Z',
        camera_axes='OpenCV: X right, Y down, Z forward',extrinsics_convention='world_to_camera',cameras=output)
    validate(value,entries)
    return value


def current(window):
    entries=window.mamma_inputs
    if not entries:raise ValueError('Open the MAMMA camera videos first')
    if window.mamma_camera_mode.currentData()=='manual' and window.mamma_calibration:
        data=original_pixels(window.mamma_calibration['cameras'],entries);data['source']='Current manual camera settings'
    else:
        root=getattr(window,'people_root',None)
        if root is None:raise ValueError('Select a completed MAMMA result or import manual calibration first')
        root=Path(root);state=json.loads((root/'project.json').read_text(encoding='utf8'))
        if state.get('pipeline',{}).get('model')!='mamma' or (root/'mamma_pending.json').exists():raise ValueError('Select a completed MAMMA result')
        workspace=(root/state['workspace_root']).resolve();work=(root/state['mamma_work']).resolve()
        if not work.is_relative_to(workspace):raise ValueError('Invalid camera result location')
        cameras=json.loads((work/'camera-world.json').read_text(encoding='utf8'))
        data=original_pixels(cameras,entries);data['source']='Saved MAMMA result; automatic camera scale/gravity may be estimated'
    data['video_files']={e['id']:Path(e['path']).name for e in entries}
    return data


def export_json(window):
    from PySide6.QtWidgets import QFileDialog
    try:
        data=current(window)
        name,_=QFileDialog.getSaveFileName(window,'Export Camera Calibration','camera_calibration.json','JSON (*.json)')
        if not name:return
        path=Path(name)
        if not path.suffix:path=path.with_suffix('.json')
        if path.suffix.lower()!='.json':raise ValueError('Choose a JSON filename')
        atomic_json(path,data);window.status.setText('Camera calibration exported in original video pixels: '+str(path))
    except (OSError,ValueError,KeyError,TypeError) as error:window.status.setText('Cannot export calibration: '+str(error))


def confirm_import(window,value):
    from PySide6.QtWidgets import QMessageBox
    import time
    # The present MAMMA adapter requires already-undistorted pinhole footage.
    for name,c in value['cameras'].items():
        if abs(c['intrinsic_matrix'][0][1])>1e-6 or np.any(np.abs(c['distortion_coeffs'])>1e-8):
            raise ValueError(name+': MAMMA needs undistorted video and zero distortion/skew; undistort footage and export its new intrinsics first')
    rows=[]
    for name,c in value['cameras'].items():
        e=np.array(c['extrinsics_matrix']);center=-e[:,:3].T@e[:,3];k=c['intrinsic_matrix']
        rows.append(f"{name}: {c['image_size'][0]} x {c['image_size'][1]}, fx/fy {k[0][0]:.2f}/{k[1][1]:.2f}, position (m) {center.round(3).tolist()}")
    message='\n'.join(rows)+'\n\nUse these cameras for the next reconstruction? Existing motion and camera results stay unchanged. Reconstruct 3D to apply the new calibration.'
    if QMessageBox.question(window,'Review Camera Calibration',message,QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)!=QMessageBox.Yes:return False
    previous=dict(camera_mode=window.mamma_camera_mode.currentData(),calibration=copy.deepcopy(window.mamma_calibration))
    root=getattr(window,'workspace_root',None) or getattr(window,'project',None)
    if root:
        backup=Path(root)/'camera-calibration-history'/f'{time.time_ns()}-before-import.json'
        atomic_json(backup,previous)
    window._mamma_previous_calibration=previous
    return True


def restore(window):
    previous=getattr(window,'_mamma_previous_calibration',None)
    if previous is None:
        window.status.setText('No calibration change to undo in this session. Previous settings are saved in camera-calibration-history.');return
    window.mamma_tokens['calibration']=window.mamma_tokens.get('calibration',0)+1
    window.mamma_calibration=copy.deepcopy(previous['calibration'])
    window.mamma_camera_mode.setCurrentIndex(window.mamma_camera_mode.findData(previous['camera_mode']))
    window._mamma_previous_calibration=None;window.mamma_settings_changed()
    window.status.setText('Previous camera settings restored for the next reconstruction.')


def open_editor(window):
    from PySide6.QtWidgets import QFileDialog
    from .retarget.runtime import blender
    import shutil,subprocess
    try:
        data=current(window)
        if any(abs(c['intrinsic_matrix'][0][1])>1e-6 for c in data['cameras'].values()):raise ValueError('Blender camera editor cannot represent skew')
        name,_=QFileDialog.getSaveFileName(window,'Create New Camera Editor Folder','Camera_Edit','Folder name (*)')
        if not name:return
        folder=Path(name)
        if folder.exists():raise ValueError('Choose a new folder so existing calibration is not overwritten')
        exe=blender();folder.mkdir()
        atomic_json(folder/'calibration.original.json',data)
        script=folder/'camera_editor.py';shutil.copyfile(Path(__file__).with_name('mamma_calibration_blender.py'),script)
        (folder/'README.txt').write_text('Move/rotate cameras in Blender. Press N > RGB MoCap > Export edited calibration. Import calibration.edited.json in RGB MoCap and reconstruct again. Camera world uses metres and Z up. To restore the panel after reopening camera_editor.blend, run its camera_editor.py text in the Scripting workspace. Camera resolution and distortion remain attached to their original camera IDs. No videos are copied.\n',encoding='utf8')
        subprocess.Popen([str(exe),'--python',str(script),'--',str(folder)],cwd=folder)
        window.status.setText('Blender camera editor opened. Import calibration.edited.json after editing.')
    except (OSError,ValueError,KeyError,TypeError) as error:window.status.setText('Cannot open camera editor: '+str(error))
