"""Validated multi-view input metadata; no models, inference or Qt imports."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np


def video_paths(paths):
    paths = [Path(p).resolve() for p in paths]
    if not paths:
        raise ValueError('Choose at least one video.')
    if len({str(p).casefold() for p in paths}) != len(paths):
        raise ValueError('The same video was selected more than once.')
    names = [p.stem for p in paths]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError('Use different video filenames so each camera has a unique name.')
    return paths


def common_media(entries, fps=30.):
    if not entries:
        raise ValueError('Open the camera videos first.')
    for e in entries:
        m = e['metadata']
        if any(not math.isfinite(float(m[k])) or float(m[k]) <= 0
               for k in ('fps', 'frames', 'duration', 'width', 'height')):
            raise ValueError('A video has invalid timing or dimensions.')
    duration = min(float(e['metadata']['duration']) for e in entries)
    frames = int(math.floor(duration * fps + 1e-6))
    if frames < 1:
        raise ValueError('The videos have no common playback interval.')
    return dict(fps=fps, frames=frames, duration=frames/fps,
                width=entries[0]['metadata']['width'], height=entries[0]['metadata']['height'])


def _rotation(matrix, name):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix.T@matrix, np.eye(3), atol=1e-4, rtol=0) or abs(np.linalg.det(matrix)-1) > 1e-4:
        raise ValueError(f'{name}: camera rotation must be a proper 3D rotation.')
    return matrix


def read_calibration(path):
    path = Path(path)
    with path.open('rb') as f:
        raw = f.read(1024*1024+1)
    if len(raw) > 1024*1024:
        raise ValueError('Camera calibration exceeds the 1 MiB limit.')
    if path.suffix.lower() == '.json':
        data = json.loads(raw.decode('utf-8-sig'))
    elif path.suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        data = yaml.safe_load(raw.decode('utf-8-sig'))
    else:
        raise ValueError('Choose a JSON or YAML camera calibration file.')
    if not isinstance(data, dict):
        raise ValueError('Camera calibration must contain named cameras.')
    if data.get('units', 'metres') not in {'m', 'meter', 'meters', 'metre', 'metres'}:
        raise ValueError('Camera translations must use metres.')
    if data.get('extrinsics_convention', 'world_to_camera') != 'world_to_camera':
        raise ValueError('JSON extrinsic matrices must use world_to_camera.')
    cameras = data.get('cameras', data)
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError('Camera calibration must contain named cameras.')
    output = {}
    for name, c in cameras.items():
        if not isinstance(name, str) or not name or len(name) > 240 or not isinstance(c, dict):
            raise ValueError('Each camera needs a name and camera parameters.')
        if c.get('units', 'metres') not in {'m', 'meter', 'meters', 'metre', 'metres'}:
            raise ValueError(f'{name}: camera translations must use metres.')
        if c.get('camera_model', 'pinhole') != 'pinhole' or c.get('distortion_model', 'radtan') != 'radtan':
            raise ValueError(f'{name}: only pinhole/radtan calibration is supported.')
        if 'rotation_quaternion' in c:
            # MAMMA YAML: camera-to-world pose, Hamilton quaternion [w,x,y,z].
            q = np.asarray(c['rotation_quaternion'], float)
            if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
                raise ValueError(f'{name}: use a unit quaternion [w,x,y,z].')
            w,x,y,z = q/np.linalg.norm(q)
            r = _rotation([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]], name)
            t = np.asarray(c['translation'], float)
            if t.shape != (3,) or not np.isfinite(t).all():
                raise ValueError(f'{name}: invalid camera position.')
            ext = np.column_stack([r.T, -r.T@t])
            values = np.asarray(c['intrinsics'], float)
            if values.shape != (4,):
                raise ValueError(f'{name}: YAML intrinsics must be [fx,fy,cx,cy].')
            fx,fy,cx,cy = values
            k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            size = c['resolution']
        else:
            k = np.asarray(c['intrinsic_matrix'], float)
            ext = np.asarray(c['extrinsics_matrix'], float)
            if ext.shape == (4,4):
                if not np.allclose(ext[3], [0,0,0,1], atol=1e-6, rtol=0):
                    raise ValueError(f'{name}: invalid homogeneous camera matrix.')
                ext = ext[:3]
            if ext.shape != (3,4) or not np.isfinite(ext).all():
                raise ValueError(f'{name}: extrinsics_matrix must be 3x4 or 4x4.')
            _rotation(ext[:,:3], name)
            size = c['image_size']
        if k.shape != (3,3) or not np.isfinite(k).all() or min(k[0,0], k[1,1]) <= 0 or abs(k[1,0])>1e-6 or not np.allclose(k[2], [0,0,1], atol=1e-6, rtol=0):
            raise ValueError(f'{name}: invalid camera intrinsics.')
        size = np.asarray(size, float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size != np.round(size)):
            raise ValueError(f'{name}: image size must be [width,height] in pixels.')
        distortion = np.asarray(c.get('distortion_coeffs', [0.]*5), float)
        if distortion.shape not in {(4,), (5,)} or not np.isfinite(distortion).all():
            raise ValueError(f'{name}: invalid radtan distortion coefficients.')
        output[name] = dict(intrinsic_matrix=k.tolist(), extrinsics_matrix=ext.tolist(),
                            image_size=size.astype(int).tolist(), distortion_coeffs=distortion.tolist())
    return dict(filename=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                units='metres', extrinsics_convention='world_to_camera', cameras=output)


def match_calibration(calibration, entries):
    cameras = calibration.get('cameras', {})
    names = [e['id'] for e in entries]
    if set(cameras) != set(names):
        raise ValueError('Calibration camera names must exactly match video filenames without extensions: '+', '.join(names))
    for e in entries:
        # Preserve original intrinsics and resolution. Future preprocessing
        # must scale K explicitly; never silently substitute image dimensions.
        if cameras[e['id']]['image_size'] != [e['metadata']['width'], e['metadata']['height']]:
            raise ValueError(e['id']+': calibration resolution does not match the original video.')
    return True


def resolved_entries(root, entries):
    result = []
    for e in entries:
        value = dict(e)
        name=e.get('id','')
        if not isinstance(name,str) or not name or len(name)>240 or any(c in name for c in '/\\:') or name in {'.','..'}:
            raise ValueError('Invalid saved camera name.')
        if e.get('media'):
            path = (Path(root)/e['media']).resolve()
            if not path.is_relative_to(Path(root).resolve()) or path == Path(root).resolve():
                raise ValueError('Invalid saved camera video location.')
        else:
            path = Path(e['path']).resolve()
        value['path'] = str(path)
        result.append(value)
    if len({e['id'].casefold() for e in result})!=len(result):
        raise ValueError('Duplicate saved camera names.')
    video_paths([e['path'] for e in result])
    return result
