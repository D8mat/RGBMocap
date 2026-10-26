"""Display metadata for the existing automatic single-person tracker.

No detector, identity recovery, motion processing or people-group conversion.
Tracking arrays and one thumbnail are read on the existing preview worker.
"""
from pathlib import Path
import json
import numpy as np
from .people_store import DEFAULT_COLOR,media_receipt

FILENAME='single_actor.json'


def preferences(project):
    record=dict(id=1,name='Person 1',color=DEFAULT_COLOR,checked=True)
    try:
        path=Path(project)/FILENAME
        if path.stat().st_size>8192:return record
        saved=json.loads(path.read_text(encoding='utf-8'))
        color=saved['color']
        if saved.get('version')!=1 or saved.get('input')!=media_receipt(project):return record
        if len(color)!=7 or color[0]!='#' or not isinstance(saved['checked'],bool):return record
        int(color[1:],16)
        record.update(color=color,checked=saved['checked'])
        from .person_names import validate_name
        record['name']=validate_name(saved.get('name','Person 1'))
    except (OSError,ValueError,KeyError,TypeError):pass
    return record


def save_preferences(project,person):
    from .storage import atomic_json
    atomic_json(Path(project)/FILENAME,dict(version=1,input=media_receipt(project),
        color=person['color'],checked=bool(person['checked']),name=person.get('name','Person 1')))


def snapshot_key(project):
    project=Path(project)
    try:
        if (project/'people.json').exists():return None
        manifest=json.loads((project/'project.json').read_text(encoding='utf-8'))
        if manifest.get('people_root') is not None:return None
        source=(project/'input.mp4').stat();tracking=(project/'tracking.npz').stat()
        if tracking.st_mtime_ns<source.st_mtime_ns:return None
        return str(project.resolve()),source.st_size,source.st_mtime_ns,tracking.st_size,tracking.st_mtime_ns,manifest['frames']
    except (OSError,ValueError,KeyError,TypeError):return None


def snapshot(project,arrays,cancelled=lambda:False):
    """Scan bounded chunks, decode a single crop, and close the capture."""
    boxes=arrays.get('boxes');observed=arrays.get('observed')
    if boxes is None or observed is None or len(boxes)!=len(observed):return None
    first=last=None;count=0;box=None
    for start in range(0,len(boxes),4096):
        if cancelled():return None
        stop=min(start+4096,len(boxes));b=np.asarray(boxes[start:stop]);seen=np.asarray(observed[start:stop],bool)
        valid=seen&np.isfinite(b).all(axis=1)&((b[:,2:]-b[:,:2])>1).all(axis=1)
        indices=np.flatnonzero(valid);count+=len(indices)
        if len(indices):
            if first is None:first=start+int(indices[0]);box=b[indices[0]].copy()
            last=start+int(indices[-1])+1
    if first is None:return None
    result=dict(first=first,last=last,observations=count,frames=len(boxes),pending=False,thumbnail=None)
    if cancelled():return None
    import cv2
    from PySide6.QtGui import QImage
    capture=cv2.VideoCapture(str(Path(project)/'input.mp4'))
    try:
        if first:capture.set(cv2.CAP_PROP_POS_FRAMES,first)
        ok,frame=capture.read()
        if ok and not cancelled():
            x1,y1,x2,y2=np.clip(np.rint(box).astype(int),[0,0,0,0],[frame.shape[1],frame.shape[0]]*2)
            if x2>x1 and y2>y1:
                crop=frame[y1:y2,x1:x2];scale=min(96/crop.shape[1],144/crop.shape[0])
                crop=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
                rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                result['thumbnail']=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()
    except (cv2.error,OSError,ValueError):pass  # A thumbnail must never block motion preview.
    finally:capture.release()
    return result
