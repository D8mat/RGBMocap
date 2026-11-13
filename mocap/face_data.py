"""Bounded validated face cache, read only on the stream sender thread."""
from collections import OrderedDict
from pathlib import Path
import json
import zipfile
import numpy as np
from .face_extract import ARKIT, source_identity


class FaceCache:
    def __init__(self):
        self.rows = OrderedDict()
        self.smoothing = False
        self.use_ai = False
        from .face_neutral import NeutralCache
        self.neutral = NeutralCache()
        self.calibration_status = 'Neutral: Off'

    def frame(self, project, frame, calibrated=True):
        self.calibration_status = 'Neutral: Off'
        self.last_status = 'No face capture available'
        neutral = np.zeros(52, dtype='<f4')
        if project is None:
            return neutral
        path = Path(project) / 'face.npz'
        try:
            stat = path.stat()
            identity = source_identity(project)
            key = (stat.st_size, stat.st_mtime_ns, json.dumps(identity, sort_keys=True))
            ai_path=Path(project)/'face_ai.npz'
            ai_stamp=ai_path.stat() if self.use_ai and calibrated and ai_path.is_file() else None
            key=key+((ai_stamp.st_size,ai_stamp.st_mtime_ns) if ai_stamp else (None,))
            row = self.rows.get(path)
            if row is None or row[0] != key:
                if stat.st_size > 96 * 1024 * 1024:
                    return neutral
                with zipfile.ZipFile(path) as archive:
                    if sum(item.file_size for item in archive.infolist()) > 96 * 1024 * 1024:
                        return neutral
                with np.load(path, allow_pickle=False) as data:
                    values = np.asarray(data['values'], dtype='<f4')
                    self.last_status = 'Invalid or stale face capture; capture again'
                    if (values.ndim != 2 or values.shape[1] != 52 or len(values) > 36000
                            or tuple(data['names'].tolist()) != ARKIT
                            or json.loads(str(data['source'])) != identity
                            or not np.isfinite(values).all()):
                        return neutral
                    valid = np.asarray(data['valid'],bool) if 'valid' in data else np.ones(len(values),bool)
                    if valid.shape != (len(values),):return neutral
                    values = np.clip(values, 0, 1)
                if ai_stamp is not None:
                    if ai_stamp.st_size>16*1024*1024:raise ValueError('Oversized AI result')
                    with zipfile.ZipFile(ai_path) as archive:
                        if sum(i.file_size for i in archive.infolist())>16*1024*1024:raise ValueError('Oversized AI result')
                    with np.load(ai_path,allow_pickle=False) as ai:
                        candidate=ai['values'].astype('<f4')
                        if (int(ai['capture_bytes'])!=stat.st_size or int(ai['capture_mtime'])!=stat.st_mtime_ns or json.loads(str(ai['source']))!=identity):raise ValueError('Stale AI result')
                        if candidate.shape!=values.shape or not np.isfinite(candidate).all():raise ValueError('Invalid AI result')
                        values=np.clip(candidate,0,1)
                self.rows[path] = (key, values, valid)
                self.rows.move_to_end(path)
                while len(self.rows) > 4:
                    self.rows.popitem(last=False)
                row = self.rows[path]
            if 0 <= frame < len(row[1]):
                self.last_status = 'Ready' if row[2][frame] or np.any(row[1][frame]) else 'No face detected in this frame'
                raw = row[1][frame]
                if not calibrated:return raw.copy()
                identity=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,source=identity)
                values = self.neutral.apply(project,raw,identity,row[1] if ai_stamp else None)
                if self.smoothing and self.last_status=='Ready' and not np.all(values==0):
                    # Symmetric, seek-independent window; never cross missing detections.
                    samples=[];weights=[]
                    for offset in range(-2,3):
                        index=frame+offset
                        if not 0<=index<len(row[1]):continue
                        lo,hi=sorted((frame,index))
                        if not row[2][lo:hi+1].all():continue
                        sample=self.neutral.apply(project,row[1][index],identity,row[1] if ai_stamp else None)
                        weight=np.exp(-.5*(offset/1.2)**2-.5*((sample-values)/.1)**2)
                        samples.append(sample*weight);weights.append(weight)
                    if weights:
                        filtered=np.sum(samples,axis=0)/np.sum(weights,axis=0)
                        for name in ('eyeBlinkLeft','eyeBlinkRight'):
                            filtered[ARKIT.index(name)]=values[ARKIT.index(name)]
                        values=filtered.astype('<f4')
                self.calibration_status = self.neutral.status
                return values
            self.last_status = 'Outside captured frames'
            return neutral
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
            return neutral
