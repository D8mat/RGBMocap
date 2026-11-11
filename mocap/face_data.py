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
                self.rows[path] = (key, values, valid)
                self.rows.move_to_end(path)
                while len(self.rows) > 4:
                    self.rows.popitem(last=False)
                row = self.rows[path]
            if 0 <= frame < len(row[1]):
                self.last_status = 'Ready' if row[2][frame] or np.any(row[1][frame]) else 'No face detected in this frame'
                raw = row[1][frame]
                if not calibrated:return raw.copy()
                values = self.neutral.apply(project,raw,dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,source=identity))
                self.calibration_status = self.neutral.status
                return values
            self.last_status = 'Outside captured frames'
            return neutral
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
            return neutral
