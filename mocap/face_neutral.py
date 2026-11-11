"""Per-take neutral reference. Captured expressions and video landmarks stay raw."""
from pathlib import Path
from collections import OrderedDict
import argparse
import json
import sys
import numpy as np
from .face_extract import source_identity
from .storage import atomic_json

FILE='face_neutral.json'


def capture_identity(project):
    stat=(Path(project)/'face.npz').stat()
    return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,source=source_identity(project))


def save_reference(project, frame=None, reset=False):
    project=Path(project)
    if reset:
        atomic_json(project/FILE,dict(version=1,enabled=False))
        return 'Neutral face calibration reset.'
    from .face_data import FaceCache
    identity=capture_identity(project)
    cache=FaceCache();values=cache.frame(project,frame,calibrated=False)
    row=cache.rows.get(project/'face.npz')
    if (row is None or cache.last_status!='Ready' or not 0<=frame<len(row[2]) or not row[2][frame]):
        raise ValueError('Choose a frame with an actually detected face; missing or filled frames cannot calibrate neutral.')
    if capture_identity(project)!=identity:raise ValueError('Face capture changed; select the frame again.')
    atomic_json(project/FILE,dict(version=1,enabled=True,capture=identity,frame=int(frame),values=values.tolist()))
    return f'Neutral face set to frame {frame+1}. All 52 output values are zero at this frame.'


class NeutralCache:
    def __init__(self):self.rows=OrderedDict();self.status='Neutral: Off'
    def apply(self,project,raw,identity):
        path=Path(project)/FILE;self.status='Neutral: Off'
        try:
            stat=path.stat();key=(stat.st_size,stat.st_mtime_ns)
            cached=self.rows.get(path)
            if cached is None or cached[0]!=key:
                if stat.st_size>16384:raise ValueError('Oversized neutral reference')
                data=json.loads(path.read_text(encoding='utf-8'))
                if data.get('version')!=1 or type(data.get('enabled')) is not bool:raise ValueError('Invalid neutral reference')
                if data['enabled']:
                    baseline=np.asarray(data['values'],np.float32)
                    if (baseline.shape!=(52,) or not np.isfinite(baseline).all() or np.any(baseline<0) or np.any(baseline>1)
                        or type(data.get('frame')) is not int or not 0<=data['frame']<36000):raise ValueError('Invalid neutral values')
                else:baseline=None
                cached=(key,data,baseline);self.rows[path]=cached;self.rows.move_to_end(path)
                while len(self.rows)>4:self.rows.popitem(last=False)
            _,data,baseline=cached
            if baseline is None:return raw
            if data['capture']!=identity:
                self.status='Neutral: capture changed; select a new reference';return raw
            self.status=f"Neutral: frame {data['frame']+1}"
            return np.clip(raw-baseline,0,1).astype('<f4',copy=False)
        except FileNotFoundError:return raw
        except (OSError,ValueError,KeyError,TypeError):
            self.status='Neutral: invalid reference; reset or select again';return raw


def main(emit):
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    choice=parser.add_mutually_exclusive_group(required=True);choice.add_argument('--frame',type=int);choice.add_argument('--reset',action='store_true')
    args=parser.parse_args(sys.argv[2:])
    try:
        message=save_reference(args.project,args.frame,args.reset)
        emit('face_neutral_done',message=message);return 0
    except (OSError,ValueError,KeyError,TypeError) as error:
        emit('failed',stage='face_neutral',message=str(error));return 1
