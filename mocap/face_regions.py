"""Bounded background loading of captured face landmarks; no GUI imports."""
from pathlib import Path
from threading import Condition,Thread
import json
import zipfile
import numpy as np
from .face_extract import source_identity

class RegionReader:
    def __init__(self):
        self.condition=Condition();self.request=None;self.output=None;self.stopping=False
        self.thread=Thread(target=self.run,name='Face video geometry',daemon=True);self.thread.start()
    def submit(self,request):
        with self.condition:
            if self.request!=request:self.request=request;self.condition.notify()
    def close(self):
        with self.condition:self.stopping=True;self.condition.notify()
        self.thread.join(5);self.output=None
    def run(self):
        stamp=None
        while True:
            with self.condition:
                self.condition.wait(.25)
                if self.stopping:return
                request=self.request
            if request is None:self.output=None;stamp=None;continue
            token,project=request
            try:
                path=Path(project)/'face.npz';stat=path.stat();identity=source_identity(project)
                key=(token,stat.st_size,stat.st_mtime_ns,json.dumps(identity,sort_keys=True))
                if key==stamp:continue
                if stat.st_size>96*1024**2:raise ValueError('Face capture is too large')
                with zipfile.ZipFile(path) as archive:
                    if sum(z.file_size for z in archive.infolist())>96*1024**2:raise ValueError('Face capture exceeds geometry limit')
                with np.load(path,allow_pickle=False) as data:
                    if json.loads(str(data['source']))!=identity:raise ValueError('Capture is stale; capture again')
                    if 'regions' not in data or 'landmarks' not in data:raise ValueError('Capture face again to create the mesh overlay')
                    valid=np.asarray(data['valid'],bool);regions=np.asarray(data['regions'],np.float32);points=np.asarray(data['landmarks'],np.float16)
                    n=len(valid)
                    if not 0<n<=36000 or regions.shape!=(n,4) or points.shape!=(n,478,2):raise ValueError('Invalid face overlay geometry')
                    if not np.isfinite(regions[valid]).all() or not np.isfinite(points[valid]).all():raise ValueError('Invalid detected face coordinates')
                with self.condition:
                    if request==self.request:self.output=(token,regions,points,valid,'');stamp=key
            except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as error:
                message='No face capture; use Capture Face from Video' if isinstance(error,FileNotFoundError) else str(error)
                self.output=(token,None,None,None,message);stamp=None
