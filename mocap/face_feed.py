"""Latest-only face data reader and optional local UDP sender, off the GUI thread."""
from threading import Condition, Thread
from pathlib import Path
import json
import socket
import time
import uuid
import numpy as np
from .face_data import FaceCache

PORT=49169


class FaceFeed:
    def __init__(self):
        self.condition=Condition();self.request=None;self.output=None;self.asset=None;self.calibration_status='Neutral: Off'
        self.use_ai=False;self.smoothing=False;self.asset_request=None;self.closed=False;self.session=uuid.uuid4().hex
        self.thread=Thread(target=self.run,name='Face preview and stream',daemon=True);self.thread.start()

    def submit(self,request):
        with self.condition:self.request=request;self.condition.notify()

    def load(self,path):
        with self.condition:self.asset_request=Path(path);self.condition.notify()

    def close(self):
        with self.condition:self.closed=True;self.condition.notify()
        self.thread.join(5)

    def run(self):
        cache=FaceCache();sequence=0;last=None;last_sent=0.;was_streaming=False
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.setblocking(False)
        try:
            while True:
                with self.condition:
                    self.condition.wait(.03)
                    if self.closed:return
                    request=self.request;asset=self.asset_request;self.asset_request=None
                if asset is not None:
                    try:
                        if asset.stat().st_size>160*1024**2:raise ValueError('Face preview cache exceeds 160 MiB.')
                        with np.load(asset,allow_pickle=False) as data:loaded={k:data[k] for k in data.files}
                        n=int(loaded['vertex_count'])
                        if not 0<n<=200000 or loaded['packed'].shape!=(53,n,3):raise ValueError('Invalid face preview geometry.')
                        loaded['key']=str(asset);self.asset=(loaded,None)
                    except Exception as error:self.asset=(None,str(error))
                if request is None:continue
                token,project,frame,label,stream=request
                now=time.monotonic()
                if request!=last or now-last_sent>=.25:
                    cache.smoothing=self.smoothing
                    cache.use_ai=self.use_ai
                    values=cache.frame(project,frame)
                    status=cache.last_status
                    self.calibration_status=cache.calibration_status
                    self.output=(token,values,status)
                    if stream or was_streaming:
                        packet=dict(version=1,session=self.session,sequence=sequence,frame=int(frame),
                                    label=str(label)[:160],valid=bool(stream and status=='Ready'),
                                    values=values.tolist() if stream else [0.]*52)
                        try:sock.sendto(json.dumps(packet,separators=(',',':'),allow_nan=False).encode(),('127.0.0.1',PORT))
                        except OSError as error:self.output=(token,values,'Stream error: '+str(error))
                        sequence+=1
                    was_streaming=stream;last=request;last_sent=now
        finally:sock.close()
