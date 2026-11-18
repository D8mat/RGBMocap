"""Asynchronous, generation-safe automatic mapping preview."""
import json,tempfile
from pathlib import Path
from PySide6.QtCore import QObject,QProcess,QTimer
from PySide6.QtWidgets import QApplication
from .paths import ROOT
from .retarget.runtime import blender


class MappingPreview(QObject):
    def __init__(self,parent,done):
        super().__init__(parent);self.done=done;self.generation=0;self.jobs=[];self.cache={}
        QApplication.instance().aboutToQuit.connect(self.stop)

    def cancel(self):
        self.generation+=1
        for process,_ in list(self.jobs):process.kill()

    def stop(self):
        self.cancel()
        for process,_ in list(self.jobs):process.waitForFinished(2000)

    def request(self,rest,bvh):
        self.cancel();generation=self.generation
        try:
            key=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in (rest,bvh))
            if key in self.cache:self.done(self.cache[key],None);return
            exe=blender()
        except (OSError,RuntimeError) as e:self.done(None,str(e));return
        temp=tempfile.TemporaryDirectory(prefix='rgbmocap-mapping-');output=Path(temp.name)/'mapping.json'
        process=QProcess(self);self.jobs.append((process,temp));tail=bytearray()
        process.setProcessChannelMode(QProcess.MergedChannels)
        def drain():
            tail.extend(bytes(process.readAllStandardOutput()))
            if len(tail)>4000:del tail[:-4000]
        process.readyReadStandardOutput.connect(drain)
        finished=[False]
        timer=QTimer(process);timer.setSingleShot(True);timer.timeout.connect(process.kill)
        def complete(code=1,*_):
            if finished[0]:return
            finished[0]=True;timer.stop();drain()
            try:
                if code:raise ValueError(tail.decode('utf-8',errors='replace')[-800:] or 'Mapping preview could not start or timed out.')
                data=json.loads(output.read_text(encoding='utf-8'))
                if generation==self.generation:
                    if len(self.cache)>=8:self.cache.clear()
                    self.cache[key]=data;self.done(data,None)
            except (OSError,ValueError) as e:
                if generation==self.generation:self.done(None,str(e))
            finally:
                self.jobs.remove((process,temp));temp.cleanup();process.deleteLater()
        process.finished.connect(complete)
        process.errorOccurred.connect(lambda error:complete() if error==QProcess.FailedToStart else None)
        process.start(str(exe),['-b','--factory-startup','--disable-autoexec','--threads','2','--python-exit-code','1','--python',str(Path(__file__).with_name('toyxyz_mapping_probe.py')),'--',str(ROOT),str(rest),str(bvh),str(output)])
        timer.start(60000)
