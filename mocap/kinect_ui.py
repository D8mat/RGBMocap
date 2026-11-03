"""Optional Kinect workspace: bounded live preview, capture, replay and exports."""
from pathlib import Path
import json,subprocess,threading,time
import numpy as np
from PySide6.QtCore import Qt,QThread,Signal,QTimer,QPointF
from PySide6.QtGui import QPainter,QColor,QPen,QImage,QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,
    QMessageBox,QSlider,QComboBox,QDoubleSpinBox,QCheckBox,QWidget,QSplitter)
from .kinect_data import bridge,packet,decode,Recording,Session,PARENTS

class Capture(QThread):
    failed=Signal(str)
    saved=Signal(str)
    def __init__(self):
        super().__init__();self.latest=None;self.process=None;self.lock=threading.Lock();self.recording=None;self.stopping=False;self.count=0
    def record(self,folder):
        with self.lock:
            if self.recording:raise ValueError('Already recording')
            self.recording=Recording(folder)
    def finish_recording(self,error=None):
        with self.lock:
            if self.recording:
                r=self.recording;self.recording=None;r.close(error);self.saved.emit(str(r.folder))
    def stop(self):
        self.stopping=True
        if self.process and self.process.poll() is None:
            try:self.process.stdin.write(b'\n');self.process.stdin.flush()
            except OSError:pass
    def run(self):
        error=None
        try:
            exe=bridge()
            if self.stopping:return
            self.process=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while not self.stopping:
                raw=packet(self.process.stdout)
                if raw is None:
                    detail=self.process.stderr.read(8192).decode(errors='replace');raise RuntimeError(detail or 'Kinect capture ended')
                frame=decode(raw)
                with self.lock:
                    if self.recording:self.recording.append(raw,frame)
                self.latest=frame;self.count+=1
        except Exception as exc:
            error=str(exc);self.failed.emit(error)
        finally:
            self.finish_recording(error)
            if self.process and self.process.poll() is None:
                self.stop()
                try:self.process.communicate(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.communicate()

class Export(QThread):
    progress=Signal(str);done=Signal(str);failed=Signal(str)
    def __init__(self,args):super().__init__();self.args=args;self.cancelled=False
    def run(self):
        try:
            from .kinect_export import export
            path=export(**self.args,progress=self.progress.emit,cancel=lambda:self.cancelled);self.done.emit(str(path))
        except Exception as exc:self.failed.emit(str(exc))

class DepthView(QWidget):
    def __init__(self):
        super().__init__();self.frame=None;self.yaw=0.;self.pitch=0.;self.zoom=170.;self.drag=None;self.setMinimumSize(360,280)
    def mousePressEvent(self,e):self.drag=e.position()
    def mouseMoveEvent(self,e):
        if self.drag is not None:
            d=e.position()-self.drag;self.yaw+=d.x()*.007;self.pitch=float(np.clip(self.pitch+d.y()*.007,-1.4,1.4));self.drag=e.position();self.update()
    def mouseReleaseEvent(self,e):self.drag=None
    def wheelEvent(self,e):self.zoom=float(np.clip(self.zoom*np.exp(e.angleDelta().y()/1200),20,1500));self.update()
    def paintEvent(self,e):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#181b21'))
        if self.frame is None:p.setPen(Qt.white);p.drawText(20,30,'Depth / skeleton preview — drag to orbit, wheel to zoom');return
        cy,sy=np.cos(self.yaw),np.sin(self.yaw);cx,sx=np.cos(self.pitch),np.sin(self.pitch)
        r=np.array([[cy,0,sy],[sx*sy,cx,-sx*cy],[-cx*sy,sx,cx*cy]])
        def project(xyz):
            q=(np.asarray(xyz)-[0,0,2])@r.T
            return np.column_stack((self.width()/2-q[:,0]*self.zoom,self.height()/2-q[:,1]*self.zoom))
        cloud=self.frame['points'][::2];xy=project(cloud['xyz'])
        for point,c in zip(xy,cloud['rgba']):
            p.setPen(QColor(int(c[0]),int(c[1]),int(c[2])));p.drawPoint(QPointF(*point))
        for i,(key,b) in enumerate(self.frame['bodies'].items()):
            xy=project(b['joints']);color=QColor.fromHsv((i*65)%360,220,255)
            for j,parent in enumerate(PARENTS):
                if parent>=0 and b['states'][j] and b['states'][parent]:
                    p.setPen(QPen(color,3,Qt.SolidLine if min(b['states'][j],b['states'][parent])==2 else Qt.DashLine));p.drawLine(QPointF(*xy[j]),QPointF(*xy[parent]))
            p.setPen(color);p.drawText(QPointF(*xy[3]),key)

class KinectDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('Kinect Studio — RGB MoCap');self.setWindowModality(Qt.WindowModal);self.resize(1150,780)
        self.capture=None;self.job=None;self.session=None;self.folder=None;self.ai_path=None;self.last_time=None;self.closing=False
        box=QVBoxLayout(self);buttons=QHBoxLayout();box.addLayout(buttons)
        def button(label,slot):
            b=QPushButton(label);b.clicked.connect(slot);buttons.addWidget(b);return b
        self.start=button('Start Kinect',self.start_capture);self.stop_button=button('Stop Kinect',self.stop_capture)
        self.record_button=button('Record...',self.record);self.end_button=button('Stop Recording',self.end_record)
        self.open_button=button('Open Capture...',self.open_capture)
        split=QSplitter();self.image=QLabel('RGB preview');self.image.setAlignment(Qt.AlignCenter);self.image.setMinimumSize(320,180);self.view=DepthView();split.addWidget(self.image);split.addWidget(self.view);box.addWidget(split,1)
        self.slider=QSlider(Qt.Horizontal);self.slider.valueChanged.connect(self.seek);box.addWidget(self.slider)
        self.info=QLabel('Start Kinect for live depth, RGB and body tracking. Captures use their own folder.');self.info.setWordWrap(True);box.addWidget(self.info)
        opts=QHBoxLayout();box.addLayout(opts);self.body=QComboBox();opts.addWidget(QLabel('Kinect person:'));opts.addWidget(self.body)
        self.pick_ai=QPushButton('Choose AI motion...');self.pick_ai.clicked.connect(self.choose_ai);opts.addWidget(self.pick_ai)
        self.offset=QDoubleSpinBox();self.offset.setRange(-3600,3600);self.offset.setDecimals(3);self.offset.setSuffix(' s');opts.addWidget(QLabel('AI frame 0 at capture time:'));opts.addWidget(self.offset)
        self.follow=QCheckBox('Follow Kinect pelvis');self.follow.setChecked(True);opts.addWidget(self.follow)
        self.ai_label=QLabel('Optional: select a prepared *_mixamo.json result of the same performance. Alignment creates a separate export.');self.ai_label.setWordWrap(True);box.addWidget(self.ai_label)
        self.use_ai=QCheckBox('Include depth-anchored AI motion (experimental)');box.addWidget(self.use_ai)
        self.export_button=QPushButton('Export Blender + FBX / BVH + Point Cloud...');self.export_button.clicked.connect(self.export_capture);box.addWidget(self.export_button)
        self.cancel_button=QPushButton('Cancel Export');self.cancel_button.clicked.connect(self.cancel_export);box.addWidget(self.cancel_button)
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(100);self.refresh()
    def error(self,text):self.info.setText(text);QMessageBox.warning(self,'Kinect',text)
    def start_capture(self):
        if self.capture and self.capture.isRunning():return
        self.capture=Capture();self.capture.failed.connect(self.error);self.capture.saved.connect(self.saved);self.capture.start();self.last_time=None;self.info.setText('Opening Kinect. Keep the sensor stationary.');self.refresh()
    def stop_capture(self):
        if self.capture:self.capture.stop()
    def record(self):
        parent=QFileDialog.getExistingDirectory(self,'Choose parent folder for a new Kinect capture')
        if not parent:return
        import datetime
        folder=Path(parent)/('Kinect_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        try:self.capture.record(folder);self.info.setText('Recording to '+str(folder))
        except Exception as exc:self.error(str(exc))
    def end_record(self):
        if self.capture:self.capture.finish_recording()
    def saved(self,folder):
        self.folder=Path(folder);self.info.setText('Capture saved: '+folder+' — stop Kinect to review/export.')
        try:self.session=Session(folder);self.slider.setRange(0,len(self.session.times)-1)
        except ValueError as exc:self.info.setText(str(exc))
    def open_capture(self):
        folder=QFileDialog.getExistingDirectory(self,'Open Kinect capture folder')
        if folder:
            try:self.session=Session(folder);self.folder=Path(folder);self.body.clear();self.slider.setRange(0,len(self.session.times)-1);self.seek(0)
            except Exception as exc:self.error(str(exc))
    def display(self,frame):
        self.view.frame=frame;self.view.update();im=QImage.fromData(frame['jpeg'],'JPEG');self.image.setPixmap(QPixmap.fromImage(im).scaled(self.image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        selected=self.body.currentText()
        for key in frame['bodies']:
            if self.body.findText(key)<0:self.body.addItem(key)
        if selected:self.body.setCurrentText(selected)
    def seek(self,i):
        if not self.session or (self.capture and self.capture.isRunning()):return
        try:
            self.display(self.session.frame(i));self.info.setText('Capture %.3f s | frame %d / %d | %s'%(self.session.times[i],i+1,len(self.session.times),self.folder))
        except Exception as exc:self.error(str(exc))
    def refresh(self):
        live=bool(self.capture and self.capture.isRunning());busy=bool(self.job and self.job.isRunning());record=bool(live and self.capture.recording)
        self.start.setEnabled(not live and not busy);self.stop_button.setEnabled(live);self.record_button.setEnabled(live and not record and self.capture.latest is not None);self.end_button.setEnabled(record)
        self.open_button.setEnabled(not live and not busy);self.slider.setEnabled(not live and not busy);self.export_button.setEnabled(self.session is not None and not live and not busy);self.cancel_button.setEnabled(busy)
        if live and self.capture.latest is not None:
            frame=self.capture.latest
            if frame['time']!=self.last_time:
                self.display(frame);self.last_time=frame['time'];self.info.setText(('RECORDING | ' if record else 'Live | ')+str(len(frame['bodies']))+' tracked people | '+str(self.capture.count)+' received frames')
        if self.closing and not live and not busy:self.accept()
    def choose_ai(self):
        p,_=QFileDialog.getOpenFileName(self,'Choose prepared AI skeleton metadata','','AI skeleton (*_mixamo.json)')
        if p:self.ai_path=p;self.ai_label.setText(p);self.use_ai.setChecked(True)
    def export_capture(self):
        ai=None
        if self.use_ai.isChecked():
            if not self.ai_path or not self.body.currentText():self.error('Choose AI motion and the matching Kinect person first.');return
            ai=dict(path=self.ai_path,body_id=self.body.currentText(),offset=self.offset.value(),follow=self.follow.isChecked())
        path,_=QFileDialog.getSaveFileName(self,'Choose a NEW export folder name',str(self.folder.parent/(self.folder.name+'_Blender')),'Export folder (*)')
        if not path:return
        self.job=Export(dict(folder=self.folder,destination=path,snapshot=self.slider.value(),ai=ai));self.job.progress.connect(self.info.setText);self.job.failed.connect(self.error);self.job.done.connect(lambda p:self.info.setText('Export complete: '+p));self.job.start();self.refresh()
    def cancel_export(self):
        if self.job:self.job.cancelled=True
    def reject(self):self.close()
    def closeEvent(self,event):
        self.stop_capture();self.cancel_export()
        if (self.capture and self.capture.isRunning()) or (self.job and self.job.isRunning()):
            self.closing=True;self.info.setText('Finishing capture/export safely...');event.ignore()
        else:event.accept()

def open_kinect(parent):
    if getattr(parent,'kinect_dialog',None) is None:
        parent.kinect_dialog=KinectDialog(parent)
    parent.kinect_dialog.closing=False
    parent.kinect_dialog.show();parent.kinect_dialog.raise_();parent.kinect_dialog.activateWindow()
