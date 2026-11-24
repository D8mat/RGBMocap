"""Docked local face model preview and independent Blender face streaming."""
from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
from PySide6.QtCore import QProcess,QTimer,Qt
from PySide6.QtWidgets import QDockWidget,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QCheckBox,QFileDialog
from .paths import ROOT,HOME
from .face_feed import FaceFeed,PORT
from .face_preview_gl import FaceView


class FacePanel(QDockWidget):
    def __init__(self,window):
        super().__init__('Face Preview and Stream',window)
        self.setObjectName('facePreviewDock');self.window=window;self.feed=FaceFeed()
        self.process=None;self.pending=None;self.token=None;self.closed=False
        panel=QWidget();layout=QVBoxLayout(panel);buttons=QHBoxLayout()
        self.open_button=QPushButton('Load Face FBX...');buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.model_label=QLabel('Load an ARKit face model for a 3D preview.');self.model_label.setWordWrap(True);layout.addWidget(self.model_label)
        self.view=FaceView();layout.addWidget(self.view,1)
        neutral_row=QHBoxLayout();self.neutral_button=QPushButton('Set Current Frame as Neutral');self.neutral_reset=QPushButton('Reset Neutral')
        neutral_row.addWidget(self.neutral_button);neutral_row.addWidget(self.neutral_reset);layout.addLayout(neutral_row)
        self.neutral_label=QLabel('Neutral: Off');self.neutral_label.setWordWrap(True);layout.addWidget(self.neutral_label)
        self.neutral_button.setToolTip('Pause on a relaxed face with eyes open. Use the same Track Face neutral calibration as the MetaHuman stream.')
        self.neutral_button.clicked.connect(window.set_current_neutral_face)
        self.neutral_reset.clicked.connect(window.reset_current_neutral_face)
        self.stream=QCheckBox('Stream selected face to Blender');layout.addWidget(self.stream)
        self.stream.setToolTip(f'Local UDP port {PORT}. Enable Connect in the RGB MoCap Face Stream Blender add-on. Uses the original Track Face playback data.')
        self.status=QLabel('Run Track Face, select a person, then play or scrub the timeline.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.setWidget(panel);self.resize(420,550)
        self.open_button.clicked.connect(self.choose_model)
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(33)
        self.visibilityChanged.connect(self.visibility_changed)
        self.load_saved_model()

    def visibility_changed(self,visible):
        if not visible:self.stream.setChecked(False);self.view.set_values(np.zeros(52,np.float32))

    def load_saved_model(self):
        from .resource_safety import path,read_settings
        try:
            filename=read_settings(path()).get('face_preview_fbx','')
            if filename and Path(filename).is_file():QTimer.singleShot(0,lambda:self.import_model(filename))
        except (OSError,ValueError):pass

    def choose_model(self):
        filename,_=QFileDialog.getOpenFileName(self,'Load ARKit Face Model','','FBX (*.fbx)')
        if filename:self.import_model(filename)

    def import_model(self,filename):
        if self.process is not None:return
        try:
            from .retarget.runtime import blender
            source=Path(filename).resolve();stamp=source.stat()
            script=ROOT/'mocap/face_model_export.py'
            identity=str(source)+str(stamp.st_size)+str(stamp.st_mtime_ns)+hashlib.sha256(script.read_bytes()).hexdigest()
            folder=HOME/'cache/face_preview';folder.mkdir(parents=True,exist_ok=True)
            target=folder/(hashlib.sha256(identity.encode()).hexdigest()+'.npz')
            self.model_label.setText(source.name+' · loading...')
            if target.is_file():self.feed.load(target);self.save_model_path(source);return
            partial=folder/(uuid.uuid4().hex+'.partial.npz')
            process=QProcess(self);self.process=process;self.pending=(source,target,partial)
            process.setProcessChannelMode(QProcess.MergedChannels);self.import_log=''
            process.readyReadStandardOutput.connect(self.import_output)
            process.errorOccurred.connect(self.import_error)
            process.finished.connect(self.import_finished)
            self.open_button.setEnabled(False)
            process.start(str(blender()),['--background','--factory-startup','--disable-autoexec','--threads','2','--python-exit-code','1','--python',str(script),'--','--source',str(source),'--output',str(partial)])
        except (OSError,ValueError,RuntimeError) as error:self.model_label.setText(str(error))

    def save_model_path(self,source):
        from .resource_safety import path,read_settings
        from .storage import atomic_json
        try:
            target=path();settings=read_settings(target);settings['face_preview_fbx']=str(source)
            target.parent.mkdir(parents=True,exist_ok=True);atomic_json(target,settings)
        except (OSError,ValueError) as error:self.window.log.append('Face preview preference: '+str(error))

    def import_output(self):
        if self.process:self.import_log=(self.import_log+bytes(self.process.readAllStandardOutput()).decode('utf-8','replace'))[-8192:]

    def import_error(self,error):
        if error==QProcess.FailedToStart:self.import_finished(1,None)

    def import_finished(self,code,status):
        if self.process is None:return
        self.import_output();process=self.process;self.process=None;source,target,partial=self.pending;self.pending=None
        try:
            if code!=0 or not partial.is_file():raise RuntimeError('Face model import failed: '+self.import_log[-1200:])
            partial.replace(target);self.feed.load(target);self.save_model_path(source)
        except (OSError,RuntimeError) as error:self.model_label.setText(str(error))
        finally:
            partial.unlink(missing_ok=True);process.deleteLater();self.open_button.setEnabled(True)

    def tick(self):
        if self.closed:return
        from .face_preview_data import preview_sample
        w=self.window
        values,frame,label,status,neutral=preview_sample(w,visible=self.isVisible())
        ready=status in ('Ready','Held expression') and not w.contact_saving
        self.neutral_button.setEnabled(ready and status=='Ready')
        self.neutral_reset.setEnabled(ready and neutral is not None)
        self.neutral_label.setText('Track Face · Neutral: '+(str(neutral+1) if neutral is not None else 'Default'))
        self.view.set_values(values)
        token=(str(w.project),label,frame)
        self.feed.submit((token,tuple(float(v) for v in values),status,frame,label,self.stream.isChecked() and self.isVisible()))
        suffix=f' · Sending UDP {PORT}' if self.stream.isChecked() else ''
        self.status.setText(f'{label} · frame {frame+1} · {status}'+suffix)
        asset=self.feed.asset
        if asset is not None:
            self.feed.asset=None;data,error=asset
            if error:self.model_label.setText(error)
            else:
                self.view.set_asset(data);rows=json.loads(str(data['report']))
                channels=set(n for row in rows for n in row['channels'])
                self.model_label.setText(f'{len(rows)} meshes · {len(channels)}/52 ARKit channels · studio shading')

    def shutdown(self):
        if self.closed:return
        self.closed=True
        self.timer.stop();self.stream.setChecked(False);self.feed.close()
        if self.process is not None:
            self.process.blockSignals(True);self.process.kill();self.process.waitForFinished(3000)
            if self.pending:self.pending[2].unlink(missing_ok=True)
            self.process.deleteLater();self.process=None
        self.view.release()
