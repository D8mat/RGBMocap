"""Known AprilTag board: per-camera lens videos, then a common floor snapshot."""
import copy
import json
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QComboBox,QDoubleSpinBox,QFormLayout,QFileDialog,QCheckBox,QTableWidget,
    QTableWidgetItem,QAbstractItemView)
from .apriltag_board import BOARD, video_lens, camera_positions
from .storage import atomic_json


class BoardWorker(QThread):
    result=Signal(object);failure=Signal(str);progress=Signal(str)
    def __init__(self,task,parent):super().__init__(parent);self.task=task
    def run(self):
        try:
            import cv2
            cv2.ocl.setUseOpenCL(False)
            value=self.task(self.isInterruptionRequested,self.progress.emit)
            if not self.isInterruptionRequested():self.result.emit(value)
        except Exception as exc:self.failure.emit(str(exc))


class AprilTagBoardDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('AprilTag Board - Calibration from Videos');self.resize(820,650)
        self.owner=parent
        while self.owner is not None and not hasattr(self.owner,'mamma_inputs'):self.owner=self.owner.parent()
        self.worker=None;self.value=None;self.closing=False;self.lenses={}
        self.entries=copy.deepcopy(getattr(self.owner,'mamma_inputs',[]))
        for e in self.entries:e['lens_path']=e['path']
        layout=QVBoxLayout(self)
        label=QLabel('Use the RGBMoCap 3x2 AprilTag board: 570 x 390 mm, IDs 0-5, black edges 150 mm, gaps 30 mm.\n'
            '1. Fixed cameras: move, tilt in BOTH axes and vary board distance. Calibrate each lens.\n'
            '2. Leave the board stationary, face up on the floor, visible in every camera. Enter that common video time.\n'
            'No manual marker coordinates or focal lengths. Cameras, focus, zoom, crop and resolution must stay fixed. '
            'This does not solve freely scattered tags or camera motion.')
        label.setWordWrap(True);layout.addWidget(label)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Camera ID','Lens calibration video','Floor pose video'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows);self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        row=QHBoxLayout();layout.addLayout(row);self.controls=[]
        def button(text,slot,where):
            b=QPushButton(text);b.clicked.connect(slot);where.addWidget(b);self.controls.append(b);return b
        button('Use Open Cameras',self.use_open,row)
        button('Choose Camera Videos...',self.choose_videos,row)
        button('Set Lens Video...',lambda:self.set_video('lens_path'),row)
        button('Set Floor Video...',lambda:self.set_video('path'),row)
        form=QFormLayout();layout.addLayout(form)
        def seconds(label,value=0,minimum=0):
            b=QDoubleSpinBox();b.setRange(minimum,86400);b.setDecimals(3);b.setValue(value);b.setSuffix(' s');form.addRow(label,b);self.controls.append(b);return b
        self.start_time=seconds('Lens video start')
        self.end_time=seconds('Lens video end (0 = end)')
        self.interval=seconds('Sample every',1,.1)
        self.model=QComboBox();self.model.addItems(['Real camera: estimate lens distortion','Unreal / known pinhole: no distortion'])
        form.addRow('Lens model',self.model);self.controls.append(self.model)
        self.model.currentIndexChanged.connect(self.reset_lenses)
        row=QHBoxLayout();layout.addLayout(row)
        button('Calibrate Selected Lens',self.lens,row)
        button('Save Selected Lens...',self.save_lens,row)
        button('Load Selected Lens...',self.load_lens,row)
        self.floor_time=seconds('Common floor snapshot time (original videos)')
        self.floor_time.valueChanged.connect(self.invalidate)
        self.confirm=QCheckBox('Board is stationary, face up on the floor at this time in ALL videos; videos are synchronized.')
        self.confirm.toggled.connect(self.invalidate);self.controls.append(self.confirm);layout.addWidget(self.confirm)
        button('Calculate Camera Calibration',self.positions,layout)
        self.save=button('Save Camera Calibration...',self.export,layout);self.save.setEnabled(False)
        self.stop=QPushButton('Stop');self.stop.clicked.connect(self.cancel);self.stop.setEnabled(False);layout.addWidget(self.stop)
        self.status=QLabel('Select a camera row, then calibrate its lens. Repeat for each camera.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.rebuild()

    def invalidate(self,*_):self.value=None;self.save.setEnabled(False)
    def reset_lenses(self,*_):self.lenses.clear();self.invalidate();self.rebuild()
    def rebuild(self):
        selected=max(0,self.table.currentRow());self.table.setRowCount(len(self.entries))
        for i,e in enumerate(self.entries):
            for col,key in enumerate(('id','lens_path','path')):
                text=str(e[key])+(' [lens ready]' if col==0 and e['id'] in self.lenses else '')
                item=QTableWidgetItem(text);item.setToolTip(str(e[key]));self.table.setItem(i,col,item)
        self.table.resizeColumnsToContents()
        if self.entries:self.table.selectRow(min(selected,len(self.entries)-1))
    def use_open(self):
        self.entries=copy.deepcopy(getattr(self.owner,'mamma_inputs',[]))
        for e in self.entries:e['lens_path']=e['path']
        self.reset_lenses()
    def choose_videos(self):
        names,_=QFileDialog.getOpenFileNames(self,'Synchronized camera videos','','Videos (*.mp4 *.mov *.avi *.mkv *.m4v)')
        if not names:return
        if len({Path(n).stem for n in names})!=len(names):self.status.setText('Video basenames must be unique camera IDs.');return
        self.entries=[dict(id=Path(n).stem,path=n,lens_path=n) for n in names];self.reset_lenses()
    def selected(self):
        i=self.table.currentRow()
        if not 0<=i<len(self.entries):self.status.setText('Select a camera row first.');return None
        return self.entries[i]
    def set_video(self,key):
        e=self.selected()
        if e is None:return
        name,_=QFileDialog.getOpenFileName(self,'Choose video for '+e['id'],'','Videos (*.mp4 *.mov *.avi *.mkv *.m4v)')
        if name:
            e[key]=name
            if key=='lens_path':self.lenses.pop(e['id'],None)
            self.invalidate();self.rebuild()
    def lens(self):
        e=self.selected()
        if e is None:return
        ident=e['id'];args=(e['lens_path'],self.start_time.value(),self.end_time.value(),self.interval.value(),self.model.currentIndex()==1)
        self.lenses.pop(ident,None);self.rebuild()
        self.run(lambda cancel,progress:dict(kind='lens',id=ident,data=video_lens(*args,cancel,progress)))
    def positions(self):
        if not self.confirm.isChecked():self.status.setText('Confirm the stationary face-up floor board and synchronized videos first.');return
        entries=copy.deepcopy(self.entries);lenses=copy.deepcopy(self.lenses);seconds=self.floor_time.value()
        self.run(lambda cancel,progress:dict(kind='cameras',data=camera_positions(entries,lenses,seconds,cancel,progress)))
    def run(self,task):
        if self.worker:return
        self.invalidate()
        for c in self.controls:c.setEnabled(False)
        self.table.setEnabled(False);self.stop.setEnabled(True);self.status.setText('Calculating on CPU...')
        self.worker=BoardWorker(task,self);self.worker.result.connect(self.received)
        self.worker.progress.connect(self.status.setText);self.worker.failure.connect(self.status.setText)
        self.worker.finished.connect(self.done_work);self.worker.start()
    def received(self,result):
        if result['kind']=='lens':
            value=result['data'];self.lenses[result['id']]=value;self.rebuild()
            q=value['quality'];self.status.setText(f"{result['id']}: {q['accepted_views']} views, RMS {q['rms_px']:.3f} px. Lens ready.")
        else:
            self.value=result['data'];message='\n'.join(f"{name}: pose RMS {c['quality']['rms_px']:.3f} px" for name,c in self.value['cameras'].items())
            if self.value['requires_undistortion']:message+='\nReal-lens distortion is preserved. MAMMA requires undistorted videos before importing this calibration; do not erase the coefficients.'
            self.status.setText(message)
    def done_work(self):
        worker=self.worker;self.worker=None;worker.deleteLater()
        for c in self.controls:c.setEnabled(True)
        self.table.setEnabled(True);self.stop.setEnabled(False);self.save.setEnabled(self.value is not None)
        if self.closing:self.close()
    def write(self,value,title,default):
        name,_=QFileDialog.getSaveFileName(self,title,default,'JSON (*.json)')
        if not name:return
        try:
            p=Path(name);p=p if p.suffix else p.with_suffix('.json');atomic_json(p,value);self.status.setText('Saved '+str(p))
        except OSError as exc:self.status.setText(str(exc))
    def export(self):
        if self.value is not None:self.write(self.value,'Save Camera Calibration','apriltag_board_calibration.json')
    def save_lens(self):
        e=self.selected()
        if e is None:return
        if e['id'] not in self.lenses:self.status.setText('Calibrate this lens first.');return
        self.write(self.lenses[e['id']],'Save Lens Calibration','apriltag_board_lens.json')
    def load_lens(self):
        e=self.selected()
        if e is None:return
        name,_=QFileDialog.getOpenFileName(self,'Load previously calibrated lens','','JSON (*.json)')
        if not name:return
        try:
            value=json.loads(Path(name).read_text(encoding='utf-8-sig'))
            from .calibration_solver import array
            if value.get('board')!=BOARD:raise ValueError('Lens file uses a different board')
            array(value['intrinsic_matrix'],(3,3));array(value['distortion_coeffs'],(5,));array(value['image_size'],(2,))
            if 'quality' not in value:raise ValueError('Missing calibration quality')
            self.lenses[e['id']]=value;self.invalidate();self.rebuild();self.status.setText('Loaded lens. Use only with identical camera, lens, focus, zoom, crop and resolution.')
        except (OSError,ValueError,KeyError,TypeError) as exc:self.status.setText(str(exc))
    def cancel(self):
        if self.worker:self.worker.requestInterruption();self.status.setText('Stopping after the current OpenCV operation...')
    def closeEvent(self,event):
        if self.worker:self.closing=True;self.cancel();event.ignore()
        else:event.accept()
    def reject(self):self.close()
