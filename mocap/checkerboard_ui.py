"""Review each camera's board orientation before saving a common-world result."""
import math
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QLabel,QPushButton,QLineEdit,
    QCheckBox,QFileDialog,QComboBox,QDoubleSpinBox)
from .calibration_solver_ui import Worker
from .checkerboard_pose import solve_board,calibration
from .image_io import read_image
from .storage import atomic_json


def detect_photo(name,cols,rows,cancel):
    path=Path(name);before=(path.stat().st_size,path.stat().st_mtime_ns)
    im=read_image(path)
    result=detect_image(im,cols,rows,cancel)
    if before!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Photo changed during detection')
    return result


def detect_snapshot(entry,seconds,cols,rows,cancel):
    if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid video time')
    path=Path(entry['path']);before=(path.stat().st_size,path.stat().st_mtime_ns)
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1,cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,2000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000])
    try:
        if cancel():raise InterruptedError('Cancelled')
        fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(fps) or fps<=0 or not math.isfinite(count) or count<1:raise ValueError('Cannot read video')
        frame=round(seconds*fps)
        if frame>=count:raise ValueError('Time is outside this video')
        if not cap.set(cv2.CAP_PROP_POS_FRAMES,frame):raise ValueError('Cannot seek frame')
        ok,im=cap.read()
        if not ok:raise ValueError('Cannot decode frame')
        result=detect_image(im,cols,rows,cancel)
        if before!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Video changed during detection')
        result['source']=dict(camera_id=entry['id'],source_file=path.name,source_stamp=list(before),frame_index=frame,fps=fps,requested_seconds=seconds)
        return result
    finally:cap.release()


def detect_image(im,cols,rows,cancel):
    if im is None:raise ValueError('Cannot read image')
    if cancel():raise InterruptedError('Cancelled')
    ok,p=cv2.findChessboardCornersSB(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(cols,rows))
    if not ok:raise ValueError('Whole checkerboard not detected')
    size=[im.shape[1],im.shape[0]];scale=min(1,900/size[0],500/size[1])
    preview=cv2.resize(im,None,fx=scale,fy=scale) if scale<1 else im
    return dict(points=p.reshape(-1,2),size=size,preview=preview,scale=scale)


class BoardDialog(QDialog):
    def __init__(self,parent,cols,rows,mm):
        super().__init__(parent);self.setWindowTitle('Checkerboard - Fixed Camera Positions')
        self.cols,self.rows,self.mm=cols,rows,mm;self.data=None;self.worker=None;self.cameras={};self.closing=False
        self.entries=copy.deepcopy(getattr(parent,'entries',[]))
        layout=QVBoxLayout(self)
        label=QLabel('First calibrate each lens from photos/video in the previous window.\n'
            'For camera positions, put the board face up on the floor and DO NOT MOVE it between cameras.\n'
            'Photograph the entire board with each fixed camera. Mark a physical inner corner outside the pattern.\n'
            'In every view, O must be that same corner, and +X must point along the same physical edge.\n'
            'Board surface becomes Z=0. Lens settings and photo resolution must match the video.')
        label.setWordWrap(True);layout.addWidget(label)
        self.ident=QLineEdit();self.ident.setPlaceholderText('Camera ID = video filename without extension');layout.addWidget(self.ident)
        self.load=QPushButton('Load Camera Photo...');self.load.clicked.connect(self.photo);layout.addWidget(self.load)
        self.open_camera=QComboBox()
        for entry in self.entries:self.open_camera.addItem(entry['id'],entry)
        layout.addWidget(self.open_camera)
        self.seconds=QDoubleSpinBox();self.seconds.setRange(0,86400);self.seconds.setDecimals(3);self.seconds.setSuffix(' s (original video time)');layout.addWidget(self.seconds)
        self.snapshot=QPushButton('Detect Board in Open Camera');self.snapshot.setEnabled(bool(self.entries));self.snapshot.clicked.connect(self.video);layout.addWidget(self.snapshot)
        self.reverse=QCheckBox('Reverse corner order (180 degrees)');self.reverse.toggled.connect(self.preview);layout.addWidget(self.reverse)
        self.image=QLabel('Load a stationary-board photo');self.image.setAlignment(Qt.AlignCenter);layout.addWidget(self.image)
        self.confirm=QCheckBox('I checked: same physical O and +X direction; board was stationary on the floor');layout.addWidget(self.confirm)
        self.add=QPushButton('Load Lens JSON and Add Camera');self.add.clicked.connect(self.add_camera);self.add.setEnabled(False);layout.addWidget(self.add)
        self.save=QPushButton('Save Cameras JSON...');self.save.clicked.connect(self.export);self.save.setEnabled(False);layout.addWidget(self.save)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
    def photo(self):
        name,_=QFileDialog.getOpenFileName(self,'Stationary board photo','','Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not name:return
        self.ident.setReadOnly(False)
        self.start_detection(lambda cancel:detect_photo(name,self.cols,self.rows,cancel))
    def video(self):
        entry=self.open_camera.currentData()
        if not entry:return
        entry=copy.deepcopy(entry);seconds=self.seconds.value()
        self.ident.setText(entry['id']);self.ident.setReadOnly(True)
        self.start_detection(lambda cancel:detect_snapshot(entry,seconds,self.cols,self.rows,cancel))
    def start_detection(self,task):
        if self.worker:return
        self.data=None;self.add.setEnabled(False);self.confirm.setChecked(False);self.load.setEnabled(False)
        self.snapshot.setEnabled(False);self.open_camera.setEnabled(False);self.seconds.setEnabled(False)
        self.status.setText('Detecting checkerboard...')
        self.worker=Worker(task,self)
        self.worker.result.connect(self.received);self.worker.failure.connect(self.status.setText);self.worker.finished.connect(self.finished_work);self.worker.start()
    def received(self,value):
        self.data=value;self.reverse.setChecked(False);self.preview();self.add.setEnabled(True)
    def finished_work(self):
        worker=self.worker;self.worker=None;worker.deleteLater();self.load.setEnabled(True)
        self.snapshot.setEnabled(bool(self.entries));self.open_camera.setEnabled(True);self.seconds.setEnabled(True)
        if self.closing:self.close()
    def preview(self):
        self.confirm.setChecked(False)
        if self.data is None:return
        p=self.data['points'][::-1] if self.reverse.isChecked() else self.data['points']
        im=self.data['preview'].copy();q=np.rint(p*self.data['scale']).astype(int)
        origin=tuple(q[0]);x=tuple(q[self.cols-1]);y=tuple(q[(self.rows-1)*self.cols])
        cv2.circle(im,origin,8,(0,0,255),2);cv2.putText(im,'O',origin,cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
        for end,label in [(x,'+X'),(y,'-Y')]:
            cv2.arrowedLine(im,origin,end,(0,255,255),2);cv2.putText(im,label,end,cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(image));self.status.setText('Check O and +X before adding this camera.')
    def add_camera(self):
        name=self.ident.text().strip()
        if not name or name in self.cameras:self.status.setText('Enter a unique camera ID.');return
        if not self.confirm.isChecked():self.status.setText('Verify the physical origin, direction and stationary floor board first.');return
        path,_=QFileDialog.getOpenFileName(self,'Lens calibration for this camera','','JSON (*.json)')
        if not path:return
        try:
            lens=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            p=self.data['points'][::-1].copy() if self.reverse.isChecked() else self.data['points']
            result=solve_board(p,lens,self.data['size'],self.cols,self.rows,self.mm)
            if 'source' in self.data:result['board_observation']=self.data['source']
            self.cameras[name]=result;self.save.setEnabled(len(self.cameras)>=2)
            self.status.setText('Added '+name+f': RMS {result["quality"]["rms_px"]:.3f} px. Cameras: '+', '.join(self.cameras))
            self.data=None;self.add.setEnabled(False);self.confirm.setChecked(False);self.ident.clear()
        except (ValueError,KeyError,TypeError,OSError,cv2.error) as exc:self.status.setText(str(exc))
    def export(self):
        path,_=QFileDialog.getSaveFileName(self,'Save checkerboard camera calibration','checkerboard_cameras.json','JSON (*.json)')
        if not path:return
        try:
            output=calibration(self.cameras)
            from .mamma_calibration_export import validate
            validate(output)
            p=Path(path);p=p if p.suffix else p.with_suffix('.json');atomic_json(p,output)
            self.status.setText('Saved '+str(p)+(' — videos must be undistorted before MAMMA import.' if output['requires_undistortion'] else ''))
        except (ValueError,OSError) as exc:self.status.setText(str(exc))
    def closeEvent(self,event):
        if self.worker:self.closing=True;self.worker.requestInterruption();event.ignore()
        else:event.accept()
    def reject(self):self.close()
