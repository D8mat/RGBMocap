"""Explicit save-only calibration wizard; never changes active reconstruction."""
import copy
import json
import math
import numpy as np
from pathlib import Path
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout)
from .calibration_solver import lens_points, solve
from .image_io import read_image
from .storage import atomic_json


class Worker(QThread):
    result = Signal(object)
    failure = Signal(str)
    def __init__(self, task, parent):
        super().__init__(parent); self.task = task
    def run(self):
        try:
            cv2.ocl.setUseOpenCL(False)
            result = self.task(self.isInterruptionRequested)
            if not self.isInterruptionRequested():self.result.emit(result)
        except Exception as exc:
            self.failure.emit(str(exc))


def calibrate_photos(names, cols, rows, mm, cancel):
    points = []; size = None; skipped = []
    for name in names:
        if cancel():raise InterruptedError('Cancelled')
        im = read_image(name)
        if im is None:raise ValueError('Cannot read '+name)
        current = (im.shape[1], im.shape[0])
        if size is not None and size != current:raise ValueError('All lens photos must have the same resolution')
        size = current
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
        if ok:points.append(corners.reshape(-1,2))
        else:skipped.append(Path(name).name)
    if cancel():raise InterruptedError('Cancelled')
    result = lens_points(points, size, cols, rows, mm)
    result['skipped_photos'] = skipped
    return result


def calibrate_video(name, cols, rows, mm, start, end, interval, cancel):
    if not all(math.isfinite(v) for v in (start,end,interval)) or start < 0 or end < 0 or interval < .1:
        raise ValueError('Invalid video sampling settings')
    path = Path(name); stamp = (path.stat().st_size,path.stat().st_mtime_ns)
    cap = cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1,
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,2000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000])
    points = []; accepted = []; skipped = []; size = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS); count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError('Cannot read video frame rate/duration')
        duration = count/fps; stop = duration if end == 0 else min(end,duration)
        if start >= stop:raise ValueError('Start must precede end of video range')
        samples = math.ceil((stop-start)/interval)
        if samples > 300:raise ValueError('Maximum 300 sampled frames; increase interval or shorten range')
        indices = sorted({min(int(count)-1,round((start+i*interval)*fps)) for i in range(samples) if start+i*interval < stop})
        for index in indices:
            if cancel():raise InterruptedError('Cancelled')
            if not cap.set(cv2.CAP_PROP_POS_FRAMES,index):raise ValueError('Cannot seek video frame '+str(index))
            ok, im = cap.read()
            if not ok:raise ValueError('Cannot decode video frame '+str(index))
            current = (im.shape[1],im.shape[0])
            if size is not None and current != size:raise ValueError('Video resolution changed')
            size = current
            if cancel():raise InterruptedError('Cancelled')
            ok,corners = cv2.findChessboardCornersSB(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(cols,rows))
            if not ok:skipped.append(dict(frame=index,reason='board not detected'));continue
            p = corners.reshape(-1,2)
            # Symmetric checkerboards can reverse their corner ordering between views.
            if any(min(np.sqrt(np.mean(np.sum(((p-q)/np.array(size))**2,axis=1))),
                       np.sqrt(np.mean(np.sum(((p-q[::-1])/np.array(size))**2,axis=1)))) < .01 for q in points):
                skipped.append(dict(frame=index,reason='near-duplicate board view'));continue
            points.append(p);accepted.append(index)
        if cancel():raise InterruptedError('Cancelled')
        if stamp != (path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Video changed during calibration')
        if len(points) < 10:raise ValueError(f'Only {len(points)} distinct board views accepted; need 10+. Move and tilt the board, keeping the camera fixed.')
        result = lens_points(points,size,cols,rows,mm)
        result['video_sampling'] = dict(source_file=path.name,source_stamp=list(stamp),fps=fps,
            start_seconds=start,end_seconds=stop,interval_seconds=interval,
            accepted_frame_indices=accepted,skipped_frames=skipped)
        return result
    finally:cap.release()


class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle('Calibration Board'); self.resize(650,380)
        owner=parent
        while owner is not None and not hasattr(owner,'mamma_inputs'):owner=owner.parent()
        self.settings_owner=owner
        self.entries=copy.deepcopy(getattr(owner,'mamma_inputs',[]))
        self.worker = None; self.value = None; self.closing = False
        layout = QVBoxLayout(self)
        label = QLabel('1. Calibrate each lens from varied checkerboard photos or video. Save one lens JSON per camera.\n'
                       '2. Calculate camera positions from a stationary checkerboard on the floor.\n'
                       'Results are saved for review, never applied automatically. Distorted videos need undistortion before MAMMA import.')
        label.setWordWrap(True); layout.addWidget(label)
        form = QFormLayout(); layout.addLayout(form)
        self.cols = QSpinBox(); self.cols.setRange(3,30); self.cols.setValue(9); form.addRow('Inner corner columns',self.cols)
        self.rows = QSpinBox(); self.rows.setRange(3,30); self.rows.setValue(6); form.addRow('Inner corner rows',self.rows)
        self.mm = QDoubleSpinBox(); self.mm.setRange(1,1000); self.mm.setValue(25); self.mm.setSuffix(' mm'); form.addRow('Checkerboard square edge',self.mm)
        self.photos = QPushButton('Calibrate Lens from Photos...'); self.photos.clicked.connect(self.lens); layout.addWidget(self.photos)
        self.video_start = QDoubleSpinBox(); self.video_start.setRange(0,86400); self.video_start.setSuffix(' s'); form.addRow('Video start',self.video_start)
        self.video_end = QDoubleSpinBox(); self.video_end.setRange(0,86400); self.video_end.setSpecialValueText('End of video'); self.video_end.setSuffix(' s'); form.addRow('Video end',self.video_end)
        self.video_interval = QDoubleSpinBox(); self.video_interval.setRange(.1,60); self.video_interval.setValue(1); self.video_interval.setSuffix(' s'); form.addRow('Sample every',self.video_interval)
        self.video = QPushButton('Calibrate Lens from Video...'); self.video.clicked.connect(self.lens_video); layout.addWidget(self.video)
        self.open_camera=QComboBox()
        for entry in self.entries:self.open_camera.addItem(entry['id'],entry)
        layout.addWidget(self.open_camera)
        self.open_video=QPushButton('Calibrate Lens from Open Camera');self.open_video.setEnabled(bool(self.entries));self.open_video.clicked.connect(self.lens_open);layout.addWidget(self.open_video)
        self.board = QPushButton('Camera Positions from Checkerboard...'); self.board.clicked.connect(self.board_positions); layout.addWidget(self.board)
        self.save = QPushButton('Save Result JSON...'); self.save.clicked.connect(self.export); self.save.setEnabled(False); layout.addWidget(self.save)
        self.stop = QPushButton('Stop'); self.stop.clicked.connect(self.cancel); self.stop.setEnabled(False); layout.addWidget(self.stop)
        self.status = QLabel('Checkerboard settings only. AprilTag calibration is a separate window.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        settings=getattr(self.settings_owner,'board_calibration_settings',{})
        for name in ('cols','rows','mm','video_start','video_end','video_interval'):
            if name in settings:getattr(self,name).setValue(settings[name])
        index=self.open_camera.findText(settings.get('camera_id',''))
        if index>=0:self.open_camera.setCurrentIndex(index)
    def remember_settings(self):
        if self.settings_owner is not None:
            value={name:getattr(self,name).value() for name in ('cols','rows','mm','video_start','video_end','video_interval')}
            value['camera_id']=self.open_camera.currentText()
            self.settings_owner.board_calibration_settings=value
    def lens(self):
        names,_ = QFileDialog.getOpenFileNames(self,'One camera: varied checkerboard photos','','Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if names:
            cols, rows, mm = self.cols.value(), self.rows.value(), self.mm.value()
            self.start(lambda cancel:calibrate_photos(names,cols,rows,mm,cancel))
    def lens_video(self):
        name,_ = QFileDialog.getOpenFileName(self,'One camera: checkerboard video','','Videos (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)')
        if name:
            args = (name,self.cols.value(),self.rows.value(),self.mm.value(),self.video_start.value(),self.video_end.value(),self.video_interval.value())
            self.start(lambda cancel:calibrate_video(*args,cancel))
    def lens_open(self):
        entry=self.open_camera.currentData()
        if entry:
            args=(entry['path'],self.cols.value(),self.rows.value(),self.mm.value(),self.video_start.value(),self.video_end.value(),self.video_interval.value())
            self.start(lambda cancel:calibrate_video(*args,cancel))
    def board_positions(self):
        from .checkerboard_ui import BoardDialog
        BoardDialog(self,self.cols.value(),self.rows.value(),self.mm.value()).exec()
    def start(self,task):
        if self.worker:return
        self.value = None; self.save.setEnabled(False)
        for widget in (self.open_camera,self.open_video,self.board,self.photos,self.video,self.video_start,self.video_end,self.video_interval,self.cols,self.rows,self.mm):widget.setEnabled(False)
        self.stop.setEnabled(True); self.status.setText('Calculating on CPU...')
        self.worker = Worker(task,self); self.worker.result.connect(self.received); self.worker.failure.connect(self.status.setText)
        self.worker.finished.connect(self.done_work); self.worker.start()
    def received(self,value):
        self.value = value
        if 'cameras' in value:
            message = '\n'.join(f'{name}: RMS {c["quality"]["rms_px"]:.3f} px' for name,c in value['cameras'].items())
            if value['requires_undistortion']:message += '\nUndistort videos before MAMMA import; do not discard distortion coefficients.'
        else:message = f'Lens RMS {value["quality"]["rms_px"]:.3f} px; {value["quality"]["accepted_views"]} accepted views.'
        self.status.setText(message)
    def done_work(self):
        worker = self.worker; self.worker = None; worker.deleteLater()
        for widget in (self.open_camera,self.open_video,self.board,self.photos,self.video,self.video_start,self.video_end,self.video_interval,self.cols,self.rows,self.mm):widget.setEnabled(True)
        self.open_video.setEnabled(bool(self.entries));self.open_camera.setEnabled(bool(self.entries))
        self.stop.setEnabled(False); self.save.setEnabled(self.value is not None)
        if self.closing:self.close()
    def export(self):
        name,_ = QFileDialog.getSaveFileName(self,'Save Calibration Result','calibration_result.json','JSON (*.json)')
        if name:
            try:
                path = Path(name)
                if not path.suffix:path = path.with_suffix('.json')
                atomic_json(path,self.value); self.status.setText('Saved '+str(path))
            except OSError as exc:self.status.setText(str(exc))
    def cancel(self):
        if self.worker:self.worker.requestInterruption(); self.status.setText('Stopping after current OpenCV operation...')
    def closeEvent(self,event):
        self.remember_settings()
        if self.worker:self.closing=True; self.cancel(); event.ignore()
        else:event.accept()
    def reject(self):self.close()


def run_job(path,cancel=lambda:False):
    def read(name):return json.loads((path.parent/name).read_text(encoding='utf-8-sig'))
    job = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(job,dict) or not all(key in job for key in ('detections','marker_map','lenses')):
        raise ValueError('Select a calibration JOB JSON referencing detections, marker_map and lenses, not the detections file. For the known 3x2 board, use AprilTag Board - Calibrate from Videos instead.')
    session = read(job['detections'])
    if session.get('cancelled') or session.get('errors'):raise ValueError('Use a complete detection session without errors')
    observations = session['observations']
    if {o['camera_id'] for o in observations} != set(job['lenses']):raise ValueError('Lens IDs must exactly match detection camera IDs')
    return solve(observations,{key:read(value) for key,value in job['lenses'].items()},read(job['marker_map']),cancel)


class AprilTagSolveDialog(QDialog):
    """AprilTag-only camera solver; lens files are selected by its job JSON."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('AprilTag Calibration - Calculate Cameras')
        self.worker=None;self.value=None;self.closing=False
        layout=QVBoxLayout(self)
        text=QLabel('Select an AprilTag calibration job JSON containing detections, measured tag map and one lens JSON per camera. '
                    'Marker size and snapshot time belong to AprilTag Calibration. Checkerboard settings are configured separately in Calibration Board.')
        text.setWordWrap(True);layout.addWidget(text)
        self.job=QPushButton('Select AprilTag Job and Calculate...');self.job.clicked.connect(self.positions);layout.addWidget(self.job)
        self.save=QPushButton('Save AprilTag Camera Calibration...');self.save.setEnabled(False);self.save.clicked.connect(self.export);layout.addWidget(self.save)
        self.stop=QPushButton('Stop');self.stop.setEnabled(False);self.stop.clicked.connect(self.cancel);layout.addWidget(self.stop)
        self.status=QLabel('Requires known lens intrinsics and a measured 3D tag map.');self.status.setWordWrap(True);layout.addWidget(self.status)
    def positions(self):
        name,_=QFileDialog.getOpenFileName(self,'AprilTag calibration job JSON','','JSON (*.json)')
        if name:self.start(lambda cancel:run_job(Path(name),cancel))
    def start(self,task):
        if self.worker:return
        self.value=None;self.save.setEnabled(False);self.job.setEnabled(False);self.stop.setEnabled(True)
        self.status.setText('Calculating AprilTag camera positions...')
        self.worker=Worker(task,self);self.worker.result.connect(self.received);self.worker.failure.connect(self.status.setText)
        self.worker.finished.connect(self.done_work);self.worker.start()
    received=CalibrationDialog.received
    export=CalibrationDialog.export
    cancel=CalibrationDialog.cancel
    reject=CalibrationDialog.reject
    def done_work(self):
        worker=self.worker;self.worker=None;worker.deleteLater();self.job.setEnabled(True);self.stop.setEnabled(False);self.save.setEnabled(self.value is not None)
        if self.closing:self.close()
    def closeEvent(self,event):
        if self.worker:self.closing=True;self.cancel();event.ignore()
        else:event.accept()
