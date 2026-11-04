"""Read-only, cancellable AprilTag snapshots; runs outside the GUI thread."""
import copy,json
from pathlib import Path
import cv2
from PySide6.QtCore import QThread,Signal,Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,
    QDoubleSpinBox,QComboBox,QFileDialog,QDialogButtonBox)
from .apriltag_detection import detect
from .image_io import read_image
from .storage import atomic_json


class DetectWorker(QThread):
    result=Signal(object,object);failure=Signal(str)
    def __init__(self,entries,seconds,size,parent=None):
        super().__init__(parent);self.entries=copy.deepcopy(entries);self.seconds=seconds;self.size=size
    def run(self):
        # OpenCV settings are thread-local for OpenCL. Do not change global thread counts.
        cv2.ocl.setUseOpenCL(False)
        for e in self.entries:
            if self.isInterruptionRequested():break
            cap=None
            try:
                p=Path(e['path']);before=[p.stat().st_size,p.stat().st_mtime_ns]
                if e.get('image'):
                    im=read_image(p);frame=None;fps=None
                else:
                    cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1,cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,2000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000])
                    fps=cap.get(cv2.CAP_PROP_FPS)
                    if not cap.isOpened() or not fps>0:raise ValueError('Cannot open video')
                    frame=round(self.seconds*fps)
                    if frame>=cap.get(cv2.CAP_PROP_FRAME_COUNT):raise ValueError('Requested time is outside this video')
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES,frame):raise ValueError('Cannot seek requested frame')
                    ok,im=cap.read()
                    if not ok:raise ValueError('Cannot decode requested frame')
                if im is None:raise ValueError('Cannot read image')
                if self.isInterruptionRequested():break
                value,preview=detect(im,self.size)
                if before!=[p.stat().st_size,p.stat().st_mtime_ns]:raise ValueError('Source file changed during detection')
                value.update(camera_id=e['id'],source_file=p.name,source_stamp=before,requested_seconds=None if e.get('image') else self.seconds,frame_index=frame,fps=fps)
                rgb=cv2.cvtColor(preview,cv2.COLOR_BGR2RGB);image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()
                if not self.isInterruptionRequested():self.result.emit(value,image)
            except Exception as exc:
                if not self.isInterruptionRequested():self.failure.emit(e['id']+': '+str(exc))
            finally:
                if cap is not None:cap.release()


class DetectionDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.setWindowTitle('AprilTag Calibration - 36h11');self.resize(900,700)
        self.entries=copy.deepcopy(getattr(parent,'mamma_inputs',[]));self.worker=None;self.closing=False;self.results=[];self.images=[];self.errors=[];self.cancelled=False
        layout=QVBoxLayout(self)
        text=QLabel('Detect AprilTags in photos or snapshots from the loaded camera videos. This exports 2D observations, not calibrated cameras. Print IDs must match this detector; ordinary checkerboards are not AprilTags. Camera videos must already be synchronized.');text.setWordWrap(True);layout.addWidget(text)
        row=QHBoxLayout();layout.addLayout(row)
        row.addWidget(QLabel('Black square edge'));self.size=QDoubleSpinBox();self.size.setRange(1,5000);self.size.setDecimals(2);self.size.setValue(150);self.size.setSuffix(' mm');row.addWidget(self.size)
        row.addWidget(QLabel('Original video time'));self.seconds=QDoubleSpinBox();self.seconds.setRange(0,86400);self.seconds.setDecimals(3);self.seconds.setSuffix(' s');row.addWidget(self.seconds)
        self.photos=QPushButton('Detect Photos...');self.photos.clicked.connect(self.choose_photos);row.addWidget(self.photos)
        self.videos=QPushButton('Detect Camera Snapshots');self.videos.setEnabled(bool(self.entries));self.videos.clicked.connect(lambda:self.start(self.entries));row.addWidget(self.videos)
        self.views=QComboBox();self.views.currentIndexChanged.connect(self.show_view);layout.addWidget(self.views)
        self.preview=QLabel('Choose photos or camera snapshots.');self.preview.setAlignment(Qt.AlignCenter);self.preview.setMinimumHeight(300);layout.addWidget(self.preview,1)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        from .calibration_solver_ui import AprilTagSolveDialog
        self.calculate=QPushButton('Calculate AprilTag Camera Positions...');self.calculate.clicked.connect(lambda:AprilTagSolveDialog(self).exec());layout.addWidget(self.calculate)
        from .apriltag_board_ui import AprilTagBoardDialog
        self.board_video=QPushButton('AprilTag Board - Calibrate from Videos...')
        self.board_video.clicked.connect(lambda:AprilTagBoardDialog(self).exec());layout.addWidget(self.board_video)
        buttons=QHBoxLayout();layout.addLayout(buttons)
        self.stop=QPushButton('Stop');self.stop.setEnabled(False);self.stop.clicked.connect(self.cancel);buttons.addWidget(self.stop)
        self.save=QPushButton('Export Detections JSON...');self.save.setEnabled(False);self.save.clicked.connect(self.export);buttons.addWidget(self.save)
        pdf=QPushButton('Printable Markers PDF');pdf.clicked.connect(self.open_pdf);buttons.addWidget(pdf)
        close=QPushButton('Close');close.clicked.connect(self.close);buttons.addWidget(close)
        settings=getattr(parent,'apriltag_calibration_settings',{})
        self.size.setValue(settings.get('size_mm',150));self.seconds.setValue(settings.get('seconds',0))
    def open_pdf(self):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'docs/samples/apriltags_36h11_150mm_ID0-5.pdf')))
    def choose_photos(self):
        names,_=QFileDialog.getOpenFileNames(self,'Select Marker Photos','','Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if names:self.start([dict(id=str(i+1)+': '+Path(n).stem,path=n,image=True) for i,n in enumerate(names)])
    def start(self,entries):
        if self.worker is not None:return
        self.results=[];self.images=[];self.errors=[];self.cancelled=False;self.views.clear();self.save.setEnabled(False);self.photos.setEnabled(False);self.videos.setEnabled(False);self.stop.setEnabled(True);self.size.setEnabled(False);self.seconds.setEnabled(False)
        self.requested=[e['id'] for e in entries]
        self.calculate.setEnabled(False)
        self.board_video.setEnabled(False)
        self.status.setText('Detecting at original resolution...');self.preview.clear()
        self.worker=DetectWorker(entries,self.seconds.value(),self.size.value(),self)
        self.worker.result.connect(self.received);self.worker.failure.connect(self.failed);self.worker.finished.connect(self.on_worker_finished);self.worker.start()
    def received(self,value,image):
        self.results.append(value)
        # Cache at most 16 previews; all observations remain exportable.
        self.images.append(image if len(self.images)<16 else None)
        self.views.addItem(value['camera_id']+' - '+str(len(value['markers']))+' markers')
        self.views.setCurrentIndex(len(self.results)-1)
    def failed(self,message):
        self.errors.append(message);self.status.setText(self.status.text()+'\n'+message)
    def show_view(self,index):
        if not 0<=index<len(self.results):return
        image=self.images[index]
        if image is not None:self.preview.setPixmap(QPixmap.fromImage(image).scaled(850,480,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else:self.preview.setText('Observations recorded; preview cache limit reached.')
        value=self.results[index];ids=[m['id'] for m in value['markers']]
        self.status.setText('Detected IDs: '+str(ids)+('. Duplicate IDs: use each ID only once in a shared scene.' if any(m['duplicate_id'] for m in value['markers']) else '')+(' No tags found; check focus, tag family, full black border and white margin.' if not ids else ''))
    def cancel(self):
        if self.worker:self.cancelled=True;self.worker.requestInterruption();self.status.setText('Stopping after the current image...')
    def on_worker_finished(self):
        self.calculate.setEnabled(True)
        self.board_video.setEnabled(True)
        worker=self.worker;self.worker=None;worker.deleteLater();self.stop.setEnabled(False);self.photos.setEnabled(True);self.videos.setEnabled(bool(self.entries));self.save.setEnabled(bool(self.results));self.size.setEnabled(True);self.seconds.setEnabled(True)
        self.status.setText(self.status.text()+f'\nCompleted {len(self.results)}/{len(self.requested)} views; errors: {len(self.errors)}; stopped: {self.cancelled}.')
        if self.closing:self.close()
    def export(self):
        name,_=QFileDialog.getSaveFileName(self,'Export AprilTag Observations','apriltag_detections.json','JSON (*.json)')
        if not name:return
        path=Path(name)
        if not path.suffix:path=path.with_suffix('.json')
        try:atomic_json(path,dict(schema='rgbmocap-apriltag-session-v1',requested_views=self.requested,cancelled=self.cancelled,errors=self.errors,observations=self.results));self.status.setText('Saved '+str(path))
        except OSError as exc:self.status.setText('Cannot save: '+str(exc))
    def closeEvent(self,event):
        if self.parent() is not None:self.parent().apriltag_calibration_settings=dict(size_mm=self.size.value(),seconds=self.seconds.value())
        if self.worker is not None:self.closing=True;self.cancel();event.ignore()
        else:event.accept()
    def reject(self):self.close()


def show(parent):
    if getattr(parent,'busy',False):parent.status.setText('Finish processing before opening AprilTag detection.');return
    DetectionDialog(parent).exec()
