"""Paint original Track Face observations over their matching decoded video."""
from pathlib import Path
import json
import numpy as np
from PySide6.QtCore import Qt, QRectF, QLineF, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QCheckBox


def observation(archive, person, seconds):
    """Never hold an old landmark overlay across a missing detection."""
    slot=archive.index.get(person)
    frame=round((seconds-archive.info['start'])*archive.info['fps'])
    if slot is None or not 0<=frame<archive.valid.shape[1] or not archive.valid[slot,frame]:
        return None
    if archive.landmarks is None:return None
    return np.asarray(archive.landmarks[slot,frame],dtype=np.float32)


class FaceVideoOverlay:
    def __init__(self, window, header):
        self.window=window;self.view=window.video_view;self.archive=None
        self.mesh=QCheckBox('Tracked Face Overlay');self.crop=QCheckBox('Face Crop')
        for control in (self.mesh,self.crop):
            control.setChecked(True);control.toggled.connect(self.refresh);header.addWidget(control)
        self.mesh.setToolTip('Original Track Face landmarks for the selected person. Older tracks need Track Face again once.')
        self.crop.setToolTip('Magnified face from the currently displayed video frame.')
        self.topology=json.loads(Path(__file__).with_name('face_topology.json').read_text())
        self.groups=[('FACE_LANDMARKS_TESSELATION','#88959da0',.65),('FACE_LANDMARKS_FACE_OVAL','#eeeeee',1.4),
                     ('FACE_LANDMARKS_LIPS','#eeeeee',1.2),('FACE_LANDMARKS_RIGHT_EYE','#ff4545',1.8),
                     ('FACE_LANDMARKS_LEFT_EYE','#59f74c',1.8)]
        self.view.face_overlay=self.paint
        self.timer=QTimer(window);self.timer.timeout.connect(self.refresh);self.timer.start(100)
        self.message='';self.allowed=set();self.key=None;self.closed=False

    def refresh(self):
        if self.closed:return
        w=self.window;self.archive=None;self.message=''
        if w._pear_active or w.busy or not (self.mesh.isChecked() or self.crop.isChecked()):
            self.view.update();return
        try:
            root=w.face_project()
            if root is None:return
            archive=w.face_playback_archive(root)
            if archive is None:self.message='Track Face: enable Track Face and wait for tracking.'
            elif archive.landmarks is None:self.message='Track Face: run Track Face again once to prepare the video overlay.'
            else:
                self.archive=archive
                key=(id(archive),str(w.project))
                if key!=self.key:
                    from .video import select_preview
                    from .project_storage import source_video
                    from .face_data import manifest
                    allowed={archive.media.resolve(),select_preview(archive.root).resolve(),source_video(archive.root,archive.info).resolve()}
                    if w.project is not None:
                        project=Path(w.project);state=manifest(project)
                        # Take clips retain the source image coordinates (resize only).
                        allowed.update(((project/'input.mp4').resolve(),select_preview(project).resolve(),source_video(project,state).resolve()))
                    self.allowed=allowed;self.key=key
        except (OSError,ValueError,KeyError,TypeError) as error:self.message='Track Face: '+str(error)
        self.view.update()

    def paint(self,painter,rect):
        view=self.view;w=self.window
        if self.closed or view.external_mode or view.preview_rotation%360 or not (self.mesh.isChecked() or self.crop.isChecked()):return
        if w._pear_active or w.busy:return
        key=view._source_key
        if key is None or view.frame is None:return
        points=None;message=self.message
        archive=self.archive
        if archive is not None:
            if Path(key[0]).resolve() not in self.allowed:
                message='Track Face overlay is available only on the tracked source camera.'
            else:
                person=w.current_person() if w.people_state else None
                ident=person['id'] if person else (1 if not w.people_state else None)
                seconds=w.source_offset+view.index/float((w.info or {}).get('fps',archive.info['fps']))
                points=observation(archive,ident,seconds)
                if points is None:message='Track Face: no detection for this person in this frame.'
        painter.save()
        try:
            if points is None:
                if message:
                    painter.setPen(QColor('#ffd28a'));painter.drawText(QRectF(10,10,view.width()-20,44),Qt.AlignLeft|Qt.TextWordWrap,message)
                return
            if self.mesh.isChecked():
                xy=points*[rect.width(),rect.height()]+[rect.x(),rect.y()]
                painter.setClipRect(rect)
                for name,color,width in self.groups:
                    painter.setPen(QPen(QColor(color),width));painter.drawLines([QLineF(*xy[a],*xy[b]) for a,b in self.topology[name]])
                painter.setClipping(False)
            if self.crop.isChecked():
                lo=points.min(axis=0);hi=points.max(axis=0);pad=(hi-lo)*.15
                x0,y0=np.clip(lo-pad,0,1);x1,y1=np.clip(hi+pad,0,1)
                source=QRectF(x0*view.frame.width(),y0*view.frame.height(),(x1-x0)*view.frame.width(),(y1-y0)*view.frame.height())
                if source.width()<=0 or source.height()<=0:return
                width=min(145.,view.width()*.25);height=min(190.,width*source.height()/source.width());width=height*source.width()/source.height()
                inset=QRectF(view.width()-width-12,32,width,height)
                painter.fillRect(inset.adjusted(-3,-22,3,3),QColor('#e820252d'));painter.drawImage(inset,view.frame,source)
                painter.setPen(QPen(QColor('#65ecc5'),1.3));painter.drawRect(inset)
                painter.drawText(inset.adjusted(0,-21,0,-height),Qt.AlignLeft|Qt.AlignVCenter,'Track Face')
        finally:painter.restore()

    def close(self):
        self.closed=True;self.timer.stop();self.view.face_overlay=None;self.archive=None
