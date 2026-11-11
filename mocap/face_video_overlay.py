"""Read captured face geometry off-thread; paint only the displayed video frame."""
from pathlib import Path
from threading import Condition,Thread
import json
import zipfile
import numpy as np
from PySide6.QtCore import Qt,QRectF,QLineF,QTimer
from PySide6.QtGui import QColor,QPen
from PySide6.QtWidgets import QCheckBox
from .face_extract import source_identity


from .face_regions import RegionReader


class FaceVideoOverlay:
    def __init__(self,window,header):
        self.window=window;self.view=window.video_view;self.reader=RegionReader();self.token=None;self.data=None
        self.mesh=QCheckBox('Face Mesh');self.crop=QCheckBox('Face Crop')
        for control in (self.mesh,self.crop):
            control.setChecked(True);control.toggled.connect(self.view.update);header.addWidget(control)
        self.mesh.setToolTip('Show the captured face mesh on the matching video frame. Capture Face first; playback does not run AI.')
        self.crop.setToolTip('Show a magnified crop of the selected detected face. No detection means no frozen crop.')
        self.topology=json.loads(Path(__file__).with_name('face_topology.json').read_text())
        self.groups=[('FACE_LANDMARKS_TESSELATION','#88959da0',.65),('FACE_LANDMARKS_FACE_OVAL','#eeeeee',1.4),('FACE_LANDMARKS_LIPS','#eeeeee',1.2),
                     ('FACE_LANDMARKS_RIGHT_EYE','#ff4545',1.8),('FACE_LANDMARKS_RIGHT_EYEBROW','#ff4545',1.8),('FACE_LANDMARKS_RIGHT_IRIS','#ff4545',1.8),
                     ('FACE_LANDMARKS_LEFT_EYE','#59f74c',1.8),('FACE_LANDMARKS_LEFT_EYEBROW','#59f74c',1.8),('FACE_LANDMARKS_LEFT_IRIS','#59f74c',1.8)]
        self.view.face_overlay=self.paint;self.timer=QTimer(window);self.timer.timeout.connect(self.refresh);self.timer.start(100)
        self._paint_key=None;self._lines=[]
    def refresh(self):
        w=self.window
        project=w.project if not w._pear_active and not w.busy and w.model_choice.currentData()!='mamma' else None
        if not (self.mesh.isChecked() or self.crop.isChecked()):project=None
        token=(str(project),self.view.generation)
        if token!=self.token:self.token=token;self.data=None;self._paint_key=None;self.view.update()
        self.reader.submit((token,project) if project is not None else None)
        output=self.reader.output
        if output is not None and output[0]==token and output is not self.data:
            self.data=output;self._paint_key=None;self.view.update()
    def paint(self,painter,rect):
        if self.data is None or self.view.external_mode or self.view.preview_rotation%360:return
        token,regions,points,valid,error=self.data
        if token!=self.token or str(self.window.project)!=token[0] or self.view.generation!=token[1]:return
        if not self.mesh.isChecked() and not self.crop.isChecked():return
        painter.save()
        try:
            frame=self.view.index
            if error or valid is None or not 0<=frame<len(valid) or not valid[frame]:
                painter.setPen(QColor('#ffd28a'))
                label='Face: '+(error if error else 'not detected in this frame')
                # Short, unobtrusive status; no old geometry remains on missing frames.
                painter.drawText(QRectF(10,10,self.view.width()-20,42),Qt.AlignLeft|Qt.TextWordWrap,label[:140]);return
            if self.mesh.isChecked():
                key=(id(points),frame,rect.x(),rect.y(),rect.width(),rect.height())
                if key!=self._paint_key:
                    xy=np.asarray(points[frame],float)*[rect.width(),rect.height()]+[rect.x(),rect.y()]
                    self._lines=[([QLineF(*xy[a],*xy[b]) for a,b in self.topology[name]],color,width) for name,color,width in self.groups]
                    self._paint_key=key
                painter.setClipRect(rect)
                for lines,color,width in self._lines:
                    painter.setPen(QPen(QColor(color),width));painter.drawLines(lines)
                painter.setClipping(False)
            if self.crop.isChecked():
                x0,y0,x1,y1=np.clip(regions[frame],0,1)
                source=QRectF(x0*self.view.frame.width(),y0*self.view.frame.height(),(x1-x0)*self.view.frame.width(),(y1-y0)*self.view.frame.height())
                if source.width()<=0 or source.height()<=0:return
                width=min(145.,self.view.width()*.25);height=min(190.,width*source.height()/source.width())
                width=height*source.width()/source.height()
                inset=QRectF(self.view.width()-width-12,32,width,height)
                painter.fillRect(inset.adjusted(-3,-22,3,3),QColor('#e820252d'))
                painter.drawImage(inset,self.view.frame,source)
                painter.setPen(QPen(QColor('#65ecc5'),1.3));painter.drawRect(inset)
                painter.drawText(inset.adjusted(0,-21,0,-height),Qt.AlignLeft|Qt.AlignVCenter,'Face')
                if not self.mesh.isChecked():
                    painter.drawRect(QRectF(rect.x()+x0*rect.width(),rect.y()+y0*rect.height(),(x1-x0)*rect.width(),(y1-y0)*rect.height()))
        finally:painter.restore()
    def close(self):
        self.timer.stop();self.reader.close();self.view.face_overlay=None;self.data=None
