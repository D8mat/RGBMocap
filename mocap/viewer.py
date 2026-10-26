from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget
from .video_preview import PreviewDecoder
from .viewport import MotionView

COCO_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


class VideoView(QWidget):
    frame_presented = Signal(int)
    frame_painted = Signal(int)
    decode_error = Signal(str)
    metadata_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(440, 280)
        cv2.setNumThreads(1)  # Preview uses one decoder, not a pool for each resize.
        self.frame = None
        self.boxes = self.observed = self.keypoints = self.confidence = None
        self.index = 0
        self.requested_index = 0
        self.decoder = None
        self.source_size = (1, 1)
        self.generation = 0
        self.logical_size = None
        self.show_observations = True
        self.actor_color = None
        self.zoom = 1.0
        self.pan = QPointF()
        self._pan_position = None

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF()
        self._end_pan()
        self.update()

    def image_rect(self):
        """One source-coordinate mapping for the image and every observation."""
        w, h = self.source_size
        scale = min(self.width() / w, self.height() / h) * self.zoom
        return QRectF((self.width() - w * scale) / 2 + self.pan.x(),
                      (self.height() - h * scale) / 2 + self.pan.y(),
                      w * scale, h * scale)

    def wheelEvent(self, event):
        if self.frame is None:
            event.ignore()
            return
        delta = event.angleDelta().y() / 120
        if not delta:
            delta = event.pixelDelta().y() / 120
        if not delta:
            event.ignore()
            return
        old_zoom = self.zoom
        new_zoom = max(0.25, min(16.0, old_zoom * math.pow(1.2, max(-8, min(8, delta)))))
        # Keep the source point beneath the mouse fixed as the image grows.
        anchor = event.position() - QPointF(self.width() / 2, self.height() / 2)
        self.pan = anchor - (anchor - self.pan) * (new_zoom / old_zoom)
        self.zoom = new_zoom
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton and self.frame is not None:
            self._pan_position = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_position is not None:
            if event.buttons() & Qt.MiddleButton:
                self.pan += event.position() - self._pan_position
                self._pan_position = event.position()
                self.update()
                event.accept()
                return
            self._end_pan()
        super().mouseMoveEvent(event)

    def _end_pan(self):
        self._pan_position = None
        self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._pan_position is not None:
            self._end_pan()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.frame is not None:
            self.reset_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def hideEvent(self, event):
        self._end_pan()
        super().hideEvent(event)

    def open(self, path: Path, source_size=None):
        self.reset_view()
        self.logical_size = source_size
        self.frame = None
        self.boxes = self.observed = self.keypoints = self.confidence = None
        if self.decoder is None:
            self.decoder = PreviewDecoder(parent=self)
            self.decoder.available.connect(self._available)
            self.decoder.opened.connect(self._opened)
            self.decoder.failed.connect(self._failed)
            self.decoder.start()
        self.generation = self.decoder.change_source(path)
        self.seek(0)

    def _opened(self, generation, metadata):
        if self.decoder is not None and self.sender() is self.decoder and generation == self.generation:
            self.source_size = self.logical_size or (metadata["width"], metadata["height"])
            self.metadata_ready.emit(metadata)

    def _failed(self, generation, message):
        if self.decoder is not None and self.sender() is self.decoder and generation == self.generation:
            self.decode_error.emit(message)

    def shutdown(self):
        self._end_pan()
        if self.decoder:
            decoder = self.decoder
            self.decoder = None
            self.generation += 1
            decoder.stop()
            decoder.wait()
            decoder.deleteLater()
        self.frame = None

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def _available(self, generation, index):
        if self.decoder is not None and self.sender() is self.decoder and generation == self.generation and index == self.requested_index:
            image = self.decoder.image(index)
            if image is not None:
                self._present(index, image)

    def _present(self, index, image):
        self.index, self.frame = index, image
        self.update()
        self.frame_presented.emit(index)

    def set_observations(self, arrays):
        self.boxes, self.observed = arrays.get("boxes"), arrays.get("observed")
        self.keypoints = arrays.get("keypoints")
        confidence = arrays.get("confidence")
        self.confidence = confidence
        self.update()

    def seek(self, index: int):
        if self.decoder is None:
            return
        self.requested_index = index
        image = self.decoder.request(index)
        if image is not None:
            self._present(index, image)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#202020"))
        if self.frame is None:
            painter.setPen(QColor("#afbdd2"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a video to begin\nMP4 · MOV · AVI")
            return
        rect = self.image_rect()
        ratio = rect.width() / self.source_size[0]
        offset_x, offset_y = rect.x(), rect.y()
        painter.drawImage(rect, self.frame)
        if self.show_observations and self.boxes is not None and self.index < len(self.boxes):
            box = self.boxes[self.index]
            seen = bool(self.observed[self.index])
            pen = QPen(QColor((self.actor_color or "#40e7bf") if seen else "#ffbd62"), 2)
            if not seen:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            if np.isfinite(box).all():
                painter.drawRect(QRectF(offset_x + box[0] * ratio, offset_y + box[1] * ratio,
                                        (box[2] - box[0]) * ratio, (box[3] - box[1]) * ratio))
        if self.show_observations and self.keypoints is not None and self.index < len(self.keypoints):
            points = self.keypoints[self.index]
            confidence = self.confidence[self.index].reshape(-1)
            painter.setPen(QPen(QColor("#7cecff"), 3))
            for a, b in COCO_BONES:
                if min(confidence[a], confidence[b]) >= 0.25:
                    painter.drawLine(QPointF(offset_x + points[a, 0] * ratio, offset_y + points[a, 1] * ratio),
                                     QPointF(offset_x + points[b, 0] * ratio, offset_y + points[b, 1] * ratio))
            for point, score in zip(points, confidence):
                if score >= 0.25:
                    painter.setBrush(QColor("#fff6cf"))
                    painter.drawEllipse(QPointF(offset_x + point[0] * ratio, offset_y + point[1] * ratio), 3, 3)
        painter.end()
        self.frame_painted.emit(self.index)


