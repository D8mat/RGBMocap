"""Capture the real GL framebuffer rather than QWidget.grab's damaged GL copy."""
from PySide6.QtCore import QPoint, QRectF
from PySide6.QtGui import QPainter


def capture_window(window):
    result = window.grab().toImage()
    view = window.motion_view
    if hasattr(view,"grabFramebuffer") and view.isValid():
        frame = view.grabFramebuffer()
        painter = QPainter(result)
        origin = view.mapTo(window,QPoint(0,0))
        painter.drawImage(QRectF(origin.x(),origin.y(),view.width(),view.height()),frame)
        painter.end()
    return result
