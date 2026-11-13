"""Pick the face on a frozen copy of the actually displayed video image."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QDialogButtonBox

def pick_face(parent,image):
    dialog=QDialog(parent);dialog.setWindowTitle('Select face to track')
    layout=QVBoxLayout(dialog)
    layout.addWidget(QLabel('Click inside the face to track. Capture will start from this frame.'))
    pixmap=QPixmap.fromImage(image).scaled(1000,700,Qt.KeepAspectRatio,Qt.SmoothTransformation)
    class Picker(QLabel):
        def mousePressEvent(self,event):
            if event.button()==Qt.LeftButton:
                dialog.point=(event.position().x()/self.width(),event.position().y()/self.height());dialog.accept()
    label=Picker();label.setPixmap(pixmap);label.setFixedSize(pixmap.size());layout.addWidget(label)
    buttons=QDialogButtonBox(QDialogButtonBox.Cancel);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
    dialog.point=None
    return dialog.point if dialog.exec()==QDialog.Accepted else None
