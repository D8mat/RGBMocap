"""Options for exporting the displayed source skeleton."""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox,
                              QCheckBox, QLabel, QDialogButtonBox)


class SourceExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export Motion')
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.format = QComboBox()
        self.format.addItem('FBX (.fbx)', 'fbx')
        self.format.addItem('Blender project + video (.blend)', 'blend')
        self.format.addItem('BVH — 3ds Max Biped (.bvh)', 'bvh_biped')
        self.format.addItem('BVH — Generic / Blender (.bvh)', 'bvh')
        form.addRow('Format', self.format)
        self.tpose = QCheckBox('Insert T-pose at frame 0')
        self.tpose.setStyleSheet("""
            QCheckBox { background:#35465a; color:#ffffff; border:1px solid #7399bf;
                        border-radius:6px; padding:12px; spacing:12px; font-weight:600; }
            QCheckBox::indicator { width:22px; height:22px; border:2px solid #d6e6f5;
                                   border-radius:3px; background:#18222e; }
            QCheckBox::indicator:checked { background:#38bdf8; border:2px solid #ffffff; }
            QCheckBox:focus { border:2px solid #38bdf8; }
        """)
        self.tpose.toggled.connect(lambda checked: self.tpose.setText(
            'Insert T-pose at frame 0 — ON' if checked else 'Insert T-pose at frame 0 — OFF'))
        layout.addWidget(self.tpose)
        note = QLabel('Adds one reference frame before the complete motion. '
                      'Original frame 0 becomes frame 1. Importers may number '
                      'the first frame as 1. BVH units: metres; up axis: Y.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Save…')
        destination_note = QLabel('Next: choose the output filename and folder. '
                                  'For multiple takes, choose an output folder.')
        destination_note.setWordWrap(True)
        layout.addWidget(destination_note)
        self.blend_note=QLabel('Blender: creates a new subfolder with the project and video.mp4. Video is a camera-view background, not a render background. Uses the stored reference camera, without new camera tracking.')
        self.blend_note.setWordWrap(True)
        self.blend_note.setVisible(False)
        layout.addWidget(self.blend_note)
        self.format.currentIndexChanged.connect(lambda: self.blend_note.setVisible(self.format.currentData()=='blend'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setMinimumWidth(420)

    def options(self):
        return dict(format='bvh' if self.format.currentData()=='bvh_biped' else self.format.currentData(),
                    bvh_profile='biped' if self.format.currentData()=='bvh_biped' else 'generic',
                    prepend_tpose=self.tpose.isChecked())
