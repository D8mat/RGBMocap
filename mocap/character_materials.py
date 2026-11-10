"""Per-character preview material overrides; imported assets and exports stay intact."""
import copy
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QLabel,QComboBox,QPushButton,
    QCheckBox,QDialogButtonBox,QColorDialog,QMessageBox)


def edit(window, character):
    source=str(window.character_path(character['asset']).resolve())
    textures=next((data['asset'].get('textures') for data in window.character_assets.values()
                   if data['asset'].get('textures',{}).get('key')==source),None)
    if not textures:
        QMessageBox.information(window,'Material Colors','Show the character and wait for its materials to load, then try again.')
        return
    original=copy.deepcopy(character.get('preview_materials',{}))
    changes=copy.deepcopy(original)
    dialog=QDialog(window);dialog.setWindowTitle('Material Colors — '+character['name']);dialog.setMinimumWidth(410)
    layout=QVBoxLayout(dialog)
    layout.addWidget(QLabel('Preview and playblast only. Original model and FBX export are unchanged.'))
    selector=QComboBox();layout.addWidget(selector)
    for i,batch in enumerate(textures['batches']):
        selector.addItem(batch.get('material_name') or f'Material {i+1}'+(' · texture' if batch['image'] else ''),str(i))
    choose=QPushButton('Choose Color...');layout.addWidget(choose)
    tint=QCheckBox('Tint existing texture');layout.addWidget(tint)
    reset=QPushButton('Restore selected material');layout.addWidget(reset)
    reset_all=QPushButton('Restore all original materials');layout.addWidget(reset_all)
    buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons)
    buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
    def redraw():
        character['preview_materials']=copy.deepcopy(changes)
        window.motion_view.update()
    def refresh():
        key=selector.currentData();value=changes.get(key)
        batch=textures['batches'][selector.currentIndex()]
        rgb=value['color'] if value else batch['color'][:3]
        color=QColor.fromRgbF(*[min(1.,max(0.,float(v))) for v in rgb])
        choose.setStyleSheet('background-color:'+color.name()+'; color:'+('black' if color.lightnessF()>.5 else 'white'))
        tint.blockSignals(True);tint.setChecked(bool(value and value.get('tint')));tint.blockSignals(False)
        tint.setEnabled(value is not None and batch['image'] is not None)
        reset.setEnabled(value is not None)
    def pick():
        key=selector.currentData();batch=textures['batches'][selector.currentIndex()]
        rgb=changes.get(key,{}).get('color',batch['color'][:3])
        color=QColorDialog.getColor(QColor.fromRgbF(*rgb),dialog,'Material Color')
        if color.isValid():
            changes[key]=dict(color=[color.redF(),color.greenF(),color.blueF()],tint=tint.isChecked())
            redraw();refresh()
    def set_tint(checked):
        key=selector.currentData()
        if key in changes:changes[key]['tint']=checked;redraw()
    def restore():
        changes.pop(selector.currentData(),None);redraw();refresh()
    def restore_all():
        changes.clear();redraw();refresh()
    selector.currentIndexChanged.connect(refresh);choose.clicked.connect(pick);tint.toggled.connect(set_tint)
    reset.clicked.connect(restore);reset_all.clicked.connect(restore_all);refresh()
    accepted=dialog.exec()==QDialog.Accepted
    if accepted:
        try:window.save_characters()
        except Exception as error:
            character['preview_materials']=original
            QMessageBox.warning(window,'Material Colors',str(error))
    else:character['preview_materials']=original
    window.motion_view.update()
