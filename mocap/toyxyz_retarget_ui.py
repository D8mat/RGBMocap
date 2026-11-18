"""Optional retarget controls; execution uses the existing cancellable character job."""
import json
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QDoubleSpinBox,QPushButton,QLabel,QMessageBox,QDialog


class RetargetPanel(QWidget):
    def __init__(self,w):
        super().__init__();self.w=w;self.signature=None;self.rows=[];self.mapping_value=None
        box=QVBoxLayout(self);box.addWidget(QLabel('Toyxyz Retargeter 0.4.30 · Blender 4.5'))
        note=QLabel('Apply prepared mocap to an imported character. The result appears in the existing viewport and character export.');note.setWordWrap(True);box.addWidget(note)
        form=QFormLayout();box.addLayout(form)
        self.source=QComboBox();self.target=QComboBox();form.addRow('Source',self.source);form.addRow('Target',self.target)
        self.puppet=QCheckBox('Puppet V5: Hips → Reference');self.puppet.setToolTip('Explicit target mapping requested for Puppet V5. Off uses automatic anatomical mapping.');box.addWidget(self.puppet)
        self.strength=self.number(form,'Solver strength',0,100,100,' %')
        self.hands=self.number(form,'Hand reach',0,100,0,' %');self.feet=self.number(form,'Foot reach',0,100,100,' %')
        self.hip=self.number(form,'Hip offset',-1,1,0,' m');self.hip.setDecimals(3);self.hip.setSingleStep(.01)
        self.travel=QComboBox()
        for label,key in [('Scale to proportions','PROPORTIONAL'),('Match source distance','SOURCE'),('In place','IN_PLACE')]:self.travel.addItem(label,key)
        form.addRow('Root travel',self.travel)
        self.floor=QCheckBox('Floor contact');self.floor.setChecked(True);box.addWidget(self.floor)
        self.mapping_button=QPushButton('Bone Mapping…');self.mapping_button.clicked.connect(self.edit_mapping);box.addWidget(self.mapping_button)
        self.run=QPushButton('Apply Toyxyz Retarget');self.run.clicked.connect(self.apply);box.addWidget(self.run)
        cancel=QPushButton('Cancel Job');cancel.clicked.connect(w.cancel_job);box.addWidget(cancel)
        self.note=QLabel();self.note.setWordWrap(True);box.addWidget(self.note);box.addStretch()
        self.timer=QTimer(self);self.timer.setInterval(500);self.timer.timeout.connect(self.refresh);self.timer.start()
        self.target.currentIndexChanged.connect(self.restore)
        self.source.currentIndexChanged.connect(self.clear_mapping)

    def number(self,form,label,low,high,value,suffix):
        field=QDoubleSpinBox();field.setRange(low,high);field.setValue(value);field.setSuffix(suffix);form.addRow(label,field);return field

    def refresh(self):
        w=self.w
        if not hasattr(w,'characters') or not hasattr(w,'status'):return
        ready=not w.busy and not w.contact_saving
        if ready:
            self.rows=w.character_sources()
            signature=(tuple((s['name'],tuple(t['project'] for t in s['takes'])) for s in self.rows),tuple((c['id'],c['name']) for c in w.characters))
            if signature!=self.signature:
                self.signature=signature;source=self.source.currentText();target=self.target.currentData()
                self.source.blockSignals(True);self.source.clear()
                for i,s in enumerate(self.rows):self.source.addItem(s['name'],i)
                index=self.source.findText(source)
                if index>=0:self.source.setCurrentIndex(index)
                self.source.blockSignals(False)
                self.target.blockSignals(True);self.target.clear()
                for c in w.characters:self.target.addItem(c['name'],c['id'])
                index=self.target.findData(target)
                if index>=0:self.target.setCurrentIndex(index)
                self.target.blockSignals(False);self.restore()
        self.run.setEnabled(ready and self.source.count()>0 and self.target.count()>0 and w.model_choice.currentData()!='gemx')
        self.mapping_button.setEnabled(self.run.isEnabled())
        self.note.setText('Native77 currently uses the original retargeter.' if w.model_choice.currentData()=='gemx' else 'Import a character and prepare source motion first.' if not self.source.count() or not self.target.count() else '')

    def restore(self,*_):
        c=next((c for c in getattr(self.w,'characters',[]) if c['id']==self.target.currentData()),{})
        settings=c.get('toyxyz_options',{})
        self.mapping_value=settings.get('mapping');self.mapping_label()
        self.puppet.setChecked(settings.get('puppet_reference','puppet' in c.get('name','').lower() and 'v5' in c.get('name','').lower()))
        for field,key,default in [(self.strength,'solver_strength',100),(self.hands,'hand_position_reach',0),(self.feet,'foot_position_reach',100),(self.hip,'hip_offset',0)]:field.setValue(settings.get(key,default))
        self.floor.setChecked(settings.get('floor_contact',True));self.travel.setCurrentIndex(max(0,self.travel.findData(settings.get('travel','PROPORTIONAL'))))

    def apply(self):
        self.refresh();w=self.w
        if not self.run.isEnabled():return
        c=next(c for c in w.characters if c['id']==self.target.currentData())
        options=dict(puppet_reference=self.puppet.isChecked(),solver_strength=self.strength.value(),hand_position_reach=self.hands.value(),foot_position_reach=self.feet.value(),hip_offset=self.hip.value(),floor_contact=self.floor.isChecked(),travel=self.travel.currentData())
        if self.mapping_value is not None:
            from .toyxyz_mapping import validate
            try:options['mapping']=validate(self.mapping_value,self.mapping_names())
            except (OSError,ValueError,KeyError,IndexError) as e:QMessageBox.warning(self,'Bone Mapping',str(e));return
        w.retarget_character(c,self.rows[self.source.currentIndex()],toyxyz_options=options)

    def mapping_label(self):
        self.mapping_button.setText('Bone Mapping… (custom)' if self.mapping_value is not None else 'Bone Mapping… (automatic)')

    def clear_mapping(self,*_):
        self.mapping_value=None;self.mapping_label()

    def mapping_names(self):
        c=next(c for c in self.w.characters if c['id']==self.target.currentData())
        project=Path(self.rows[self.source.currentIndex()]['takes'][0]['project'])
        source=json.loads((project/(self.w.active_result+'_mixamo.json')).read_text(encoding='utf-8'))
        target=json.loads((self.w.character_path(c['asset'])/(c['kind']+'-rig.json')).read_text(encoding='utf-8'))
        return dict(source=source['names'],target=[b['name'] for b in target['bones']])

    def edit_mapping(self):
        from .toyxyz_mapping_ui import MappingDialog
        try:dialog=MappingDialog(self,self.mapping_names(),self.mapping_value,self.puppet.isChecked())
        except (OSError,ValueError,KeyError,IndexError) as e:QMessageBox.warning(self,'Bone Mapping','Prepare source motion first. '+str(e));return
        if dialog.exec()==QDialog.Accepted:self.mapping_value=dialog.value();self.mapping_label()
