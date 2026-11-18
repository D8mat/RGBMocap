"""Manual role mapping with searchable source/target bone selectors."""
import json
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QHeaderView,QVBoxLayout,QHBoxLayout,QTabWidget,QTableWidget,QComboBox,QSpinBox,QLabel,QPushButton,QDialogButtonBox,QMessageBox,QFileDialog
from .toyxyz_mapping import MAIN,FINGERS,TOES,FORMAT,validate


class MappingEditor(QWidget):
    def __init__(self,parent,names,value=None,puppet=False):
        super().__init__(parent);self.names=names;self.fields={};self.counts={};self.chains={};self.automatic={};self.puppet=puppet
        self.setMinimumHeight(440)
        box=QVBoxLayout(self)
        note=QLabel('Auto keeps the retargeterâ€™s bone matching. Choose a bone to override it, or None to disable an optional role. Manual mapping takes precedence over the Puppet preset.');note.setWordWrap(True);box.addWidget(note)
        tabs=QTabWidget();tabs.setMinimumHeight(320);box.addWidget(tabs)
        for title,roles in [('Main Bones',MAIN),('Fingers',FINGERS),('Toes',TOES)]:
            table=QTableWidget(len(roles),3);table.setHorizontalHeaderLabels(['Role','Source','Target']);table.verticalHeader().hide();table.horizontalHeader().setStretchLastSection(True);table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            from PySide6.QtWidgets import QTableWidgetItem
            for row,role in enumerate(roles):
                item=QTableWidgetItem(role);item.setFlags(item.flags() & ~Qt.ItemIsEditable);table.setItem(row,0,item)
                for column,side in enumerate(('source','target'),1):
                    combo=self.combo(side);self.fields[side,role]=combo;table.setCellWidget(row,column,combo)
            tabs.addTab(table,title)
        for chain,minimum in [('spine',1),('neck',0)]:
            table=QTableWidget(13,3);table.setHorizontalHeaderLabels([chain.title()+' chain','Source','Target']);table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);table.horizontalHeader().setStretchLastSection(True)
            from PySide6.QtWidgets import QTableWidgetItem
            table.setItem(0,0,QTableWidgetItem('Count (-1 = Auto)'))
            for col,side in enumerate(('source','target'),1):
                spin=QSpinBox();spin.setRange(-1,12);spin.setValue(-1);spin.setSpecialValueText('Auto');table.setCellWidget(0,col,spin);self.counts[side,chain]=spin
                for i in range(12):
                    combo=self.combo(side,auto=False);table.setCellWidget(i+1,col,combo);self.chains[side,chain,i]=combo
                    table.setItem(i+1,0,QTableWidgetItem(str(i+1)))
                spin.valueChanged.connect(lambda _,s=side,c=chain:self.chain_changed(s,c))
                self.chain_changed(side,chain)
            tabs.addTab(table,chain.title())
        row=QHBoxLayout();box.addLayout(row)
        for label,callback in [('Reset to Auto',self.reset),('Import JSON',self.import_json),('Export JSON',self.export_json)]:
            button=QPushButton(label);button.clicked.connect(callback);row.addWidget(button)
        if value:self.load(value)
        self.show_automatic()

    def combo(self,side,auto=True):
        c=QComboBox();c.setEditable(True);c.setInsertPolicy(QComboBox.NoInsert)
        if auto:c.addItem('Auto',None)
        c.addItem('None','')
        for name in self.names[side]:c.addItem(name,name)
        c.completer().setFilterMode(Qt.MatchContains);return c

    def chain_changed(self,side,chain):
        count=self.counts[side,chain].value()
        for i in range(12):
            c=self.chains[side,chain,i];c.setEnabled(i<count)
            if count==-1:
                names=self.automatic.get(side,{}).get(chain,[])
                c.setCurrentIndex(0);c.setItemText(0,'Auto — '+(names[i] if i<len(names) else '—'))
            else:c.setItemText(0,'None')

    def reset(self):
        for c in self.fields.values():c.setCurrentIndex(0)
        for c in self.counts.values():c.setValue(-1)

    def selection(self,c):
        i=c.findText(c.currentText(),Qt.MatchExactly)
        if i<0:raise ValueError('Choose an existing bone: '+c.currentText())
        return c.itemData(i)

    def value(self):
        data=dict(format=FORMAT,version=1)
        for side in ('source','target'):
            data[side]=dict(bones={role:self.selection(c) for (s,role),c in self.fields.items() if s==side and self.selection(c) is not None})
            for chain in ('spine','neck'):
                count=self.counts[side,chain].value()
                data[side][chain]=None if count==-1 else [self.selection(self.chains[side,chain,i]) for i in range(count)]
                if count>0 and any(not n for n in data[side][chain]):raise ValueError('Assign every '+side+' '+chain+' bone.')
        return validate(data,self.names)

    def load(self,data):
        data=validate(data,self.names);self.reset()
        for side in ('source','target'):
            for role,name in data[side]['bones'].items():
                c=self.fields[side,role];c.setCurrentIndex(c.findData(name))
            for chain in ('spine','neck'):
                names=data[side][chain]
                if names is not None:
                    self.counts[side,chain].setValue(len(names))
                    for i,name in enumerate(names):
                        c=self.chains[side,chain,i];c.setCurrentIndex(c.findData(name))

    def import_json(self):
        name,_=QFileDialog.getOpenFileName(self,'Import Bone Mapping','','JSON (*.json)')
        if not name:return
        try:
            if Path(name).stat().st_size>1024*1024:raise ValueError('Mapping JSON is too large.')
            self.load(json.loads(Path(name).read_text(encoding='utf-8-sig')))
        except (OSError,ValueError,TypeError) as e:QMessageBox.warning(self,'Bone Mapping',str(e))

    def export_json(self):
        try:data=self.value()
        except ValueError as e:QMessageBox.warning(self,'Bone Mapping',str(e));return
        name,_=QFileDialog.getSaveFileName(self,'Export Bone Mapping','bone-mapping.json','JSON (*.json)')
        if name:
            try:Path(name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError as e:QMessageBox.warning(self,'Bone Mapping',str(e))

    def show_automatic(self,data=None,puppet=None):
        if data is not None:self.automatic=data
        if puppet is not None:self.puppet=puppet
        for (side,role),combo in self.fields.items():
            name=self.automatic.get(side,{}).get('bones',{}).get(role)
            if side=='target' and role=='Hips' and self.puppet and 'Reference' in self.names['target']:name='Reference'
            index=combo.currentIndex();text=combo.currentText()
            combo.setItemText(0,'Auto — '+(name or ('unmapped' if self.automatic else 'loading…')))
            if index!=0:combo.setEditText(text)
        for (side,chain),spin in self.counts.items():
            names=self.automatic.get(side,{}).get(chain)
            spin.setSpecialValueText('Auto'+(' ('+str(len(names))+')' if names is not None else ''))
            self.chain_changed(side,chain)
