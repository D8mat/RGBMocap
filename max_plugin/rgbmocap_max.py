"""RGB MoCap for 3ds Max 2025: run this file using Scripting > Run Script."""
from pathlib import Path
import sys,json,queue,struct,time
folder=str(Path(__file__).resolve().parent)
if folder not in sys.path:sys.path.insert(0,folder)
from PySide6 import QtWidgets,QtCore
import qtmax
from pymxs import runtime as rt
from rgbmocap_max_protocol import Client,validate,snapshot,matrix
from rgbmocap_max_scene import Session,descendants,suffix

class Panel(QtWidgets.QDialog):
    def __init__(self):
        super().__init__(qtmax.GetQMaxMainWindow())
        self.setWindowTitle('RGB MoCap — FBX Live Link');self.resize(600,720)
        self.client=None;self.schemas={};self.slots={};self.session=Session();self.bake=None
        self.targets=[];self.last_record={};self.record_clock=0
        layout=QtWidgets.QVBoxLayout(self)
        line=QtWidgets.QHBoxLayout();self.port=QtWidgets.QSpinBox();self.port.setRange(1024,65535);self.port.setValue(49169)
        line.addWidget(QtWidgets.QLabel('Local port'));line.addWidget(self.port)
        self.connect_button=QtWidgets.QPushButton('Connect');self.connect_button.clicked.connect(self.connect_stream);line.addWidget(self.connect_button)
        stop=QtWidgets.QPushButton('Disconnect');stop.clicked.connect(self.disconnect);line.addWidget(stop);layout.addLayout(line)
        note=QtWidgets.QLabel('In RGB MoCap: Stream > 3ds Max, check people. Import an FBX in its reference pose, select its skeleton root, then map it below. Biped/CAT and constrained rigs are not supported.')
        note.setWordWrap(True);layout.addWidget(note)
        self.actors=QtWidgets.QComboBox();self.actors.currentIndexChanged.connect(self.actor_changed);layout.addWidget(self.actors)
        row=QtWidgets.QHBoxLayout()
        select=QtWidgets.QPushButton('Use Selected Skeleton / Auto Map');select.clicked.connect(self.load_targets);row.addWidget(select)
        bind=QtWidgets.QPushButton('Bind Mapping');bind.clicked.connect(self.bind);row.addWidget(bind);layout.addLayout(row)
        self.table=QtWidgets.QTableWidget(0,2);self.table.setHorizontalHeaderLabels(['Source bone','Target bone']);self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch);layout.addWidget(self.table)
        line=QtWidgets.QHBoxLayout();self.scale=QtWidgets.QDoubleSpinBox();self.scale.setRange(.01,100);self.scale.setDecimals(3);self.scale.setValue(1)
        line.addWidget(QtWidgets.QLabel('Root travel scale'));line.addWidget(self.scale)
        self.live=QtWidgets.QCheckBox('Live Preview');self.live.toggled.connect(self.live_changed);line.addWidget(self.live);layout.addLayout(line)
        line=QtWidgets.QHBoxLayout();self.start_frame=QtWidgets.QSpinBox();self.start_frame.setRange(-100000,100000)
        line.addWidget(QtWidgets.QLabel('Target start frame'));line.addWidget(self.start_frame);layout.addLayout(line)
        line=QtWidgets.QHBoxLayout();self.first=QtWidgets.QSpinBox();self.last=QtWidgets.QSpinBox()
        self.first.setRange(0,35999);self.last.setRange(0,35999);self.last.setValue(35999)
        line.addWidget(QtWidgets.QLabel('Bake source range'));line.addWidget(self.first);line.addWidget(self.last);layout.addLayout(line)
        line=QtWidgets.QHBoxLayout()
        record=QtWidgets.QPushButton('Record');record.clicked.connect(self.record);line.addWidget(record)
        keep=QtWidgets.QPushButton('Stop / Keep Record');keep.clicked.connect(self.keep_record);line.addWidget(keep)
        bake=QtWidgets.QPushButton('Bake Selected Person');bake.clicked.connect(self.begin_bake);line.addWidget(bake)
        cancel=QtWidgets.QPushButton('Cancel / Restore');cancel.clicked.connect(self.cancel);line.addWidget(cancel);layout.addLayout(line)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress)
        self.status=QtWidgets.QLabel('Disconnected');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(16)
    def message(self,text):self.status.setText(str(text))
    def error(self,exc):self.message('Error: '+str(exc))
    def actor(self):return self.actors.currentData()
    def connect_stream(self):
        self.disconnect();self.client=Client(self.port.value());self.schemas={};self.slots={};self.actors.clear();self.message('Connecting…')
    def disconnect(self):
        if self.session.mode!='idle':self.session.finish(False)
        self.bake=None;self.live.setChecked(False);self.session.stop_preview()
        if self.client:self.client.close();self.client=None
        self.message('Disconnected; unfinished recording/bake restored')
    def live_changed(self,on):
        if not on and self.session.mode=='idle':self.session.stop_preview()
    def actor_changed(self,*args):self.fill_table()
    def load_targets(self):
        try:
            if self.session.mode!='idle':raise ValueError('Stop Record/Bake first')
            selected=list(rt.selection)
            if len(selected)!=1:raise ValueError('Select exactly one skeleton root in Max')
            self.targets=descendants(selected[0]);self.fill_table();self.message('Review mapping, then Bind Mapping in the reference pose')
        except Exception as exc:self.error(exc)
    def fill_table(self):
        schema=self.schemas.get(self.actor());self.table.setRowCount(0)
        if not schema:return
        self.table.setRowCount(len(schema['bones']))
        for i,b in enumerate(schema['bones']):
            label=QtWidgets.QTableWidgetItem(b[0]);label.setFlags(label.flags() & ~QtCore.Qt.ItemIsEditable);self.table.setItem(i,0,label)
            combo=QtWidgets.QComboBox();combo.addItem('(unmapped)',None)
            matches=[]
            for j,n in enumerate(self.targets):
                combo.addItem(str(n.name),j)
                if suffix(n.name)==suffix(b[0]):matches.append(j+1)
            if len(matches)==1:combo.setCurrentIndex(matches[0])
            self.table.setCellWidget(i,1,combo)
    def bind(self):
        try:
            actor=self.actor();schema=self.schemas[actor]
            nodes={i:self.targets[self.table.cellWidget(i,1).currentData()] for i in range(self.table.rowCount()) if self.table.cellWidget(i,1).currentData() is not None}
            self.session.bind(actor,schema,nodes,self.scale.value());self.message(f'Bound {len(nodes)} bones to {schema["name"]}')
        except Exception as exc:self.error(exc)
    def record(self):
        try:
            if not self.client:raise ValueError('Connect first')
            self.session.start('record',list(self.session.bindings));self.record_clock=time.monotonic();self.record_start=self.start_frame.value();self.last_record={}
            self.message('Recording incoming live samples. Stop keeps keys; Cancel restores previous animation.')
        except Exception as exc:self.error(exc)
    def keep_record(self):
        if self.session.mode=='record':
            count=self.session.recorded;self.session.finish(True);self.live.setChecked(False)
            if self.last_record:self.extend_range(self.record_start,max(self.last_record.values()))
            self.message(f'Recorded {count} samples. Keys kept.')
    def extend_range(self,first,last):
        rt.animationRange=rt.interval(min(int(rt.animationRange.start.frame),int(first)),max(int(rt.animationRange.end.frame),int(last)+1))
    def begin_bake(self):
        try:
            actor=self.actor();schema=self.schemas[actor]
            if not self.client:raise ValueError('Connect first')
            if self.first.value()>self.last.value():raise ValueError('Invalid source range')
            self.session.start('bake',[actor]);self.bake={'actor':actor,'rig':validate(schema),'offset':self.start_frame.value(),'first':self.first.value(),'last':self.last.value(),'received':0,'seen':set()}
            self.client.send(b'B'+bytes.fromhex(actor)+bytes.fromhex(self.bake['rig']));self.message('Requesting complete animation from RGB MoCap…')
        except Exception as exc:
            if self.session.mode=='bake':self.session.finish(False);self.bake=None
            self.error(exc)
    def cancel(self):self.disconnect()
    def packet(self,tag,data):
        if tag=='connected':self.message('Connected. Check people in RGB MoCap.');return
        if tag=='error':raise ConnectionError(data)
        if tag=='R':
            schema=json.loads(data);schema['key']=validate(schema);actor=schema['actor']
            existing=self.session.bindings.get(actor)
            if existing and validate(existing.schema)!=schema['key']:
                if self.session.mode!='idle':raise ValueError('Rig changed during capture')
                existing.restore();del self.session.bindings[actor]
            self.schemas[actor]=schema;self.slots[schema['slot']]=schema
            if self.actors.findData(actor)<0:self.actors.addItem(schema['name'],actor)
        elif tag=='S':
            poses=snapshot(data,self.slots)
            if self.session.mode=='bake':return
            for actor,source_frame,pose in poses:
                binding=self.session.bindings.get(actor)
                if binding is None:continue
                if self.session.mode=='record':
                    key=self.record_start+round((time.monotonic()-self.record_clock)*float(rt.frameRate))
                    if self.last_record.get(actor)==key:continue
                    binding.apply(pose,key);self.last_record[actor]=key;self.session.recorded+=1
                elif self.live.isChecked():binding.apply(pose)
        elif tag=='H':
            if self.bake is None:return
            header=json.loads(data)
            if header['actor']!=self.bake['actor'] or header['rig']!=self.bake['rig']:raise ValueError('Bake skeleton mismatch')
            if not 0<header['frames']<=36000 or not .1<=header['fps']<=1000:raise ValueError('Invalid Bake duration/rate')
            self.bake.update(fps=float(header['fps']),count=int(header['frames']))
            self.progress.setRange(0,header['frames']);self.progress.setValue(0)
        elif tag=='D':
            if self.bake is None:return
            job=self.bake
            if 'fps' not in job:raise ValueError('Bake data arrived before header')
            b=self.session.bindings[job['actor']];n=len(b.source);size=4+48*n
            if len(data)%size:raise ValueError('Truncated Bake chunk')
            for offset in range(0,len(data),size):
                frame,=struct.unpack_from('<I',data,offset)
                if frame>=job['count'] or frame in job['seen']:raise ValueError('Duplicate/out-of-range Bake frame')
                job['seen'].add(frame)
                if job['first']<=frame<=job['last']:
                    vals=struct.unpack_from('<'+'f'*(n*12),data,offset+4)
                    pose=[matrix(vals[i:i+12]) for i in range(0,len(vals),12)]
                    key=job['offset']+(frame-job['first'])*float(rt.frameRate)/job['fps']
                    b.apply(pose,key);job['received']+=1;job['last_key']=key
                self.progress.setValue(frame+1)
            self.message(f'Baking: {job["received"]} frames keyed')
        elif tag=='Z':
            if self.bake is None:return
            job=self.bake
            if data!=b'\x01' or job['received']==0:raise ValueError('Bake returned no animation in selected range')
            self.session.finish(True);self.bake=None;self.live.setChecked(False)
            self.extend_range(job['offset'],job['last_key'])
            self.client.send(b'A'+bytes.fromhex(job['actor']))
            self.message(f'Bake complete: {job["received"]} frames. Existing keys outside the range are preserved.')
        elif tag=='F':raise ValueError(data.decode('utf-8',errors='replace'))
    def poll(self):
        if not self.client:return
        began=time.monotonic();processed=False
        try:
            for _ in range(12):
                try:tag,data=self.client.events.get_nowait()
                except queue.Empty:break
                self.packet(tag,data);processed=True
                if time.monotonic()-began>.025:break
            if processed:rt.redrawViews()
        except Exception as exc:self.disconnect();self.error(exc)
    def closeEvent(self,event):self.disconnect();self.timer.stop();event.accept()

def show():
    global _panel
    try:_panel.close()
    except (NameError,RuntimeError):pass
    _panel=Panel();_panel.show();return _panel

if __name__=='__main__':show()
