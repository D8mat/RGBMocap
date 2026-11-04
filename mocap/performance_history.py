"""Bounded history of actual worker measurements; missing metrics stay unknown."""
import json,time
from pathlib import Path
from .storage import atomic_json


def read(project):
    if project is None:return []
    try:
        p=Path(project)/'performance-history.json'
        if p.stat().st_size>1024*1024:return []
        value=json.loads(p.read_text(encoding='utf8'))
        return value[-100:] if isinstance(value,list) else []
    except (OSError,ValueError):return []


def record(project,event):
    if project is None:return
    fields=('stage','elapsed','cached','peak_ram_gib','peak_cuda_allocated_gib','peak_cuda_reserved_gib')
    row={k:event[k] for k in fields if k in event};row['time']=time.strftime('%Y-%m-%d %H:%M:%S')
    atomic_json(Path(project)/'performance-history.json',(read(project)+[row])[-100:])


def show(parent):
    from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QDialogButtonBox
    d=QDialog(parent);d.setWindowTitle('Performance');d.resize(850,420);layout=QVBoxLayout(d)
    label=QLabel('Recent completed steps for this project. RAM/VRAM values are worker-reported peaks, not whole-PC usage. A dash means unavailable. Cached steps reuse verified results; interrupted steps do not appear as completed.');label.setWordWrap(True);layout.addWidget(label)
    rows=read(getattr(parent,'project',None));table=QTableWidget(len(rows),6);table.setHorizontalHeaderLabels(['Time','Step','Seconds','Cache reused','RAM GiB','CUDA allocated GiB'])
    for i,row in enumerate(reversed(rows)):
        for j,key in enumerate(('time','stage','elapsed','cached','peak_ram_gib','peak_cuda_allocated_gib')):
            value=row.get(key,'—')
            if isinstance(value,float):value=f'{value:.3f}'
            table.setItem(i,j,QTableWidgetItem(str(value)))
    table.setEditTriggers(QTableWidget.NoEditTriggers);table.resizeColumnsToContents();layout.addWidget(table)
    try:
        data=json.loads((Path(parent.project)/'unreal-export-performance.json').read_text(encoding='utf8'))
        detail=QLabel('Latest Unreal export: %.2f s preparation; %.2f s build; prepared data reused: %s; output reused: %s. Blender sampled peak RAM: %s bytes.' % (data['preparation_seconds'],data['build_seconds'],data['reused_preparation'],data['reused_output'],data.get('resources',{}).get('blender_sampled_peak_ram_bytes','—')));detail.setWordWrap(True);layout.addWidget(detail)
    except (AttributeError,TypeError,OSError,ValueError,KeyError):pass
    buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(d.reject);layout.addWidget(buttons);d.exec()
