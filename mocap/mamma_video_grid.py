"""Up to four independently decoded videos with source-timed MAMMA overlays."""
import json
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QComboBox,QSpinBox,QLabel,QSplitter
from .multi_preview import MammaVideoPair
from .viewer import VideoView
from .mamma_overlay import MammaOverlay


class MammaVideoGrid(MammaVideoPair):
    def __init__(self,owner,primary):
        super().__init__(owner,primary)
        self.enabled.hide()
        self.count=QComboBox();self.count.addItems(['1 video','2 videos','3 videos','4 videos'])
        self.controls.layout().insertWidget(0,self.count)
        self.mode=QComboBox()
        for label,data in [('Video only','video'),('Mesh overlay','mesh'),('Skeleton overlay','skeleton')]:self.mode.addItem(label,data)
        self.mode.setToolTip('Use the selected MAMMA result and saved camera calibration. Display only; no reconstruction.')
        self.opacity=QSpinBox();self.opacity.setRange(0,100);self.opacity.setValue(65);self.opacity.setSuffix(' %')
        self.opacity.setToolTip('Overlay opacity')
        self.overlay_controls=QWidget();row=QHBoxLayout(self.overlay_controls);row.setContentsMargins(4,2,4,2)
        row.addWidget(self.mode,1);row.addWidget(QLabel('Opacity'));row.addWidget(self.opacity)
        self.layout().insertWidget(1,self.overlay_controls)
        self.top=QSplitter(Qt.Horizontal);self.bottom=QSplitter(Qt.Horizontal)
        self.top.addWidget(primary);self.top.addWidget(self.second)
        self.splitter.addWidget(self.top);self.splitter.addWidget(self.bottom)
        self.extra=[];self.extra_choices=[];self.extra_keys=[None,None]
        self.extra_panels=[]
        for ordinal in (2,3):
            pane=QWidget();box=QVBoxLayout(pane);box.setContentsMargins(0,0,0,0)
            choice=QComboBox();box.addWidget(choice)
            video=VideoView();video.setMinimumSize(120,90);box.addWidget(video,1)
            self.bottom.addWidget(pane)
            self.extra.append(video);self.extra_choices.append(choice);self.extra_panels.append(pane)
            video.metadata_ready.connect(lambda _,v=video:v.seek(primary.index))
            video.decode_error.connect(lambda msg,n=ordinal:self.failed('Camera '+str(n+1)+': '+msg))
            choice.currentIndexChanged.connect(self.refresh)
        self.second.setMinimumSize(120,90)
        self.overlay=MammaOverlay(self)
        self.all_videos=[primary,self.second,*self.extra]
        for ordinal,video in enumerate(self.all_videos):
            video.mamma_overlay=lambda painter,rect,v=video,n=ordinal:self.overlay.paint(v,painter,rect,self.camera_id(n))
        self.count.currentIndexChanged.connect(self.layout_changed)
        self.mode.currentIndexChanged.connect(self.overlay_changed)
        self.opacity.valueChanged.connect(self.repaint_videos)
        self._extra_entries=None;self._overlay_source=None
        self.refresh()

    def camera_id(self,n):
        inputs=getattr(self.owner,'mamma_inputs',[])
        index=getattr(self.owner,'mamma_active_camera',0) if n==0 else (
            self.choice.currentIndex() if n==1 else self.extra_choices[n-2].currentIndex())
        return inputs[index]['id'] if 0<=index<len(inputs) else None

    def overlay_wanted(self):
        return hasattr(self.owner,'model_choice') and self.owner.model_choice.currentData()=='mamma' and self.mode.currentData()!='video'

    def repaint_videos(self,*_):
        for v in self.all_videos:v.update()
        tools=getattr(self.owner,'camera_tools',None)
        if tools:tools.canvas.update()

    def overlay_changed(self,*_):
        self.overlay.clear()
        self.opacity.setEnabled(self.mode.currentData()!='video')
        tools=getattr(self.owner,'camera_tools',None)
        if tools:tools.sync()
        self.repaint_videos()

    def layout_changed(self,*_):
        self.enabled.setChecked(self.count.currentIndex()>0)
        self.refresh()
        self.top.setSizes([1000,1000]);self.bottom.setSizes([1000,1000]);self.splitter.setSizes([1000,1000])

    def refresh(self,*_):
        super().refresh()
        if not hasattr(self,'extra'):return
        inputs=getattr(self.owner,'mamma_inputs',[])
        available=hasattr(self.owner,'model_choice') and self.owner.model_choice.currentData()=='mamma' and len(inputs)>1
        self.overlay_controls.setVisible(available)
        count=min(self.count.currentIndex()+1,len(inputs)) if available else 1
        for i in range(4):self.count.model().item(i).setEnabled(i<max(1,len(inputs)))
        if available and self.count.currentIndex()+1>len(inputs):
            self.count.setCurrentIndex(len(inputs)-1);return
        orientation=Qt.Vertical if count==2 else Qt.Horizontal
        if self.top.orientation()!=orientation:
            self.top.setOrientation(orientation)
            self.top.setSizes([1000,1000])
        self.choice.setVisible(count>1)
        self.bottom.setVisible(count>2)
        entries=tuple((v['id'],str(v['path'])) for v in inputs)
        if entries!=self._extra_entries:
            self._extra_entries=entries
            for n,choice in enumerate(self.extra_choices):
                choice.blockSignals(True);choice.clear()
                for ident,path in entries:choice.addItem(ident,path)
                choice.setCurrentIndex(min(n+2,len(entries)-1));choice.blockSignals(False)
        identity=(str(getattr(self.owner,'project',None)),id(getattr(self.owner,'preview_data',None)),available)
        if identity!=self._overlay_source:
            self._overlay_source=identity;self.overlay.clear()
        for n,(video,choice,pane) in enumerate(zip(self.extra,self.extra_choices,self.extra_panels)):
            active=count>n+2;pane.setVisible(active)
            if not active:
                if video.decoder is not None:video.shutdown()
                self.extra_keys[n]=None;continue
            if not getattr(self.owner,'info',None):continue
            try:
                project=getattr(self.owner,'project',None)
                state_path=Path(project)/'project.json' if project else None
                stamp=state_path.stat().st_mtime_ns if state_path else None
                key=(entries,choice.currentIndex(),str(project),stamp,self.owner.info['fps'])
                if key==self.extra_keys[n]:continue
                entry=inputs[choice.currentIndex()];path=Path(entry['path']);first=0
                if state_path:
                    state=json.loads(state_path.read_text(encoding='utf8'))
                    if state.get('pipeline',{}).get('model')=='mamma':
                        saved=next((v for v in state.get('mamma_preview_videos',[]) if v['id']==entry['id']),None)
                        if saved:
                            path=(Path(project)/saved['path']).resolve();first=saved['first']
                            root=getattr(self.owner,'workspace_root',None)
                            if root is None or not path.is_relative_to(Path(root).resolve()):raise ValueError('Invalid saved MAMMA preview camera path')
                self.extra_keys[n]=key
                video.open(path,timeline_fps=self.owner.info['fps'],timeline_start=first)
                video.seek(self.primary.index)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                video.shutdown();self.failed(str(exc))
        if self.overlay_wanted():self.repaint_videos()

    def seek(self,index):
        super().seek(index)
        for v in getattr(self,'extra',[]):
            if not self.closed and v.isVisible():v.seek(index)

    def shutdown(self):
        super().shutdown()
        for v in self.extra:v.shutdown()
        for v in self.all_videos:v.mamma_overlay=None
        self.overlay.clear()
