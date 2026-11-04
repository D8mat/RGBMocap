"""Optional preview panes: shared motion data, independently owned cameras/renderers."""
from pathlib import Path
import json
import numpy as np
from PySide6.QtCore import Qt, QObject, QEvent, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QComboBox, QCheckBox, QPushButton, QLabel
from .viewport import MotionView
from .viewer import VideoView


class SecondaryMotionView(MotionView):
    # Borrow data only. Never share GL objects, owned mappings, or camera state.
    DATA = '''joints rotations mesh_frames mesh_asset mesh_binding path single_transform
        body_color floor body_height scene_span people_scene people_mode scene_message
        single_actor_visible scene_visible scene_colors scene_selected scene_time scene_names
        actor_name character_workspace render_mode show_bones target_frames target_asset
        raw_asset native_vertices native_required index foot_lock_activity hand_lock_activity
        correction_floor correction_floor_auto correction_floor_pending correction_floor_enabled
        camera_records camera_message camera_translation_scale live_active live_data live_asset
        live_message live_key show_grid checker_floor checker_floor_opacity show_correction_floor
        show_trajectory show_foot_locks show_names show_cameras'''.split()

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.setMinimumSize(180, 180)
        self.follow = False
        self._scene_identity = None

    def sync_data(self):
        source = self.source
        identity = (id(source.people_scene), source.path)
        for name in self.DATA:
            setattr(self, name, getattr(source, name))
        if identity != self._scene_identity:
            self._scene_identity = identity
            self.scene_center = source.scene_center.copy()
            self.mesh_error = None
            self.fit_view()

    def target(self):
        if self.follow and self.people_scene is not None:
            selected = [(take, frame) for take, frame in self.scene_actors()
                        if take.ident == self.scene_selected]
            if selected:
                take, frame = selected[0]
                return self.scene_joint_points(take, frame)[0] + self.pan
        return super().target()

    def fit_view(self):
        super().fit_view()
        if self.follow:
            self.distance = max(1., self.body_height * .60 / np.tan(np.radians(20)) /
                                min(1., max(.4, self.width()/max(1, self.height()))))

    def paintEvent(self, event):
        self.sync_data()
        super().paintEvent(event)


class MultiViewport(QObject):
    def __init__(self, owner, splitter, controls):
        super().__init__(owner)
        self.owner, self.splitter = owner, splitter
        self.views, self.panels = [], []
        self.count = QComboBox()
        self.count.addItems(['1 viewport', '2 viewports', '3 viewports'])
        self.count.setToolTip('Independent 3D cameras; shared animation and timeline. Extra views use additional GPU memory.')
        controls.addWidget(self.count)
        self.count.currentIndexChanged.connect(self.set_count)
        owner.motion_view.installEventFilter(self)
        owner.video_view.frame_presented.connect(self.refresh)

    def set_count(self, index):
        self.owner.motion_view.setMinimumWidth(180 if index else 350)
        while len(self.views) < index:
            view = SecondaryMotionView(self.owner.motion_view)
            panel = QWidget()
            box = QVBoxLayout(panel); box.setContentsMargins(0, 0, 0, 0)
            bar = QHBoxLayout()
            bar.addWidget(QLabel(f'View {len(self.views)+2}'))
            choice = QComboBox(); choice.addItems(['Perspective','Front','Back','Right','Left','Top','Bottom'])
            choice.currentTextChanged.connect(view.set_view)
            def changed(name, combo=choice):
                combo.blockSignals(True); combo.setCurrentText(name); combo.blockSignals(False)
            view.view_changed.connect(changed)
            bar.addWidget(choice)
            box.addLayout(bar); box.addWidget(view, 1)
            footer = QHBoxLayout()
            follow = QCheckBox('Follow Actor'); follow.setToolTip('Follow the actor selected in People. Turn off for a fixed scene view.')
            follow.toggled.connect(view.set_follow)
            frame = QPushButton('Frame All'); frame.clicked.connect(view.frame_all)
            footer.addWidget(follow); footer.addWidget(frame); box.addLayout(footer)
            self.splitter.addWidget(panel)
            self.views.append(view); self.panels.append(panel)
        for i, (view, panel) in enumerate(zip(self.views, self.panels)):
            panel.setVisible(i < index)
            if i < index:view.sync_data(); view.update()
            else:view.cleanup_gl()
        size = max(1, self.splitter.width()//(index+1))
        self.splitter.setSizes([size]*(index+1)+[0]*(len(self.views)-index))

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Paint:self.refresh()
        return False

    def refresh(self, *_):
        for view in self.views:
            if view.isVisible():view.update()

    def close(self):
        self.owner.motion_view.removeEventFilter(self)
        self.owner.video_view.frame_presented.disconnect(self.refresh)
        for view in self.views:view.hide(); view.close()


class MammaVideoPair(QWidget):
    """A second bounded decoder follows the primary presented timeline."""
    def __init__(self, owner, primary):
        super().__init__()
        self.owner, self.primary = owner, primary
        self.key = None
        self.entries_key = None
        self.closed = False
        box = QVBoxLayout(self); box.setContentsMargins(0,0,0,0)
        self.controls = QWidget(); bar = QHBoxLayout(self.controls)
        bar.setContentsMargins(4,2,4,2)
        self.enabled = QCheckBox('Second camera')
        self.choice = QComboBox(); self.choice.setMinimumWidth(0)
        bar.addWidget(self.enabled); bar.addWidget(self.choice,1)
        box.addWidget(self.controls)
        self.splitter = QSplitter(Qt.Vertical); self.splitter.setHandleWidth(5)
        self.splitter.addWidget(primary)
        self.second = VideoView(); self.second.setMinimumSize(180,120)
        self.splitter.addWidget(self.second); self.second.hide()
        box.addWidget(self.splitter,1)
        self.note = QLabel(); self.note.setWordWrap(True); self.note.hide(); box.addWidget(self.note)
        self.second.metadata_ready.connect(lambda _: self.second.seek(primary.index))
        self.second.decode_error.connect(self.failed)
        primary.frame_presented.connect(self.seek)
        self.enabled.toggled.connect(self.refresh)
        self.choice.currentIndexChanged.connect(self.refresh)
        self.timer = QTimer(self); self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh); self.timer.start()
        self.controls.hide()

    def failed(self, message):
        self.note.setText('Second camera: '+message); self.note.show()

    def refresh(self, *_):
        if self.closed:return
        w = self.owner
        inputs = getattr(w, 'mamma_inputs', [])
        available = bool(hasattr(w,'model_choice') and w.model_choice.currentData()=='mamma' and len(inputs)>1)
        self.controls.setVisible(available)
        entries = tuple((item['id'], str(item['path'])) for item in inputs)
        if entries != self.entries_key:
            self.entries_key = entries
            self.choice.blockSignals(True); self.choice.clear()
            for ident, path in entries:self.choice.addItem(str(ident), path)
            self.choice.setCurrentIndex(min(1,len(entries)-1)); self.choice.blockSignals(False)
        active = available and self.enabled.isChecked()
        self.second.setVisible(active)
        if not active:
            if self.second.decoder is not None:self.second.shutdown()
            self.key = None; self.note.hide(); return
        if not getattr(w, 'info', None):return
        # Read project metadata only on source/project selection changes.
        index = self.choice.currentIndex()
        project = getattr(w,'project',None)
        state_path = Path(project)/'project.json' if project else None
        try:
            stamp = state_path.stat().st_mtime_ns if state_path else None
            key = (entries, index, str(project), stamp, w.info['fps'])
            if key == self.key:return
            self.key = key
            entry = inputs[index]; path = Path(entry['path']); first = 0
            if state_path:
                state = json.loads(state_path.read_text(encoding='utf-8'))
                if state.get('pipeline',{}).get('model')=='mamma':
                    saved = next((v for v in state.get('mamma_preview_videos',[]) if v['id']==entry['id']), None)
                    if saved:
                        path = (project/saved['path']).resolve(); first = saved['first']
                        root = getattr(w,'workspace_root',None)
                        if root is None or not path.is_relative_to(Path(root).resolve()):
                            raise ValueError('Invalid saved MAMMA preview camera path.')
            self.note.hide()
            self.second.open(path, timeline_fps=w.info['fps'], timeline_start=first)
            self.second.seek(self.primary.index)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.second.shutdown(); self.failed(str(exc))

    def seek(self, index):
        if not self.closed and self.second.isVisible():self.second.seek(index)

    def shutdown(self):
        self.closed = True; self.timer.stop()
        self.primary.frame_presented.disconnect(self.seek)
        self.second.shutdown()
