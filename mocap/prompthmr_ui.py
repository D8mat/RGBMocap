"""Lightweight PromptHMR controls and worker queue; no model imports in Qt."""
import json,uuid
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QVBoxLayout,QGroupBox,QLabel,QSpinBox,QCheckBox


class PromptHMRWorkspace:
    def build_prompthmr_settings(self):
        page=QWidget();page.setProperty('motion_tool','prompthmr')
        layout=QVBoxLayout(page);layout.setContentsMargins(0,0,0,0);layout.setSpacing(10);layout.setAlignment(Qt.AlignTop)
        group=QGroupBox('PromptHMR Settings');box=QVBoxLayout(group);box.setSpacing(7);layout.addWidget(group)
        box.addWidget(QLabel('People'))
        self.prompthmr_people=QSpinBox();self.prompthmr_people.setRange(0,8);self.prompthmr_people.setValue(2)
        self.prompthmr_people.setToolTip('Maximum number of people to track together. 0 detects automatically, up to 8 in this app. Start with 2 for a two-person interaction. The intended people should be visible near the beginning; this is not an identity selector. Applies on Reconstruct 3D.')
        box.addWidget(self.prompthmr_people)
        self.prompthmr_interaction=QCheckBox('Interaction Prompt');self.prompthmr_interaction.setChecked(True)
        self.prompthmr_interaction.setToolTip('Ask PromptHMR to consider the people together. May improve relative placement during interaction; it does not guarantee hand contact or prevent body intersections. Off follows the official default video path. Applies on Reconstruct 3D.')
        box.addWidget(self.prompthmr_interaction)
        self.prompthmr_notice=QLabel();self.prompthmr_notice.setWordWrap(True);self.prompthmr_notice.setTextFormat(Qt.PlainText);box.addWidget(self.prompthmr_notice)
        group.setToolTip('Fixed-camera video only in this Windows integration. SAM2 tracking, PromptHMR image/video inference and native world post-optimization. No animated finger inference. Low Memory retains the same one-image batch; additional offloading is not implemented.')
        self.prompthmr_people.valueChanged.connect(self.prompthmr_settings_changed)
        self.prompthmr_interaction.toggled.connect(self.prompthmr_settings_changed)
        self.model_pages.addWidget(page)
        self.model_choice.setItemData(self.model_choice.findData('prompthmr'),'PromptHMR: shared-scene single-video capture, especially for people interacting. Fixed cameras only; no animated fingers. Separate optional runtime.',Qt.ToolTipRole)

    def prompthmr_options(self):
        return dict(people=self.prompthmr_people.value(),interaction=self.prompthmr_interaction.isChecked())

    def restore_prompthmr_options(self,value):
        from .prompthmr_pipeline import options
        value=options(value)
        for control,setting in ((self.prompthmr_people,value['people']),(self.prompthmr_interaction,value['interaction'])):
            control.blockSignals(True)
            if isinstance(control,QCheckBox):control.setChecked(setting)
            else:control.setValue(setting)
            control.blockSignals(False)

    def prompthmr_settings_changed(self,*_):
        if hasattr(self,'readiness'):self.update_readiness()

    def sync_prompthmr_controls(self):
        if self.model_choice.currentData()!='prompthmr':return
        self.find_person_button.hide()
        self.people_help_dialog.setText('PromptHMR tracks people together during Reconstruct 3D; Find person is not needed. Use a fixed camera and set People to 2 for a two-person interaction, or 0 for automatic detection (up to 8). The intended people should be visible near the start.\n\nAfter reconstruction, check rows to show people and select a row to edit contacts. All people keep one inferred world. Interaction Prompt does not guarantee contact or prevent intersections. No animated fingers are generated.')
        self.prompthmr_people.setEnabled(not self.busy);self.prompthmr_interaction.setEnabled(not self.busy)
        message=''
        try:
            from .prompthmr_runtime import configuration
            configuration()
            if self.project is not None:
                state=json.loads((self.project/'project.json').read_text(encoding='utf8'));saved=state.get('prompthmr_options',{})
                if saved!=self.prompthmr_options():message='People/interaction settings changed. Reconstruct 3D to apply; previous motion is still displayed.'
                elif state.get('pipeline',{}).get('model')=='prompthmr':
                    receipt=self.project/'prompthmr_source.json'
                    if receipt.is_file() and not json.loads(receipt.read_text(encoding='utf8')).get('native',{}).get('hand_pose_policy'):
                        message='Legacy PromptHMR finger placeholders. Reconstruct 3D to refresh hands; verified inference caches will be reused.'
        except (OSError,ValueError,RuntimeError) as error:
            message=str(error);self.run_button.setEnabled(False)
        self.prompthmr_notice.setText(message);self.prompthmr_notice.setVisible(bool(message))
        self.run_button.setToolTip('Track and reconstruct people together from one fixed-camera video. Preserves the shared inferred scene, then optionally applies Motion Correction. No moving-camera or finger inference in this integration.')

    def run_prompthmr(self):
        if self.busy or self.contact_saving or not self.video_path or not self.info:return
        try:
            from .prompthmr_runtime import configuration
            from .models import missing
            from .paths import HOME,PROJECTS
            from .storage import atomic_json,project_key
            from .prompthmr_pipeline import STAGES
            configuration()
            if missing('prompthmr_assets'):
                self.status.setText('Prepare the missing PromptHMR files in Model Manager.');self.open_model_manager();return
            first,end=self.processing_range()
            root=self.workspace_root or PROJECTS/(self.video_path.stem[:80]+'_'+project_key(self.video_path,0,self.info['duration'])+'_project')
            folder=HOME/'cache/prompthmr-jobs';folder.mkdir(parents=True,exist_ok=True);request=folder/(uuid.uuid4().hex+'.json')
            atomic_json(request,dict(version=1,root=str(root),video=str(self.video_path),start=first,end=end,options=self.prompthmr_options(),
                correction_enabled=self.motion_correction.isChecked(),correction=self.selected_correction_options()))
            self.start_jobs([(stage,['--request',str(request)]) for stage in STAGES])
        except (OSError,ValueError,RuntimeError) as error:self.status.setText('Cannot start PromptHMR: '+str(error))
