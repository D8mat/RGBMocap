"""Offline face capture uses the normal cancellable job queue."""
from pathlib import Path
from .paths import ROOT, worker_command


class FaceWorkspace:
    def refresh_face_capture_action(self):
        # Keep the entry usable without a reconstructed project, including before
        # model-specific early returns. Clicking explains missing input/support.
        self.face_capture_action.setEnabled(not self.busy and not self.contact_saving)
        for name in ('face_neutral_action','face_neutral_reset_action'):
            action=getattr(self,name,None)
            if action is not None:action.setEnabled(not self.busy and not self.contact_saving)
        self.face_capture_action.setToolTip(
            'Capture expressions from the selected take or opened video. '
            'For a new single-camera video, detect the face directly; '
            '3D reconstruction is not required. PEAR live capture is not supported.')

    def extract_face(self):
        if self.busy or self.contact_saving:
            self.status.setText('Wait for the current task or contact save to finish.')
            return
        if self._pear_active:
            self.status.setText('Face capture uses recorded video. Select ExMR, FlowHMR or GVHMR and open a video.')
            return
        try:
            worker_command('face')
        except (OSError, RuntimeError) as error:
            self.status.setText('Run Setup_Face.cmd. ' + str(error))
            return
        take = Path(self.project) if self.project is not None else None
        if take is not None and (take / 'input.mp4').is_file():
            self.start_jobs([('face', ['--take', str(take)])])
            return
        if self.people_state:
            self.status.setText('Select a person/take with a prepared video and tracking in the People list first.')
            return
        if self.model_choice.currentData() not in ('exmr', 'flowhmr', 'gvhmr'):
            self.status.setText('For a new face capture, select ExMR, FlowHMR or GVHMR and open one video.')
            return
        # Prepare the clip, then detect faces directly; body tracking is optional.
        self.input_jobs(['face'])

    def open_face_preview(self):
        from PySide6.QtCore import Qt
        from .face_panel import FacePanel
        panel=getattr(self,'_face_panel',None)
        if panel is None:
            panel=self._face_panel=FacePanel(self)
            self.addDockWidget(Qt.RightDockWidgetArea,panel)
        panel.show();panel.raise_()

    def face_source_at_display(self):
        if self._pear_active or self.model_choice.currentData()=='mamma':return None,0,'Unavailable'
        if self.people_state and self.scene_data is not None:
            current=self.scene_data.take_at(self.active_person_id,self.motion_view.scene_time,self.project)
            if current is None:return None,0,'No take at this time'
            take,frame=current;person=self.current_person()
            return take.project,frame,person['name'] if person else 'Selected person'
        return self.project,int(self.video_view.index),'Selected take'

    def calibrate_face_neutral(self, reset=False):
        if self.busy or self.contact_saving:return
        # Snapshot the actually presented image/take, not a pending slider request.
        project,frame,_=self.face_source_at_display()
        if project is None:
            self.status.setText('Open a captured face take and select its neutral video frame first.');return
        self.stop_playback()
        args=['--project',str(project)] + (['--reset'] if reset else ['--frame',str(frame)])
        self.start_jobs([('face_neutral',args)])
