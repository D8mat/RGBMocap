"""Source-motion export options; existing character exports remain separate."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QLabel,QDialogButtonBox,QDoubleSpinBox

class SourceExportDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('Export');self.setMinimumWidth(660)
        self.model=parent.model_choice.currentData() if parent is not None and hasattr(parent,'model_choice') else None
        layout=QVBoxLayout(self);layout.setContentsMargins(18,18,18,18);layout.setSpacing(12)
        form=QFormLayout();form.setVerticalSpacing(12);layout.addLayout(form)
        self.format=QComboBox()
        self.format.addItem('FBX (.fbx)','fbx');self.format.addItem('BVH (.bvh)','bvh')
        self.format.addItem('Blender project + video (.blend)','blend')
        form.addRow('Format',self.format)
        self.fbx_profile=QComboBox();self.fbx_profile.addItem('Source skeleton only','generic')
        if self.model in ('mamma','exmr','gvhmr','prompthmr','flowhmr'):self.fbx_profile.addItem('Unreal Engine — skinned body + skeleton + animation','unreal')
        self.fbx_label=QLabel('FBX profile');form.addRow(self.fbx_label,self.fbx_profile)
        self.fbx_profile.currentIndexChanged.connect(self.update_format)
        self.unreal_z=QDoubleSpinBox();self.unreal_z.setRange(-100000,100000);self.unreal_z.setDecimals(4);self.unreal_z.setValue(0);self.unreal_z.setSuffix(' cm')
        self.unreal_z_label=QLabel('Additional Unreal Z offset');form.addRow(self.unreal_z_label,self.unreal_z)
        self.unreal_ground=QCheckBox('Place feet on floor (Z = 0)');self.unreal_ground.setChecked(True);layout.addWidget(self.unreal_ground)
        self.unreal_ground.setToolTip('Ground the reference/T-pose and first motion frame using the skinned foot surface. Apply one constant offset to the whole original animation; preserve jumps. Additional Z offset is applied afterward.')
        self.unreal_heading=QCheckBox('Align animation facing to reference skeleton');layout.addWidget(self.unreal_heading)
        self.unreal_heading.setToolTip('Rotate the whole animation around the initial pelvis to match the reference heading. Preserves height and relative motion; useful for MetaHuman retargeting. Does not change the bind pose or IK chain mapping.')
        self.unreal_resume=QCheckBox('Resume completed export steps');self.unreal_resume.setChecked(True);layout.addWidget(self.unreal_resume)
        self.unreal_resume.setToolTip('Keep verified prepared data and completed output in the project export cache. Retry with the same destination after interruption. An unfinished step restarts; changed inputs invalidate it.')
        self.unreal_preview=QCheckBox('Preview only: first 3 seconds as Blender project');layout.addWidget(self.unreal_preview)
        self.unreal_preview.setToolTip('Save a short .blend with the exact export skeleton, skin, heading, floor offset and optional T-pose. Review it in Blender before exporting the full FBX.')
        self.unreal_performance=QLabel();self.unreal_performance.setWordWrap(True);layout.addWidget(self.unreal_performance)
        try:
            import json
            from pathlib import Path
            report=json.loads((Path(parent.project)/'unreal-export-performance.json').read_text(encoding='utf8'))
            self.unreal_performance.setText('Last Unreal export: %.2f s preparation, %.2f s build; reused preparation: %s, output: %s.' % (report['preparation_seconds'],report['build_seconds'],report['reused_preparation'],report['reused_output']))
        except (AttributeError,TypeError,OSError,ValueError,KeyError):self.unreal_performance.setText('Export timings and checkpoint reuse will appear here after the first export and in the Console.')
        self.profile=QComboBox();self.profile.addItem('Generic / Blender','generic')
        if self.model!='gemx':self.profile.addItem('3ds Max Biped','biped')
        self.profile.setToolTip('Generic / Blender keeps anatomical names and original rotation orders. 3ds Max Biped uses Character Studio names and uniform ZXY rotations. Both keep metre units and Y up.')
        self.profile_label=QLabel('BVH profile');form.addRow(self.profile_label,self.profile)
        self.whole_scene=QCheckBox('MAMMA whole scene: body meshes and all cameras');layout.addWidget(self.whole_scene)
        self.whole_scene.setToolTip('Export checked people, exact animated surfaces, reference skeletons and every calibrated camera/video. Keeps the original timeline. Mesh caches can be large.')
        self.tpose=QCheckBox('Add T-pose before motion');self.tpose.setChecked(False)
        self.tpose.setToolTip('Insert a T-pose at the first motion frame root position. Original frames shift by one; their poses and timing intervals stay unchanged.')
        layout.addWidget(self.tpose)
        self.place_at_origin=QCheckBox('Place at origin');layout.addWidget(self.place_at_origin)
        self.place_at_origin.setToolTip("Center each person's first exported frame on world X/Z. Height and travel are preserved. People are centered independently; their relative placement changes.")
        self.symmetric_rest=QCheckBox('Symmetrical rest pose');layout.addWidget(self.symmetric_rest)
        self.symmetric_rest.setToolTip('Use mirrored left/right rest bones with averaged paired lengths and a straight vertical spine. Animation rotations and timing are retained; changed proportions can move hands and feet. Exported files only.')
        self.animate_camera=QCheckBox('Export saved camera tracking');self.animate_camera.setChecked(True);layout.addWidget(self.animate_camera)
        self.background_refine=QCheckBox('Refine Camera Rotation');layout.addWidget(self.background_refine)
        from .camera_ui import BACKGROUND_HELP
        self.background_refine.setToolTip(BACKGROUND_HELP)
        if parent is not None and hasattr(parent,'camera_tools'):self.background_refine.setChecked(parent.camera_tools.background_options().get(self.model,False))
        self.translation_scale=QDoubleSpinBox();self.translation_scale.setRange(0,100);self.translation_scale.setDecimals(3);self.translation_scale.setValue(1);self.translation_scale.setSingleStep(.1)
        if parent is not None and hasattr(parent,'camera_tools'):self.translation_scale.setValue(parent.camera_tools.scale.value())
        self.translation_scale.setToolTip('Approximate saved camera movement. 0 fixes the camera center; 1 uses the estimate. This does not change the source animation.')
        self.scale_label=QLabel('Camera movement scale');form.addRow(self.scale_label,self.translation_scale)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        layout.addStretch(1)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Continue...');buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        for signal in (self.format.currentIndexChanged,self.whole_scene.toggled,self.animate_camera.toggled):signal.connect(self.update_format)
        self.update_format()
    def update_format(self):
        fmt=self.format.currentData();blend=fmt=='blend';whole=blend and self.model=='mamma' and self.whole_scene.isChecked()
        unreal=fmt=='fbx' and self.fbx_profile.currentData()=='unreal'
        self.fbx_label.setVisible(fmt=='fbx');self.fbx_profile.setVisible(fmt=='fbx')
        self.unreal_z_label.setVisible(unreal);self.unreal_z.setVisible(unreal);self.unreal_ground.setVisible(unreal);self.unreal_heading.setVisible(unreal);self.unreal_resume.setVisible(unreal);self.unreal_preview.setVisible(unreal);self.unreal_performance.setVisible(unreal)
        self.profile_label.setVisible(fmt=='bvh');self.profile.setVisible(fmt=='bvh')
        self.whole_scene.setVisible(blend and self.model=='mamma');self.tpose.setEnabled(not whole);self.tpose.setVisible(not whole)
        for control in (self.place_at_origin,self.symmetric_rest):control.setVisible(not blend and not unreal);control.setEnabled(not blend and not unreal)
        self.animate_camera.setVisible(blend and not whole)
        camera=blend and not whole and self.model in ('flowhmr','exmr')
        self.background_refine.setVisible(camera);self.background_refine.setEnabled(self.animate_camera.isChecked())
        self.scale_label.setVisible(camera);self.translation_scale.setVisible(camera);self.translation_scale.setEnabled(self.animate_camera.isChecked())
        if self.model=='gemx':
            self.symmetric_rest.hide();self.symmetric_rest.setEnabled(False)
            if blend:self.tpose.hide();self.animate_camera.hide()
            self.note.setText('Exports the native SOMA skeleton, exact animated surface and synchronized video in a new Blender bundle folder. Keep the files together.' if blend else
                              'Exports the selected SOMA Native77 skeleton and motion at 30 fps. BVH uses metres and Y up. Add T-pose inserts one frame using the native rest pose.')
            self.layout().activate()
            return
        text=('Creates a new folder containing the .blend, videos and mesh caches. Keep the folder together. Bodies are baked PC2 surfaces; skeletons are separate editable references. No T-pose is inserted.' if whole else
              'Creates a new folder containing the .blend and synchronized video. Keep both together. The video is a camera-view background, not a render background. Saved camera movement is approximate.' if blend else
              'Exports the selected source skeleton. BVH uses metres and Y up. The T-pose is an extra animation frame; it does not replace the character bind pose.')
        if unreal:text='Creates a skinned SMPL-X / SMPL-H body, native deformation skeleton and baked animation for Unreal. Import as a new Skeletal Mesh with Skeleton=None. Pose-corrective surface detail is not included; this is not the UE Manny skeleton. Prepare the mesh preview first.'
        self.note.setText(text)
        self.layout().activate();self.adjustSize()
    def options(self):
        if self.model=='gemx' and self.format.currentData()=='blend':return dict(format='blend',animate_camera=False)
        fmt=self.format.currentData();whole=fmt=='blend' and self.model=='mamma' and self.whole_scene.isChecked()
        value=dict(format='blend_scene' if whole else fmt,bvh_profile=self.profile.currentData() if fmt=='bvh' else 'generic',prepend_tpose=self.tpose.isChecked() and not whole)
        value['unreal']=fmt=='fbx' and self.fbx_profile.currentData()=='unreal'
        value['unreal_z_offset_cm']=self.unreal_z.value()
        value['unreal_ground_feet']=self.unreal_ground.isChecked()
        value['unreal_align_heading']=self.unreal_heading.isChecked()
        value['unreal_resume']=self.unreal_resume.isChecked()
        value['unreal_preview']=value['unreal'] and self.unreal_preview.isChecked()
        value.update(place_at_origin=self.place_at_origin.isChecked() and fmt in ('fbx','bvh') and not value['unreal'],symmetric_rest=self.model!='gemx' and self.symmetric_rest.isChecked() and fmt in ('fbx','bvh') and not value['unreal'])
        if fmt=='blend':value.update(animate_camera=self.animate_camera.isChecked(),background_refine=self.background_refine.isChecked() and self.animate_camera.isChecked() and self.model in ('flowhmr','exmr'),camera_translation_scale=self.translation_scale.value())
        return value
