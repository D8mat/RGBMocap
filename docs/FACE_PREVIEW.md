# Face Preview from Track Face

Use the original **Track Face** checkbox, then **File > Face Preview & Stream**.
Select a person in People and play/scrub the timeline. Load an ARKit blendshape FBX
in the preview dock. The dock reads the exact prepared expression sample used by
Unreal Live Link, including upstream smoothing, neutral calibration, person and
source-frame timing. Channel names are remapped to the preview model order.
Missing data clears the preview; upstream held expressions are labelled explicitly.
MAMMA uses the same selected-person/take resolution as Live Link.

The custom Capture Face action, click-to-select face capture, SAiD candidate and
separate smoothing controls have been removed. The video header offers Tracked Face Overlay and Face Crop using original Track Face
landmarks; the 3D face preview dock is retained. No legacy
face.npz, face_ai.npz, saved model, project or installed runtime is deleted.
Old worker/setup entry points explain that Track Face replaces Capture Face.

Set/reset neutral in the dock uses the original Track Face controls. MetaHuman
body/face streaming and its protocol are unchanged. Optional face-only Blender UDP
and the Blender body stream now also use original Track Face data.

## Video overlay

Enable Tracked Face Overlay / Face Crop above the video. Select a person in People.
Old tracks need Track Face once again (toggle Track Face off/on) to save original
2D landmarks. Tracking expressions and MetaHuman streaming use the same original
algorithm. The overlay appears on the tracked source camera, not unrelated camera
views. Missing detections clear geometry rather than holding it on a later frame.
