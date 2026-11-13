# MetaHuman face streaming and preview

## RGB MoCap

1. Restart RGB MoCap after the source update. Open a video or a prepared take.
2. Use **File > Capture Face from Video (ARKit)...** if it has no face capture.
3. Open **File > Face Preview & Stream...**. **Load Face FBX...** selects a local
   model. The last path is remembered in local settings, never in motion identities.
4. Play or scrub the timeline. The dock shows the selected face, independently
   of body reconstruction. Drag to orbit, wheel to zoom, double-click to reset.
5. Enable **Stream selected face to Blender**. This sends only facial values to
   `127.0.0.1:49169` (UDP). The separate body streamer continues to use its own port.
   Only the selected take/person is sent by the face-only stream.

Closing/hiding the dock turns face streaming off. Stop or close the dock before
binding the same face to another sender. During processing, missing/stale capture
and timeline gaps send neutral values with invalid status. Paused valid values
are refreshed at least every 0.25 seconds. The sender does not know whether a
receiver is listening; 'Sending' is not a connection acknowledgement.

## Blender

1. Install `blender_addon/rgbmocap_face_stream.zip` using **Preferences > Add-ons >
   Install from Disk**, then enable **RGB MoCap Face Stream**.
2. In the 3D View sidebar (**N**), open **RGB MoCap > RGB MoCap Face Stream**.
3. Use **Import Face FBX...** for `metaHumanHead_52shapekeys_01.fbx`. Alternatively,
   select an already imported head, both eyes and teeth and click **Bind Selected
   Meshes**. Selecting their common parent includes its child meshes.
4. Click **Connect Face Stream**, then enable the app's face stream checkbox.

Binding uses exact ARKit names, never the order of keys in the FBX. The supplied
model has 51 channels on `head_lod0_ORIGINAL`, four eye channels on each eye and
five jaw/mouth channels on the teeth. The `tongueOut` channel is absent and skipped.
The same named channels drive every bound object that supports them. No reparenting,
bone replacement, keyframe writing or mesh topology edits are performed.

The receiver applies data only in its connected scene/view layer and Object Mode.
A one-second timeout resets mapped channels to neutral. Disconnect restores the
values recorded at Connect. Stop before editing shape keys; live updates override
manual values, and existing animated/driven shape-key values may compete with the
stream. This version previews facial motion; it does not record/bake it to an Action.

## Preview implementation and limits

The app imports the selected FBX through its existing isolated Blender runtime.
Only relative ARKit shape keys are supported; vertex-group-masked keys are refused.
The original FBX stays unchanged. Private derived geometry stays under
`cache/face_preview`, with cache identity tied to the source path, size, timestamp
and importer source. The model itself is not included in the add-on ZIP.

The 3D preview retains all 29,835 vertices of the supplied four meshes. It uses
GPU morph blending and recomputes normals; only 52 weights change per frame.
Colors are neutral studio shading, not the original texture/material rendering.
The window reports imported channel coverage and missing/stale capture.

Face data reading, model-cache reading and UDP sending run off the GUI thread with
latest-only requests. Geometry upload happens once per loaded model. Source changes
and paused seeks reject older preview responses. During playback the dock accepts
recent same-source responses up to six frames behind, displaying its own frame
number; this avoids dropping every response due to asynchronous read latency.

## Verified locally, 2026-11-11

- Imported the supplied FBX in Blender 4.5; matched four meshes and 51 unique channels.
- Real UDP packet drove head, eyes and teeth independently by channel name.
- Duplicate sequence and NaN rejection, one-second timeout and disconnect restoration.
- GPU deformation of 29,835 vertices matched the CPU sum within 2.99e-8 at tested values.
- Actual RGB MoCap dock imported the FBX, displayed captured data and emitted UDP
  packets without body motion loaded. The model path was saved for the user.
- Automated face sender test covers stale tracking, neutral output and thread shutdown.

No webcam face capture, facial recording/baking, texture fidelity or face-expression
accuracy against ground truth is claimed. The existing video capture supplies values.

## Neutral face calibration

In **File > Face Preview & Stream...**, pause/scrub the video to a relaxed face
with eyes open and the mouth at rest. Click **Set Current Frame as Neutral**.
The equivalent File menu action is **Use Current Frame as Neutral Face**.
The click pauses playback and freezes the actually displayed image's frame and
selected take, even when the timeline has an outstanding decode request.
The status shows the saved frame number (1-based). **Reset Neutral** or the
File menu's **Reset Neutral Face** restores uncalibrated output.

For each channel, output is `clip(raw - reference, 0, 1)`. Every one of the 52
channels is exactly zero at the reference frame. There is no rescaling or division:
maximum expression amplitude can decrease by the reference offset. Selecting an
expressive rather than relaxed reference intentionally removes that expression.
Choose a genuinely detected frame; missing or interpolated detections are refused.

The per-take `face_neutral.json` sidecar records the raw 52-channel reference,
frame and capture/source identity. The original `face.npz`, video, body results
and face-landmark overlay are unchanged. Repeated calibration always uses raw
capture values, never an already calibrated result. Preview, face-only UDP and
face data in the body stream use the same calibration. Source/take selection
keeps calibrations separate; Save Project As copies the sidecar with the take.
A new/replaced capture invalidates the old reference; the dock asks for a new one.

Calibration runs in the normal cancellable worker queue without model inference.
Preview and stream readers refresh the small sidecar without reloading the
capture arrays. Three regressions cover exact zeros, clamping, repeated selection,
reset, invalid frame rejection, recapture invalidation, independent takes and real
UDP zero values. An actual GUI/worker test verified displayed-frame selection,
zero preview values and reset while preserving the raw capture bytes. The existing
capture, stream and geometry tests also passed (nine regression tests in total).

## Experimental local SAiD editing

The prepared local installation exposes **SAiD: refine next 2 seconds** in the
face dock. It uses the displayed frame and a separate Python environment specified
by `face_ai_said_runtime` in local settings. The runtime folder contains
`venv_said/Scripts/python.exe`, pinned SAiD sources in `SAiD`, and `SAiD.pth`.
It is not installed by Setup_Face.cmd or bundled with the repository.
The adapter is `tools/face_ai_said.py`. Source revision:
`bb7b18018358d3156569d8051f386943744a5ea0`; checkpoint revision:
`2e901c38bee43ee6c0933eae467f64d6a2152d5b`.

Original source audio is hash-checked and offset by the take start time. The model
edits 32 lower-face channels using 20% diffusion strength and 35% output mixing,
with fades at both interval boundaries. Eyes and brows are retained. Each run
replaces the previous candidate with a fresh edit of raw capture; edits do not
accumulate. `face_ai.npz` stores the candidate separately. Toggle **Preview SAiD
candidate** to compare. It affects face preview and face-only UDP, not the body
stream, exports or landmark overlay. Neutral calibration remains zero at its
reference frame. Smooth face curves uses a five-frame symmetric adaptive filter
and preserves blink channels; it is not AI and depends on capture frame rate.

Local validation: actual CUDA inference on RTX 5060 Ti; two-second GoPro candidate
created; ranges, preserved eye/brow channels and neutral zero checked. No visual
quality improvement is claimed yet. SemanticFace is not integrated: its author
model download endpoints returned HTTP 403/404 during preparation.
