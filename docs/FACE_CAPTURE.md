# Face capture from video

Run `Setup_Face.cmd` once with RGB MoCap closed. This optional CPU MediaPipe
runtime has its own NumPy/OpenCV packages; body runtimes stay unchanged.
The installer verifies wheel hashes and the Google Face Landmarker checksum.

Open a video with ExMR, FlowHMR or GVHMR selected, then use **File > Capture Face from Video (ARKit)...**. A new video is prepared and faces are detected directly; body tracking and 3D reconstruction are not required. Existing person tracking, when available, scopes the detection to that actor. An existing selected take with video and tracking is captured directly.
Capture uses that take's `input.mp4` and optional `tracking.npz`, and publishes `face.npz`
atomically. Use the normal Cancel button to stop. Repeat for other takes/people.
When tracking exists, its frame count must match the video; PEAR live
capture and multiview facial fusion are not implemented.

Install the updated `blender_addon/rgbmocap_bone_stream.zip` in Blender and
connect through the existing Stream tab. Each actor receives 52 ARKit-named
custom properties (`arkit_jawOpen`, etc.). Meshes parented beneath that actor
with exactly matching ARKit shape-key names receive those values automatically.
An armature without a facial mesh exposes values but cannot show facial deformation.
The app's body preview and FBX/BVH/VMD exports do not include facial animation.
The add-on's Bake command still bakes body bones only.

Missing channels (including unsupported tongue motion) stay zero. Interior
losses of at most 0.2 seconds hold the nearest detected expression; longer losses
and ends are neutral. `valid` retains the actual detection mask. Small/occluded
faces and inverted poses may fail. Full person crops preserve aspect ratio,
with a head crop fallback for distant faces. Without body tracking, ambiguous
multi-face frames are rejected. This is not identity-proof facial recognition. No universal accuracy is claimed.

The stream rejects nonfinite, oversized, malformed or stale data. Video/tracking
size and modification time are checked, not cryptographic source identity.
Moving a project with tools that change timestamps requires recapturing its face.
Face arrays are loaded on the stream thread in a bounded four-take cache. Older
receivers retain body streaming through explicit face capability negotiation.

Adapted from mayalhc/Chamiseul_RGBMocap commit
`f5eccc5ff7c64d6f420415068c4a75e27b58fe69` (face capture/setup only).
Upstream license follows:

```text
MIT License

Copyright (c) 2026 RGB MoCap contributors
Copyright (c) 2026 Chamiseul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

----------------------------------------------------------------------------
Scope and third-party components

The MIT grant above applies to original code and documentation contributed to
RGB MoCap, except the Blender add-on in blender_addon/, which is licensed under
GPL-3.0-or-later (see blender_addon/LICENSE). It does not relicense third-party code, modifications derived from
third-party code, dependencies, model weights, registered body models or other
external assets, wherever they are located. Those materials retain their
respective licenses and notices.

See THIRD_PARTY_NOTICES.md for the major components and model conditions.
This file does not grant permission to redistribute restricted materials or
resolve copyleft obligations for a combined application distribution.

```

## Local validation (2026-11-11)

- Optional pinned runtime installed; real Face Landmarker initialized successfully.
- Real extraction on a 12-frame synthetic clip made from Google's MediaPipe
  portrait test asset detected a face in all 12 frames. This checks execution,
  not accuracy on body-motion footage.
- Three regression tests cover bounded gap filling, stale/corrupt results,
  frame limits, legacy socket negotiation and two-person face/frame pairing.
- Blender 4.5 background test applied two distinct actor values to custom
  properties and parented ARKit shape keys; nonfinite payloads were rejected.
- Offscreen GUI initialized with the new menu action. The base runtime registry
  still matches the installation receipt.
- Real user-video accuracy, interactive playback throughput and face animation
  export were not validated. Facial export is not implemented.

## Independent face preview and streaming

Use **File > Face Preview & Stream...** for a local 3D FBX head preview and face-only Blender streaming, even without body reconstruction. See [FACE_STREAM.md](FACE_STREAM.md) for the separate face add-on and setup.

## Video face mesh and crop

The Movie Clip header now has independent **Face Mesh** and **Face Crop** switches.
They display the selected captured face on the currently displayed video frame.
Face Mesh uses MediaPipe's tessellation, a white outline and lips, and red/green
eye, iris and eyebrow contours. Face Crop shows a magnified inset from that same
video image. These controls are independent of the body/camera **Overlays** switch.

Capture version 2 stores 478 normalized 2D landmarks and face bounds alongside
expressions. Older captures remain usable for streaming but must be captured again
for the mesh overlay. No AI, decoder, model loading or filesystem access runs in
painting. One background reader checks source identities; geometry loading is
bounded to 96 MiB uncompressed and 36000 frames. Painting uses cached arrays and
the exact displayed index, including seeks, video scaling, zoom and panning.
Missing detections show a status message rather than keeping a stale mesh/crop.
PEAR live feeds and multi-camera MAMMA video overlays remain outside this feature.

A 1382-frame working portrait clip that previously yielded zero faces now yielded
1382 detected faces. Five widely spaced samples confirmed the original head crop
missed the face while the full person crop found it. This measures detection
coverage on this clip, not expression accuracy or universal reliability. Actual
GUI tests covered overlays at frame 345, toggling, zoom and a reverse seek to frame 0.
Two geometry-reader tests plus the existing capture and stream regressions passed.

Topology in `mocap/face_topology.json` is adapted from MediaPipe 1.2.2, Copyright
2023 The MediaPipe Authors, Apache-2.0. See `docs/licenses/mediapipe-LICENSE.txt`
and `docs/licenses/mediapipe-NOTICE.txt`; dataclass edges were converted to unique,
undirected JSON pairs. No MediaPipe Python modules are imported in the GUI.

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
