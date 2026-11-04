# Editable MAMMA camera calibration

The MAMMA settings panel adds **Export Camera Data...**, **Edit Cameras in
Blender...**, and **Restore Previous Camera Settings** beside the existing import.
They are disabled while processing. No existing reconstruction is modified.

JSON export uses original video resolution, metres, a Z-up MAMMA world and
OpenCV camera axes (X right, Y down, Z forward). `extrinsics_matrix` is a 3x4
world-to-camera transform: camera_point = R * world_point + t. Camera position is
-R^T * t, not t. `intrinsic_matrix` contains pixel fx/fy/cx/cy. Camera IDs must
match original video filenames without extensions. Saved automatic calibration
is rescaled from the processed image size to the original video size before
export; extrinsics and distortion are preserved. This does not make estimated
VGGT scale/gravity into measured calibration.

Blender editor creates a new folder with immutable `calibration.original.json`,
`camera_editor.py`, and `camera_editor.blend`. Move/rotate cameras and adjust focal
length or shift. In the N sidebar, RGB MoCap > Export edited calibration writes
`calibration.edited.json`; previous edited files receive timestamped backups.
Keep camera IDs, unit scale, perspective type and horizontal sensor fit. The
custom fx/fy field preserves non-square pixel aspect. Resolution and distortion
stay associated with the original camera. The Blender preview does not simulate
lens distortion or copy source videos. After reopening the saved blend, run its
camera_editor.py text in the Scripting workspace to restore the export panel;
no automatic embedded script execution is enabled.

Import the edited JSON in RGB MoCap. The review shows camera positions and focal
lengths. Accepting stores the previous settings under
`camera-calibration-history/` and selects Manual Camera Data for the **next**
Reconstruct 3D. Restore Previous Camera Settings undoes the most recent import in
the current session; persistent backup JSONs retain earlier settings. Importing
never changes camera receipts or meshes of an existing result.

The present MAMMA adapter requires already-undistorted pinhole footage with zero
skew/distortion. Import rejects unsupported nonzero coefficients early. A measured
lens calibration generally requires an undistortion step and its updated camera
matrix before use here. No board-image calibration solver is added in this update.

## Printable checkerboards

- [A4 landscape, 25 mm squares](samples/calibration_A4_25mm.pdf)
- [A3 landscape, 35 mm squares](samples/calibration_A3_35mm.pdf)

Both contain 10 x 7 squares / 9 x 6 internal corners. Print at 100% / Actual Size,
not Fit to Page; measure the 100 mm reference line and actual squares. Mount flat
on a rigid matte board. Keep focus, zoom, capture resolution and digital cropping
fixed after calibration. Use multiple sharp views spanning the frame and varied
board angles for lens calibration; shared observations across cameras are needed
for their relative poses. A4 is a starting board, not guaranteed sufficient at a
full-stage distance. Retain white margins. These are checkerboards, not AprilTags.

OpenCV reference: https://docs.opencv.org/4.12.0/dc/dbb/tutorial_py_calibration.html

Validation (2026-11-04, isolated staging): four CPU tests cover original-pixel
rescaling, wrong names/resolutions, invalid rotations and nonfinite data. Two
synthetic cameras round-tripped through Blender with intrinsic error below
0.001 px and extrinsic error below 1e-6. A 0.25 m camera movement survived JSON
export and the MAMMA importer accepted it. UI review cancel/accept, backup,
restore and busy controls passed. OpenCV detected all 54 internal corners in
both rendered PDFs. Physical printed-board calibration and reconstruction with
edited cameras have not been tested.

## AprilTag observations

MAMMA > Camera Calibration > **Detect AprilTags...** opens a read-only snapshot
inspector. **Detect Photos...** accepts photos, or **Detect Camera Snapshots** reads
one frame at the chosen time (seconds from the start of each original camera
video). Videos must already be synchronized; the nearest native-frame timestamp
may differ across frame rates. It does not open webcams or start AI inference.

The detector uses OpenCV `DICT_APRILTAG_36h11`, refines corners, labels IDs and
flags duplicate IDs in one image. Detection runs in a cancellable background
thread, releases video handles and keeps at most 16 preview images. JSON includes
original-resolution corners, image dimensions, camera/source name and file stamp,
requested time/frame, OpenCV version and dictionary fingerprint. Missing views,
errors and cancellation are explicit. The supplied tag size is metadata, not a
measurement or camera-pose estimate. No calibration or existing motion changes.

[Printable AprilTags: six A4 pages, IDs 0-5, 150 mm black edge](samples/apriltags_36h11_150mm_ID0-5.pdf)

Print at 100% / Actual Size. Measure the 100 mm check line and the 150 mm outer
black edge, excluding the white margin. Keep the white border, mount flat on a
rigid matte surface and use each physical ID once in a shared scene. The included
PDF IDs were verified with the app dictionary; labels in externally generated
AprilTag PDFs may use different numbering and should not be assumed equivalent.
Checkerboards are not AprilTags. This first version exports 2D observations; it
does not implement a marker-map calibration solver, continuous tracking, or lens
undistortion. Camera calibration still uses the explicit import workflow above.

Validation: detector tests passed for rotations, perspective corners, duplicate
IDs and blank images. Every page of the new PDF detected its expected ID 0-5;
vector geometry uses a 150 mm black square on A4. The background worker was tested
with a Unicode photo filename, synthetic video frame selection, a failed view,
JSON export and the dialog. Physical camera/printed-tag accuracy remains untested.
# Calculate calibration from checkerboards and measured AprilTags

Open **MAMMA > Camera Calibration > Detect AprilTags > Calculate Calibration**.
This CPU worker saves a reviewable result and does not apply it to a project.

1. For each fixed camera, capture 15-25 sharp checkerboard photos with varied
   position, distance and tilt, at the exact video resolution, zoom and focus.
   Choose **Calibrate Lens from Photos**, inner corners **9 x 6**, square edge
   **25 mm** for our A4 board or **35 mm** for A3. At least 10 detected views are
   required. Save one lens JSON per camera. Fisheye lenses are not supported.
2. Mount unique 150 mm AprilTags rigidly, and measure their four outer black
   corners in a common coordinate system in metres, with Z upward. Pick the
   origin on the floor. The detection preview labels corners **0,1,2,3**; preserve
   that exact ID-specific order even for a rotated tag. Do not assume image
   top-left equals corner 0. Use `docs/samples/marker_map.example.json` as a
   format example, replacing its invented coordinates with your measurements.
3. Detect and export one complete snapshot session of the static mapped tags.
   Each camera needs at least three mapped tags. Cameras can see different tags
   because the measured map already supplies their common coordinate system.
   Include tags at different heights/depths where possible; flat configurations
   are more sensitive to viewing angle and depth ambiguity.
4. Copy `docs/samples/calibration_job.example.json`; set the detection JSON,
   measured map JSON and lens file for every exact camera ID from the session.
   Paths are relative to the job file, or absolute. Select it using
   **Calculate Camera Positions from Job JSON**. Review the per-camera RMS
   reprojection errors, then **Save Result JSON**.

The solver uses OpenCV calibrateCamera (five radtan coefficients), PnP RANSAC
and LM refinement. It rejects duplicate IDs, mismatched resolutions/dictionaries,
invalid square maps, insufficient views or tilt, too few consistent corners,
negative depths and RMS above 2 pixels. At least 80% of supplied corners must
be inliers; reported maximum error includes rejected corners. A low residual
does not establish physical accuracy: verify an independent measured distance
and held-out board views. Bad measurements, near-planar ambiguity or poor lens
coverage can still produce a plausible solution. No unknown tag-map estimation,
bundle adjustment, camera synchronization or moving-camera tracking is provided.

**MAMMA limitation:** computed lens distortion is preserved, never silently
zeroed. Current MAMMA requires already-undistorted videos and corresponding
intrinsics; this tool does not yet create those videos. A nonzero-distortion
result is valid calibration data but must not be imported as though its footage
were distortion-free. No existing videos or reconstruction results are changed.

Validation: `runtime/python.exe -X utf8 -B tools/test_calibration_solver.py`
tests synthetic two-camera metric recovery, lens recovery, parser compatibility,
cancellation and invalid-input guards. Physical camera accuracy is not yet tested.

## Lens calibration from video

In **Calculate Calibration**, choose **Calibrate Lens from Video**. Keep the
camera fixed and move/tilt the printed checkerboard through the image, including
its edges. Set inner corners and square size exactly as for photos. Choose a
start/end time (end=0 means end of video) and sampling interval (default 1 second).
The original resolution is preserved; source videos are read only.

At most 300 candidate frames are sampled. Shorten the range or increase the
interval for longer videos. Missing boards and near-duplicate board views are
skipped; at least 10 distinct accepted views are required. The duplicate heuristic
compares all corner positions with a 1% normalized-image-distance threshold,
including reversed ordering of a symmetric checkerboard. It does not guarantee
sufficient coverage or eliminate motion blur. Prefer sharp, varied board poses.

Stop is checked between decoding/detection operations. The saved lens JSON also
contains source filename/stamp, FPS, accepted frame indices and skipped-frame
reasons. The same measured AprilTag-map workflow then calculates camera poses.
Video does not remove the need to measure that map or undistort MAMMA footage.

Validation used a generated 25-frame MJPEG video with perspective checkerboards:
actual decoding, checkerboard detection and lens fitting passed (25 accepted
views, 0.057 px RMS), along with cancellation and offscreen dialog construction.
This is synthetic validation, not a physical lens accuracy measurement.

## Full checkerboard workflow (no AprilTag measurements)

Open **MAMMA > Camera Calibration > Calculate Calibration**.

1. Calibrate each lens from the existing photo or video option. Use the actual
   square edge (A4: 25 mm, A3: 35 mm) and 9 x 6 inner corners. Save each lens JSON.
2. Place the board flat, face up on the floor. Keep it completely stationary
   while capturing a photo from each fixed camera; use the same lens settings
   and resolution as the motion videos. Every camera must see the entire board.
   For this static setup the photos need not be simultaneous. Do not move the
   board to make it visible to another camera: use a larger board or reposition
   cameras before starting calibration instead.
3. Choose **Camera Positions from Checkerboard**. Enter the camera ID exactly
   matching the motion video filename without extension; load its board photo.
4. Review the overlaid **O**, **+X**, **-Y** arrows. Identify the same physical
   inner corner and X direction in every camera. Mark the physical reference
   outside the pattern before shooting. Use **Reverse corner order** when
   necessary. Check the orientation confirmation, load that camera's lens JSON,
   and add the camera. Repeat for every camera, then save the camera JSON.

The first inner corner is the world origin; +X follows the first row, +Y is
opposite the direction toward later rows, +Z points above the board. The board
surface defines Z=0, so board thickness offsets the actual floor by that amount.
Rectangular boards are required; square grids are rejected because they add
90-degree ordering ambiguity. User confirmation is essential: an incorrect
180-degree choice can have a low reprojection error but a wrong world frame.

The planar IPPE solver checks both pose candidates, refines them, requires
positive depth and an above-floor camera, rejects RMS above 2 px, small image
coverage (<0.5%) and materially different near-equal pose candidates. Increase
board size/visibility if rejected. This single stationary-board pose is more
sensitive to noise than multi-pose stereo bundle adjustment; physical accuracy
has not yet been validated. No automatic unknown-board-motion matching is done.
Nonzero lens distortion is retained: MAMMA still requires separately undistorted
footage and matching intrinsics. The board workflow does not remove that limit.

Tests: `tools/test_checkerboard_pose.py` checks metric pose recovery, reversed
corner handling and invalid inputs. Offscreen UI testing covered orientation
review, adding two cameras and exporting parser-compatible JSON.

## Reuse camera videos already open in MAMMA

The calibration dialog snapshots the loaded MAMMA camera list when opened.
Select a camera and choose **Calibrate Lens from Open Camera** to use that video
with the existing start/end/interval controls. No second file selection is needed.

In **Camera Positions from Checkerboard**, select the open camera, set the
**original video time**, and choose **Detect Board in Open Camera**. The camera ID
is filled automatically and locked for that snapshot. Review O/+X as before,
load its lens JSON and add the camera. Repeat for each open camera. These times
are relative to original files, not project trims or synchronization offsets.
Choose frames showing the same stationary floor-board placement; the software
does not infer synchronization or compensate for a board moved between frames.

Snapshots use independent read-only decoders, preserve original pixel resolution,
check file changes and release decoder handles on errors/cancellation. Source
filename, timestamp/size, selected frame, FPS and requested time accompany each
camera result. This does not seek the app playback or modify source videos.

Validation: generated-video snapshot decoding and corner detection, out-of-range
rejection, lens fitting, inherited camera-list controls and two-camera JSON export
passed. No physical multi-camera calibration accuracy is claimed.

## Separate calibration methods

MAMMA now exposes two independent entry points:

- **Calibration Board...**: checkerboard dimensions/square edge, lens calibration
  from photos, chosen videos or open cameras, and stationary-board camera poses.
- **AprilTag Calibration...**: tag36h11 marker size, snapshot time, detection
  photos/open-camera snapshots and **Calculate AprilTag Camera Positions...**
  using its measured-map job JSON. It has no checkerboard parameter controls.

Board dimensions, video sampling times and camera selection are remembered
separately from AprilTag size/time for the current app session. Reopening either
window restores its own settings; these are not saved to a project or persisted
across app restarts. Calibration outputs are saved separately by explicit Save.
AprilTags still need lens intrinsics; existing lens JSON files (including those
created with Calibration Board) are explicitly referenced in the AprilTag job.
This separation does not add automatic lens estimation from arbitrary tags.

Earlier references to Calculate Calibration / Detect AprilTags in this guide
refer to the two renamed entry points above. Offscreen tests verified method
control separation, independent settings across reopen, worker completion and
checkerboard camera export.
