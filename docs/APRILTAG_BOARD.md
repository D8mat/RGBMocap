# AprilTag board: calibration from videos

Open **AprilTag Calibration > AprilTag Board - Calibrate from Videos...**.
This workflow is independent of the checkerboard and the free-marker job-JSON solver.
It uses the rigid RGBMoCap 3x2 board, not six independently placed tags.

## Board

- AprilTag 36h11 IDs 0,1,2 across the top; 3,4,5 across the bottom.
- Black tag edges: 150 mm. Gap between black edges: 30 mm. Outer white margin: 30 mm.
- Printed rectangle: 570 x 390 mm. Print/mount flat, at actual size.
- Origin: center of printed face; +X right, +Y toward the top row, +Z out of the face.
- Machine-readable layout: [apriltag_board_3x2.json](samples/apriltag_board_3x2.json).

## Recording / Unreal test

Keep every camera fixed. Lock focal length/zoom, focus, crop, resolution and stabilization.
Move the rigid board for 20-30 seconds: cover the image, move closer/farther, tilt
around BOTH horizontal and vertical axes (roughly +/-20-40 degrees). Aim for 20-30
sharp distinct views per camera; at least three whole tags must remain visible in
an accepted frame. Keep other copies of IDs 0-5 out of view. Translations alone
cannot constrain lens calibration reliably.

Finally place the board face up on the floor and keep it stationary for several
seconds, visible in all cameras. Record synchronized camera videos. The floor
snapshot defines the coordinate system; it does not recover an external Unreal
world origin. The printed surface is Z=0 (a rigid board's thickness raises that
plane slightly above the physical floor).

## Calculate

1. Load the camera videos in MAMMA, then open the new dialog. **Use Open Cameras**
   copies their camera IDs. Alternatively **Choose Camera Videos...** uses file
   basenames as camera IDs; these must match the eventual MAMMA camera IDs.
2. Select a camera row. If needed, **Set Lens Video...** assigns a separate
   calibration clip; **Set Floor Video...** assigns its common floor/reference clip.
3. Set lens start/end to the moving-board interval. End=0 means end of video.
   Sample every 0.5-1 second is a useful starting point. At most 300 samples per run.
4. Choose **Real camera: estimate lens distortion**, or explicitly choose
   **Unreal / known pinhole: no distortion** for renders without lens distortion.
   The pinhole option is not a workaround for real lens distortion.
5. **Calibrate Selected Lens**, repeat for every camera. Lens results can be saved
   and loaded for reuse with the identical camera/lens settings and resolution.
6. Set **Common floor snapshot time** in the original videos. Confirm the board is
   stationary, face up and the clips synchronized. **Calculate Camera Calibration**.
7. Review errors and **Save Camera Calibration...**. Nothing is applied to an
   active reconstruction automatically. Import the saved calibration through MAMMA.

Real-lens output includes distortion. The current MAMMA importer requires
undistorted videos and matching intrinsics before importing nonzero-distortion
calibration. This dialog does not yet create those undistorted videos. Never clear
the distortion coefficients to bypass that requirement. Pinhole renders can be
imported directly.

## Validation and limits

The solver matches tag IDs to metric corners, fits variable visible point sets,
reserves every fifth accepted view for validation, checks image coverage, focal
uncertainty and tilt diversity. Minimum 12 accepted views; fitted RMS <=2 px,
held-out RMS <=2.5 px. Floor pose checks planar ambiguity, front-side visibility,
RMS <=2 px and maximum corner error <=4 px. Small RMS does not guarantee accuracy;
check a known distance and camera arrangement after import.

Video decoding runs off the UI thread, one frame at a time, with a maximum of 300
samples. Stop/close cancels between frames and OpenCV operations. Input videos and
existing reconstruction data are not modified. Changing the lens model invalidates
computed lenses. Lens calculations from separate clips require unchanged cameras.

Tests cover metric recovery, real distortion, board texture corner ordering,
invalid/degenerate inputs, two synthetic compressed videos through detection and
camera export, and asynchronous UI completion/cancellation. Physical camera
accuracy and a direct Unreal import have not yet been measured.
