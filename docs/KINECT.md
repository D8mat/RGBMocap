# Kinect v2 capture and depth reference

Open **File > Kinect Studio...**. This optional workspace uses the Kinect for
Xbox One / Kinect for Windows v2 sensor, powered USB 3.0 adapter and Microsoft's
Kinect SDK 2.0. It does not support Xbox 360 Kinect or Azure Kinect. No Python
package installation is needed. A small C# bridge is compiled locally using the
Windows .NET Framework compiler and the installed SDK; binaries are cached locally.

## Live capture

1. Close other applications using Kinect. Keep the sensor stationary.
2. Click **Start Kinect**. The left pane shows RGB and the right pane shows
   registered color/depth points and up to six SDK skeletons. Drag to orbit and
   use the mouse wheel to zoom. Dashed bones indicate inferred joints.
3. **Record...** chooses a parent folder and creates a new timestamped capture.
4. **Stop Recording**, then **Stop Kinect**. Use the timeline to review the capture.
   **Open Capture...** reopens an existing capture folder.

The recorder retains actual sensor timestamps, 1920 x 1080 JPEG color, 512 x 424
millimetre depth, tracked joint positions/orientations/confidence and a registered
colored point cloud sampled every fourth depth pixel. Tracking IDs distinguish
people; reacquiring a lost person can create a new ID. Invalid depth samples are
excluded from the cloud. This is a 25-joint SDK body, not detailed finger mocap.

Capture uses a worker process/thread and a latest-only preview. Recording is
streamed to disk without an unbounded frame queue. It stops below 1 GiB free space.
The effective frame rate depends on the computer, SDK and USB connection; it is
not guaranteed to be 30 fps. Raw captures can occupy tens of GiB per hour. A
`recording.pending` marker indicates an unfinished capture and prevents replay
as a completed session. Do not edit capture indexes or remove this marker manually.

## Depth anchoring of AI motion (experimental)

Choose the matching recorded Kinect person and **Choose AI motion...** to select
a prepared `*_mixamo.json` result from the same performance. Set **AI frame 0 at
capture time** in seconds; this is a manual synchronization offset, not automatic
video synchronization. Enable **Include depth-anchored AI motion**.

The export fits a proper rotation, translation and uniform scale from corresponding
tracked body joints, using the first sufficiently tracked synchronized pose.
It rejects degenerate poses, implausible scale, matches beyond 60 ms and a
calibration residual above 20 cm. **Follow Kinect pelvis** additionally anchors
the AI pelvis to the SDK pelvis on matched frames; AI relative articulation and
finger rotations are retained. Missing/low-confidence pelvis samples are marked
unavailable rather than filled from a different person. Review the result and
the calibration RMS in `ai.json`; passing the threshold does not prove identity
or accurate calibration. This is world-placement assistance, not a new neural
depth-fusion model or an improvement guarantee. It assumes a stationary Kinect
and a coherent AI coordinate system; moving-camera AI output must already be
expressed in a coherent world frame.

Original AI projects, model files and source motion are never modified. The
aligned animation is saved separately as `ai_frames.npy` and in the Blender scene.

## Blender, motion and environment export

Stop live capture, select the desired environment snapshot with the timeline,
then choose **Export Blender + FBX / BVH + Point Cloud...** and a new folder name.
The bundle contains:

- `Kinect Scene.blend`: animated SDK rigs, an optional aligned AI rig, a static
  colored depth point cloud and the Kinect RGB reference camera/video.
- `video.mp4`: a 30 fps presentation resampled from actual capture timestamps;
  this does not turn a lower-rate capture into 30 independently measured poses.
- `Kinect Motion.fbx`, when tracked bodies are available.
- Per-person BVH takes, split at tracking gaps; filenames include the start frame.
  These use Kinect joint names and six channels per joint to preserve measured
  positions and orientations. They are not the 3ds Max Biped profile.
- `environment.ply` and `environment.npz`: colored points in Kinect camera metres,
  from the selected snapshot, not a watertight mesh or a fused room scan.
- `scene.json`, alignment metadata when used, and export logs.

The Blender camera location matches the sensor origin, but the RGB lens is
currently approximate. The video background is a reference, not a calibrated
pixel-perfect RGB overlay. Kinect depth-to-color registration for point colors
uses the SDK coordinate mapper. Blender converts camera Y-up coordinates to Z-up.
The point cloud is displayed/rendered as points through Geometry Nodes.
Tracking gaps are hidden in Blender and split in BVH; FBX consumers may ignore
visibility animation, so use the split BVH takes when gaps matter. Bone lengths
can vary with SDK estimates. Blender rigs use measured joint directions; raw SDK
joint orientations remain available in the capture and BVH.

Exports run in a background worker and support cancellation. An interrupted or
failed export is retained in its uniquely named staging folder with `ERROR.txt`,
never published over an existing output folder. Completed bundles use relative
video paths. Optional AI export reads only the explicitly selected source files.

## Validation on this installation

On 2026-11-03, actual Kinect v2 RGB/depth/body capture produced 96 recorded frames
with one body ID. After optimizing depth serialization, a short second live test
measured approximately 15 fps. The recorded capture exported to Blender/FBX/BVH
and a color point cloud. Seventy-five Blender joint positions were checked against
the capture at three times within 0.1 mm. A controlled synthetic transform test
verified AI scale/rotation/pelvis anchoring and rejected degenerate calibration.
Real AI/Kinect fusion accuracy, six-person capture and long-session endurance
remain unverified.
