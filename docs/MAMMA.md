# MAMMA multi-view reconstruction

MAMMA reconstructs SMPL-X human motion from multiple camera views, including
multi-person interactions. The [official input guide](https://github.com/cuevhv/mamma/blob/main/docs/YOUR-DATA.md)
organizes footage and calibration by camera; it does not define four cameras as
the maximum. The [official dataset documentation](https://github.com/cuevhv/mamma/blob/main/docs/DATASETS.md)
includes 16- and 32-camera captures as well as four-iPhone captures. The four-view
example is not a general model limit.

**RGB MoCap implementation:** reconstruction accepts two or more synchronized
fixed-camera videos without a fixed camera-count maximum. Available RAM and
VRAM determine practical capacity. Existing system-memory monitoring, CUDA
allocator budgeting and OOM failure handling remain active; removing the input
count limit does not guarantee that every camera/clip combination fits memory.
Four columns in the thumbnail strip are only a layout choice; more views add
rows. Input/preparation/save/restore was checked with up to 65 tiny synthetic
camera files, not 65-view AI fitting. Full model inference has only the scope
recorded in [validation](VALIDATION.md); official multi-camera datasets are not
evidence that every such workload fits this adapter or a particular GPU.
See [all model guides](models/README.md) for the other capture workflows.

It has a separate optional runtime; ExMR, FlowHMR and GVHMR keep their existing
environments. Run the normal `install.cmd` first.

## Before installing

Obtain the official [MAMMA code and registered assets](https://mamma.is.tue.mpg.de/)
under its own license. The MAMMA software, weights, dataset and locked-head
SMPL-X body data have separate restricted-use terms. They are not redistributed
in this repository; the root MIT license covers the original connector only.
This integration does not grant commercial-use or redistribution permission.

Use 64-bit Windows and a compatible NVIDIA GPU/driver. This path builds native
CUDA components locally. Install the following tools separately before running
the MAMMA setup:

- **Visual Studio 2022 Build Tools**, with **Desktop development with C++** and
  a Windows SDK. Select the 2022 release.
- **[NVIDIA CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)**.
  The NVIDIA driver alone does not include the CUDA compiler.

Compiler tools add disk usage beyond the runtime and may require administrator
installation. The app and its runtime are installed in the existing RGB MoCap
folder. Only one Windows/NVIDIA PC has been tested; a clean Windows installation
and other GPUs have not been verified.

| Item | Storage |
| --- | ---: |
| Installed MAMMA runtime on the test PC | 6.19 GiB |
| Model files, when the shared VGGT-Omega model is already installed | 3.42 GiB additional |
| Model files including VGGT-Omega | 7.69 GiB |
| Free space needed during runtime setup, before models and compiler tools | About 18 GiB |

The setup downloads approximately 3.57 GiB of package/source archives, excluding
Python and models. Retained downloads, videos and generated projects use extra
space. These figures are separate from the base installation's storage estimate.

## Install the runtime

1. Complete the [base installation](USER_GUIDE.md#install) with `install.cmd`,
   then install the compiler tools above.
2. Save your project and close RGB MoCap. In its folder, run:

   ```cmd
   Setup_MAMMA.cmd
   ```

3. Read and accept the upstream license prompt if its terms permit your use.
   Setup downloads hash-pinned packages and official source, compiles SAM 2/SDF
   for this PC's GPU, tests actual CUDA kernels and checks installed package
   hashes before enabling the runtime. Provider passwords are not requested.
4. Run `RGBMoCap.cmd`, select **MAMMA** in **AI Motion Model**, and open
   **Model Manager**. **Runtime Setup...**
   shows readiness, prerequisites and a command you can copy. Prepare models
   separately as described below.

Keep the setup window open until it reports completion. **Runtime Setup...**
shows instructions and can copy the command; it does not run installation inside
the app. You do not need to install Python packages by hand. The installer uses
its separate pinned Python 3.11/PyTorch CUDA environment without replacing the
other model runtimes.

MAMMA and pytorch_sdf prohibit ordinary redistribution under their provider
terms. Consequently, their source/native modules are downloaded and built on
the user's machine rather than shipped as prebuilt files by RGB MoCap.

## Prepare model files

Select **MAMMA** before opening **Model Manager**. Use automatic download for
public files. For registered files, choose **Manual Download...**, register or
sign in on the provider's site and accept its terms, then use **Import...** in
that window or in the model list. The app does not ask for provider passwords.

The manager lists nine assets. The paths below are relative to the app's
`models/` folder; importing places the file in its expected location.

| Asset | How to obtain it | Destination inside `models/` |
| --- | --- | --- |
| MAMMA checkpoint | [MAMMA website](https://mamma.is.tue.mpg.de/), manual download | `mamma/mamma_mask_full_cvpr.ckpt` |
| Surface landmarks | MAMMA website, manual download | `mamma/verts_512.pkl` |
| Locked-head SMPL-X Neutral, Male and Female (three files) | [SMPL-X website](https://smpl-x.is.tue.mpg.de/), manual download | `mamma/smplx/SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` |
| SAM 2.1 Hiera Large | Automatic download | `mamma/sam2.1_hiera_large.pt` |
| YOLO12x | Automatic download | `mamma/yolo12x.pt` |
| OpenCLIP ViT-B/32 | Automatic download | `mamma/open_clip_model.safetensors` |
| VGGT-Omega 512 | Model Manager's manual download instructions; shared with FlowHMR | `flowhmr/vggt_omega_1b_512.pt` |

For the MAMMA checkpoint and surface landmarks, open the
[official MAMMA website](https://mamma.is.tue.mpg.de/) and click **Data**.
Register if needed, review and accept the provider terms, then sign in to access
the download page. Download `mamma_mask_full_cvpr.ckpt` and `verts_512.pkl`.
Return to Model Manager and use **Import** in each matching row, or directly
from its **Manual Download** window. The app does not need your password or token.

On the separate SMPL-X site, choose **"SMPL-X with removed head bun (NPZ, 392 MB)"**.
Extract the archive and import each of its three NPZ files. The ordinary SMPL-X
v1.1 files used by other models cannot replace these locked-head files.
Supported files are checked against expected SHA-256 hashes. If an automatic
download fails, use the row's manual download/import actions. With compatible
**Manual Camera Data**, the VGGT weight is not required; the other eight assets
are still needed.

## Prepare your videos

- Use two or more fixed-camera views of the **same performance**, with overlapping views
  of the people. Unrelated videos are not multi-view input.
- Synchronize their starting times before opening them. The app does not align
  audio or provide separate camera start offsets.
- Use distinct, short filenames such as `camera_01.mp4` and `camera_02.mp4`.
  Filenames without extensions identify cameras, including in manual calibration.
- Ensure the intended people are visible in the first camera, which initializes
  their identities. A view in which a person is completely hidden supplies little
  useful evidence for that person.

The app uses a common 30 fps timeline, limited by the shortest input. The current
adapter accepts 4–1,800 common frames (up to 60 seconds at 30 fps) and at most
eight people. Start with a short trimmed section to check the cameras and masks
before processing a long clip. These are adapter limits, not a guarantee that
every clip fits memory; the full 60-second limit has not been validated.

## Reconstruct and export

1. Select **MAMMA** and use **File > Open Video** to select two or more
   synchronized, fixed-camera videos of the same performance.
2. Click camera thumbnails to change the main preview. Up to four thumbnails
   fit per row. One common 30 fps timeline and trim applies to every video;
   duration is limited by the shortest input. Drag the ends of the timeline's
   **Video** track to trim all views together without modifying the originals.
3. Keep **VGGT (Automatic)** or use **Import Camera Data...**. **People: 0**
   automatically detects the count; specify 1–8 if someone is missed or background people interfere.
   MAMMA finds people during reconstruction, so **Find person** is hidden.
   Enter only the number in **Reference Height (m)**, for example `1.70` for
   170 cm. It sets approximate VGGT
   scale using initially standing people;
   it is an assumption, not a measured height or a reliable gravity estimate for
   an initially crouching scene.
4. **Reconstruct 3D** prepares aspect-preserving inputs within 1920×1080, creates
   SAM 2.1 masks, predicts FP32 landmarks, estimates/loads cameras, runs the full
   MAMMA body fit, then prepares native people, correction and export bones.
   Mask fragments below eight pixels are removed when a substantial component
   exists; soft visibility weights reduce the influence of occluded views.
5. Review the people together in one shared world. Check/uncheck **People** rows
   to show or hide them; select a row to edit that person's contacts.
   **Enable Correction** is on
   by default; turning it off selects the preserved original motion. Edit contact
   timing and use **Apply Correction** as for the other models. These operations
   do not need new MAMMA inference. Foot correction is on and hand correction is
   off by default; enable **Hand** when palm support is needed. The viewport
   **Lock** checkbox only shows or hides contact markers.
6. Use **File > Export FBX** for the displayed, applied motion. It uses the
   existing 52-bone source-export convention. Use **File > Save Project As** to
   keep a named copy; reopen its `project.json` with **File > Open Project**.

Resource monitoring remains active. Fitting can take several minutes even for
a short clip; an animated progress indicator is used when the stage cannot
report a meaningful percentage. The app uses SAM 2.1 and FP32 fitting; SAM 3.1
standalone tests remain separate. No quantization, TF32, iteration reduction or
physics simulation is enabled in this path.

Low Memory does not select smaller MAMMA camera sets, fewer fitting iterations
or a lower-precision model. The shared allocator budget and system-memory guard
still apply; shorten the common clip or reduce input views yourself if needed.

## Manual cameras

Selecting Manual Camera Data disables the reference-height control: metric
calibration supplies scale. With VGGT, Reference Height accepts 0.50–2.50 m
(default 1.70 m); it is an approximate scene-scale assumption, not a measurement.
Reconstruct after changing camera, count or height settings; Apply Correction
does not re-estimate cameras or poses.

### JSON example and required fields

Open [mamma-camera-example.json](samples/mamma-camera-example.json) for a complete,
importable two-camera example. It is an original RGB MoCap **synthetic format
example**, covered by this repository's MIT license. Its numbers are not measured
calibration for MAMMA's videos or your own footage. Changing camera names alone
does not calibrate a video; use VGGT when you do not have matching calibration.

| Field | Required structure and meaning |
| --- | --- |
| `units` | `"metres"`; translations use metres. |
| `extrinsics_convention` | `"world_to_camera"`. |
| `cameras` | Object containing one entry for every opened video. Each key must exactly match its filename without the extension: `A001.mp4` → `A001`. Remove unused camera entries when opening only a subset of views. |
| `intrinsic_matrix` | 3×3 matrix `[[fx,0,cx],[0,fy,cy],[0,0,1]]`. Focal lengths `fx,fy` and optical center `cx,cy` are in pixels for the original input resolution. |
| `extrinsics_matrix` | 3×4 matrix `[R | t]`, or 4×4 with last row `[0,0,0,1]`. `R` is a proper rotation; `t` is a translation in metres. Maps a world point to camera coordinates: `p_camera = R @ p_world + t`. |
| `image_size` | `[width,height]` in pixels, matching the original opened video, not the app's resized working copy. |
| `distortion_coeffs` | Optional `[k1,k2,p1,p2]` or `[k1,k2,p1,p2,k3]`. Omitted means zero. The current app adapter requires all zeros and undistorted pinhole footage. |

The example has `A001` and `B001` entries at 1920×1080. It describes two hypothetical
cameras at `[0,-4,1.6]` and `[4,0,1.6]` metres, looking toward the world origin.
Camera position is **not** the last column of the extrinsic matrix:
`camera_position_world = -R.T @ t`.

Use a **Z-up world** (ground at world Z=0) and OpenCV-style camera axes: X right,
Y down, Z forward. Zero intrinsic skew and zero lens distortion are required;
unsupported calibration is rejected rather than silently discarded. Intrinsics
are explicitly resized to the working input dimensions. Import is limited to
1 MiB per calibration file.

Open the matching videos, choose **Manual Camera Data**, click **Import Camera
Data...**, select the JSON, then **Reconstruct 3D**. A camera-name or resolution
mismatch is rejected. The app cannot confirm that supplied calibration measures
the correct real camera arrangement. VGGT ignores supplied manual calibration.

### Official MAMMA sample calibration

The official four-camera `pushing_and_lifting_from_ground` demo pairs its
`A001.mp4`, `B001.mp4`, `C001.mp4` and `D001.mp4` videos with
[iphones_outdoors.yaml](https://github.com/cuevhv/mamma/blob/588492f18876e2ed6888b2d26929047cb6b575e7/configs/examples/calib/iphones_outdoors.yaml).
This is **YAML**, not a camera JSON. Obtain it directly from MAMMA under its
[upstream license](https://github.com/cuevhv/mamma/blob/588492f18876e2ed6888b2d26929047cb6b575e7/LICENSE);
RGB MoCap does not redistribute the sample's original or converted calibration.
Use the corresponding original-resolution videos; if you independently resize
them, scale the calibration's focal lengths and optical center as well as its
resolution. The official sample specifies 3840×2160.

For MAMMA YAML, each entry under `cameras` has `intrinsics: [fx,fy,cx,cy]`,
`resolution: [width,height]`, `translation` (camera position in the world), and
`rotation_quaternion` (camera-to-world, Hamilton order `[w,x,y,z]`). The app
converts this to world-to-camera matrices. JSON uses the matrix convention above.
MAMMA's `configs/examples/captures/*.json` files describe input datasets and
processing paths; they are **not camera calibration files** for this button.

## Saved data and limitations

**Save Project As** copies all camera videos and project-private body data.
Keep the whole project folder together. MAMMA data lives under
`<project>/mamma/inference/<generation>/` (working videos, masks, landmarks,
cameras, raw fitted parameters) and `mamma/results/<generation>/` (native
people/takes, contacts, previews and exports). Deleting these folders removes
saved reconstruction work, not just disposable thumbnails.

Prepared MAMMA people can also be used as character-retargeting sources. The
retargeter reads the actor's project-private body, not the ordinary SMPL-X body
used by ExMR/GVHMR. That extra standard body is not required for MAMMA-only
retargeting. See [character import, retargeting and export](USER_GUIDE.md#character-retargeting).

Each actor retains all sixteen shape coefficients exactly: the final six are
folded into its private body template for the editor's ten-slot skin format.
Animated poses, joint centers and hand means are preserved. Original raw motion
is retained, while correction is saved separately. Body-derived files remain
restricted registered data and must not be published with source code.

Verified inference outputs are reused only with matching source, model, engine
and connector identities plus output receipts. Interrupted attempts retain a
pending generation; a retry cannot publish mixed outputs. Camera/person changes
require reconstruction and leave the previous displayed motion in place.

Occlusion, segmentation/identity mistakes, weak camera baselines and uncertain
scale can still produce poor poses. Contact correction reduces sliding but does
not repair lost identities, bad camera geometry or missing observations. This
integration was tested on one Windows/NVIDIA PC. A three-view, two-person manual
test processed 413 frames at 30 fps through correction/preview in **712.08 s**:
masks 257.80 s, landmarks 89.72 s, cameras 14.42 s, fitting 294.42 s and final
correction/preview 31.97 s; the remainder was video preparation/startup.
No fatal traceback or OOM appeared in that log. This is one observation, not a
speed guarantee. Clean Windows, other GPUs and the full 60-second limit have
not been validated.

## Updates and troubleshooting

- **Runtime missing or out of date:** save your work, close the app and run
  `Setup_MAMMA.cmd`. `update.cmd` maintains the base runtimes; it does not install
  this optional runtime. Re-run the MAMMA setup when its dependency notice asks
  you to, or after changing GPU architecture.
- **Compilation or download failure:** check the setup message, free space,
  compiler prerequisites and `.setup/logs/mamma-*.log`, then retry. Verified
  downloads can be reused. A failed setup retains the previously enabled runtime,
  models and projects. Do not delete active `runtime_mamma/` or `.setup/` state.
- **Models not ready:** use the exact supported file and **Import...**. Recheck
  that the SMPL-X archive is the locked-head version and that all three bodies
  were imported. Manual camera mode can run without the VGGT weight.
- **Poor poses or jitter:** inspect camera overlap, synchronization, identity
  and masks first. Try a clearer first camera or an explicit **People** count.
  Camera, people-count or reference-height changes require **Reconstruct 3D**;
  **Apply Correction** only updates contact correction.

The verified runtime is stored in an immutable generation under
`runtime_mamma/`, referenced by `.setup/mamma-runtime.json`. It is configured for
this installation's paths; copying that folder alone is not a supported setup
method. For development environments and detailed checks, see
[the agent installation/native-build runbook](MAMMA_AGENT_SETUP.md),
[Build from source](BUILD.md) and [Validation](VALIDATION.md).


## Local integration notes

The MAMMA camera worker includes the current upstream automatic memory routing,
Low Memory block offload and fast camera meta/assign loader. Other existing model
workers keep this fork's current policies. The original guide above describes
upstream usage; recorded upstream validation is not a test of this local port.
Our Blender bundles use the primary calibrated camera and its source video.
