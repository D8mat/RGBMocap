# RGB MoCap user guide

[Overview](../README.md) · [Build from source](BUILD.md)

A native Windows app for turning video into human motion, reviewing the body mesh
and animation bones, correcting hand/foot contacts, retargeting motion to humanoid
characters, and exporting animation for Blender or MMD. Processing runs locally
in an English interface.

## Character retargeting

Transfer a reconstructed person's motion to a character without running AI
inference again. Supported targets are **Mixamo FBX**, **VRM** and **PMX** humanoids.

1. Prepare the source person's motion and apply any contact edits.
2. Choose **File > Import Character...** and select the character file. It appears
   at the origin in its rest pose, below the source people in the **People** list.
3. Right-click the character and choose **Retarget > [source person]**. The status
   bar shows processing progress. Play or scrub the timeline to review the result.
4. Right-click the retargeted character and choose **Export Character...**, or
   select its row and use that command in **File**.

Up to **8 characters** can be imported per project. Each character can use a
different source person. Check or uncheck its row to show or hide it; use
**Remove Character** in its right-click menu to remove it from the project.
Character data and assigned motion are saved inside the project.

Characters are drawn in front of the SMPL source meshes, with **Bone Overlay**
above both. **Name** shows the list name above each visible person or character,
including characters still in their rest pose. Names stay upright and the same
screen size; crowded labels are separated. These controls change the preview only.

Retargeting uses the selected source motion and its applied contact timings.
**Apply Correction** also refreshes characters linked to the corrected source,
including hidden characters. After other source changes, run **Retarget** again.
A character marked **Needs retarget** must be updated before export.
Unreachable hand/foot targets
are not solved by stretching bones. Large differences in body proportions may
require further editing in animation software. Travel distance can differ from
the source because motion is scaled to the target character's leg length;
shorter-legged characters may cover less ground, and longer-legged characters
may cover more. Source contact timings are retained.

Only mapped body/finger bones and necessary body helpers receive animation;
the armature object stays unkeyed.
Hair, clothing and secondary physics are not simulated.

| Export option | Result |
| --- | --- |
| **FBX + Include character mesh** | Character mesh, skeleton and animation. |
| **FBX, mesh unchecked** | Skeleton and animation only. |
| **VMD (PMX characters only)** | Body/finger motion sampled at 30 fps for the same PMX model in MMD. Facial animation and physics are not generated. |

Separate takes produce separate files. Both FBX options preserve the imported
character's original bone hierarchy and rest pose separately from its animation;
the first animation frame does not replace the rest pose. Existing retargeted
characters can be exported again to use the corrected FBX rest-pose handling.

Known humanoid mappings are automatic; custom bone mapping is not yet available.
Files missing required body bones are rejected with a message. Viewport materials
use a simplified color preview.

### Included Blender runtime

Character-enabled packages include a separate **Blender 4.5 LTS** runtime in
`runtime_character`. It runs in the background to import characters, evaluate rig
constraints, bake bone animation and export character FBX/VMD files. You do not
need to install or open Blender separately when using these packages. Keep this
folder with the app.

RGB MoCap performs the retargeting/contact calculations and viewport playback.
The Blender backend also provides a foundation for future format support and
animation-processing features; these are possibilities, not additional features
currently provided by the app. Hair/clothing physics remains unsupported.

The Git checkout does not include Blender binaries. `install.cmd` downloads the
pinned Blender runtime automatically. Development runs may set
`RGBMOCAP_BLENDER` to a compatible `blender.exe`; see [Build from source](BUILD.md).
Blender retains its GPL license and bundled notices; see [Third-party notices](../THIRD_PARTY_NOTICES.md).

## Install

**Tested configuration:** NVIDIA GeForce RTX 5090 (32 GB VRAM), AMD Ryzen 9
9950X, 96 GB RAM, Windows 11 Pro (64-bit). This is the tested configuration,
not a minimum specification. Other GPUs, drivers and clean Windows installations
have not yet been verified; compatibility reports are welcome.

1. Install [Git for Windows](https://git-scm.com/download/win). Open **Command
   Prompt** in a writable folder, then run:

   ```cmd
   git clone https://github.com/toyxyz/toyxyz_RGBMocap.git
   cd toyxyz_RGBMocap
   ```

   Avoid protected folders such as Program Files. The commands below are for
   Command Prompt; in PowerShell, prefix each `.cmd` filename with `.\`.
2. Run `install.cmd` with an internet connection. It downloads and verifies
   pinned Python environments, Blender 4.5.0, FFmpeg 7.1.1 and small upstream
   support files, and installs the included prebuilt motion-correction module.
   The AI runtimes remain separate. Downloads and free disk space are needed;
   the install can take time.
3. Run `RGBMoCap.cmd`. Keep its console open while the GUI runs; launch errors
   and exit codes are displayed there. Processing messages and app **Console** output
   are also sent there, but very long or high-volume output can be truncated or
   dropped. Use **Help > Check System** if needed.
4. In **Model Manager**, download or import the selected AI model files. Some
   models require registration and agreement to their provider's terms and must
   be downloaded manually.

To update, close RGB MoCap and run `update.cmd` in the repository folder. It
refuses to overwrite tracked local edits, requires the `main` branch, pulls only
fast-forward changes and checks the runtimes. Back up or commit local changes
before updating. A direct `git pull` updates source only; close the app first,
and run `install.cmd` if dependencies change. The launcher will ask for setup if
it detects an incomplete update or a dependency mismatch.

If a download or installation fails, read the console message, check the internet
connection and free disk space, and rerun `install.cmd`. A completed download may
be reused. Do not delete `models`, `projects` or the remaining `.setup` state to
retry. If Git reports a conflict or local changes, resolve or back them up before
running `update.cmd` again.

On the tested PC, a separate-folder first install took about 10 min 32 sec with
previously verified package downloads reused; Blender and FFmpeg were downloaded.
A subsequent code update and runtime verification took about 1 min 55 sec. These
are single local tests, not estimates for a download with an empty cache or for
other PCs; connection speed and storage performance affect the time.

Allow at least **32.4 GiB free** for first-time runtime installation before
models. All supported model files previously measured about 29 GiB. Download
caches, videos, projects, characters and exports require additional space, and
actual sizes can vary. No clean-Windows or other-GPU test is claimed.

## Storage and cleanup

In the installed app, `<install folder>` means the folder containing
`RGBMoCap.cmd`. Close RGB MoCap and any installation or update before deleting files. Use **File →
Open Project Folder** to locate the current project; a **Save Project As** copy
may be outside the installation folder.

| Location | What you can remove and what happens |
| --- | --- |
| `<install folder>\cache\` | Its contents can be deleted. This includes `huggingface`, `torch` and ExMR support caches. They are recreated or downloaded again when needed; the next run may take longer and require internet access. Saved projects and the separate `models` library remain. |
| `<install folder>\projects\<project folder>\` | Delete an entire unwanted project folder. This removes its model results, prepared videos, previews, contact edits and imported/retargeted characters. Reprocessing requires the source video and AI inference; manual edits are lost. Keep projects you still need. |
| Your **Save Project As** destination | The same rule applies to each saved project folder at your chosen location. It contains a copy of the video and project data. Deleting it removes that copy; cleaning the default `projects` folder does not clean external copies. |
| `<install folder>\character_scenes\<session folder>\` | Contains characters imported before a project was opened. Old session folders can be deleted once those imports are no longer needed or have been copied into a saved project. Unsaved session imports are lost. |
| `<install folder>\.setup\downloads\`, `.setup\uv-cache\` and `.setup\tmp\` | Downloaded wheels and temporary cache can be removed only after installation or update completes. They are reused for repair and updates; deleting them may require downloading files again. Keep the rest of `.setup`, including the private Python and installation state. |

Repeated work mainly grows project data: each model has its own `exmr`, `flowhmr`
or `gvhmr` folder, different processing ranges can add `clips` subfolders, and
character imports/retargeting add character data. These are saved work, not just
disposable caches. Do not delete individual model folders, `characters`, motion
tensors, preview files or receipt files from a project you want to keep; its
metadata can still reference them. Export or back up needed work, then remove an
unwanted project as a whole. Exported FBX/VMD files in other folders must be
managed separately.

Keep `models`, `runtime*`, `_internal`, `mocap`, `third_party`, bundled assets and
the remaining `.setup` repair files. Deleting these can remove required model
files or break the app. Downloaded model archives stored elsewhere can be removed after a successful installation/import if you do not
need an offline copy; keep a backup of manually acquired restricted models.

## Start here

1. **File → Open Video**, or drop a video into the workspace.
2. Optionally trim the timeline's **Video** track by dragging its ends.
3. Choose an **AI Motion Model**, then use **Model Manager** to prepare its files.
4. For several people, use **Find person** and check who to reconstruct.
5. Click **Reconstruct 3D**. Review the video, mesh and **Bone Overlay** together.
6. Adjust correction settings or contact timings, then click **Apply Correction**.
7. **File → Export FBX** saves the displayed, applied animation.

To animate a character, follow [Character retargeting](#character-retargeting),
then use **Export Character...** for FBX or, for PMX, VMD.

**Open Project** restores saved results and settings. **Open Project Folder** opens
their location. **Save Project As** (`Ctrl+Shift+S`) creates a new named project
folder at your chosen location and switches to that copy. It includes the video,
motion, people and contact edits for every model. Open its `project.json` to return
to it. Each project manages `exmr`, `gvhmr` and `flowhmr` folders. Selecting a model
loads its saved motion; if it has no result, **Reconstruct 3D** creates its data in
that folder. Result selections, contact edits and settings stay separate by model.
New processing ranges stay under that model's `clips` folder. The previous project
folder format is not migrated; open the original video to create a new project.
Model weights remain in the app's model
library. Trimming does not modify the original video. Processing uses a
30 fps working clip; camera sampling does not reduce its motion frame count.

## Choose a model

These quality comparisons are **subjective observations from our test videos**,
not accuracy scores or a universal ranking. See the in-app **Model Guide** for
recorded speed and memory measurements. New sessions start with **ExMR** selected;
opening a project restores its saved model. **Prepare Preview**, below
**Reconstruct 3D**, prepares existing motion without running inference again.

| Model | Observed strengths | Finger animation |
| --- | --- | --- |
| **ExMR** | Generally fast, with good overall motion quality. | Included. |
| **FlowHMR** | Slow, but the best overall motion quality in our tests. | Included. |
| **GVHMR** | Lowest overall quality in our tests, but sometimes better for unusual poses such as backflips. | Optional **HandFlow**; not inferred by GVHMR itself. |

All three support people selection, contact correction, a 52-bone overlay and
FBX export. The viewport shows the selected source mesh: SMPL-X for ExMR/GVHMR,
SMPL-H for FlowHMR. The export skeleton follows that motion; source estimation
errors can remain.

## Recorded processing time and VRAM

Historical measurements on the tested PC above (30 fps inputs, October 2026),
recorded before the current Git-based installation was tested.
Each duration uses the same input across all three models.

| Video length | Model | Processing time | Peak VRAM (GiB) |
| --- | --- | ---: | ---: |
| 7 seconds | ExMR | 1m 08s | 11.32 |
| 7 seconds | FlowHMR | 2m 49s | 14.89 |
| 7 seconds | GVHMR | 2m 01s | 5.08 |
| 15 seconds | ExMR | 1m 47s | 11.32 |
| 15 seconds | FlowHMR | 5m 16s | 10.79 |
| 15 seconds | GVHMR | 2m 45s | 5.08 |
| 25.7 seconds | ExMR | 1m 44s | 11.32 |
| 25.7 seconds | FlowHMR | 7m 32s | 10.67 |
| 25.7 seconds | GVHMR | 3m 00s | 5.08 |

Times cover the recorded model-processing stages, excluding shared person
tracking, app contact correction and preview/export. GVHMR includes HTD, FootMR
and HandFlow. FlowHMR includes camera estimation and the original SAM path;
these timings predate Fast SAM. VRAM is peak PyTorch **allocated** memory across
workers, not total GPU usage or reserved memory. These are individual test runs,
not minimum hardware requirements or predictions for every video. Resolution,
camera motion, settings and available memory also affect results; a longer
video does not always use more VRAM or take proportionally longer.

## Prepare models

**Model Manager** lists files for the selected tool and its add-ons.

| Action | Meaning |
| --- | --- |
| **Auto Download / Download All** | Download supported missing public files; completed files are kept if you cancel or retry. |
| **Manual Download...** | Show the exact file, provider link and download instructions. Use for files requiring registration or access approval. |
| **Import...** | Validate and register a file you already downloaded. |

Automatic downloads can fail, especially for files hosted on **Google Drive**.
If a download fails, use **Manual Download...** on that model's row, open the
provided link in your browser and download the exact file listed in the
instructions. Use **Import...** directly in the manual download window, or on
the same model row, to select the downloaded file. The instructions can stay
open while you download and import. If it comes in an archive, extract the
required file first. Import checks
the file and places it in the correct model folder; check that the row is ready
before reconstructing.

Some models **do not support automatic download** because their providers
require account registration, access approval and/or acceptance of license terms
before downloading. This is separate from a temporary download failure. Use
**Manual Download...** to visit the provider's site, complete its requirements
in your browser, download the required file, then use **Import...** on the
matching row. The app cannot complete these steps on your behalf.

Registered body/hand models must be obtained under their provider's terms.
Follow the manager's exact file instructions; model families are not interchangeable.
The app does not need your provider password or access token.

| Used by | Registered or access-controlled files |
| --- | --- |
| ExMR / GVHMR | [SMPL-X v1.1 neutral](https://smpl-x.is.tue.mpg.de/download.php): import `models/smplx/SMPLX_NEUTRAL.npz` from the downloaded archive. The app stores it under `models/body_models/smplx/`. |
| FlowHMR | [AMASS SMPL-H neutral](https://mano.is.tue.mpg.de/download.php): Extended SMPL+H model, `smplh/neutral/model.npz` (16 shape components). |
| ExMR / GVHMR HandFlow | [MANO](https://mano.is.tue.mpg.de/download.php): `MANO_RIGHT.pkl` from `mano_v1_2.zip`. |
| ExMR / FlowHMR | [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3): supported `model.ckpt`, `model_config.yaml` and `assets/mhr_model.pt`, each in its matching row. |
| FlowHMR moving camera | [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega): supported `vggt_omega_1b_512.pt`. Not needed in Static Camera mode. |

### Reuse models in a new installation

After installing the app in a new location, close both copies of the app and
copy the **entire `models` folder** from the old installation into the new
installation folder, beside the new **RGBMoCap.cmd**:

```text
Old installation/models/  →  New installation/models/
```

For example, copy `D:\RGBMoCap-Old\models` into `E:\RGBMoCap` so the result is
`E:\RGBMoCap\models`, not `E:\RGBMoCap\models\models`. Keep all subfolders,
configuration files, `.receipt.json` files and license files together. Copy
first and keep the old files until the new installation works.

Open the new app and check **Model Manager** for each model you use. If a copied
file is not marked ready, use **Import...** on its row and select that file
from the copied folder to validate it again. Copying can change file timestamps,
and newer releases may require a different model version; follow the row's
**Manual Download...** instructions if validation fails.

Let `install.cmd` create the new runtime folders; do not copy the old `runtime*`
folders over them. The `models` folder contains model files only. Saved work
is in separate project folders, which can be copied separately and opened
with **File → Open Project**.

## Model and processing settings

Changes to camera, smoothing or feature settings take effect on **Reconstruct 3D**.
GVHMR refinement checkboxes also select saved preview/export results. If the
selected result is missing, reconstruct it; the app reports it as unavailable.

| Setting | Function and when to use it |
| --- | --- |
| **ExMR → Camera** | Static Camera for a camera that stays still without zooming; Moving Camera / SimpleVO when it moves. |
| **ExMR → Smooth Motion** | Reduce small changes between frames. Off keeps the unsmoothed result. |
| **FlowHMR → Fast SAM** | Speed up body/hand image processing with the same model files. Small numerical differences can affect motion and automatic contacts. |
| **FlowHMR → Camera / Sampling** | Start with Moving / Unknown - Auto for moving or uncertain cameras. Static Camera skips camera estimation for locked footage. Explicit frame intervals control camera analysis only; denser sampling uses more memory. |
| **GVHMR → HTD Refine** | Improve body-motion consistency over time. Runs before FootMR when both are enabled. |
| **GVHMR → FootMR** | Refine ankle angles from foot observations. Separate from foot contact locking. |
| **GVHMR → HandFlow** | Add finger tracking while preserving the selected body and wrist motion. Requires additional weights and MANO. |
| **Analyze 2D** | Prepare GVHMR refinement observations separately. Reconstruct 3D includes this step when needed. |
| **Resource Mode** | Automatic balances speed and available memory. Low Memory uses smaller image batches while retaining inference quality. |
| **Prepare Preview** | Rebuild mesh/skeleton previews from existing motion without running the AI model again. Useful for older projects. |
| **Cancel Job** | Stop queued processing; completed results are kept. |

Processing shows the current step, progress and elapsed time. **Step left** estimates
the current step; **Total left** estimates the queue when timing data is available.
Use **Console** for details and errors. Loading or unmeasured work shows activity
instead of an invented percentage. FlowHMR is limited to 5,000 frames per take
(about 166.7 seconds); available memory may require a shorter clip.

## Motion Correction

Contact correction reduces sliding when feet or palms rest on a support. It uses
existing motion without rerunning AI inference. Original and corrected results
are kept separately. Enable Correction and Foot default on for new work; Hand
defaults off. Existing projects retain saved selections.

| Setting | Function and meaning |
| --- | --- |
| **Enable Correction** | Show the corrected result; off shows the uncorrected result. |
| **Correction Mode → Flat Ground** | Use one correction floor for level ground. |
| **Correction Mode → Multi Level** | Keep contacts at their inferred heights for stairs, slopes and changing levels. The single floor controls are inactive. Terrain geometry is not reconstructed. |
| **Foot / Hand** | Correct only checked contact types. Disabling one retains its saved timing edits. Hand correction adjusts the arm for support contacts while retaining hand orientation. |
| **Lock Preset** | Weak allows more original movement; Normal balances holding and smooth transitions; Strong holds more uncertain contacts and may resist intentional sliding. |
| **Auto Floor Height** | Estimate the contact floor. Turn off to enter a manual height. |
| **Floor Height (World Y)** | Floor height in metres along the vertical axis. Higher values raise the correction floor; the gray reference grid stays at zero. |
| **Pelvis Height Offset (from Auto)** | Zero keeps automatic hip adjustment. Positive values undo some automatic lowering; negative values lower the hips further during contact. Raising can weaken foot locks. |
| **Apply Correction** | Commit current settings and timeline timings, update the source preview/export, and recalculate linked characters. |

Edit **L foot / R foot / L hand / R hand** tracks below the video track. Drag empty
space to add an interval, drag it to move, or drag its edges to resize. Delete
removes a selected interval; Ctrl+Z / Ctrl+Y undo/redo. **Reset Locks** restores
automatic timings. The timeline's **?** has full editing instructions.

Apply pending changes before export; stale corrected export is blocked. Automatic
hand detection is intended for stable support such as a floor, or raised supports
in Multi Level. Inspect its timing. Hanging poses, object interaction and larger
animation edits are best handled in a 3D animation editor. Contact detection is
an estimate and can leave residual sliding or penetration.

## People, preview and navigation

| Control | Function |
| --- | --- |
| **Find person** | Discover people for explicit selection. Without it, reconstruction tracks one person automatically and may switch identity when people overlap. |
| **People checkboxes / selected row** | Checked people are processing/display targets. The selected row controls timeline and contact editing. |
| **Character checkboxes / selected row** | Show or hide imported characters independently of source people. Select a character to export it; right-click it for Retarget, Export Character or Remove Character. |
| **People → ?** | Open instructions for people selection, names, colors and takes. The list footer shows only current status or warnings. |
| **Rename... / color square** | Right-click a row to rename it; click its square to change display color. The name is also used for FBX filenames and rig names. |
| **Take selector** | Choose a visible section when a person leaves and reappears. Long unseen gaps are separate takes. |
| **Overlays / Bone Overlay** | Show video observations / source export bones. Bone Overlay is drawn above source meshes and imported characters. Display only. |
| **View direction / Frame All** | Choose a camera direction / fit visible people in the 3D view. Left drag orbits, middle/right drag pans, wheel zooms. |
| **Floor Grid / Correction Floor** | Show the gray zero-height reference grid / orange correction floor preview. Neither applies correction. |
| **Motion Path / Follow Actor** | Show the person's path / move the viewing camera with the person. |
| **Lock** | Show active hand/foot contacts as translucent horizontal discs in timeline colors. Hand discs are at palm height, including raised supports. Display only. |
| **Name** | Show list names above visible people and imported characters. Text stays upright and the same screen size as you move the camera. Rename a person from the list's right-click menu. |
| **Timeline ruler / blue frame box** | Click or drag to seek. Track editing does not seek. Wheel zooms; middle drag or Shift+wheel pans. |
| **Playback / frame number / speed** | Space plays/pauses; arrows step one frame, Shift+arrows one second. Enter a frame number to jump. Playback speed affects preview only. |
| **Timeline Frame All / Reset Trim** | Fit the full timeline / restore the full clip range. |

Checked people are processed independently and shown in a shared scene. They
share correction mode/floor; timings and pelvis offsets remain per person/take.
Apply Correction updates all available takes, including hidden ones. Relative
depth and contacts between people are estimates; collision avoidance is not provided.

## FBX and Blender

**File → Export FBX** follows the displayed, applied result. Each checked
person/take gets a separate named file and rig; duplicate filenames get suffixes.
People exports use the viewport's shared placement. Takes retain their own frame
counts; align their start times when combining different source intervals.

Source-person exports use a 52-bone hierarchy with **Hips** as root and plain
anatomical names. It retains
actor lengths, SMPL-neutral curved limbs/feet, calibrated rest axes and straight
open fingers. FBX stores rest transforms separately; no T-pose frame is inserted.
BVH is internal compatibility data only.

**Export Character...** preserves the imported target character's own hierarchy
and original rest pose; it does not replace them with the 52-bone source rig.
Character FBX animation is stored on bones, with no armature-object keyframes.

In Blender use **Scale 1**, **Automatic Bone Orientation off**, **Use Pre/Post
Rotation on**, **Ignore Leaf Bones on** and **Animation Offset 1**. The file supplies
frame rate and axes. **Armature Data → Rest Position** shows the rest pose.
The first frame in **Pose Position** shows animation, which can differ from rest.
Use the character workflow above for supported humanoids, or retarget the source
FBX in your animation software.

## Development and license

See [Build from source](BUILD.md) for environment setup, editing and release builds.
Original RGB MoCap code/documentation uses the [MIT License](../LICENSE).
Third-party code, derived modifications, weights and body data keep their own
terms; see [Third-party notices](../THIRD_PARTY_NOTICES.md).
