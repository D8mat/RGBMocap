# RGB MoCap: agent operating and development guide

This guide is for agents helping users operate or modify RGB MoCap. It describes the Git/CMD version. User requests and newer verified behavior take precedence over historical benchmarks. The English GUI is the supported user interface; do not assume a general command-line inference or retargeting API exists. Avoid asking for the user's videos or model weights if the existing local project is sufficient.

## Start and update

- Windows 64-bit, compatible NVIDIA GPU/driver and OpenGL 4.3 are required for the documented inference workflow. Other systems and GPUs were not comprehensively verified; see [validation](VALIDATION.md).
- In a writable checkout, run `install.cmd` once. It sets up separate pinned Python/model profiles, Blender 4.5, FFmpeg and the prebuilt native motion-correction module. Downloading and disk space are required. Models are obtained separately.
- Run `RGBMoCap.cmd`. Keep its command window open; launch errors, processing messages and app Console output are also sent there, but long messages can be truncated and a saturated output queue can drop messages. Within the GUI, **Help > Check System** checks the system.
- Close the app before `update.cmd`. The updater requires a clean tracked worktree and the `main` branch, uses `git pull --ff-only`, then verifies/repairs runtime configuration. It refuses tracked local edits. A plain `git pull` does not perform runtime repair; use `install.cmd` after dependency changes when needed. Do not discard local edits or force an update.
- For setup failure, read the message, check connectivity and free space, then retry `install.cmd`. Preserve models, projects and `.setup` state. Do not repeatedly reinstall or redownload models without evidence.
- Installation currently estimates at least 32.4 GiB free before models; all supported weights were previously measured around 29 GiB. Project data and download caches add space. Measurements are not guarantees. See [user guide](USER_GUIDE.md#install).

## Models and model files

| Model | Practical use | Hands / options |
| --- | --- | --- |
| ExMR | Default; generally fast, good quality in the project's subjective tests. SMPL-X. | Fingers included; static/moving camera and smoothing options. |
| FlowHMR | Slower; best overall quality in those tests, not a universal ranking. SMPL-H. | Fingers included; Fast SAM and camera/sampling choices. Moving camera needs VGGT-Omega. |
| GVHMR | Lower overall quality in those tests but sometimes better for unusual poses such as backflips. SMPL-X. | Fingers via optional HandFlow. HTD refines body motion; FootMR refines ankles. Both are separate from contact correction. |

The [model guide](USER_GUIDE.md#choose-a-model) includes observed timings, conditions and VRAM; these are historical measurements, not minimum specifications. FlowHMR has a 5,000-frame per-take limit and available memory may require less. Do not change numerical precision, temporal context or model settings merely to improve a benchmark.

Use **Model Manager** for the selected model and add-ons. Automatic download may fail, especially on Google Drive. Gated models require the user to register, accept upstream terms and obtain files; use **Manual Download...** and **Import...**. Do not handle provider passwords or redistribute licensed weights. Imports validate files; several model receipts tie validity to file size, timestamp and expected hash. If a copied file is not marked ready, import it again rather than assuming it is missing or editing receipts.

Model files are in `<installation>/models/` by default. To reuse a legitimate installation, close both apps, copy the complete models folder and its receipt/license files beside `RGBMoCap.cmd`, and check readiness in Model Manager. Do not copy old `runtime*` folders. The optional SMPL Neutral body model is not required for the three primary workflows. Exact requirements are defined by `mocap/models.py` and the chosen stages, not by guessed filenames.

## Standard GUI workflow

1. **File > Open Video**, optionally trim the Video track; select an AI Motion Model and prepare its models. New work defaults to ExMR.
2. For multiple people, use **Find person** and check the desired rows; without it, automatic tracking selects one person and can switch identity on overlap. A checked row controls processing/visibility; the selected row controls the current timeline and contact edits.
3. Choose camera and processing settings, then **Reconstruct 3D**. Processing runs on a 30 fps working clip, without changing the original video. Progress and cancellation are in the app. **Prepare Preview** rebuilds previews from existing motion without rerunning inference.
4. Review motion and contacts. **Enable Correction** selects corrected output; Foot defaults on and Hand off for new work. Flat Ground uses a common floor; Multi Level retains inferred contact heights and does not reconstruct terrain. Inspect automatic contacts; hand contacts represent support, not hanging/object tracking.
5. Edit L/R foot and hand intervals on the timeline, then **Apply Correction**. Pending changes are not applied until then; stale corrected export is blocked. Applying also refreshes linked retargeted characters. Correction and preview do not require fresh AI inference.
6. **File > Export FBX** exports displayed applied source-person motion. Multiple checked people/takes produce separate named files. Source FBX uses a 52-bone Hips-root hierarchy; BVH files may exist internally but are not the public export option.

See [the full workflow and controls](USER_GUIDE.md#start-here). Do not confuse viewport toggles, automatic detection and applied motion corrections. No general scene geometry or collision reconstruction is implied.

## Character retargeting and export

- **File > Import Character...** accepts Mixamo-style FBX, VRM and PMX, and adds the character in rest pose below people. Up to eight characters per project are supported.
- Right-click the character and choose **Retarget > [source person]**. Review via timeline; row checkboxes control visibility. Custom bone mapping is not implemented. Unsupported or incomplete mappings must not be silently relabeled.
- The source's applied contact timings are used. **Apply Correction** refreshes linked characters; after other source changes retarget again. Targets that cannot be reached do not cause bone stretching. Different limb lengths can change travel distance, and editing may be required in a 3D tool.
- **Export Character...** offers FBX with or without character mesh, and VMD for PMX characters. VMD is sampled at 30 fps for the same PMX model; facial motion and hair/clothing physics are not generated. Character FBX preserves its imported rest hierarchy and keys body bones, not the armature object. The character viewport uses original-resolution base-color textures with simplified lighting. Texture caches are prepared off the GUI thread, independently of motion; existing characters prepare them on first display.
- A separate Blender runtime is used for character import, rig evaluation and baking/export; the user need not open Blender for the app workflow. Preserve upstream licenses and the bundled Blender environment. See [character retargeting](USER_GUIDE.md#character-retargeting) and [FBX/Blender](USER_GUIDE.md#fbx-and-blender).

## Projects, state and protection

- Default projects live under `<installation>/projects/`; **Save Project As** creates and switches to a copy at a user-selected location. Open the saved `project.json` using **File > Open Project**. A project contains model-specific `exmr`, `flowhmr`, `gvhmr` result folders; model switching loads existing results when available. People, takes, contact edits and imported characters are project data.
- `models/`, `projects/`, `cache/`, `character_scenes/`, `runtime*/` and `.setup/` are local data, not source artifacts. User project folders can also be outside the checkout. Never assume `.gitignore` protects a path that was already tracked or will be force-added. Check `git status` and tracked files before committing; do not upload model weights, video, personal data, credentials, generated motion or caches. Explicitly approved README sample assets are exceptions.
- Preserve source videos, registered weights and project data. Do not delete project subfolders, results or receipts to fix a problem: they can be referenced by saved metadata. Cache cleanup and setup-download cleanup are distinct from deleting saved work; see [storage and cleanup](USER_GUIDE.md#storage-and-cleanup). Obtain authorization for destructive cleanup.
- `mocap/paths.py` defines `ROOT`, `RGB_MOCAP_DATA`, models and projects; normal portable startup sets its own data location. When diagnosing an actual run inspect the effective environment rather than assuming a path.

## Resources and performance

Use **Resource Mode > Automatic** by default. **Low Memory** uses smaller image batches; it is a separate choice, not a promise every clip will fit. GPU/CPU memory availability, resolution, clip duration and camera settings affect workloads. Model processing is not rejected using predicted RAM/VRAM requirements. The desktop checks actual available RAM and Windows allocation headroom every 0.5 seconds and stops the worker tree when these become low; actual allocation failures also produce an error. Close competing applications or shorten the clip after a memory-related stop. Keep genuine model frame limits, bounded buffers and adaptive batch/attention choices; do not silently change model quality or truncate motion. Avoid expensive AI reruns for operations that can use existing results.

GPU allocator limits share ComfyUI's external-use margin: 600 MiB on Windows, 700 MiB above 15 GiB GPU capacity (400 MiB on other systems). Subtract it from actual free VRAM without an additional percentage cap or fixed FlowHMR/ExMR 3/2 GiB reserve. This margin is separate from inference working space, which stays inside the allocator budget. Do not copy ComfyUI's model offload or quality-changing policies into these models.

The app uses separate runtime profiles to maintain compatibility, with bounded batches, previews and background workers. Keep video decoding, large I/O and numerical work off the GUI thread. Playback responsiveness, synchronization and resource cleanup are correctness requirements. If changes affect these, measure frame-time outliers and repeated open/close behavior; do not infer universal performance from a short test.

## Code map and verification

- GUI, people, controls and logging: `mocap/gui.py`, `workspace_ui.py`, `people_ui.py`, `character_ui.py`, `console_log.py`, `tooltips.py`.
- Model selection and loading: `mocap/motion_tools.py`, `models.py`, `model_guide.py`; worker routing and runtimes: `worker.py`, `worker_progress.py`, `runtime_registry.py`, `compute_policy.py`.
- Project storage: `project_storage.py`, `project_workspace.py`, `project_save.py`; contact correction: `correction.py`, `contact_trajectory.py`, `contact_pose_plan.py`, `contact_arm_plan.py`.
- Source mesh / bone fit and preview: `source_animation.py`, `source_surface.py`, `surface_roll.py`, `preview_data.py`, `viewer.py`, `viewport.py`; source export: `fbx_export.py`, `fbx_writer.py`, `scene_export.py`.
- Character pipeline: `mocap/retarget/` plus `character_data.py` and `character_ui.py`; Blender backend and model-specific constraints must be considered separately from source-person export.
- Installation/update: `install.cmd`, `update.cmd`, `tools/git_bootstrap.ps1`, `tools/git_managed_install.py`, `tools/git_update.py`, `installer/runtime-lock.json`. Native correction source/build instructions: [developer-build/motion-correction](../developer-build/motion-correction/README.md).

Inspect [BUILD.md](BUILD.md), [VALIDATION.md](VALIDATION.md) and relevant code before edits. The developer `.venv` workflow is separate from ordinary Git/CMD installation. Run targeted checks for modified behavior; real model inference, real GPU use, Blender/MMD import and clean-install compatibility must only be reported when actually tested. This public checkout may not include private regression tests or licensed assets.
