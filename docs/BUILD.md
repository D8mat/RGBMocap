# Windows source and release build

The source distribution uses `install.cmd`, `RGBMoCap.cmd`, and `update.cmd`
for ordinary installation, launch and updates. The executable installer build
steps below describe the older packaging path; they are not required for Git
installation. For the native motion-correction module see
[contributor build instructions](../contributors/motion-correction/README.md).

## Prerequisites

- Native 64-bit Windows and a compatible NVIDIA driver for inference.
- An initial Python 3.10+ installation; `setup_dev.py` obtains pinned Python
  3.10.22 and 3.12.10 through uv 0.13.1.
- Visual Studio 2022 Build Tools, Desktop development with C++, Windows SDK and
  CMake. Git is needed if CMake fetches its pinned pybind11/Eigen source.
- FFmpeg and ffprobe on PATH, or a folder passed with `--ffmpeg-dir`.
- The complete [Blender 4.5 LTS Windows distribution](https://www.blender.org/download/lts/4-5/),
  installed or extracted separately. Keep its Python, libraries and license files.
- Internet access for dependencies; enough disk space for downloads and builds.

No Ubuntu, WSL, Docker or system CUDA Toolkit is required for the reviewed wheel
stack. GPU compatibility still depends on the NVIDIA driver and installed wheels.

## Environments

Run `python tools/setup_dev.py --plan` to review groups. Then run the same command
without `--plan` to install them. `--profiles shared exmr` prepares only those
profiles. Shared is added automatically for overlay groups. Every process returns
a failure if downloads, hashes or installation fail.

| Development folder | Purpose |
| --- | --- |
| `.venv` | GUI, GVHMR and shared CPU correction |
| `.venv-flow` | FlowHMR overlay on the shared dependency base |
| `.venv-hand` | HandFlow overlay |
| `.venv-htd` | HTD overlay |
| `.venv-exmr` | Independent ExMR Python 3.12 environment |

The authoritative source is `installer/runtime-lock.json`, not a broad pip
resolver over one combined environment. Small pure-Python packages without a
matching public wheel are retained as source under `installer/runtime_sources/`.
Build tools reconstruct wheels locally and write a derived lock under `build/`.
The source lock is never silently replaced from the developer's installed pip set.

The root `requirements-windows.txt`, `requirements-flow-windows.txt` and
`requirements-hand-windows.txt` are historical cache-fingerprint inputs, retained
byte-for-byte so existing worker receipts remain comparable. They are bundled
with the application and checked by both release builders. Do not use them to
install the managed runtimes; use `tools/setup_dev.py` and the runtime lock above.

When changing model dependencies, review its profile and overlay relationship in
`mocap/runtime_registry.py`, update the corresponding lock deliberately and rerun
model-specific checks. Do not upgrade a shared dependency for a new model without
checking existing models.

## Editing application features

| Area | Main files |
| --- | --- |
| Workspace and controls | `mocap/gui.py`, `workspace_ui.py`, `people_ui.py`, `tooltips.py` |
| Model list and file import | `mocap/motion_tools.py`, `models.py`, `model_guide.py` |
| Worker routing | `mocap/worker.py`, `worker_progress.py`, `runtime_registry.py` |
| Model processing | `mocap/exmr_backend.py`, `flow_backend.py`, `engines.py` |
| Contact correction | `mocap/correction.py`, `contact_trajectory.py`, `contact_pose_plan.py`, `contact_arm_plan.py` |
| Skeleton and surface fitting | `mocap/source_animation.py`, `source_surface.py`, `surface_roll.py` |
| Preview and playback | `mocap/viewport.py`, `preview_data.py`, `playback.py`, `video_preview.py` |
| FBX | `mocap/fbx_export.py`, `fbx_bind.py`, `fbx_writer.py`, `scene_export.py` |
| Setup | `installer/Setup.cs`, `FolderPicker.cs`, `engine.py`, `runtime-lock.json` |

Run `app.py` from `.venv` while editing. Worker changes are loaded by their own
profile processes; C++ changes require rebuilding the native correction module.
Keep source inference, contact correction and derived preview/export generations
consistent. A change to painting or labels should not silently rerun inference.

Character import/baking runs in Blender's own Python, separately from the five AI
profiles. For development, point `RGBMOCAP_BLENDER` to its executable:

```powershell
$env:RGBMOCAP_BLENDER = 'C:\Tools\Blender-4.5\blender.exe'
.venv\Scripts\python.exe app.py
```

Character UI and processing live in `mocap/character_ui.py`, `character_data.py`
and `mocap/retarget/`. Standard Mixamo, VRM and PMX mappings are automatic.
Custom bone mapping and hair/clothing simulation are not implemented.

## Required model and auxiliary data

Use Model Manager for weights, SMPL-X/SMPL-H/MANO and gated supporting models.
Registration and license acceptance remain the user's responsibility.

Some upstream auxiliary regressors, mean-pose and mapping arrays are also kept
out of Git. Exact paths and hashes are in `installer/external-assets.json`.
Obtain the matching public auxiliary files from the pinned upstream repositories,
retaining their license conditions:

```powershell
python tools/stage_assets.py --download
```

This downloads only the small hash-pinned files listed in the auxiliary manifest,
not inference weights or registered body models. If an upstream link is unavailable,
obtain the exact files from the versions in `third_party/SOURCES.json` and import
an existing properly acquired tree with:

```powershell
python tools/stage_assets.py --from-root D:\AcquiredAssets
```

That folder must contain paths such as
`third_party\FlowHMR\assets\body_models\smpl_neutral_J_regressor.pt`.
The importer checks exact sizes and SHA256 values. It does not import weights or
registered body models. These locally acquired arrays remain ignored by Git.
Keep the source package's pinned version when acquiring them; do not replace
arrays with similarly named files from another body model.

`third_party/character_import` includes the actual format add-on source, licenses,
OpenCC dictionaries and editable shader libraries. The five small `.blend` shader
libraries and ten standard toon ramps are explicitly listed and hash checked in
`installer/source-resources.json`; they are not sample characters or motion files.
OpenCC's pure-Python wheel is retained as unpacked source under
`installer/runtime_sources/` and reconstructed only in release staging. The add-ons
are loaded by the worker; do not install these source folders through Blender's
extension installer.

## Compile and build

```powershell
.venv\Scripts\python.exe tools/check_source.py
.venv\Scripts\python.exe tools/build_release.py
```

The builder:

1. Freezes the current GUI from `app.py` using PyInstaller.
2. Stages current application and adapted upstream source; no earlier release
   or installed application is used.
3. Verifies auxiliary data and builds the CPython 3.10 native correction module.
4. Adds FFmpeg tools/notices and a complete separate `runtime_character` copied
   from the supplied Blender 4.5 distribution, then writes the application manifest.
5. Compiles the C# Windows Setup and appends its verified application payload,
   runtime lock and helper. AI runtimes are downloaded by Setup; Blender's character
   runtime is embedded. Model weights are not embedded. Disk estimates include
   the additional character runtime.

Example with explicit paths:

```powershell
.venv\Scripts\python.exe tools/build_release.py `
  --asset-root D:\AcquiredAssets `
  --ffmpeg-dir C:\Tools\ffmpeg\bin `
  --blender-dir C:\Tools\Blender-4.5 `
  --output releases\RGBMoCap-1.0.1-setup-windows-x64.exe
```

`--app-only` creates the staged app without Setup. `--work` selects temporary
build output. `--native-deps` points to a local Eigen/pybind11 source cache;
otherwise CMake fetches its pinned dependencies. `--native-module` is intended
for an already reviewed build of this source for CPython 3.10 x64. All build
products are ignored by Git.

`mocap/__init__.py` supplies the application version. Keep `pyproject.toml` and
the runtime lock's release version consistent. The builder records this version
in the application metadata, Windows executable properties and Setup; it rejects
a mismatched staged app. Dependency profile pins remain independently reviewed.

The builder uses `--blender-dir`, `RGBMOCAP_BLENDER`, PATH or a standard Blender 4.5
installation, in that order. It rejects other Blender versions. It does not copy
personal preferences or extensions from the user's Windows profile. Retain the
runtime's Blender license/source information when distributing the resulting app.

Native CMake compilation is defined in
`third_party/KimodoMotionCorrection/CMakeLists.txt`. Source changes to that backend
require a rebuild. The core currently uses AVX compiler options; do not claim
compatibility with processors that have not been checked.

## Before sharing a release

Run the new EXE from the staged app, then test the new Setup in a separate folder.
Check cancellation/retry, disk space, model imports, one clip per supported model,
preview playback, correction and FBX import in Blender. A source/build smoke check
does not replace inference tests on the target hardware.

Retain upstream licenses, required corresponding source/build materials and any
applicable model restrictions. The exact FFmpeg binary and PyAV libraries determine
their license obligations. Copy the relevant FFmpeg notices/source offer alongside
your distribution. For Qt, preserve applicable replacement/relinking rights.
See `THIRD_PARTY_NOTICES.md`; the project MIT grant does not relicense these parts.

## Git upload

Upload this checkout's source and documentation. Do not add ignored environments,
models, external arrays, build outputs, installers or personal projects. Inspect
the staged file list before the first push. Host Setup separately and add its real
download URL to README when available. This source preparation does not create
a GitHub repository or upload anything.

Git attributes preserve exact source bytes for the source manifest, hash-pinned
runtime registry and unpacked wheel contents. Avoid automatic line-ending rewrites
of those files. When intentionally changing the registry, review and refresh the
lock's registry hash; the development bootstrap rejects an unreviewed mismatch.
