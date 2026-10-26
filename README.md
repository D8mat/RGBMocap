# RGB MoCap

A native Windows app that turns video into editable human animation. AI processing
runs locally, with an English interface for reviewing and exporting motion.

## Install and run

Clone this repository with Git into a writable folder, run `install.cmd`, then
run `RGBMoCap.cmd`. To update, close the app and run `update.cmd`. Installation
downloads the pinned runtimes, Blender and FFmpeg; model weights are separate.
See the [installation guide](docs/USER_GUIDE.md#install) for prerequisites and recovery.

## Features

- Reconstruct one or several people and preview the body mesh and animation bones.
- Reduce sliding with foot/palm contact correction and editable contact timings.
- Retarget motion to Mixamo FBX, VRM and PMX humanoid characters.
- Export source animation as FBX; export characters as FBX with or without meshes,
  or PMX character motion as VMD for MMD.
- Save projects with separate results for each AI model.

## How it works

1. Open a video, select an AI model and prepare its files in **Model Manager**.
2. Choose the people to capture and click **Reconstruct 3D**.
3. Review the motion, edit contacts if needed and **Apply Correction**.
4. Export FBX, or import a character, retarget the motion and export it.

## Supported AI models

| Model / original project | Body representation | Fingers |
| --- | --- | --- |
| [ExMR](https://github.com/twehrbein/ExMR) | SMPL-X | Included |
| [FlowHMR](https://github.com/flowhmr/flowhmr) | SMPL-H | Included |
| [GVHMR](https://github.com/zju3dv/GVHMR) | SMPL-X | Optional HandFlow add-on |

Requires 64-bit Windows, a compatible NVIDIA GPU/driver and OpenGL 4.3.
Model files are acquired separately; some require registration and license acceptance.

**Disk space:** allow about **45 GB (42 GiB)** for the app, all AI runtimes,
Blender and all supported model files, including optional add-ons. Videos,
projects, imported characters, exports, generated caches and retained download
archives need additional space. See the [storage details](docs/USER_GUIDE.md#install).

[Cache and project cleanup: paths and deletion effects](docs/USER_GUIDE.md#storage-and-cleanup).

## Documentation

- [User guide: installation, controls and settings](docs/USER_GUIDE.md)
- [Development and builds](docs/BUILD.md)
- [Source validation](docs/VALIDATION.md)
- [Third-party notices and model terms](THIRD_PARTY_NOTICES.md)

Original RGB MoCap code and documentation use the [MIT License](LICENSE).
Third-party components and model files retain their own licenses.
