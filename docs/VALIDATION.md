# Source package validation

The Git-based distribution was tested on the Windows PC listed in the
[user guide](USER_GUIDE.md#install). It has not been tested on a clean Windows
installation or on other GPUs or drivers.

## Recorded checks

- A separate local Git checkout installed all six pinned runtime profiles using
  `install.cmd`. Blender and FFmpeg were downloaded and hash checked; previously
  verified package downloads were reused. Installation took approximately 10 min
  32 sec on the tested PC. CMD launch and launch-error reporting were checked.
- ExMR, FlowHMR and GVHMR each processed a one-second video, including inference,
  correction, rig generation and FBX export. Thirteen worker fingerprint inputs
  were checked. This does not establish long-video performance on other systems.
- The included correction module and a contributor rebuild produced identical
  root positions and local rotation values on the tested clips: 30 frames for
  each model and 210 additional frames for ExMR and FlowHMR, with automatic,
  manual and disabled foot contact. Other motions and platforms were not covered.
- One Mixamo FBX, one VRM and one PMX character were imported and retargeted.
  Mesh-inclusive and animation-only FBX files were reimported into Blender; bone
  animation and mesh options were present without armature-object transform keys.
  A PMX VMD was reimported with Blender MMD Tools. Visual equivalence in the
  standalone MMD application was not established by that check.
- A local Git update retained test model and project data and relaunched the GUI;
  runtime verification took approximately 1 min 55 sec. Twenty automated tests
  covered simulated download errors, unsafe archive paths, update guards and
  interrupted replacement. Actual power loss and disk exhaustion were not tested.
- The public source snapshot was checked for forbidden generated/model payloads,
  and its manifest hashes matched. Pattern checks found no typical private-path or
  credential formats in scanned text; such scans cannot prove absence of secrets.

These checks describe the recorded examples, not a guarantee for every video,
character or computer. Model weights and registered body data remain separate and
subject to their providers' terms. Review the applicable third-party notices and
corresponding source/build requirements before redistributing binaries.

## Historical packaging issue

The earlier v1.0.0 standalone installer omitted historical requirements files
used by worker fingerprints. The current source checkout includes those files;
13 fingerprint inputs were checked in the separate-install test. The current Git
installation uses `install.cmd`, not the old standalone installer.
