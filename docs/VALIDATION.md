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
- The included correction module and a developer rebuild produced identical
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

## Latest published-source audit (2026-10-27)

The audit started from Git commit `1daf49c` and used the existing installed
runtimes and legitimately acquired local model files. All test scripts and
outputs were kept outside the public checkout.

- 150 targeted regression checks passed for project saving, model selection,
  character handling, export options, runtime guards and update transactions.
- Each of the three AI models processed 30 new frames through correction and FBX
  export. Blender imported all three 52-bone source rigs with `Hips` as root;
  sampled integer and fractional animation frames contained finite transforms.
- Mixamo FBX, VRM and PMX imports, retargeting and six character FBX exports
  passed. Reimported FBX rest-joint positions matched the imported character's
  original rest pose within 0.1 mm; no armature-object transform keys were found.
  The PMX VMD also loaded with Blender MMD Tools.
- The source checker was corrected to accept the approved README samples and
  root agent guide, ignore untracked local model data, and still reject model
  files accidentally tracked by Git. Archive and Git-checkout cases were tested.
- Local documentation links resolved. The source manifest was refreshed after
  the small documentation and checker fixes.

This audit did not repeat fresh installation, long-clip performance testing,
interactive playback, standalone MMD playback or testing on other GPUs.

## Character preview and long-path fixes (2026-10-27)

- Final preview windows no longer reload themselves. Loader checks covered
  1, 2, 10, 111, 112 and 400 frames, exact frame retrieval, bounded windows and
  worker shutdown. An isolated five-second character viewport check loaded once
  and performed no redraws after warm-up. This is an idle-loop check, not a
  playback FPS or whole-desktop performance guarantee.
- A 474-frame PMX animation was baked and reopened under a 295-character data
  path. All 215 animation channels retained exactly the same key values as the
  short-path baseline; no armature-object animation keys were added.
- Animation-only FBX and PMX VMD exports succeeded at 289-character destination
  paths. Blender reimport checks found finite sampled FBX poses and no object
  transform keys. Short- and long-path VMD outputs were byte-identical.
- Test scripts, character assets and generated outputs remained outside the
  public checkout. These checks did not repeat standalone MMD playback or test
  other GPUs, network shares or every possible Windows path.

These checks describe the recorded examples, not a guarantee for every video,
character or computer. Model weights and registered body data remain separate and
subject to their providers' terms. Review the applicable third-party notices and
corresponding source/build requirements before redistributing binaries.
