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

## Original-resolution character textures (2026-10-27)

- Mixamo FBX, VRM and PMX samples retained their original image dimensions and UVs.
  A pre-existing PMX asset prepared its texture cache without reimport or retargeting.
  Sampled GPU-deformed vertex buffers were unchanged, including a 474-frame take.
- Isolated 1920 x 1080 GPU checks ran 600 draws per sample, excluding the first 50
  from timing statistics. Original-texture median / p95 draw times were
  0.108 / 0.264 ms (FBX), 0.331 / 0.536 ms (VRM), 0.466 / 0.697 ms (PMX rest),
  and 0.411 / 0.617 ms (PMX animation). The largest recorded draw was 2.50 ms.
  These are isolated renderer measurements, not complete application frame times.
- Twelve renderer load/close cycles kept the handle count constant. Process RAM
  rose about 13 MiB during warm-up and stabilized in the last three cycles.
  Three loader playback/seek cycles had no missing animation windows, constant
  handles and less than 0.1 MiB additional RAM. Timer lateness reached 15.7 ms;
  these checks do not establish video/audio synchronization or universal FPS.
  Cancelling background texture preparation shut the loader down in 0.107 seconds.
- Original images are cached once per character on CPU and uploaded once per
  loaded renderer, with mipmaps for distant viewing. Motion buffers remain bounded.
  Full PBR materials and general translucent-material sorting are not implemented.
  Tests and generated character assets remained outside the public checkout.

## Runtime memory-pressure policy (2026-10-28)

- Removed predicted RAM/VRAM admission failures from shared workers, FlowHMR and
  ExMR. FlowHMR's 5,000-frame limit, adaptive batches, FP32 temporal context,
  allocator reserves and the existing GVHMR full-key query-split path remain.
  Estimates can still select a computation path; they no longer reject it.
- A real 12-frame 1920 x 1080 input completed moving-camera VGGT, Fast SAM and
  FlowHMR generation while worker policies were given 24 GiB available RAM.
  This simulated the reported availability; the physical test PC had more RAM.
  VGGT peak process RAM was 8.89 GiB and peak Torch CUDA allocation 6.32 GiB.
  This does not establish a minimum RAM requirement or behavior on other GPUs.
- Qt timer/worker-tree integration tests simulated low available RAM, low Windows
  allocation headroom and critical pressure. Workers and their children stopped
  in 0.65–1.14 seconds; stage errors were saved and prior committed test data
  survived. One low sample followed by recovery did not stop a job. The largest
  observed 16 ms UI-timer gap was 31.8 ms in these short tests.
- Injected CPU MemoryError and CUDA allocation failures returned explicit failed
  events and retained existing motion files. A real FlowHMR retry completed and
  cleared its pending marker. Actual system RAM was never exhausted deliberately.
- Exact policy-only file-hash aliases preserve preceding generic/GVHMR signatures
  and all three ExMR stage identities. Later byte changes invalidate them normally.
  ExMR identity checks used synthetic input stamps and installed model statistics;
  full ExMR and GVHMR inference were not repeated. GVHMR allocation/retry control
  flow was exercised with a test model and simulated allocator observations.
- Tests and generated inputs/results remain outside the source checkout. Runtime
  monitoring cannot guarantee intervention before every sudden large allocation.

## ComfyUI-style external VRAM margin (2026-10-28)

- Shared, FlowHMR and ExMR CUDA allocator budgets now subtract 600 MiB from
  current free VRAM on Windows, or 700 MiB above 15 GiB GPU capacity, matching
  [ComfyUI's external-use margin](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py).
  The old FlowHMR/ExMR fixed 3/2 GiB deductions and extra 95/90% caps are removed.
  ComfyUI's separate 0.8 GiB inference planning allowance is not another allocator
  deduction: weights, activations and temporary buffers share the allowed budget.
  This does not implement ComfyUI's offloading or guarantee free memory against
  other applications or non-PyTorch allocations.
- Boundary checks covered simulated 4/8/12/15/16/24/32 GiB capacities, the exact
  15 GiB transition, exhausted free memory and all four allocator call sites.
  Existing batch backoff and GVHMR dense-OOM/full-key retry checks passed.
- On the RTX 5090 test PC, all three isolated runtime profiles applied a 700 MiB
  margin, performed CUDA allocations, raised a real allocator OOM under a tiny
  test cap and successfully allocated again after restoring the normal budget.
  The test deliberately limited the allocator rather than exhausting the GPU.
- A private 12-frame 1080 x 1920 clip completed FlowHMR camera/Fast SAM/generation,
  GVHMR observations/generation and ExMR keypoints/features/generation using the
  installed model weights. Other GPUs and long clips were not tested here.
  FlowHMR generation from identical cached features produced identical motion
  tensors with the old and new allocator caps.
- Existing generic/GVHMR cache fingerprints and all three ExMR stage identities
  were retained. Test code, inputs, generated motion and logs remain outside the
  public checkout under the private validation artifacts directory.

## Memory-policy follow-up review (2026-10-28)

- Fixed GVHMR's dense-OOM retry report to compute `fits` from observed capacity
  rather than always reporting a shortfall. The retry still preserves FP32 and
  the full key/temporal context and does not reject work using that estimate.
- Made worker cancellation idempotent while its Windows termination helper is
  active. Late helper callbacks check that the worker still exists and only clear
  their own helper reference. Four callback/worker-lifetime combinations, each
  with five duplicate requests, passed without Qt callback exceptions.
- ExMR now replaces a previous failed stage status after successful computation
  or validated cache reuse. Its explicit force flag also bypasses the cache.
  All three cached ExMR stages repaired failed status without altering tensors;
  a forced 12-frame generation was separately checked for a non-cached completion.
- Each model's actual worker entry reported injected CPU allocation failures and
  real CUDA allocator OOMs with an error, exit code 1 and unchanged prior motion.
  Tiny allocator limits triggered CUDA failures without exhausting system VRAM.
  Subsequent retries completed using existing preprocessing data.
- Four Qt timer/worker-tree tests simulated low RAM, low Windows allocation
  headroom and critical pressure. Stops completed in 0.66-1.17 seconds, preserved
  committed results and left no test child worker running. The largest observed
  16 ms UI heartbeat gap was 29.12 ms; a transient low sample did not stop work.
- Exact RAM/VRAM boundaries, Windows commit observations, traceback cleanup,
  exhausted-batch errors, existing inference cache identities and source checks
  passed. This is a single-PC regression check, not a low-memory hardware test or
  a guarantee against sudden allocations or competing GPU applications.

These checks describe the recorded examples, not a guarantee for every video,
character or computer. Model weights and registered body data remain separate and
subject to their providers' terms. Review the applicable third-party notices and
corresponding source/build requirements before redistributing binaries.
