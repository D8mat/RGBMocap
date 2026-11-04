# Unreal Engine skeletal body export

Choose **File > Export > FBX > Unreal Engine — skinned body + skeleton + animation**.
Prepare the current motion and mesh preview first. This profile is available for
all five offline model modes: MAMMA, ExMR, GVHMR and PromptHMR use SMPL-X;
FlowHMR uses its native SMPL-H body and 52 deformation joints. The separate Kinect
workspace has its own capture export; PEAR does not record/export offline results.

Import the resulting FBX into a new Content Browser folder. Enable skeletal mesh
and animation import and leave **Skeleton = None** on the first import. Unreal
creates a Skeletal Mesh, Skeleton and Animation Sequence. Do not select Manny,
Quinn or an unrelated existing skeleton. The body has a native 55-joint SMPL-X or 52-joint SMPL-H
deformation hierarchy plus one fixed root. Its bone names are convenient for
retargeting, but its hierarchy is not the Unreal mannequin hierarchy; use IK
Rig / IK Retargeter for an existing game character.

The export uses the registered body's skin weights, at most eight normalized
influences per vertex, a fixed bind shape, centimetre units, smoothing groups
and baked animation. Pelvis movement stays in the pelvis track; the added root
is stationary, so this is not automatically an extracted root-motion clip.
The original motion, models and project remain unchanged. **Add T-pose before motion** optionally inserts a true straight-arm pose at frame
0, at the first pelvis position/orientation; all original frames move forward by
one. The bind pose is retained. **Place feet on floor (Z = 0)** defaults on. It measures skinned foot vertices,
not bone pivots. The bind pose, optional T-pose and first original motion frame
are grounded independently; the whole original animation receives one constant
translation. This preserves subsequent vertical motion and does not fix foot
sliding, penetration later in the recording, or force both feet down when one is
raised. If the first motion frame is airborne, disable automatic grounding to
retain its original height. **Additional Unreal Z offset** defaults to **0 cm**;
a nonzero value intentionally raises/lowers the grounded result. Disable grounding
and enter -128.5652 cm to reproduce the previous fixed-offset mode.

The fixed skin does not export SMPL-X pose-corrective blend shapes or animated
shape/expression coefficients. It therefore preserves skeletal deformation but
does not reproduce the exact PC2 surface at every frame. The MAMMA Blender whole
scene export remains the exact-surface option. Its PC2 meshes and reference
armatures are separate and cannot become an Unreal Skeletal Mesh simply by
selecting both and exporting FBX.

The source-only FBX profile exports animation without a skinned body and is not
the profile to use for importing a new Unreal character. A smoothing-group
warning is distinct from an FBX parser failure or an empty export.

Validation on 2026-11-03: a real 1,004-frame MAMMA actor exported with 10,475
vertices, 55 deformation bones and one root. Blender FBX round-trip skin checks
at the first, middle and last frame differed from the intended normalized LBS
surface by less than 0.004 mm. Direct Interchange imports in installed Unreal
5.7 and 5.8 created Skeletal Mesh, Skeleton and AnimSequence successfully, without
FBX parser or smoothing warnings. Gameplay retargeting, root-motion extraction
and all other model/result combinations have not been independently tested.

Extended validation: native FlowHMR FK matched saved joints within 0.001 mm over 1,004 frames. Both SMPL-H and SMPL-X exports with a T-pose and -128.5652 cm Z offset passed Interchange import in UE 5.7 and 5.8; animation duration confirmed all 1,005 samples. FBX round-trip mesh checks included first/middle/last motion and the FlowHMR T-pose.

Optional **Align animation facing to reference skeleton** rotates the entire motion
and inserted T-pose around the initial pelvis vertical axis to match the bind
skeleton heading. It defaults off and is available for all five Unreal export
models. Height, timing and relative motion are preserved; horizontal trajectory
is rotated. It does not modify the bind pose or configure MetaHuman IK chains.
Use it for opposite-facing retargets; leave it off to retain scene orientation.

## Faster export, checkpoints and preview

Unreal export writes animation curves in bulk after converting world poses to
local bone transforms, without updating the Blender dependency graph per bone.
The FBX keeps the native skeleton, skin weights, frame rate, heading and floor
options. This optimization does not change reconstruction or contact correction.

**Resume completed export steps** defaults on. Retry the same export destination
after cancellation or failure. Completed prepared motion and completed export
files are SHA-256 verified before reuse. Source file identity, scene placement,
export code and Blender executable identity invalidate old checkpoints; export
option changes rebuild the output while retaining valid prepared motion.
An unfinished step restarts from its beginning, not its last frame. There is no
automatic restart after an error. Disabling this option uses temporary files.
Checkpoints stay in the project's `.unreal-export-cache` directory, consume disk
space and can be removed when no export is running. They are derived data.
Existing model-specific inference receipts continue to control reconstruction
reuse; this feature adds checkpoints to Unreal export, not mid-inference resume.

**Preview only: first 3 seconds as Blender project** writes a standalone `.blend`
instead of the full FBX. It uses the same skin/rig transformation code, including
floor alignment, heading and optional T-pose. Open it in Blender and play the
short timeline before choosing a full FBX export. No root-motion extraction or
in-place mode is introduced. The viewport is framed on the initial body.

**Help > Performance** displays up to 100 completed worker steps for the active
project: elapsed time, cache reuse, and reported process RAM/CUDA allocation
peaks. Missing readings remain unknown; these are not whole-system peaks.
The latest Unreal export additionally reports preparation/build timings and a
sampled Blender RAM maximum (polling can miss brief peaks). The source export
options also show the latest export timings. Files are stored as
`performance-history.json` and `unreal-export-performance.json` inside the project.

Validation in the separate staging checkout (2026-11-04): a 100-frame FlowHMR
fixture plus T-pose took 2.305 s with the previous builder and 0.753 s with bulk
curves, including Blender startup, with two CPU threads. This is one local run,
not a general speed guarantee. Comparing all 101 frames after FBX reimport gave
maximum vertex error 0.00983 mm. A short fixture retained floor alignment and a
50 cm lift. Checkpoint tests covered interruption, retry, reuse, corrupted output,
changed options/inputs, locking and publication. The preview saved a 91-frame
Blender project at 30 fps. UI option routing passed for all five offline models.
No new AI inference, GPU benchmark or Unreal Engine import was run for this update.

Export checkpoints now live under `<installation data>/cache/unreal-export/`, keyed by project, destination and input identity. Temporary non-resumable builds use the same short cache root. This avoids Blender failing to open job files under deeply nested project/take folders on Windows. Old project-local `.unreal-export-cache` data is preserved but no longer reused. A 286-character failing job path was reproduced from an existing project; the same data exported successfully from a short workspace (1,225 frames including T-pose).

## Recenter Character

The timeline action **Recenter Character** places the selected source take's
pelvis at horizontal scene origin at the current frame (app X=0, Z=0; Y height
unchanged). One constant translation applies to the complete take, retaining
root trajectory, rotations and frame timing. **Reset Position** removes this
placement. Select the source person, not an imported target, and place the
playhead inside its take. Other takes/people are not moved.

The placement is stored as `character_position.json` in that take, separately
from inference and correction outputs, and used in single/shared preview, source
FBX/BVH, Unreal and portable Blender exports (including MAMMA mesh caches).
Existing retargeted takes linked to that source receive the same translation in
their saved placement; future retarget jobs inherit it too. Original mocap and
model files are not rewritten. Shared cameras remain in their original positions,
so deliberately moving one actor changes its alignment to reference video.
The Unreal export's separate heading alignment/grounding options still apply.

Validation: `tools/test_character_position.py` covers repeat/reset/persistence,
linked target placement and BVH motion preservation; an offscreen check covers
single-preview and shared-scene transforms. These are not an Unreal editor test.

An additional real-source three-frame FBX export was reimported into Blender: centered horizontal pelvis, unchanged height and unchanged inter-frame root displacement were verified numerically.
