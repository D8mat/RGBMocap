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
