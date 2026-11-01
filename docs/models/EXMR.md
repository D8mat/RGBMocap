# ExMR

[Original project](https://github.com/twehrbein/ExMR) · [All model guides](README.md)

ExMR reconstructs SMPL-X body and finger motion from one video. It is the default
model for new work. In this project's subjective tests it was generally fast
with good motion quality; results depend on the footage and settings.

## Prepare files

Run the normal `install.cmd`; no separate ExMR setup command is needed.
Select **ExMR > Model Manager** and prepare the listed files. Key assets are:

| Asset | Location under the app's `models/` folder |
| --- | --- |
| ExMR weights | `exmr/exmr.safetensors` |
| Sapiens2 pose weights | `exmr/sapiens2_1b_pose.safetensors` |
| SMPL-X v1.1 neutral body | `body_models/smplx/SMPLX_NEUTRAL.npz` |
| MANO right hand | `body_models/mano/MANO_RIGHT.pkl` |
| Shared SAM 3D Body model, config and MHR | `flowhmr/sam/model.ckpt`, `flowhmr/sam/model_config.yaml`, `flowhmr/sam/assets/mhr_model.pt` |
| HaMeR and hand mean | `handflow/hamer.ckpt`, `handflow/mano_mean_params.npz` |

Model Manager also lists shared tracking dependencies. Use its exact supported
files and import checks; do not substitute arbitrary checkpoint versions.
Registered SMPL-X/MANO and gated SAM 3D Body files may require manual download.
The shared `flowhmr` and `handflow` folder names do not mean you must run those
models or enable GVHMR's HandFlow add-on. ExMR uses these assets directly.
MAMMA's locked-head SMPL-X bodies are different files and cannot replace ExMR's
standard body. See [model download and reuse](../USER_GUIDE.md#prepare-models).

## Controls

| Control | Default for new work | Meaning |
| --- | --- | --- |
| **Camera** | Static Camera | Use for a camera that stays still without zooming. Choose Moving Camera / SimpleVO when it moves. ExMR does not use FlowHMR's VGGT camera stage. |
| **Smooth Motion** | On | Reduce small frame-to-frame changes; off preserves the unsmoothed result. |
| **Resource Mode** | Automatic | Adapt supported work batches to available memory. Low Memory is a separate option. |

Camera and smoothing changes take effect on **Reconstruct 3D**, not on
**Apply Correction**. Existing projects retain saved settings.

## Capture and export

1. Select ExMR and use **File > Open Video**. Trim the **Video** track as needed.
2. For explicit identity selection or several people, run **Find person** and
   check who to process. The selected row controls preview and contact editing.
3. Prepare models, choose camera settings and run **Reconstruct 3D**.
4. Review the original and corrected motion. Edit foot/palm support intervals
   and **Apply Correction** when needed. This reuses inference results.
5. Use **File > Export FBX** for the displayed applied source motion, or
   [import and retarget a character](../USER_GUIDE.md#character-retargeting).

Processing uses a 30 fps working clip and preserves the source video. Saved
results are under the project's `exmr/` folder. Switching models loads that
model's own available results; it does not convert one model's inference into
another's. **Save Project As** copies saved work to the selected location.

## Limitations and resources

Occlusion, motion blur, poor visibility and identity changes can produce wrong
poses or fingers. Automatic contact detection can also miss intentional support;
edit its timings rather than expecting correction to recover missing evidence.
Contact correction is distinct from Smooth Motion and preserves original results.

RAM/Windows allocation-headroom monitoring and the shared CUDA allocator budget
remain active. Allocation failure stops the job with an error; shorten footage
or free resources if necessary. Historical [processing/VRAM measurements](../USER_GUIDE.md#recorded-processing-time-and-vram)
are individual runs, not minimum specifications. See [validation](../VALIDATION.md)
for the tested PC and scope, and [licenses](../../THIRD_PARTY_NOTICES.md) for
upstream restrictions.
