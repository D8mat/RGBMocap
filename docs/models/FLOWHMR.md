# FlowHMR

[Original project](https://github.com/flowhmr/flowhmr) · [All model guides](README.md)

FlowHMR reconstructs SMPL-H body and finger motion from one video. In this
project's subjective tests it was slower than ExMR but gave the best overall
motion quality. This is not a universal accuracy ranking.

## Prepare files

Run the normal `install.cmd`, then select **FlowHMR > Model Manager**.
Key assets are:

| Asset | Location under the app's `models/` folder |
| --- | --- |
| Supported FlowHMR latest checkpoint and configuration | `flowhmr/flowhmr_latest.ckpt`, `flowhmr/config.yml` |
| SAM 3D Body model, configuration and MHR | `flowhmr/sam/model.ckpt`, `flowhmr/sam/model_config.yaml`, `flowhmr/sam/assets/mhr_model.pt` |
| AMASS SMPL-H neutral body, 16 shape components | `body_models/smplh/neutral/model.npz` |
| VGGT-Omega 512, for camera estimation | `flowhmr/vggt_omega_1b_512.pt` |

Model Manager also lists shared tracking dependencies. On the MANO/SMPL-H
provider's site, follow the manager's instructions for the **Extended SMPL+H**
model. A SMPL-X file cannot replace it. Gated models require the provider's
registration/access steps; use **Manual Download... > Import...** for the exact
file. The current checkpoint is `flowhmr_latest`, not an arbitrary base model.
See [model preparation](../USER_GUIDE.md#prepare-models).

## Controls

| Control | Default for new work | Meaning |
| --- | --- | --- |
| **Fast SAM** | On | Speed up feature extraction using the same supported weights. Small numerical differences can affect motion or automatic contact timings. |
| **Camera / Sampling** | Moving / Unknown - Auto | Automatically choose camera-estimation sampling when the camera moves or its behavior is uncertain. |
| **Camera / Sampling > Static Camera** | Optional | Skip camera estimation for a fixed camera without zooming. This path does not require the VGGT weight. |
| **Camera / Sampling > Every ... Frames** | Optional | Analyze the camera every 1, 2, 5, 10 or 30 frames. Denser camera sampling uses more memory. It does not reduce motion frame count. |
| **Resource Mode** | Automatic | Adapt supported image batches to current memory; Low Memory is separately selectable. |

These options apply on **Reconstruct 3D**. Fast SAM is an extraction option,
not MAMMA's SAM 2.1 mask stage or the standalone SAM 3.1 experiments.

## Capture and export

1. Select FlowHMR and **File > Open Video**, then trim the **Video** track.
2. Use **Find person** if you need explicit selection or several people. Check
   processing targets; select a row to inspect its timeline and contacts.
3. Prepare the required files and choose the camera option, then **Reconstruct 3D**.
4. Review contact correction, edit timings and **Apply Correction** if needed.
   Original motion is retained; applying contact changes does not rerun FlowHMR.
5. Use **File > Export FBX** for the displayed applied motion, or
   [retarget an imported character](../USER_GUIDE.md#character-retargeting).

The working clip is 30 fps; the source video is preserved. The app supports up
to **5,000 frames per take** (about 166.7 seconds at 30 fps), but available memory
may require a shorter take. Results are kept under the project's `flowhmr/`
folder. **Save Project As** copies saved work; switching models keeps their
results separate.

## Limitations and resources

Camera estimation, occlusion and image features can introduce pose, trajectory
or finger errors. A fixed-camera option cannot compensate for a camera that
actually moves. Contact correction reduces supported-foot/palm sliding but does
not repair bad camera geometry or identity swaps.

The app retains RAM/Windows allocation-headroom monitoring, CUDA allocator
budgets and allocation-error reporting. Camera analysis and long motion context
can still exceed available memory. Historical [time/VRAM records](../USER_GUIDE.md#recorded-processing-time-and-vram)
include camera estimation and predate Fast SAM; they are not predictions or
minimum specifications. See [validation](../VALIDATION.md) and
[upstream licenses](../../THIRD_PARTY_NOTICES.md).
