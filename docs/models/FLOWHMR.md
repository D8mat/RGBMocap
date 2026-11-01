# FlowHMR

[Original project](https://github.com/flowhmr/flowhmr) · [All model guides](README.md)

FlowHMR reconstructs SMPL-H body and finger motion from one video. In this
project's subjective tests it was slower than ExMR but gave the best overall
motion quality. This is not a universal accuracy ranking.

## Prepare files

Run the normal `install.cmd`, then select **FlowHMR > Model Manager**.
FlowHMR does not require `Setup_MAMMA.cmd`, `Setup_PromptHMR.cmd` or their
C++/CUDA native-build prerequisites. Key assets are:

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
| **Fast SAM Acceleration > 4-Frame Batching** | On | Process up to four frames together. Short tests measured about 2–2.4× faster SAM features, with about 0.4 GiB more allocated VRAM (0.7 GiB more reserved). Small numerical differences are possible. Low Memory uses one frame per batch. |
| **Fast SAM Acceleration > GPU Hand Crops** | On | Prepare hand crops on the GPU. Tests on top of 4-frame batching measured about 8–11% less feature time, with unchanged allocated VRAM at that batch size. Image interpolation differs from CPU crops, so the resulting motion can differ slightly. |
| **Camera / Sampling** | Moving / Unknown - Auto | Automatically choose camera-estimation sampling when the camera moves or its behavior is uncertain. |
| **Camera / Sampling > Static Camera** | Optional | Skip camera estimation for a fixed camera without zooming. This path does not require the VGGT weight. |
| **Camera / Sampling > Every ... Frames** | Optional | Analyze the camera every 1, 2, 5, 10 or 30 frames. Denser camera sampling uses more memory. It does not reduce motion frame count. |
| **Resource Mode** | Automatic | Use the fast GPU path when memory is sufficient; otherwise offload SAM/VGGT transformer blocks to CPU. Low Memory forces offloading. This can be slower and use more RAM; precision and motion frame count stay unchanged. |

These options apply on **Reconstruct 3D**. Fast SAM is an extraction option,
not MAMMA's SAM 2.1 masks, PromptHMR's SAM2 tiny tracker or the standalone
SAM 3.1 experiments. Those checkpoints are not interchangeable.

The two acceleration switches work independently and are saved with the project.
They are used only while **Fast SAM** is on. They do not lower model precision,
reduce image resolution or skip motion frames. In the short validation clips,
GPU crops changed final joint positions by about 1 mm at the 95th percentile,
with differences up to 5 mm; this measures disagreement with CPU crops, not
accuracy against ground truth. Speed figures describe feature extraction after
warm-up, not the whole reconstruction, and vary with the PC and video. CUDA
memory pressure can retry with Low Memory and a smaller batch.

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

Single-person reconstruction connects feature extraction and motion generation
in one worker, releasing the feature model before loading the motion model.
Multiple people retain stage-scoped model reuse. Automatic may prepare up to
two CPU image batches ahead when memory is available; Low Memory disables this
overlap. This does not change the Fast SAM switches, resolution or precision.

### Measured effects

Automatic's VGGT camera-only loader skips temporary initial weights using meta
initialization and direct checkpoint assignment. A paired camera-component test
reduced loading from **7.24 to 2.34 s**, with identical camera outputs and peak
allocated VRAM; peak RAM working set fell **8.88 to 5.40 GiB**. Low Memory and
Automatic fallback retain the previous loader, avoiding the experimental increase
in Windows private allocation. This does not change SAM or motion inference.
See [loading measurements and scope](../PERFORMANCE.md#vggt-camera-checkpoint-loading).

Measured on Windows with RTX 5090 32 GB, Ryzen 9 9950X and 96 GB RAM.
The rows use different stages/baselines; do not combine their speedups.

| Comparison | Measured change |
| --- | --- |
| Automatic → Low Memory, 60 frames including cameras/correction/preview preparation | Allocated VRAM **6.10 → 2.77 GiB** (55% less); RAM working set **8.42 → 8.75 GiB**; **95.24 → 96.15 s**. This mode check predates the latest acceleration. |
| 4-frame batching, warm SAM features | About **2.0–2.4× faster**; allocated VRAM **3.44 → 3.84 GiB**, reserved VRAM **3.74 → 4.45 GiB**; RAM about **6.47 GiB** in both. |
| GPU hand crops on top of 4-frame batching | Two clips: **6.10 → 5.44 s** and **10.87 → 9.81 s**, about 10% less feature time; a repeated alternating test found 8%. Allocated VRAM remained **3.84 GiB**. |
| Connected phases, one person / 60 frames | **55.99 → 42.65 s** (24% less); allocated VRAM **3.84 GiB** in both. Includes worker startup/saving, not discovery, cameras, correction or export. |
| Bounded CPU prefetch, warm SAM features / 450 frames | **42.96 → 40.79 s** (5% less); VRAM unchanged at **3.84 GiB**; RAM **2.45 → 2.49 GiB**. Disabled in Low Memory. |

Offloading can noticeably slow individual stages despite similar whole-job
times in a single short mode check. The shared VGGT component's warm inference
took about **3.3× as long** with offloading while its peak VRAM fell by 52%.
For model reuse across people, exact test scopes and separate output-difference
checks, see [resource modes and measured optimizations](../PERFORMANCE.md).

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

## Citation

[FlowHMR paper](https://arxiv.org/abs/2610.03691) · [Research citations and BibTeX](../CITATIONS.md#flowhmr)
