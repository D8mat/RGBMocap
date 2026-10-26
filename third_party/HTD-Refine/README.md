<div align="center">
  <h1>HTD-Refine: Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos</h1>
  <p>
    <a href="https://kune1008.github.io/"><strong>Dingkun Wei</strong></a><sup>1,2,*</sup> ·
    <a href="https://zehongs.github.io/"><strong>Zehong Shen</strong></a><sup>2,*</sup> ·
    <a href="https://scholar.isshikih.top/"><strong>Yan Xia</strong></a><sup>3</sup> ·
    <a href="https://geopavlakos.github.io/"><strong>Georgios Pavlakos</strong></a><sup>3</sup>
    <br>
    <a href="https://shenyujun.github.io/"><strong>Yujun Shen</strong></a><sup>2</sup> ·
    <a href="https://www.xzhou.me/"><strong>Xiaowei Zhou</strong></a><sup>1</sup>
    <br>
    <sup>1</sup> Zhejiang University · <sup>2</sup> Ant Group · <sup>3</sup> The University of Texas at Austin · <sup>*</sup> Equal contribution
  </p>
  <p><strong>CVPR 2026 · 🎉 Oral · 🏆 Best Paper Candidate</strong></p>
  <p>
    <a href="https://arxiv.org/abs/2605.26879"><img src="assets/badges/arxiv.svg" alt="arXiv" /></a>
    <a href="https://zju3dv.github.io/htd-refine/"><img src="assets/badges/project-page.svg" alt="Project Page" /></a>
  </p>
</div>

<p align="center">
  <strong>TL;DR:</strong> HTD-Refine uses estimated 3D velocity and acceleration to recover natural global human motion from existing HMR results.
</p>

https://github.com/user-attachments/assets/e888e690-2762-4230-ba6a-d35213017e65

## 1. Required Inputs 📥

The pipeline runs PVA-Net followed by full-sequence optimization.

### 1.1 PVA-Net Inputs

| Input | Format |
| --- | --- |
| RGB frames | A video, an image, or `imgs`/`images` tensors |
| Person boxes | `bbx_xyxy` or `bbx_xys`; optional when the YOLO fallback is enabled |
| Camera intrinsics | `K` with shape `[3,3]`, `[L,3,3]`, or `[B,L,3,3]` |

### 1.2 Optimization Inputs

| Input | Format | Example |
| --- | --- | --- |
| PVA-Net result | 2D keypoints, 3D velocities, and 3D accelerations | Output of `htd-refine-pva` |
| Initial HMR result | An SMPL-X or SMPL sequence | [GVHMR](https://github.com/zju3dv/GVHMR) · [TRAM](https://github.com/yufu-wang/tram) |
| Camera intrinsics | Camera matrix `K`; use the same values as PVA-Net | [HumanFoV](https://github.com/pixelite1201/CameraHMR) |
| Camera extrinsics | Explicit per-video `[L,4,4] T_w2c`; omit only for a static camera | [TRAM](https://github.com/yufu-wang/tram) · [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) |

> **Camera intrinsics matter.** Use the same calibrated `K` for both stages, or
> estimate it with HumanFoV. If no usable `K` is available, HTD-Refine assumes a 53.13° diagonal
> FOV, roughly comparable to an iPhone at about 1.8× zoom.

## 2. Method ⚒️

<p align="center">
  <a href="https://zju3dv.github.io/htd-refine/">
    <img src="assets/method.png" alt="HTD-Refine pipeline" width="100%" />
  </a>
</p>

## 3. Installation 📦

The provided requirements were validated on Linux with Python 3.10, CUDA 12.1,
and PyTorch 2.3.0. A different CUDA stack requires matching PyTorch,
torchvision, and PyTorch3D builds.

```bash
conda create -y -n htd-refine python=3.10
conda activate htd-refine

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

## 4. Demo Assets and Checkpoints 📂

The repository includes one 30 FPS example and its GVHMR initial-motion input:

```text
assets/demo/
├── demo.mp4
└── gvhmr/hmr4d_results.pt
```

Model checkpoints and licensed body models are not stored in Git. Place them
under `inputs/checkpoints/`:

| File | When it is needed | Download |
| --- | --- | --- |
| `pvanet.pt` | PVA-Net inference and the end-to-end demo | [Google Drive](https://drive.google.com/drive/folders/1Rg6ajhTRwebUZ_coSvfTlxUfPMD0CR9c?usp=sharing) |
| `yolo/yolov8x.pt` | Optional; only when per-frame person boxes are not provided | [Google Drive](https://drive.google.com/drive/folders/1Rg6ajhTRwebUZ_coSvfTlxUfPMD0CR9c?usp=sharing) *(third-party license applies)* |
| `body_models/smpl/SMPL_NEUTRAL.pkl` | SMPL refinement and rendering | Register and download from the [official SMPL website](https://smpl.is.tue.mpg.de/) |
| `body_models/smplx/SMPLX_NEUTRAL.npz` | SMPL-X refinement and rendering | Register and download from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) |

> By downloading or using the third-party YOLO checkpoint, you agree to the
> corresponding [Ultralytics license](https://www.ultralytics.com/license). SMPL
> and SMPL-X model files are **not redistributed**; download them from their
> official websites after accepting their licenses.

## 5. Run the Demo 🚀

> **Note:** The current demo supports only 30 FPS input videos. Resample other
> videos to 30 FPS before running HTD-Refine.

> **Scope:** The automatic person selection assumes a single visible primary
> subject. In multi-person videos, provide person boxes and an HMR track for the
> same subject.

The bundled example uses a static camera and includes a GVHMR initialization:

```bash
htd-refine-demo \
  --video assets/demo/demo.mp4 \
  --source gvhmr \
  --device cuda:0 \
  --run_optimization
```

Add `--overwrite` to recompute existing PVA, optimization, and rendered files.
The default PVA window is 120 frames with 30 frames of overlap. The selected
`--device` is used by PVA-Net, YOLO, optimization, and rendering.

## 6. Run Individual Stages ⚙️

Generate PVA targets from a video:

```bash
htd-refine-pva \
  --input assets/demo/demo.mp4 \
  --camera_intrinsics /path/to/intrinsics.pt \
  --output outputs/my_video/pva_results.pt
```

The PVA command also accepts an image or a `.pt` input dictionary. Tensor inputs
may provide `imgs`/`images`, `bbx_xyxy`/`bbx_xys`, and
`K_fullimg`/`camera_intrinsics`. When boxes are absent, a raw image, video, or
frame array must be available for the YOLO fallback.

Run refinement with a non-default GVHMR file:

```bash
htd-refine-opt \
  --video /path/to/video.mp4 \
  --pva_result outputs/my_video/pva_results.pt \
  --source gvhmr \
  --motion /path/to/hmr4d_results.pt \
  --camera_intrinsics /path/to/intrinsics.pt \
  --camera_traj /path/to/T_w2c.pt \
  --output_root outputs/my_video/optimization \
  --run_optimization \
  --render
```

For TRAM, set `--source tram --motion /path/to/tram_sequence_directory`.
TRAM's `frame` array is used to align the motion track to PVA-Net and video
frames. If the track contains gaps, optimization uses its longest consecutive
span and reports the selected frame range.

`--camera_traj` accepts only an explicit `[L,4,4] T_w2c` tensor or a dictionary
with a `T_w2c` entry. Its row `i` must correspond to video frame `i`.

## 7. Citation and License 📝

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{wei2026htdrefine,
  title     = {Natural Human Motion Recovery by Aligning High-Order Temporal Dynamics from Monocular Videos},
  author    = {Wei, Dingkun and Shen, Zehong and Xia, Yan and Pavlakos, Georgios and Shen, Yujun and Zhou, Xiaowei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

The code is released under the
[GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`).

Copyright (c) 2026 Dingkun Wei, Zehong Shen, Yan Xia, Georgios Pavlakos,
Yujun Shen, and Xiaowei Zhou.

Third-party models, checkpoints, and dependencies remain subject to their own
licenses.
