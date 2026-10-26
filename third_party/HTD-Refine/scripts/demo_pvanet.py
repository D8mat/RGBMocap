"""Run PVA-Net inference on an image, video, or tensor sample."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from htd_refine.runtime import (
    DEFAULT_MODEL_PATH,
    load_full_model,
    project_path,
    resolve_device,
    resolve_existing_project_path,
    to_cpu_tree,
    torch_load,
)
from utils.camera import load_camera_intrinsics, normalize_camera_intrinsics

DIRECT_KEY_ALIASES = {
    "imgs": ("imgs", "images"),
    "bbx_xyxy": ("bbx_xyxy", "bounding_boxes_xyxy", "boxes_xyxy", "bbox_xyxy", "bbox", "bbx"),
    "bbx_xys": ("bbx_xys", "bbox_xys"),
    "K_fullimg": ("K_fullimg", "camera_intrinsics", "intrinsics", "K"),
}
RAW_FRAME_ALIASES = ("raw_images", "raw_imgs", "frames", "video_frames")
PATH_ALIASES = ("video_path", "video", "input_video", "image_path", "image", "input_image")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TORCH_SUFFIXES = {".pt", ".pth"}
YOLO_CHECKPOINT_PATH = "inputs/checkpoints/yolo/yolov8x.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="assets/demo/demo.mp4",
        help="Direct .pt sample, image, or video. Relative paths resolve from the project root.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Serialized PVA-Net model checkpoint.")
    parser.add_argument("--output", default="outputs/demo_output.pt", help="Where to save demo predictions.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--camera_intrinsics",
        "--intrinsics",
        default=None,
        help="Optional .pt/.npy/.npz intrinsics with [3,3], [L,3,3], or [B,L,3,3] values.",
    )
    parser.add_argument(
        "--no_yolo",
        action="store_true",
        help="Disable YOLO fallback when the demo input has no bbx_xyxy/bbx_xys.",
    )
    parser.add_argument("--save_heatmaps", action="store_true", help="Also save the dense heatmap tensor.")
    parser.add_argument(
        "--window_length",
        type=int,
        default=120,
        help="PVA-Net sliding-window length. Use <=0 to run the full sequence in one pass.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=30,
        help="Number of overlapping frames between PVA-Net windows.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if --output already exists.")
    return parser.parse_args()


def _first_present(data: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    for key in aliases:
        if key in data:
            return data[key]
    return None


def _resolve_existing_any_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [project_path(path), Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ensure_batch_dim(tensor: torch.Tensor, batched_ndim: int) -> torch.Tensor:
    if tensor.ndim == batched_ndim - 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == batched_ndim:
        return tensor
    raise ValueError(f"Expected {batched_ndim - 1}D or {batched_ndim}D tensor, got shape {tuple(tensor.shape)}")


def _frames_from_tensor(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"Expected raw frames as [L,H,W,3] or [L,3,H,W], got {array.shape}")
    if array.shape[1] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB raw frames with 3 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    return array


def _load_frames_from_path(path: Path) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return bgr[..., ::-1][None], "image"
    if suffix in VIDEO_SUFFIXES:
        from utils.video_io import read_video_np

        return read_video_np(path), "video"
    raise ValueError(f"Unsupported raw demo input suffix {suffix!r}; expected image/video or .pt/.pth")


def _estimate_K(frames: np.ndarray, length: int) -> torch.Tensor:
    from utils.camera import estimate_K

    height, width = frames.shape[1:3]
    return estimate_K(width, height).float().repeat(length, 1, 1)


def _resolve_yolo_checkpoint() -> Path:
    yolo_path = project_path(YOLO_CHECKPOINT_PATH)
    if not yolo_path.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {yolo_path}. " f"Expected the demo-local checkpoint at {YOLO_CHECKPOINT_PATH}."
        )
    return yolo_path


def _track_bbx_with_yolo(video_path: Path, device: torch.device) -> torch.Tensor:
    from utils.net import moving_average_smooth
    from utils.seq import (
        frame_id_to_mask,
        get_frame_id_list_from_mask,
        linear_interpolate_frame_ids,
        rearrange_by_mask,
    )
    from utils.video_io import get_video_lwh
    from tqdm import tqdm
    from ultralytics import YOLO

    yolo = YOLO(_resolve_yolo_checkpoint())
    cfg = {
        "device": str(device),
        "conf": 0.5,
        "classes": 0,
        "verbose": False,
        "stream": True,
    }
    results = yolo.track(str(video_path), **cfg)
    track_history = []
    for result in tqdm(results, total=get_video_lwh(str(video_path))[0], desc="YoloV8 Tracking"):
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            bbx_xyxy = result.boxes.xyxy.cpu().numpy()
            track_history.append([{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))])
        else:
            track_history.append([])

    id_to_frame_ids = defaultdict(list)
    id_to_bbx_xyxys = defaultdict(list)
    for frame_id, frame in enumerate(track_history):
        for det in frame:
            id_to_frame_ids[det["id"]].append(frame_id)
            id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
    if not id_to_frame_ids:
        raise RuntimeError("YOLO did not detect any person tracks in the demo input.")

    for track_id, boxes in id_to_bbx_xyxys.items():
        id_to_bbx_xyxys[track_id] = np.asarray(boxes)

    _, width, height = get_video_lwh(str(video_path))
    id_area_sum = {}
    for track_id, boxes in id_to_bbx_xyxys.items():
        bbx_wh = boxes[:, 2:] - boxes[:, :2]
        id_area_sum[track_id] = (bbx_wh[:, 0] * bbx_wh[:, 1] / width / height).sum()
    track_id = max(id_area_sum, key=id_area_sum.get)

    frame_ids = torch.tensor(id_to_frame_ids[track_id])
    bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])
    mask = frame_id_to_mask(frame_ids, get_video_lwh(str(video_path))[0])
    bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)
    missing_frame_id_list = get_frame_id_list_from_mask(~mask)
    bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
    if not (bbx_xyxy_one_track.sum(1) != 0).all():
        raise RuntimeError("YOLO track interpolation left empty boxes.")

    bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
    bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
    return bbx_xyxy_one_track.float()


def _detect_bbx_per_frame(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    from ultralytics import YOLO

    yolo = YOLO(_resolve_yolo_checkpoint())
    boxes: list[torch.Tensor] = []
    missing: list[int] = []
    for idx, frame in enumerate(frames):
        frame_bgr = np.ascontiguousarray(frame[..., ::-1])
        result = yolo.predict(frame_bgr, device=str(device), conf=0.5, classes=0, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            boxes.append(torch.zeros(4))
            missing.append(idx)
            continue
        xyxy = result.boxes.xyxy.detach().cpu().float()
        areas = (xyxy[:, 2] - xyxy[:, 0]).clamp_min(0) * (xyxy[:, 3] - xyxy[:, 1]).clamp_min(0)
        boxes.append(xyxy[areas.argmax()])
    bbx = torch.stack(boxes)
    if missing:
        valid = (bbx.sum(dim=1) != 0).nonzero(as_tuple=False).flatten()
        if len(valid) == 0:
            raise RuntimeError("YOLO did not detect any person boxes in the demo input.")
        for idx in missing:
            nearest = valid[(valid - idx).abs().argmin()]
            bbx[idx] = bbx[nearest]
    return bbx


def _get_yolo_bbx(
    source_path: Path | None,
    frames: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if source_path is not None and source_path.suffix.lower() in VIDEO_SUFFIXES:
        return _track_bbx_with_yolo(source_path, device)
    return _detect_bbx_per_frame(frames, device)


def _shrink_boxes_to_crop_ratio(bbx_xyxy: torch.Tensor) -> torch.Tensor:
    from utils.bbox import shrink_bbx_xyxy_to_ratio

    return shrink_bbx_xyxy_to_ratio(bbx_xyxy.float()).float()


def _expand_detector_boxes_to_crop_ratio(bbx_xyxy: torch.Tensor) -> torch.Tensor:
    from utils.bbox import expand_bbx_xyxy_to_ratio

    return expand_bbx_xyxy_to_ratio(bbx_xyxy.float()).float()


def _crop_images_from_frames(frames: np.ndarray, bbx_xyxy: torch.Tensor) -> torch.Tensor:
    from utils.bbox import crop_and_resize_np

    if bbx_xyxy.ndim == 3 and bbx_xyxy.shape[0] == 1:
        bbx_xyxy = bbx_xyxy[0]
    if len(frames) != len(bbx_xyxy):
        raise ValueError(f"Frame count and bbox count must match, got frames={len(frames)} boxes={len(bbx_xyxy)}")
    crops = [crop_and_resize_np(frame, box.detach().cpu().numpy()) for frame, box in zip(frames, bbx_xyxy)]
    return torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0


def _to_float_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if not torch.is_floating_point(value):
        value = value.float()
    return value


def _load_raw_source(
    data: dict[str, Any], input_path: Path | None = None
) -> tuple[np.ndarray | None, Path | None, str | None]:
    source_value = _first_present(data, PATH_ALIASES)
    if source_value is not None:
        source_path = _resolve_existing_any_path(source_value)
        frames, source_kind = _load_frames_from_path(source_path)
        return frames, source_path, source_kind

    raw_value = _first_present(data, RAW_FRAME_ALIASES)
    if raw_value is not None:
        return _frames_from_tensor(raw_value), input_path, "raw_frames"

    return None, None, None


def _load_direct_sample(
    data: dict[str, Any],
    device: torch.device,
    allow_yolo: bool,
    input_path: Path | None = None,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    frames, source_path, source_kind = _load_raw_source(data, input_path=input_path)
    sample: dict[str, torch.Tensor | None] = {}
    meta: dict[str, Any] = {"input_format": "direct_tensor_sample"}

    bbx_value = _first_present(data, DIRECT_KEY_ALIASES["bbx_xyxy"])
    if bbx_value is not None:
        bbx_xyxy = _to_float_tensor(bbx_value, "bbx_xyxy")
        meta["bbx_source"] = "input_bbx_xyxy"
    else:
        bbx_xys_value = _first_present(data, DIRECT_KEY_ALIASES["bbx_xys"])
        if bbx_xys_value is not None:
            from utils.camera import get_bbx_xyxy_from_xys

            bbx_xyxy = get_bbx_xyxy_from_xys(_to_float_tensor(bbx_xys_value, "bbx_xys"))
            meta["bbx_source"] = "input_bbx_xys"
        else:
            if not allow_yolo:
                raise KeyError("Missing bbx_xyxy/bbx_xys and YOLO fallback is disabled by --no_yolo.")
            if frames is None:
                raise KeyError(
                    "Missing bbx_xyxy/bbx_xys. Add a video_path/image_path/raw_images field, "
                    "or pass an image/video directly so the PVA-Net command can run YOLO."
                )
            bbx_xyxy = _get_yolo_bbx(source_path, frames, device)
            meta["bbx_source"] = "yolo"
    if meta["bbx_source"] == "yolo":
        bbx_xyxy = _expand_detector_boxes_to_crop_ratio(bbx_xyxy)
        meta["bbx_processing"] = "expand_bbx_xyxy_to_ratio"
    else:
        bbx_xyxy = _shrink_boxes_to_crop_ratio(bbx_xyxy)
        meta["bbx_processing"] = "shrink_bbx_xyxy_to_ratio"

    imgs_value = _first_present(data, DIRECT_KEY_ALIASES["imgs"])
    if imgs_value is not None:
        imgs = _to_float_tensor(imgs_value, "imgs")
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
    else:
        if frames is None:
            raise KeyError(
                "Missing imgs/images. Add raw_images/video_path/image_path so PVA-Net can crop from the boxes."
            )
        imgs = _crop_images_from_frames(frames, bbx_xyxy)
        meta["image_source"] = f"crop_from_{source_kind or 'raw_source'}"

    K_value = _first_present(data, DIRECT_KEY_ALIASES["K_fullimg"])
    if K_value is not None:
        K_fullimg = _to_float_tensor(K_value, "K_fullimg")
    else:
        if frames is None:
            raise KeyError(
                "Missing K_fullimg/camera_intrinsics and no raw frames are available to estimate intrinsics."
            )
        K_fullimg = _estimate_K(frames, len(frames))
        meta["K_source"] = "estimate_K_from_raw_image_size"

    imgs_batched = _ensure_batch_dim(imgs, 5)
    boxes_batched = _ensure_batch_dim(bbx_xyxy, 3)
    if imgs_batched.shape[:2] != boxes_batched.shape[:2]:
        raise ValueError(
            "Image and bbox batch/sequence dimensions must match, "
            f"got imgs={tuple(imgs_batched.shape)} boxes={tuple(boxes_batched.shape)}"
        )
    batch_size, length = imgs_batched.shape[:2]
    sample["imgs"] = imgs_batched.to(device)
    sample["bbx_xyxy"] = boxes_batched.to(device)
    sample["K_fullimg"] = normalize_camera_intrinsics(
        K_fullimg,
        length=length,
        batch_size=batch_size,
    ).to(device)
    if source_path is not None:
        meta["raw_source_path"] = str(source_path)
    return sample, meta


def load_demo_sample(
    path: Path,
    device: torch.device,
    allow_yolo: bool,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    if path.suffix.lower() not in TORCH_SUFFIXES:
        if not allow_yolo:
            raise KeyError("Image/video demo inputs need YOLO boxes; remove --no_yolo or pass a .pt file with bbx.")
        frames, source_kind = _load_frames_from_path(path)
        bbx_xyxy = _expand_detector_boxes_to_crop_ratio(_get_yolo_bbx(path, frames, device))
        imgs = _crop_images_from_frames(frames, bbx_xyxy)
        sample = {
            "imgs": imgs.unsqueeze(0).to(device),
            "bbx_xyxy": bbx_xyxy.unsqueeze(0).to(device),
            "K_fullimg": _estimate_K(frames, len(imgs)).unsqueeze(0).to(device),
        }
        meta = {
            "input_format": source_kind,
            "raw_source_path": str(path),
            "bbx_source": "yolo",
            "bbx_processing": "expand_bbx_xyxy_to_ratio",
            "image_source": "crop_from_yolo_bbx",
            "K_source": "estimate_K_from_raw_image_size",
        }
        return sample, meta

    data = torch_load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Demo input must be a dict-like torch file, got {type(data)!r}")

    return _load_direct_sample(data, device, allow_yolo=allow_yolo, input_path=path)


def _repeat_to_window_length(tensor: torch.Tensor, window_length: int) -> torch.Tensor:
    length = tensor.shape[1]
    if length == window_length:
        return tensor
    if length > window_length:
        raise ValueError(f"Cannot pad sequence length {length} to smaller window_length={window_length}")
    if length <= 0:
        raise ValueError("Cannot pad an empty sequence.")
    pad_shape = list(tensor.shape)
    pad_shape[1] = window_length - length
    pad = tensor[:, -1:].expand(*pad_shape)
    return torch.cat([tensor, pad], dim=1)


def _slice_window(
    sample: dict[str, torch.Tensor | None], start: int, end: int, window_length: int
) -> dict[str, torch.Tensor | None]:
    window: dict[str, torch.Tensor | None] = {}
    for key, value in sample.items():
        if value is None:
            window[key] = None
        else:
            window[key] = _repeat_to_window_length(value[:, start:end], window_length)
    return window


def _copy_first_available(
    buffer: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
    global_start: int,
    count: int,
) -> None:
    if count <= 0:
        return
    gidx = torch.arange(global_start, global_start + count, device=mask.device)
    new = ~mask[gidx]
    if new.any():
        lidx = gidx[new] - global_start
        buffer[:, gidx[new]] = values[:, lidx]
        mask[gidx[new]] = True


def _fill_missing_by_nearest(buffer: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    missing = torch.nonzero(~mask, as_tuple=False).flatten()
    if missing.numel() == 0:
        return
    known = torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()
    if not known:
        raise RuntimeError(f"No valid predictions to fill {name}.")
    print(f"[window] fill {missing.numel()} uncovered timestep(s) for {name} by nearest neighbor")
    import bisect

    for global_idx in missing.cpu().tolist():
        pos = bisect.bisect_left(known, global_idx)
        if pos == 0:
            nearest = known[0]
        elif pos == len(known):
            nearest = known[-1]
        else:
            left, right = known[pos - 1], known[pos]
            nearest = left if (global_idx - left) <= (right - global_idx) else right
        buffer[:, global_idx] = buffer[:, nearest]
        mask[global_idx] = True


def _run_model_once(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(
        sample["imgs"],
        sample["bbx_xyxy"],
        sample["K_fullimg"],
    )


def run_pvanet_inference(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor | None],
    window_length: int,
    overlap: int,
    keep_heatmaps: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    tuple[int, ...],
    dict[str, Any],
]:
    total_len = int(sample["imgs"].shape[1])
    if window_length <= 0 or total_len <= window_length:
        outputs = _run_model_once(model, sample)
        heatmaps = outputs[-1]
        stats = {"mode": "full", "window_length": total_len, "overlap": 0, "num_windows": 1}
        return (*outputs[:-1], heatmaps if keep_heatmaps else None, tuple(heatmaps.shape), stats)

    if overlap < 2 or overlap >= window_length:
        raise ValueError(
            f"--overlap must be in [2, {window_length - 1}] so velocity and acceleration timesteps remain covered; "
            f"got {overlap}"
        )
    step = window_length - overlap
    last_start = total_len - window_length
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    print(
        f"[window] length={window_length} overlap={overlap} step={step} num_windows={len(starts)} total_frames={total_len}"
    )

    image_keypoints = keypoints_bbox01 = keypoint_confidence = None
    camera_velocities = camera_accelerations = heatmaps_full = None
    kp_mask = bi01_mask = conf_mask = heatmap_mask = vel_mask = accel_mask = None
    heatmaps_shape: tuple[int, ...] | None = None

    for window_idx, start in enumerate(starts, start=1):
        end = min(start + window_length, total_len)
        valid_frames = end - start
        window_sample = _slice_window(sample, start, end, window_length)
        out = _run_model_once(model, window_sample)
        win_kp, win_bi01, win_conf, win_vel, win_accel, win_heatmaps = out
        print(f"[window] {window_idx}/{len(starts)} frames={start}:{end} valid={valid_frames}")

        if image_keypoints is None:
            image_keypoints = win_kp.new_empty((win_kp.shape[0], total_len, *win_kp.shape[2:]))
            keypoints_bbox01 = win_bi01.new_empty((win_bi01.shape[0], total_len, *win_bi01.shape[2:]))
            keypoint_confidence = win_conf.new_empty((win_conf.shape[0], total_len, *win_conf.shape[2:]))
            camera_velocities = win_vel.new_empty((win_vel.shape[0], total_len - 1, *win_vel.shape[2:]))
            camera_accelerations = win_accel.new_empty((win_accel.shape[0], total_len - 2, *win_accel.shape[2:]))
            kp_mask = torch.zeros(total_len, dtype=torch.bool, device=win_kp.device)
            vel_mask = torch.zeros(total_len - 1, dtype=torch.bool, device=win_vel.device)
            accel_mask = torch.zeros(total_len - 2, dtype=torch.bool, device=win_accel.device)
            heatmaps_shape = (win_heatmaps.shape[0], total_len, *win_heatmaps.shape[2:])
            if keep_heatmaps:
                heatmaps_full = win_heatmaps.new_empty(heatmaps_shape)
                heatmap_mask = torch.zeros(total_len, dtype=torch.bool, device=win_heatmaps.device)
            bi01_mask = torch.zeros(total_len, dtype=torch.bool, device=win_bi01.device)
            conf_mask = torch.zeros(total_len, dtype=torch.bool, device=win_conf.device)

        assert image_keypoints is not None
        assert keypoints_bbox01 is not None
        assert keypoint_confidence is not None
        assert camera_velocities is not None
        assert camera_accelerations is not None
        assert kp_mask is not None
        assert bi01_mask is not None
        assert conf_mask is not None
        assert vel_mask is not None
        assert accel_mask is not None

        _copy_first_available(image_keypoints, kp_mask, win_kp, start, valid_frames)
        _copy_first_available(keypoints_bbox01, bi01_mask, win_bi01, start, valid_frames)
        _copy_first_available(keypoint_confidence, conf_mask, win_conf, start, valid_frames)
        if keep_heatmaps:
            assert heatmaps_full is not None
            assert heatmap_mask is not None
            _copy_first_available(heatmaps_full, heatmap_mask, win_heatmaps, start, valid_frames)
        _copy_first_available(camera_velocities, vel_mask, win_vel, start, max(valid_frames - 1, 0))
        _copy_first_available(camera_accelerations, accel_mask, win_accel, start, max(valid_frames - 2, 0))

    assert image_keypoints is not None
    assert keypoints_bbox01 is not None
    assert keypoint_confidence is not None
    assert camera_velocities is not None
    assert camera_accelerations is not None
    assert kp_mask is not None
    assert bi01_mask is not None
    assert conf_mask is not None
    assert vel_mask is not None
    assert accel_mask is not None
    assert heatmaps_shape is not None

    _fill_missing_by_nearest(image_keypoints, kp_mask, "image_keypoints")
    _fill_missing_by_nearest(keypoints_bbox01, bi01_mask, "keypoints_bbox01")
    _fill_missing_by_nearest(keypoint_confidence, conf_mask, "keypoint_confidence")
    if keep_heatmaps and heatmaps_full is not None:
        assert heatmap_mask is not None
        _fill_missing_by_nearest(heatmaps_full, heatmap_mask, "heatmaps")
    _fill_missing_by_nearest(camera_velocities, vel_mask, "camera_velocities")
    _fill_missing_by_nearest(camera_accelerations, accel_mask, "camera_accelerations")
    stats = {"mode": "sliding", "window_length": window_length, "overlap": overlap, "num_windows": len(starts)}
    return (
        image_keypoints,
        keypoints_bbox01,
        keypoint_confidence,
        camera_velocities,
        camera_accelerations,
        heatmaps_full,
        heatmaps_shape,
        stats,
    )


def main() -> None:
    args = parse_args()
    output_path = project_path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"[reuse] {output_path}")
        return
    device = resolve_device(args.device)
    input_path = _resolve_existing_any_path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Demo input not found: {input_path}\n"
            "Expected a .pt dict, image, or video. For .pt without bbox, include "
            "video_path/image_path/raw_images so the PVA-Net command can run YOLO."
        )

    sample, meta = load_demo_sample(input_path, device, allow_yolo=not args.no_yolo)
    if args.camera_intrinsics is not None:
        intrinsics_path = resolve_existing_project_path(args.camera_intrinsics)
        sample["K_fullimg"] = load_camera_intrinsics(
            intrinsics_path,
            length=sample["imgs"].shape[1],
            batch_size=sample["imgs"].shape[0],
        ).to(device)
        meta["K_source"] = str(intrinsics_path)
    model_path = resolve_existing_project_path(args.model)
    model = load_full_model(model_path, device=device)
    print(f"[model] loaded {model_path}")
    print(f"[device] {device}")
    print(f"[input] {input_path} ({meta['input_format']})")
    print(f"[bbx] source={meta.get('bbx_source', 'unknown')}")
    print(f"[intrinsics] source={meta.get('K_source', 'input')}")

    with torch.inference_mode():
        (
            image_keypoints,
            keypoints_bbox01,
            keypoint_confidence,
            camera_velocities,
            camera_accelerations,
            heatmaps,
            heatmaps_shape,
            window_stats,
        ) = run_pvanet_inference(
            model,
            sample,
            window_length=args.window_length,
            overlap=args.overlap,
            keep_heatmaps=args.save_heatmaps,
        )

    result: dict[str, Any] = {
        "format": "pvanet_output_v1",
        "model_path": str(model_path),
        "input_file": str(input_path),
        "meta": meta,
        "bbx_xyxy": sample["bbx_xyxy"].detach().cpu(),
        "K_fullimg": sample["K_fullimg"].detach().cpu(),
        "image_keypoints": image_keypoints,
        "keypoints_bbox01": keypoints_bbox01,
        "keypoint_confidence": keypoint_confidence,
        "camera_velocities": camera_velocities,
        "camera_accelerations": camera_accelerations,
        "heatmaps_shape": heatmaps_shape,
        "pvanet_window": window_stats,
    }
    if args.save_heatmaps and heatmaps is not None:
        result["heatmaps"] = heatmaps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_cpu_tree(result), output_path)
    print(f"[save] {output_path} ({output_path.stat().st_size / 1024**2:.1f} MiB)")
    for key in ["image_keypoints", "camera_velocities", "camera_accelerations"]:
        print(f"[shape] {key}: {tuple(result[key].shape)}")


if __name__ == "__main__":
    main()
