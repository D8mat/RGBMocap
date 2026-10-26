"""Visualization helpers for HTD-Refine demos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from einops import einsum

from optimization.body_model import create_body_model
from utils.camera import create_camera_sensor
from utils.draw import draw_coco17_skeleton_batch
from utils.geometry import apply_T_on_points, move_to_start_point_face_z
from utils.net import to_device
# Windows GUI supplies rendering; no compiled PyTorch3D renderer is loaded., get_global_cameras_static, get_ground_params_from_points
from utils.video_io import get_video_lwh, get_video_reader, get_writer, read_video_np, save_video

CRF = 23
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMPLX2SMPL_PATH = PROJECT_ROOT / "utils/body_model/smplx2smpl_sparse.pt"
SMPL_J_REGRESSOR_PATH = PROJECT_ROOT / "utils/body_model/smpl_neutral_J_regressor.pt"


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _trim_video_frames(
    video_path: str | Path,
    length: int,
    frame_indices: torch.Tensor | None = None,
):
    selected = None
    if frame_indices is not None:
        selected = torch.as_tensor(frame_indices, dtype=torch.long).cpu().tolist()[:length]
        if any(right <= left for left, right in zip(selected, selected[1:])):
            raise ValueError("frame_indices must be strictly increasing.")
    reader = get_video_reader(video_path)
    selected_pos = 0
    try:
        for idx, frame in enumerate(reader):
            if selected is None:
                if idx >= length:
                    break
                yield frame
            elif selected_pos < len(selected) and idx == selected[selected_pos]:
                yield frame
                selected_pos += 1
                if selected_pos == len(selected):
                    break
    finally:
        reader.close()
    if selected is not None and selected_pos != len(selected):
        raise ValueError(
            f"Video ended before aligned frame {selected[selected_pos]}; selected {selected_pos}/{len(selected)} frames."
        )


def _log_video_progress(label: str, idx: int, total: int) -> None:
    if idx == 0 or (idx + 1) % 60 == 0 or idx + 1 == total:
        print(f"[render] {label} {idx + 1}/{total}", flush=True)


def _smpl_vertices(
    params: dict[str, torch.Tensor],
    model_name: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    device = torch.device(device)
    if model_name == "smplx":
        body_model = create_body_model("smplx_pca").to(device)
        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=True).to(device)
        faces = create_body_model("smpl").faces
        with torch.no_grad():
            out = body_model(**to_device(params, device))
            verts = torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices])
    elif model_name == "smpl":
        body_model = create_body_model("smpl").to(device)
        faces = body_model.faces
        with torch.no_grad():
            verts = body_model(**to_device(params, device)).vertices
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}")
    return verts, faces


def render_kp2d_overlay(
    video_path: str | Path,
    kp2d: torch.Tensor,
    output_path: str | Path,
    overwrite: bool = False,
    frame_indices: torch.Tensor | None = None,
) -> Path:
    output_path = _ensure_parent(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    video = read_video_np(video_path)
    kp2d = kp2d.detach().cpu().float()
    if frame_indices is not None:
        frame_indices = torch.as_tensor(frame_indices, dtype=torch.long).cpu()
        if len(frame_indices) and int(frame_indices[-1]) >= len(video):
            raise ValueError(f"Video has {len(video)} frames but alignment requests frame {int(frame_indices[-1])}.")
        video = video[frame_indices.numpy()]
    length = min(len(video), len(kp2d))
    overlay = draw_coco17_skeleton_batch(video[:length], kp2d[:length], 0.0)
    save_video(overlay, output_path, fps=30, crf=CRF)
    return output_path


def render_incam_video(
    video_path: str | Path,
    smpl_params_incam: dict[str, torch.Tensor],
    K_fullimg: torch.Tensor,
    output_path: str | Path,
    model_name: str = "smplx",
    overwrite: bool = False,
    *,
    device: str | torch.device = "cuda",
    frame_indices: torch.Tensor | None = None,
) -> Path:
    output_path = _ensure_parent(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    length, width, height = get_video_lwh(video_path)
    device = torch.device(device)
    verts, faces = _smpl_vertices(smpl_params_incam, model_name, device)
    length = min(length, len(verts), len(K_fullimg))
    if frame_indices is not None:
        length = min(length, len(frame_indices))
    renderer = Renderer(width, height, device=device, faces=faces, K=K_fullimg[0])
    writer = get_writer(output_path, fps=30, crf=CRF)
    color = [0.8, 0.8, 0.8]
    try:
        frames = _trim_video_frames(video_path, length, frame_indices=frame_indices)
        for i, (frame, verts_i) in enumerate(zip(frames, verts[:length])):
            _log_video_progress("incam", i, length)
            if i > 0 and not torch.equal(K_fullimg[i], K_fullimg[0]):
                renderer.initialize_camera_params(focal_length=None, K=K_fullimg[i])
            rendered = renderer.render_mesh(verts_i.to(device), frame, color)
            writer.write_frame(rendered)
    finally:
        writer.close()
    return output_path


def render_global_video(
    video_path: str | Path,
    smpl_params_global: dict[str, torch.Tensor],
    output_path: str | Path,
    model_name: str = "smplx",
    overwrite: bool = False,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    output_path = _ensure_parent(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    length, width, height = get_video_lwh(video_path)
    device = torch.device(device)
    verts, faces = _smpl_vertices(smpl_params_global, model_name, device)
    length = min(length, len(verts))
    J_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=True).to(device)
    verts_glob = move_to_start_point_face_z(verts[:length], J_regressor, y_offset="min_all")
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
        device=device,
    )
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device=device, faces=faces, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3, device=device) * 0.8
    writer = get_writer(output_path, fps=30, crf=CRF)
    try:
        for i in range(length):
            _log_video_progress("global", i, length)
            cameras = renderer.create_camera(global_R[i], global_T[i])
            writer.write_frame(renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights))
    finally:
        writer.close()
    return output_path


def render_global_from_incam(
    video_path: str | Path,
    smpl_params_incam: dict[str, torch.Tensor],
    T_w2c: torch.Tensor,
    output_path: str | Path,
    model_name: str = "smpl",
    overwrite: bool = False,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    output_path = _ensure_parent(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    device = torch.device(device)
    verts, faces = _smpl_vertices(smpl_params_incam, model_name, device)
    length = min(len(verts), len(T_w2c), get_video_lwh(video_path)[0])
    verts_w = apply_T_on_points(verts[:length], torch.linalg.inv(T_w2c[:length].to(verts.device)))
    return _render_global_vertices(video_path, verts_w, faces, output_path, overwrite=True, device=device)


def _render_global_vertices(
    video_path: str | Path,
    verts: torch.Tensor,
    faces: np.ndarray,
    output_path: Path,
    overwrite: bool,
    device: str | torch.device,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    _, width, height = get_video_lwh(video_path)
    device = torch.device(device)
    J_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=True).to(device)
    verts_glob = move_to_start_point_face_z(verts, J_regressor, y_offset="min_all")
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
        device=device,
    )
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device=device, faces=faces, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3, device=device) * 0.8
    writer = get_writer(output_path, fps=30, crf=CRF)
    length = len(verts_glob)
    try:
        for i in range(length):
            _log_video_progress("global_from_incam", i, length)
            cameras = renderer.create_camera(global_R[i], global_T[i])
            writer.write_frame(renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights))
    finally:
        writer.close()
    return output_path


def merge_three_videos(
    video_paths: list[str | Path], output_path: str | Path, overwrite: bool = False, labels: list[str] | None = None
) -> Path:
    output_path = _ensure_parent(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    readers = [get_video_reader(path) for path in video_paths]
    _, widths, heights = zip(*(get_video_lwh(path) for path in video_paths))
    target_h = min(heights)
    target_h = target_h if target_h % 2 == 0 else target_h - 1
    target_ws = [max(2, int(round(width * target_h / height))) for width, height in zip(widths, heights)]
    target_ws = [width if width % 2 == 0 else width + 1 for width in target_ws]
    out_w = sum(target_ws)
    out_w = out_w if out_w % 2 == 0 else out_w + 1
    writer = get_writer(output_path, fps=30, crf=CRF)

    try:
        for i, frame_items in enumerate(zip(*readers)):
            if i == 0 or (i + 1) % 60 == 0:
                print(f"[merge] frame {i + 1}", flush=True)
            frames = []
            for frame, width in zip(frame_items, target_ws):
                frame = np.asarray(frame)
                frame = cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            merged = np.concatenate(frames, axis=1)
            if merged.shape[1] < out_w:
                pad = np.zeros((merged.shape[0], out_w - merged.shape[1], 3), dtype=merged.dtype)
                merged = np.concatenate([merged, pad], axis=1)
            if labels:
                x0 = 18
                for frame_width, label in zip(target_ws, labels):
                    cv2.putText(merged, label, (x0, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 240, 255), 2, cv2.LINE_AA)
                    x0 += frame_width
            writer.write_frame(merged.astype(np.uint8))
    finally:
        writer.close()
        for reader in readers:
            close = getattr(reader, "close", None)
            if close is not None:
                close()
    return output_path


__all__ = [
    "merge_three_videos",
    "render_global_from_incam",
    "render_global_video",
    "render_incam_video",
    "render_kp2d_overlay",
]
