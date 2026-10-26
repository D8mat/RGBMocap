"""Run HTD optimization from GVHMR/TRAM motion plus PVA-Net targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from htd_refine.runtime import project_path, resolve_device, resolve_existing_project_path, torch_load
from optimization import opt_vel
from utils.camera import estimate_K, load_camera_intrinsics, normalize_camera_intrinsics
from utils.gvhmr import load_gvhmr_result, motion_length, to_tensor
from utils.tram import load_tram_result
from utils.video_io import get_video_lwh
from utils.vis import render_global_from_incam, render_global_video, render_incam_video, render_kp2d_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="assets/demo/demo.mp4")
    parser.add_argument(
        "--pva_result",
        default=None,
        help="PVA-Net .pt path. Defaults to pva_results_L120_ov30.pt under the video output directory.",
    )
    parser.add_argument("--source", default="gvhmr", choices=["gvhmr", "tram"])
    parser.add_argument("--motion", default=None, help="GVHMR .pt path or TRAM sequence folder.")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--model", default="auto", choices=["auto", "smplx", "smpl"])
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device used by optimization and rendering, for example cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--camera_intrinsics",
        "--intrinsics",
        default=None,
        help="Optional .pt/.npy/.npz intrinsics. Overrides intrinsics stored in GVHMR/TRAM input.",
    )
    parser.add_argument(
        "--camera_traj",
        "--traj",
        default=None,
        help="Explicit [L,4,4] world-to-camera trajectory. If omitted, the demo uses a static camera.",
    )
    parser.add_argument("--static_cam", action="store_true", help="Force the default static-camera demo mode.")
    parser.add_argument("--run_optimization", action="store_true", help="Compute optimization if no cache exists.")
    parser.add_argument("--vel_thr", type=float, default=None, help="Override IK postprocess velocity threshold.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render kp2d/incam/global videos.")
    return parser.parse_args()


def demo_video_root(video_path: str | Path) -> str:
    return f"outputs/demo_video/{Path(video_path).stem}"


def default_pva_result(video_path: str | Path) -> str:
    return f"{demo_video_root(video_path)}/pvanet/pva_results_L120_ov30.pt"


def default_motion_path(video_path: str | Path, source: str) -> str:
    resolved_video = project_path(video_path).resolve()
    bundled_video = project_path("assets/demo/demo.mp4").resolve()
    if resolved_video == bundled_video and source == "gvhmr":
        return "assets/demo/gvhmr/hmr4d_results.pt"
    stem = Path(video_path).stem
    if source == "gvhmr":
        return f"inputs/gvhmr/{stem}/hmr4d_results.pt"
    return f"inputs/tram/{stem}"


def default_output_root(video_path: str | Path, source: str, static_cam: bool) -> str:
    camera_mode = "static" if static_cam else "dynamic"
    return f"{demo_video_root(video_path)}/optimization_{source}_{camera_mode}"


def _unwrap_batch(x: torch.Tensor) -> torch.Tensor:
    return x[0] if x.ndim >= 1 and x.shape[0] == 1 else x


def load_pva_targets(path: str | Path) -> dict[str, torch.Tensor | None]:
    data = torch_load(resolve_existing_project_path(path), map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"PVA result must be a dict, got {type(data)!r}")
    nested = data.get(0) if 0 in data and isinstance(data[0], dict) else data

    def pick(*keys: str) -> torch.Tensor:
        for key in keys:
            if key in nested:
                return _unwrap_batch(to_tensor(nested[key])).float()
            if key in data:
                return _unwrap_batch(to_tensor(data[key])).float()
        raise KeyError(f"Cannot find any of keys {keys} in PVA result.")

    conf = None
    for key in ("pred_conf", "keypoint_confidence", "pred_conf_0", "pred_kp2d_conf"):
        if key in nested:
            conf = _unwrap_batch(to_tensor(nested[key])).float()
            break
        if key in data:
            conf = _unwrap_batch(to_tensor(data[key])).float()
            break

    return {
        "kp2d": pick("pred_p2d", "image_keypoints", "pred_p2d_0", "pred_kp2d"),
        "vel": pick("vel", "vel_3d", "camera_velocities", "pred_vel_3d", "pred_vel"),
        "accel": pick("accel", "accel_3d", "camera_accelerations", "pred_accel_3d", "pred_accel"),
        "conf": conf,
    }


def load_motion(args: argparse.Namespace) -> dict[str, Any]:
    path = args.motion or default_motion_path(args.video, args.source)
    if args.source == "gvhmr":
        return load_gvhmr_result(project_path(path))
    return load_tram_result(project_path(path))


def default_model(args: argparse.Namespace) -> str:
    if args.model != "auto":
        return args.model
    return "smplx" if args.source == "gvhmr" else "smpl"


def identity_t_w2c(length: int) -> torch.Tensor:
    return torch.eye(4).repeat(length, 1, 1)


def load_camera_traj(path: str | Path, frame_indices: torch.Tensor) -> tuple[torch.Tensor, Path]:
    """Load and select an explicit per-video `[L,4,4] T_w2c` trajectory."""
    traj_path = resolve_existing_project_path(path)
    if not traj_path.exists():
        raise FileNotFoundError(f"Camera trajectory not found: {traj_path}")

    if traj_path.suffix in {".npy", ".npz"}:
        traj_obj = np.load(traj_path, allow_pickle=True)
        if isinstance(traj_obj, np.lib.npyio.NpzFile):
            archive = traj_obj
            try:
                traj_obj = {key: archive[key] for key in archive.files}
            finally:
                archive.close()
        elif isinstance(traj_obj, np.ndarray) and traj_obj.shape == ():
            traj_obj = traj_obj.item()
    else:
        traj_obj = torch_load(traj_path, map_location="cpu")

    if isinstance(traj_obj, dict):
        if "T_w2c" not in traj_obj:
            raise KeyError(
                f"Camera trajectory dictionaries must contain an explicit 'T_w2c' array: {traj_path}"
            )
        traj_obj = traj_obj["T_w2c"]

    traj = to_tensor(traj_obj).float()
    if traj.ndim != 3 or traj.shape[-2:] != (4, 4):
        raise ValueError(
            f"Camera trajectory must have shape [L,4,4] and contain T_w2c matrices, got {tuple(traj.shape)}."
        )
    if not torch.isfinite(traj).all():
        raise ValueError(f"Camera trajectory contains non-finite values: {traj_path}")
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=traj.dtype)
    if not torch.allclose(traj[:, 3], expected_bottom.expand_as(traj[:, 3]), atol=1e-5, rtol=0.0):
        raise ValueError(f"Camera trajectory has invalid homogeneous bottom rows: {traj_path}")
    rotation = traj[:, :3, :3]
    identity = torch.eye(3, dtype=traj.dtype).expand(rotation.shape[0], -1, -1)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-3, rtol=1e-3):
        raise ValueError(f"Camera trajectory rotations are not orthonormal: {traj_path}")
    if (torch.linalg.det(rotation) <= 0).any():
        raise ValueError(f"Camera trajectory rotations must have positive determinant: {traj_path}")

    frame_indices = torch.as_tensor(frame_indices, dtype=torch.long)
    if frame_indices.ndim != 1 or frame_indices.numel() == 0:
        raise ValueError("frame_indices must be a non-empty 1D tensor.")
    if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= traj.shape[0]:
        raise ValueError(
            "Camera trajectory does not cover all aligned video frames: "
            f"requested [{int(frame_indices.min())}, {int(frame_indices.max())}], length={traj.shape[0]}"
        )
    return traj[frame_indices], traj_path


def estimate_k(video_path: Path, length: int) -> torch.Tensor:
    _, width, height = get_video_lwh(video_path)
    return estimate_K(width, height).repeat(length, 1, 1).float()


MIN_OPTIMIZATION_FRAMES = 4


def slice_params(
    params: dict[str, torch.Tensor] | None,
    indices: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    if params is None:
        return None
    return {key: value[indices].float() for key, value in params.items()}


def select_alignment_indices(
    motion: dict[str, Any],
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map motion rows to original video frames, keeping one continuous 30 FPS span."""
    motion_len = motion_length(motion)
    raw_frames = motion.get("frames")
    if raw_frames is None:
        length = min(motion_len, target_length)
        indices = torch.arange(length, dtype=torch.long)
        return indices, indices.clone()

    frames = torch.as_tensor(raw_frames, dtype=torch.long)
    if frames.ndim != 1 or len(frames) != motion_len:
        raise ValueError(f"Motion frames must have shape [{motion_len}], got {tuple(frames.shape)}")
    if (frames < 0).any():
        raise ValueError("Motion frame indices must be non-negative.")
    if len(frames) > 1 and (frames[1:] <= frames[:-1]).any():
        raise ValueError("Motion frame indices must be strictly increasing.")

    best_start = best_end = 0
    run_start: int | None = None
    for idx, frame in enumerate(frames.tolist()):
        valid = frame < target_length
        continues = idx > 0 and frame == int(frames[idx - 1]) + 1
        if valid and (run_start is None or continues):
            if run_start is None:
                run_start = idx
        elif valid:
            run_start = idx
        else:
            if run_start is not None and idx - run_start > best_end - best_start:
                best_start, best_end = run_start, idx
            run_start = None
        if run_start is not None and idx + 1 - run_start > best_end - best_start:
            best_start, best_end = run_start, idx + 1

    motion_indices = torch.arange(best_start, best_end, dtype=torch.long)
    frame_indices = frames[motion_indices]
    if len(frame_indices) < motion_len:
        print(
            f"[alignment] using the longest continuous TRAM span: "
            f"motion rows {best_start}:{best_end}, video frames "
            f"{int(frame_indices[0]) if len(frame_indices) else 'none'}:"
            f"{int(frame_indices[-1]) + 1 if len(frame_indices) else 'none'}"
        )
    return motion_indices, frame_indices


def rotate_smpl_root(params: dict[str, torch.Tensor], rotation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rotate root orientation and translation into a new global coordinate frame."""
    rotation = rotation.to(device=params["global_orient"].device, dtype=params["global_orient"].dtype)
    out = {key: value.clone() for key, value in params.items()}
    root_rot = axis_angle_to_matrix(out["global_orient"])
    out["global_orient"] = matrix_to_axis_angle(rotation[None] @ root_rot)
    out["transl"] = torch.einsum("ij,fj->fi", rotation, out["transl"])
    return out


def tram_static_global_from_incam(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert TRAM camera-space SMPL roots to the static-demo y-up frame."""
    camera_to_y_up = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=params["global_orient"].dtype))
    return rotate_smpl_root(params, camera_to_y_up)


def align_inputs(
    motion: dict[str, Any],
    targets: dict[str, torch.Tensor | None],
    video_path: Path,
    static_cam: bool,
    camera_traj: str | Path | None,
    camera_intrinsics: str | Path | None,
):
    kp2d, vel, accel, conf = targets["kp2d"], targets["vel"], targets["accel"], targets["conf"]
    if kp2d is None or vel is None or accel is None:
        raise KeyError("PVA targets must contain keypoints, velocities, and accelerations.")
    video_length = get_video_lwh(video_path)[0]
    target_length = min(video_length, len(kp2d), len(vel) + 1, len(accel) + 2)
    motion_indices, frame_indices = select_alignment_indices(motion, target_length)
    length = len(frame_indices)
    if length < MIN_OPTIMIZATION_FRAMES:
        raise ValueError(
            f"Optimization needs at least {MIN_OPTIMIZATION_FRAMES} consecutive frames, got {length}."
        )

    global_params = slice_params(motion.get("pred_smpl_params_global"), motion_indices)
    incam_params = slice_params(motion.get("pred_smpl_params_incam"), motion_indices)
    if incam_params is None and global_params is None:
        raise KeyError("Motion has neither incam nor global SMPL params.")
    global_source = "input" if global_params is not None else None
    if static_cam and global_params is None and incam_params is not None and motion.get("source") == "tram":
        global_params = tram_static_global_from_incam(incam_params)
        global_source = "tram_incam_camera_to_y_up"
    if camera_intrinsics is not None:
        intrinsics_path = resolve_existing_project_path(camera_intrinsics)
        K_all = load_camera_intrinsics(intrinsics_path, length=int(frame_indices[-1]) + 1)[0]
        K = K_all[frame_indices]
        intrinsics_source = str(intrinsics_path)
    elif motion.get("K_fullimg") is not None:
        K_motion = normalize_camera_intrinsics(
            motion["K_fullimg"],
            length=motion_length(motion),
        )[0]
        K = K_motion[motion_indices]
        intrinsics_source = f"{motion.get('source', 'motion')}:K_fullimg"
    else:
        K = estimate_k(video_path, length)
        intrinsics_source = "estimate_K_from_video_size"
    if static_cam:
        T_w2c = identity_t_w2c(length)
        camera_source = "identity"
    else:
        if camera_traj is None:
            raise ValueError(
                "Dynamic optimization requires --camera_traj; motion outputs are not used as an implicit camera."
            )
        T_w2c, camera_source_path = load_camera_traj(camera_traj, frame_indices)
        camera_source = str(camera_source_path)
    return {
        "length": length,
        "frame_indices": frame_indices,
        "global_params": global_params,
        "incam_params": incam_params,
        "kp2d": kp2d[frame_indices].float(),
        "vel": vel[frame_indices[:-1]].float(),
        "accel": accel[frame_indices[:-2]].float(),
        "conf": None if conf is None else conf[frame_indices].float(),
        "K": K,
        "T_w2c": T_w2c,
        "camera_mode": "static" if static_cam else "trajectory",
        "camera_source": camera_source,
        "intrinsics_source": intrinsics_source,
        "global_source": global_source,
    }


def run_optimization(
    aligned: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    model_name: str,
    device: torch.device,
) -> dict[str, Any]:
    opt_path = out_dir / "opt_results.pt"
    preserve_camera = args.source == 'gvhmr' and args.static_cam
    if opt_path.exists() and not args.overwrite:
        cached = torch_load(opt_path, map_location="cpu")
        if not preserve_camera or cached.get('meta', {}).get('camera_alignment_version') == 1:
            return cached
        if not args.run_optimization:
            raise ValueError('Cached HTD motion uses an older camera alignment. Run refinement again.')
    if not args.run_optimization and not args.overwrite:
        raise FileNotFoundError(f"Missing cached {opt_path}; pass --run_optimization to compute it.")

    incam = (
        None
        if aligned["incam_params"] is None
        else {key: value.to(device) for key, value in aligned["incam_params"].items()}
    )
    glob = (
        None
        if aligned["global_params"] is None
        else {key: value.to(device) for key, value in aligned["global_params"].items()}
    )
    if not args.static_cam and incam is not None:

        glob_for_opt = None
        init_mode = "incam_plus_camera_traj"
    else:
        glob_for_opt = glob
        init_mode = "global_params" if glob is not None else "incam_params"
    smpl_params_global_opt, T_w2c_optimized, scale = opt_vel(
        tgt_vel_3d=aligned["vel"].to(device),
        tgt_accel_3d=aligned["accel"].to(device),
        tgt_kp2d=aligned["kp2d"].to(device),
        K=aligned["K"].to(device),
        T_w2c_tgt=aligned["T_w2c"].to(device),
        pred_smpl_params_incam=incam,
        pred_smpl_params_global=glob_for_opt,
        model=model_name,
        vid=Path(args.video).stem,
        static=args.static_cam,
        tgt_conf=None if aligned["conf"] is None else aligned["conf"].to(device),
        vel_thr=args.vel_thr,
        preserve_input_camera=preserve_camera,
    )
    result = {
        "smpl_params_global_optimized": {k: v.detach().cpu() for k, v in smpl_params_global_opt.items()},
        "smpl_params_global_input": aligned["global_params"],
        "smpl_params_incam_input": aligned["incam_params"],
        "K_fullimg": aligned["K"],
        "frame_indices": aligned["frame_indices"],
        "T_w2c_input": T_w2c_optimized.detach().cpu() if preserve_camera else aligned["T_w2c"],
        "T_w2c_optimized": T_w2c_optimized.detach().cpu(),
        "scale": float(scale) if not isinstance(scale, torch.Tensor) else float(scale.detach().cpu().item()),
        "meta": {
            "video": str(args.video),
            "source": args.source,
            "model": model_name,
            "init_mode": init_mode,
            "camera_mode": aligned["camera_mode"],
            "camera_source": 'gvhmr_global_incam_first_frame' if preserve_camera else aligned["camera_source"],
            "camera_alignment_version": 1 if preserve_camera else 0,
            "camera_locked": preserve_camera,
            "intrinsics_source": aligned["intrinsics_source"],
            "global_source": aligned["global_source"],
            "vel_thr": args.vel_thr,
            "device": str(device),
        },
    }
    torch.save(result, opt_path)
    return result


def render_outputs(
    aligned: dict[str, Any],
    opt_results: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    model_name: str,
    device: torch.device,
):
    video_path = project_path(args.video)
    kp2d_video = render_kp2d_overlay(
        video_path,
        aligned["kp2d"],
        out_dir / "0_input_kp2d.mp4",
        args.overwrite,
        frame_indices=aligned["frame_indices"],
    )
    incam_target = out_dir / "1_incam.mp4"
    global_target = out_dir / "2_global.mp4"
    incam_params = opt_results.get("smpl_params_incam_input") or aligned["incam_params"]
    if incam_params is not None:
        incam_video = render_incam_video(
            video_path,
            incam_params,
            aligned["K"],
            incam_target,
            model_name,
            args.overwrite,
            device=device,
            frame_indices=aligned["frame_indices"],
        )
    else:
        incam_video = render_global_from_incam(
            video_path,
            opt_results["smpl_params_global_optimized"],
            aligned["T_w2c"],
            incam_target,
            model_name,
            args.overwrite,
            device=device,
        )

    global_video = render_global_video(
        video_path,
        opt_results["smpl_params_global_optimized"],
        global_target,
        model_name,
        args.overwrite,
        device=device,
    )
    return {"kp2d": kp2d_video, "incam": incam_video, "global": global_video}


def main() -> None:
    args = parse_args()
    if args.static_cam and args.camera_traj is not None:
        raise ValueError("--static_cam and --camera_traj are mutually exclusive.")
    args.static_cam = args.camera_traj is None or args.static_cam
    device = resolve_device(args.device)
    video_path = project_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    model_name = default_model(args)
    pva_result = args.pva_result or default_pva_result(args.video)
    output_root = project_path(args.output_root or default_output_root(args.video, args.source, args.static_cam))
    out_dir = output_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_pva_targets(pva_result)
    motion = load_motion(args)
    aligned = align_inputs(
        motion,
        targets,
        video_path,
        static_cam=args.static_cam,
        camera_traj=args.camera_traj,
        camera_intrinsics=args.camera_intrinsics,
    )
    opt_results = run_optimization(aligned, args, out_dir, model_name, device)
    print(f"[device] {device}")
    print(
        f"[alignment] video_frames={int(aligned['frame_indices'][0])}:"
        f"{int(aligned['frame_indices'][-1]) + 1} length={aligned['length']}"
    )
    print(f"[camera] mode={aligned['camera_mode']} source={aligned['camera_source']}")
    print(f"[intrinsics] source={aligned['intrinsics_source']}")
    print(f"[optimization] opt_results={out_dir / 'opt_results.pt'}")
    if args.render:
        videos = render_outputs(aligned, opt_results, args, out_dir, model_name, device)
        for key, value in videos.items():
            if value is not None:
                print(f"[video] {key}: {value}")


if __name__ == "__main__":
    main()
