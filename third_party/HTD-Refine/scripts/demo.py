"""Run the HTD-Refine demo: PVA-Net targets, optimization, and a three-panel video."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from htd_refine.runtime import project_path
from utils.vis import merge_three_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="assets/demo/demo.mp4")
    parser.add_argument("--source", default="gvhmr", choices=["gvhmr", "tram"])
    parser.add_argument("--motion", default=None, help="GVHMR .pt path or TRAM sequence folder.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device used by PVA-Net, YOLO, optimization, and rendering.",
    )
    parser.add_argument("--pva_window_length", type=int, default=120, help="PVA-Net sliding-window length.")
    parser.add_argument("--pva_overlap", type=int, default=30, help="PVA-Net sliding-window overlap.")
    parser.add_argument(
        "--camera_intrinsics",
        "--intrinsics",
        default=None,
        help="Optional .pt/.npy/.npz camera intrinsics. If omitted, each stage uses its documented fallback.",
    )
    parser.add_argument(
        "--camera_traj",
        "--traj",
        default=None,
        help="Explicit [L,4,4] world-to-camera trajectory. If omitted, the demo uses a static camera.",
    )
    parser.add_argument("--run_optimization", action="store_true", help="Compute optimization if no cache exists.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def demo_video_root(video_path: str | Path) -> str:
    return f"outputs/demo_video/{Path(video_path).stem}"


def default_pva_output(video_path: str | Path, window_length: int, overlap: int) -> str:
    return f"{demo_video_root(video_path)}/pvanet/pva_results_L{window_length}_ov{overlap}.pt"


def default_output_root(video_path: str | Path, source: str, dynamic_camera: bool) -> str:
    camera_mode = "dynamic" if dynamic_camera else "static"
    return f"{demo_video_root(video_path)}/optimization_{source}_{camera_mode}"


def run_command(cmd: list[str]) -> None:
    print("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    video_path = project_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    dynamic_camera = args.camera_traj is not None
    pva_output = default_pva_output(args.video, args.pva_window_length, args.pva_overlap)

    pva_script = SCRIPT_ROOT / "demo_pvanet.py"
    opt_script = SCRIPT_ROOT / "demo_optimization.py"

    pva_cmd = [
        sys.executable,
        str(pva_script),
        "--input",
        args.video,
        "--output",
        pva_output,
        "--device",
        args.device,
        "--window_length",
        str(args.pva_window_length),
        "--overlap",
        str(args.pva_overlap),
    ]
    if args.camera_intrinsics is not None:
        pva_cmd.extend(["--camera_intrinsics", args.camera_intrinsics])
    if args.overwrite:
        pva_cmd.append("--overwrite")
    run_command(pva_cmd)

    opt_cmd = [
        sys.executable,
        str(opt_script),
        "--video",
        args.video,
        "--pva_result",
        pva_output,
        "--source",
        args.source,
        "--device",
        args.device,
    ]
    if args.motion is not None:
        opt_cmd.extend(["--motion", args.motion])
    if args.camera_intrinsics is not None:
        opt_cmd.extend(["--camera_intrinsics", args.camera_intrinsics])
    if args.camera_traj is not None:
        opt_cmd.extend(["--camera_traj", args.camera_traj])
    if args.run_optimization or args.overwrite:
        opt_cmd.append("--run_optimization")
    if args.overwrite:
        opt_cmd.append("--overwrite")
    opt_cmd.append("--render")
    run_command(opt_cmd)

    out_root = default_output_root(args.video, args.source, dynamic_camera)
    stem = video_path.stem
    videos = [
        project_path(f"{out_root}/{stem}/0_input_kp2d.mp4"),
        project_path(f"{out_root}/{stem}/1_incam.mp4"),
        project_path(f"{out_root}/{stem}/2_global.mp4"),
    ]
    for video in videos:
        if not video.exists():
            raise FileNotFoundError(f"Missing intermediate video: {video}")
    merged = merge_three_videos(
        videos,
        project_path(f"{out_root}/{stem}/demo_three_panel.mp4"),
        overwrite=args.overwrite,
        labels=["input + kp2d", "incam", "global"],
    )
    print(f"[merge] {merged}")


if __name__ == "__main__":
    main()
