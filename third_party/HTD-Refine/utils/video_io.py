"""Video I/O helpers for HTD-Refine demos."""

from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import torch
from video_reader import PyVideoReader


def get_video_lwh(video_path):
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0, threads=0, resize_shorter_side=None):
    options = {"threads": threads}
    options_decode = {}
    should_check_length = False
    if not (start_frame == 0 and end_frame == -1):
        options_decode["start_frame"] = start_frame
        options_decode["end_frame"] = end_frame
        should_check_length = True
    if scale != 1.0:
        if resize_shorter_side is not None:
            raise ValueError("only one option can be set: scale or resize_shorter_side")
        _, W, H = get_video_lwh(video_path)
        options["resize_shorter_side"] = int(scale * min(H, W))
    elif resize_shorter_side is not None:
        options["resize_shorter_side"] = resize_shorter_side
    frames = PyVideoReader(str(video_path), **options).decode(**options_decode)
    if should_check_length and len(frames) != end_frame - start_frame:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {end_frame - start_frame}")
    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def save_video(images, video_path, fps=30, crf=17):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)
    if images.shape[1] % 2 != 0 or images.shape[2] % 2 != 0:
        raise ValueError(f"H.264 output requires even frame size, got {images.shape[2]}x{images.shape[1]}")
    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


__all__ = ["get_video_lwh", "get_video_reader", "get_writer", "read_video_np", "save_video"]
