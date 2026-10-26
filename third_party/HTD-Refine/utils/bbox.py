"""Bounding-box and crop helpers for HTD-Refine demos."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def convert_bbx_xyxy_to_center_width_height(bbx_xyxy):
    center_x = (bbx_xyxy[..., 0] + bbx_xyxy[..., 2]) / 2
    center_y = (bbx_xyxy[..., 1] + bbx_xyxy[..., 3]) / 2
    width = bbx_xyxy[..., 2] - bbx_xyxy[..., 0]
    height = bbx_xyxy[..., 3] - bbx_xyxy[..., 1]
    return center_x, center_y, width, height


def crop_and_resize_np(img, bbx, dst_wh=(192, 256), bbx_format="xyxy"):
    if bbx_format != "xyxy":
        raise NotImplementedError(f"unsupported bbox format {bbx_format!r}")
    center_x, center_y, width, height = convert_bbx_xyxy_to_center_width_height(bbx)
    src = np.stack(
        [
            np.array([center_x - width / 2, center_y - height / 2]),
            np.array([center_x + width / 2, center_y - height / 2]),
            np.array([center_x, center_y]),
        ]
    ).astype(np.float32)
    dst = np.array([[0, 0], [dst_wh[0] - 1, 0], [dst_wh[0] / 2 - 0.5, dst_wh[1] / 2 - 0.5]], dtype=np.float32)
    return cv2.warpAffine(img, cv2.getAffineTransform(src, dst), (dst_wh[0], dst_wh[1]), flags=cv2.INTER_LINEAR)


def expand_w_h_to_ratio(w, h, width_height_ratio=(192, 256)):
    w_t, h_t = width_height_ratio
    if isinstance(w, torch.Tensor):
        h_new = torch.where(h / w < h_t / w_t, w * h_t / w_t, h)
        w_new = torch.where(h / w < h_t / w_t, w, h * w_t / h_t)
        return w_new, h_new
    if isinstance(w, np.ndarray):
        h_new = np.where(h / w < h_t / w_t, w * h_t / w_t, h)
        w_new = np.where(h / w < h_t / w_t, w, h * w_t / h_t)
        return w_new, h_new
    return max(h * w_t / h_t, w), max(w * h_t / w_t, h)


def shrink_bbx_xyxy_to_ratio(bbx_xyxy, width_height_ratio=(192, 256)):
    center_x, center_y, width, height = convert_bbx_xyxy_to_center_width_height(bbx_xyxy)
    width_new = width * (width_height_ratio[0] / width_height_ratio[1])
    height_new = height
    values = [center_x - width_new / 2, center_y - height_new / 2, center_x + width_new / 2, center_y + height_new / 2]
    if isinstance(bbx_xyxy, torch.Tensor):
        return torch.stack(values, dim=-1)
    return np.stack(values, axis=-1)


def expand_bbx_xyxy_to_ratio(bbx_xyxy, width_height_ratio=(192, 256)):
    center_x, center_y, width, height = convert_bbx_xyxy_to_center_width_height(bbx_xyxy)
    width_new, height_new = expand_w_h_to_ratio(width, height, width_height_ratio)
    values = [center_x - width_new / 2, center_y - height_new / 2, center_x + width_new / 2, center_y + height_new / 2]
    if isinstance(bbx_xyxy, torch.Tensor):
        return torch.stack(values, dim=-1)
    return np.stack(values, axis=-1)


__all__ = ["crop_and_resize_np", "expand_bbx_xyxy_to_ratio", "shrink_bbx_xyxy_to_ratio"]
