"""OpenCV drawing helpers for demo visualization."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, list):
        return np.array(x)
    return x.clone().cpu().numpy()


def draw_coco17_skeleton(img, keypoints, conf_thr=0, colors=(0, 255, 0), radius=6):
    use_conf_thr = keypoints.shape[1] == 3
    img = img.copy()
    coco_skel = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]
    for bone in coco_skel:
        kp1 = keypoints[bone[0]][:2].astype(int)
        kp2 = keypoints[bone[1]][:2].astype(int)
        if use_conf_thr:
            kp1_c = keypoints[bone[0]][2]
            kp2_c = keypoints[bone[1]][2]
            if kp1_c > conf_thr and kp2_c > conf_thr:
                img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), colors, radius - 2)
            if kp1_c > conf_thr:
                img = cv2.circle(img, (kp1[0], kp1[1]), radius, colors, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, (kp2[0], kp2[1]), radius, colors, -1)
        else:
            img = cv2.line(img, (kp1[0], kp1[1]), (kp2[0], kp2[1]), colors, radius - 2)
    return img


def draw_coco17_skeleton_batch(imgs, keypoints_batch, conf_thr=0, colors=(0, 255, 0)):
    if len(imgs) != len(keypoints_batch):
        raise ValueError(f"image/keypoint length mismatch: {len(imgs)} != {len(keypoints_batch)}")
    keypoints_batch = to_numpy(keypoints_batch)
    return [draw_coco17_skeleton(imgs[i], keypoints_batch[i], conf_thr, colors) for i in range(len(imgs))]


__all__ = ["draw_coco17_skeleton_batch"]
