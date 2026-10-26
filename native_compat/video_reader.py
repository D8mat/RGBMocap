"""Native Windows PyAV adapter for the small upstream video-reader API."""

from __future__ import annotations

import av
import cv2
import numpy as np


class PyVideoReader:
    def __init__(self, path, threads=0, resize_shorter_side=None):
        self.path, self.threads, self.resize_shorter_side = path, threads, resize_shorter_side

    def decode(self, start_frame=0, end_frame=-1):
        frames = []
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            if self.threads:
                stream.codec_context.thread_count = self.threads
            for index, frame in enumerate(container.decode(stream)):
                if index < start_frame:
                    continue
                if end_frame >= 0 and index >= end_frame:
                    break
                rgb = frame.to_ndarray(format="rgb24")
                if self.resize_shorter_side:
                    scale = self.resize_shorter_side / min(rgb.shape[:2])
                    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                frames.append(rgb)
        if not frames:
            raise ValueError("No video frames decoded")
        return np.stack(frames)
