"""RGB MoCap Windows path compatibility addition (root MIT license)."""
import cv2
import numpy as np


def read_bgr_frame(path):
    """Decode unchanged OpenCV BGR pixels without its ANSI filename handling."""
    with open(path, 'rb') as stream:
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    return image


def read_rgb_frame(path):
    return cv2.cvtColor(read_bgr_frame(path), cv2.COLOR_BGR2RGB)
