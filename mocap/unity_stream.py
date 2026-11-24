"""Local Unity transport: bounded, latest-frame snapshots in metres, Unity axes.

Wire format: little-endian uint32 byte count followed by UTF-8 JSON (v1).
Reference and pose are global row-major 3x4 matrices, including all fingers.
"""
import json
import socket
import struct

import numpy as np

from .bone_stream import BoneStream, MAX_ACTORS

LIMIT = 8 * 1024 * 1024
BLENDER_TO_UNITY = np.array(((1., 0., 0.), (0., 0., 1.), (0., 1., 0.)))


def unity_matrices(matrices):
    value = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 4)
    if not np.isfinite(value).all():
        raise ValueError('Unity stream contains non-finite bone transforms.')
    result = value.copy()
    c = BLENDER_TO_UNITY
    result[:, :, :3] = c @ value[:, :, :3] @ c.T
    result[:, :, 3] = (c @ value[:, :, 3, None])[..., 0]
    rotation = result[:, :, :3]
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=.004)
            or np.any(np.linalg.det(rotation) < .99)):
        raise ValueError('Unity streaming requires rigid bone transforms (no scaled scene transforms).')
    return result.ravel().tolist()


def snapshot(actors):
    if len(actors) > MAX_ACTORS:
        raise ValueError('Too many Unity streaming actors.')
    output = []
    for actor, name, rig, frame, world, transform, _project in actors:
        bones = rig.bones
        if not 0 < len(bones) <= 256:
            raise ValueError('Unity supports 1–256 bones per actor.')
        parents = [int(b[1]) for b in bones]
        if any(p < -1 or p >= i for i, p in enumerate(parents)):
            raise ValueError('Invalid Unity bone hierarchy.')
        pose = np.frombuffer(rig.pose(world, transform), dtype='<f4').reshape(-1, 3, 4)
        output.append(dict(id=actor, name=name, rig=rig.key, frame=int(frame),
                           names=[b[0] for b in bones], parents=parents,
                           rest=unity_matrices([b[2] for b in bones]), pose=unity_matrices(pose)))
    raw = json.dumps(dict(version=1, actors=output), ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > LIMIT:
        raise ValueError('Unity frame exceeds 8 MiB.')
    return struct.pack('<I', len(raw)) + raw


class UnityStream(BoneStream):
    """Shares the loopback listener/lifecycle, not the Blender wire protocol."""
    def _send(self, client):
        last = None
        while not self.stopping.is_set():
            self.changed.clear()
            current = self.latest
            if current is not last:
                client.sendall(snapshot(current[0]))
                last = current
            # Heartbeats allow a paused timeline to detect disconnects and reconnect.
            if not self.changed.wait(.5):
                client.sendall(struct.pack('<I', 0))
