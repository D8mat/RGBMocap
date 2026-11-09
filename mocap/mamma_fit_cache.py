"""Bounded, fit-scoped reuse of visibility smoothing; no model/precision changes."""
from collections import OrderedDict
from functools import wraps


def smooth_visibility(values):
    import torch
    if values is None:
        return None
    result = []
    for v in values:
        q = ((v.mean(-1) - .03) / .12).clamp(0, 1)
        q = q*q*(3-2*q)
        if len(q) > 1:
            kernel = v.new_tensor([1., 4., 6., 4., 1.])[None, None]/16
            for _ in range(2):
                q = torch.nn.functional.conv1d(torch.nn.functional.pad(
                    q[None, None], (2, 2), mode='replicate'), kernel)[0, 0]
        result.append(v*q[:, None])
    return result


class VisibilityCache:
    def __init__(self, max_bytes=128*1024**2, max_entries=8):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.current = None

    def prepare(self, raw, sources, skip, indices):
        import numpy as np
        self.current = None
        if raw is None or sources is None:
            return
        # Never retain an autograd graph. Unsupported/inference tensors fall back.
        if any(v.requires_grad for v in sources) or any(v.requires_grad for v in raw):
            return
        try:
            versions = tuple((id(v), v._version) for v in sources)
        except RuntimeError:
            return
        if indices is None:
            index_key = None
        elif isinstance(indices, np.ndarray):
            index_key = (indices.dtype.str, indices.shape, indices.tobytes())
        else:
            return
        key = (versions, max(0, int(skip)), index_key)
        entry = self.entries.pop(key, None)
        if entry is not None and all(a is b for a, b in zip(entry[0], sources)):
            self.hits += 1
            self.entries[key] = entry
            prepared = entry[1]
        else:
            self.misses += 1
            prepared = smooth_visibility(raw)
            size = sum(v.numel()*v.element_size() for v in prepared)
            if size <= self.max_bytes and self.max_entries > 0:
                while self.entries and (self.bytes+size > self.max_bytes or len(self.entries) >= self.max_entries):
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= old[2]
                self.entries[key] = (tuple(sources), prepared, size)
                self.bytes += size
        self.current = (raw, prepared)

    def weights(self, raw):
        if self.current is not None and self.current[0] is raw:
            return self.current[1]
        return smooth_visibility(raw)

    def clear(self):
        self.entries.clear()
        self.current = None
        self.bytes = 0


def install(optimizer_class, cache):
    """Return restoration callback; external runtime files remain untouched."""
    original = optimizer_class.process_data

    @wraps(original)
    def process(self, pts3d_pred, joints3d_pred, lndmks2d, lndmks2d_vis,
                hand_joints2d, triangulated_3d_points, triangulated_contacts,
                skip_first_t_frames=0, valid_verts_idx=None):
        result = original(self, pts3d_pred, joints3d_pred, lndmks2d, lndmks2d_vis,
                          hand_joints2d, triangulated_3d_points, triangulated_contacts,
                          skip_first_t_frames=skip_first_t_frames, valid_verts_idx=valid_verts_idx)
        cache.prepare(result[6], lndmks2d_vis, skip_first_t_frames, valid_verts_idx)
        return result

    optimizer_class.process_data = process
    def restore():
        optimizer_class.process_data = original
        cache.clear()
    return restore
