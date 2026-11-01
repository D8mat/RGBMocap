"""One resident model per worker phase; no cross-tool or cross-job global pool."""
from contextvars import ContextVar
from contextlib import contextmanager
import gc

_active = ContextVar('mocap_model_session', default=None)


def current():
    return _active.get()


def model_key(name, *paths, **settings):
    from pathlib import Path
    files = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        files.append((str(path), stat.st_size, stat.st_mtime_ns))
    return name, tuple(files), tuple(sorted(settings.items()))


def dispose(model):
    # Feature collectors must not retain GPU activations or form a hook cycle.
    for hook in getattr(model, '_hooks', ()):
        hook.remove()
    for name in ('_body_tokens','_hand_tokens'):
        values=getattr(model,name,None)
        if values is not None:values.clear()
    for values in getattr(model, '_flow_fast_collectors', ()):
        values.clear()
    close = getattr(model, 'close', None)
    if close is not None:close()
    inner = getattr(getattr(model, 'estimator', None), 'model', model)
    from .gpu_offload import release
    release(inner)


class ModelSession:
    def __init__(self):
        self.key = self.value = None
        self.resident_bytes = 0
        self.cap_set = False
        self.force_low = False

    def acquire(self, key, factory):
        if self.value is not None and key == self.key:return self.value
        self.discard()
        import torch
        before = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
        value = factory()
        self.key, self.value = key, value
        self.resident_bytes = max(0, torch.cuda.memory_allocated() - before) if torch.cuda.is_initialized() else 0
        return value

    def discard(self):
        value, self.value = self.value, None
        self.key = None
        self.resident_bytes = 0
        try:
            if value is not None:dispose(value)
        finally:
            del value
            gc.collect()
            import torch
            if torch.cuda.is_initialized():torch.cuda.empty_cache()

    @contextmanager
    def activate(self):
        if current() is not None:raise RuntimeError('Nested model sessions are not supported')
        token = _active.set(self)
        try:yield self
        finally:
            try:self.discard()
            finally:_active.reset(token)


def acquire(key, factory):
    session = current()
    return factory() if session is None else session.acquire(key, factory)


def release(model):
    session = current()
    if session is None or session.value is not model:dispose(model)
    else:
        # Clear only transient tokens; the weights and their one hook stay live.
        if hasattr(model, '_tokens'):model._tokens = None
        for name in ('_body_tokens','_hand_tokens'):
            values=getattr(model,name,None)
            if values is not None:values.clear()
        for values in getattr(model, '_flow_fast_collectors', ()):values.clear()


def reusable_bytes():
    session = current()
    return session.resident_bytes if session is not None else 0


def set_memory_budget(free, total):
    import torch
    from .compute_policy import gpu_memory_fraction
    session = current()
    # Fix the allocator cap once per phase. Reapplying a cap from free memory
    # after each actor would subtract the retained weights repeatedly.
    if session is None or not session.cap_set:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(free, total))
        if session is not None:session.cap_set = True
