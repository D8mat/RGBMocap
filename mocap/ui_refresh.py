"""Batch synchronous control changes without caching project validity."""
from functools import wraps


def batch_readiness(method):
    """Validate controls once after a nested GUI operation has finished.

    This scope never outlives the synchronous call. Async completions and
    export actions still perform their own fresh checks against disk.
    """
    @wraps(method)
    def batched(self, *args, **kwargs):
        self._readiness_depth = getattr(self, '_readiness_depth', 0) + 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._readiness_depth -= 1
            if not self._readiness_depth and getattr(self, '_readiness_pending', False):
                self._readiness_pending = False
                self.update_readiness()
    return batched
