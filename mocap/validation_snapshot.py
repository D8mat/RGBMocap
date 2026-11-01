"""Share expensive read-only checks within one synchronous GUI update only."""
from contextvars import ContextVar
from functools import wraps

_snapshot = ContextVar('preview_validation_snapshot', default=None)


def read_only_checks(method):
    """No Qt event processing, writes or retained cache inside this scope."""
    @wraps(method)
    def scoped(*args, **kwargs):
        if _snapshot.get() is not None:
            return method(*args, **kwargs)
        token = _snapshot.set({})
        try:
            return method(*args, **kwargs)
        finally:
            _snapshot.reset(token)
    return scoped


def snapshot_value(check):
    @wraps(check)
    def checked(*args, **kwargs):
        cache = _snapshot.get()
        if cache is None:
            return check(*args, **kwargs)
        key = (check, args, tuple(sorted(kwargs.items())))
        if key not in cache:
            value = check(*args, **kwargs)
            if len(cache) < 64:
                cache[key] = value
            return value
        return cache[key]
    return checked
