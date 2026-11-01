"""Bounded CPU preparation; inference stays on the consuming worker thread."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

MAX_PREPARED_BYTES = 256 * 1024**2


def preparation_depth(batch_bytes=0):
    import psutil
    from .compute_policy import prefetch_plan
    memory = psutil.virtual_memory()
    depth = prefetch_plan(memory.available, memory.total)
    if batch_bytes > 0:
        depth = min(depth, MAX_PREPARED_BYTES // batch_bytes)
    return max(0, min(2, depth))


class _PreparedBatches:
    def __init__(self, source, depth):
        # DataLoader iterator creation can draw a seed. Keep it on the consumer
        # before starting CPU work so model randomness retains its ordering.
        self.source = iter(source)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cpu-prep')
        self.pending = deque()
        self.closed = False
        try:
            for _ in range(depth):
                self.pending.append(self.pool.submit(next, self.source))
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        try:
            while self.pending:
                future = self.pending.popleft()
                try:
                    value = future.result()
                except StopIteration:
                    return
                self.pending.append(self.pool.submit(next, self.source))
                yield value
                del value
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for future in self.pending:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        close = getattr(self.source, 'close', None)
        if close is not None:
            close()


@contextmanager
def batch_source(source, *, batch_bytes=0):
    depth = preparation_depth(batch_bytes)
    if not depth:
        yield source
        return
    prepared = _PreparedBatches(source, depth)
    try:
        yield prepared
    finally:
        prepared.close()
