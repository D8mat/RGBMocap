"""Per-operation resource selection; never changes precision or saved UI settings."""
import gc
import os
from contextlib import contextmanager
from .compute_policy import GIB, gpu_reserve

VERSION = 1
GUIDANCE = ('For memory shortages, select Processing Resources > Resource Mode > Low Memory. '
            'CPU loading/offloading can be slower and use more RAM. Close other apps '
            'or trim the clip if memory is still insufficient.')


def requested_mode():
    return 'low_memory' if os.environ.get('RGB_MOCAP_COMPUTE_MODE') == 'low_memory' else 'auto'


def effective_mode():
    return os.environ.get('RGB_MOCAP_EFFECTIVE_MODE', requested_mode())


def decide(requested, available, required):
    """Advisory routing only: insufficient estimates never reject a job."""
    return 'low_memory' if requested == 'low_memory' or available < required else 'auto'


def available_vram():
    import torch
    free, total = torch.cuda.mem_get_info()
    # The allocator cap was chosen when this worker started. Other apps can
    # change free memory later; respect both limits without subtracting twice.
    cap = torch.cuda.get_per_process_memory_fraction() * total
    alloc = torch.cuda.memory_allocated()
    from .model_session import current
    session = current()
    if session is not None:
        cached = max(0, torch.cuda.memory_reserved() - alloc)
        additional = max(0, min(free + cached - gpu_reserve(total), cap - alloc))
        return additional + session.resident_bytes
    room = cap - torch.cuda.memory_reserved() + alloc
    return max(0, min(free - gpu_reserve(total), room))


def announce(stage, mode, reason, emit=None):
    selected = 'Low Memory' if mode == 'low_memory' else 'Automatic / fast path'
    message = f'Resources [{stage}]: {selected}. {reason}'
    if mode == 'low_memory':
        message += ' ' + GUIDANCE
    if emit:
        emit('resource_mode', stage=stage, mode=mode, requested=requested_mode(), message=message)
    else:
        import json
        print('@@MOCAP@@' + json.dumps(dict(type='resource_mode', stage=stage, mode=mode,
              requested=requested_mode(), message=message)), flush=True)


@contextmanager
def effective(mode):
    previous = os.environ.get('RGB_MOCAP_EFFECTIVE_MODE')
    os.environ['RGB_MOCAP_EFFECTIVE_MODE'] = mode
    try:
        yield
    finally:
        if previous is None:os.environ.pop('RGB_MOCAP_EFFECTIVE_MODE', None)
        else:os.environ['RGB_MOCAP_EFFECTIVE_MODE'] = previous


def run(stage, operation, fast_gib, emit=None):
    """Retry a fresh, scoped operation once after leaving the OOM traceback.

    operation(low_memory) must own its models/captures and publish no partial
    result. The caller commits only the returned complete CPU output.
    """
    import torch
    available = available_vram()
    from .model_session import current
    session = current()
    selected = decide('low_memory' if session is not None and session.force_low else requested_mode(), available, fast_gib * GIB)
    reason = ('Selected by the user.' if requested_mode() == 'low_memory' else
              f'Available GPU budget {available/GIB:.2f} GiB; fast-path estimate {fast_gib:.2f} GiB.')
    announce(stage, selected, reason, emit)
    failed = False
    try:
        with effective(selected):
            return operation(selected == 'low_memory')
    except torch.cuda.OutOfMemoryError:
        if selected == 'low_memory':raise
        failed = True
    if failed:
        if session is not None:
            session.discard()
            session.force_low = True
        gc.collect()
        torch.cuda.empty_cache()
        announce(stage, 'low_memory', 'Automatic recovery after CUDA ran out of memory; retrying once.', emit)
        with effective('low_memory'):
            return operation(True)


def select_stage(stage, free, total, frames, emit):
    """Existing GVHMR/add-on policies keep full temporal/precision settings."""
    from .compute_policy import gvhmr_plan
    estimate = {'track':1., 'people':1.5, 'pva':5.5, 'features':4.5,
                'refine':3., 'footmr':5., 'handflow':5.}.get(stage, 3.) * GIB
    if stage == 'gvhmr':estimate = gvhmr_plan(frames, free, total)['dense_estimated_bytes']
    mode = decide(requested_mode(), max(0, free - gpu_reserve(total)), estimate)
    os.environ['RGB_MOCAP_EFFECTIVE_MODE'] = mode
    announce(stage, mode, 'Using the stage memory budget; model precision and temporal context are unchanged.', emit)
    return mode
