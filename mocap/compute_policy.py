"""Worker resource budgets. GUI imports stay free of Torch and CUDA."""
from __future__ import annotations
import os
GIB=1024**3
MODES=('auto','low_memory')


def ram_reserve(total):
    return max(GIB,min(4*GIB,int(total*.08)))


def gpu_reserve(total):
    return max(.75*GIB,min(1.5*GIB,total*.10))


def worker_threads(stage, logical=None):
    logical=max(1,logical or os.cpu_count() or 1)
    cap=4 if stage in ('refine','handflow') else 16 if stage=='footmr' else 8
    default=2 if stage=='correct' else min(cap,max(1,logical//2))
    try:requested=int(os.environ.get('RGB_MOCAP_WORKER_THREADS',default))
    except ValueError:requested=default
    return max(1,min(requested,max(1,logical-2)))


def configure_worker_cpu(stage):
    # Do not inherit arbitrary desktop BLAS pools; large pools slow preprocessing.
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    return worker_threads(stage)


def prefetch_plan(available_ram,total_ram,logical=None,mode=None):
    """Overlap CPU crops only when the desktop has spare CPU and RAM."""
    logical=logical or os.cpu_count() or 1
    mode=mode or os.environ.get('RGB_MOCAP_COMPUTE_MODE','auto')
    return 2 if (mode!='low_memory' and logical>=8 and
                 available_ram-ram_reserve(total_ram)>=6*GIB) else 0


def gvhmr_plan(frames,free_vram,total_vram):
    """Use dense attention whenever it fits. Split queries, never keys/time context.

    Conservative estimates include model weights and whole-sequence intermediates.
    The validated query block is fixed at 2048; smaller unvalidated blocks are
    deliberately not used as an automatic escape hatch.
    """
    frames=max(1,int(frames))
    budget=max(0,free_vram-gpu_reserve(total_vram))
    fixed=.5*GIB+frames*196608
    dense=max(3*GIB,fixed+frames*frames*72)
    split=max(3*GIB,fixed+frames*min(frames,2048)*72)
    use_split=frames>2048 and dense>budget
    return dict(attention='query_split' if use_split else 'dense',query_block=2048,
                estimated_bytes=int(split if use_split else dense),
                dense_estimated_bytes=int(dense),fits=(split if use_split else dense)<=budget,
                precision='float32',full_key_context=True)


def image_plan(stage,free_vram,available_ram,mode=None):
    mode=mode or os.environ.get('RGB_MOCAP_COMPUTE_MODE','auto')
    if mode not in MODES:mode='auto'
    import sys
    torch=sys.modules.get('torch')
    initialized=bool(torch and torch.cuda.is_initialized())
    total=torch.cuda.get_device_properties(0).total_memory if initialized else free_vram
    budget=max(0,free_vram-gpu_reserve(total))
    if stage=='pva':
        # Called AFTER loading the 3.5GiB model. Measured total allocation peaks
        # at 5.08GiB even with batch 120; the temporal network sets that floor.
        batch=120 if budget>=2*GIB else 60 if budget>=1.7*GIB else 24 if budget>=1.5*GIB else 8
        if mode=='low_memory':batch=min(batch,24)
    else:
        # Real 5090 sweeps: WholeBody batch 8 is faster than 16/32, not just smaller.
        measured_gpu=initialized and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 5090'
        preferred=8 if stage=='footmr' and measured_gpu else 16
        batch=preferred if budget>=3.5*GIB else 8 if budget>=3*GIB else 4 if budget>=2.8*GIB else 1
        if mode=='low_memory':batch=min(batch,4)
    batch=min(batch,max(1,int(max(0,available_ram-GIB)/(64*1024**2))))
    return {'image_batch':batch,'core_frames':256 if mode=='low_memory' or budget<3*GIB else 1024,
            'halo_frames':720,'precision':'float32','flip_test':True,'mode':mode}


def resource_error(stage,frames,available_ram,total_ram,free_vram=None,total_vram=None):
    """Conservative admission estimates, not a claim of universal memory bounds."""
    frames=max(0,frames)
    fixed={'track':2.,'people':2.2,'person':1.,'pva':5.,'gvhmr':4.,'features':4.,'refine':2.2,
           'footmr':3.5,'handflow':10.,'correct':.8,'export':1.,'mesh':1.}.get(stage,.5)*GIB
    per_frame={'track':512,'pva':2048,'features':8192,'gvhmr':16384,'refine':8192,
               'footmr':16384,'handflow':32768,'correct':8192,'export':16384,'mesh':16384}.get(stage,4096)
    required=fixed+frames*per_frame+ram_reserve(total_ram)
    if available_ram<required:
        return (f'Not enough available RAM for {stage}. Need about {required/GIB:.1f} GiB including '
                f'the system reserve; {available_ram/GIB:.1f} GiB available. Close other apps or trim the clip.')
    if free_vram is not None:
        # Include whole-sequence intermediates and the selected attention path.
        gpu={'track':.6,'pva':5.2,'gvhmr':3.,'features':3.,'refine':.3,'footmr':3.,'handflow':3.8}.get(stage,.3)*GIB
        if stage=='pva':gpu=(3.5+.014*min(max(4,frames),120))*GIB
        # Upstream explicitly materializes an 8-head FP32 attention matrix,
        # plus another score/softmax buffer. This is quadratic, not image batching.
        if stage=='gvhmr':gpu=gvhmr_plan(frames,free_vram,total_vram or free_vram)['estimated_bytes']
        if stage=='refine':gpu+=frames*262144
        reserve=gpu_reserve(total_vram or free_vram)
        if free_vram<gpu+reserve:
            return (f'Not enough free GPU memory for {stage} and this clip length. Estimated need '
                    f'{(gpu+reserve)/GIB:.1f} GiB including the desktop reserve; '
                    f'{free_vram/GIB:.1f} GiB free. Close GPU apps or trim the clip.')
    return None


def retry_cuda(operation,batch,on_retry,oom_type=None,cleanup=None):
    """Leave the exception scope before cleanup; traceback must not retain CUDA tensors."""
    if oom_type is None:
        import torch
        oom_type=torch.cuda.OutOfMemoryError
    while True:
        failed=False
        try:
            return operation(batch)
        except oom_type:
            if batch<=1:raise RuntimeError('Not enough GPU memory even at batch 1. Close GPU apps or trim the clip.') from None
            failed=True
        if failed:
            batch=max(1,batch//2)
            if cleanup is None:
                import gc,torch
                gc.collect();torch.cuda.empty_cache()
            else:cleanup()
            on_retry(batch)


def idle_timeout(last_output,last_cpu_change,now,limit=180):
    return now-max(last_output,last_cpu_change)>limit
