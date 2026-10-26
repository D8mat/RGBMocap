"""Inference-only full-key query splitting; upstream equations and masks retained.

The local upstream hook only skips constructing the dense L*L mask. No SDPA,
reduced precision, independent clips or key-range truncation is used here.
"""
from contextlib import contextmanager
import math
from types import MethodType


def predict(model,data,project,emit):
    """Budget before allocation, and retry dense OOM once with released traceback."""
    import gc
    import torch
    from .compute_policy import gvhmr_plan,gpu_reserve
    from .storage import atomic_json
    frames=len(data['bbx_xys'])
    def capacity():
        free,total=torch.cuda.mem_get_info()
        allocated=torch.cuda.memory_allocated()
        # Respect both changing external GPU use and this process's allocator cap.
        cap=torch.cuda.get_per_process_memory_fraction()*total
        room=cap-torch.cuda.memory_reserved()+allocated
        return min(free+allocated,room+gpu_reserve(total)),total
    free,total=capacity()
    selected=gvhmr_plan(frames,free,total)
    if not selected['fits']:
        raise RuntimeError('Not enough free GPU memory for full-context GVHMR. Close GPU apps or trim the clip.')
    atomic_json(project/'gvhmr_attention_plan.json',selected)
    emit('compute_plan',**selected)
    failed=False
    try:
        with torch.inference_mode(),query_split(model,selected['attention']=='query_split'):
            result=model.predict(data,static_cam=True)
    except torch.cuda.OutOfMemoryError:
        if selected['attention']=='query_split':raise
        failed=True
    if not failed:return result
    # Never shrink the temporal/key context or move silently to a CPU backend.
    gc.collect();torch.cuda.empty_cache()
    free,total=capacity()
    low=gvhmr_plan(frames,0,total)
    if frames<=2048 or low['estimated_bytes']>free-gpu_reserve(total):
        raise RuntimeError('Not enough free GPU memory for full-context GVHMR. Close GPU apps or trim the clip.')
    low.update(fits=True,reduced_after_oom=True)
    atomic_json(project/'gvhmr_attention_plan.json',low)
    emit('compute_plan',**low)
    with torch.inference_mode(),query_split(model,True):
        return model.predict(data,static_cam=True)


def split_attention(self,x,attn_mask=None,key_padding_mask=None):
    import torch
    b,length,channels=x.shape
    q,k,v=[linear(x).reshape(b,length,self.num_heads,-1).transpose(1,2)
           for linear in (self.query,self.key,self.value)]
    q,k=self.rope.rotate_queries_or_keys(q),self.rope.rotate_queries_or_keys(k)
    output=torch.empty_like(v)
    columns=torch.arange(length,device=x.device)
    for first in range(0,length,2048):
        end=min(length,first+2048)
        scores=(q[:,:,first:end] @ k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if isinstance(attn_mask,int):
            rows=torch.arange(first,end,device=x.device)
            left=(rows-attn_mask//2).clamp(min=0,max=length-attn_mask)
            right=(rows+attn_mask//2).clamp(min=attn_mask,max=length)
            blocked=(columns[None]<left[:,None]) | (columns[None]>=right[:,None])
            scores=scores.masked_fill(blocked[None,None],-float('inf'))
        elif attn_mask is not None:
            scores=scores.masked_fill(attn_mask[None,None,first:end],-float('inf'))
        if key_padding_mask is not None:
            scores=scores.masked_fill(key_padding_mask[:,None,None,:],-float('inf'))
        weights=scores.softmax(-1)
        output[:,:,first:end]=weights @ v
        # Do not retain the preceding block while allocating the next one.
        del scores,weights
    return self.proj(output.transpose(1,2).reshape(b,length,channels))


@contextmanager
def query_split(model,enabled):
    if not enabled:
        yield
        return
    import torch
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE
    from hmr4d.network.base_arch.transformer.encoder_rope import RoPEAttention
    if model.training or torch.is_grad_enabled():
        raise RuntimeError('GVHMR query splitting is only supported for inference')
    saved=[]
    try:
        for module in model.modules():
            if isinstance(module,NetworkEncoderRoPE):
                saved.append((module,'_mocap_split_queries',module.__dict__.get('_mocap_split_queries')))
                module._mocap_split_queries=True
            elif isinstance(module,RoPEAttention):
                saved.append((module,'forward',module.__dict__.get('forward')))
                module.forward=MethodType(split_attention,module)
        if not any(name=='_mocap_split_queries' for _,name,_ in saved):
            raise RuntimeError('GVHMR temporal encoder is incompatible with query splitting')
        yield
    finally:
        for module,name,value in reversed(saved):
            if value is None:delattr(module,name)
            else:setattr(module,name,value)
