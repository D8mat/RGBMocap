"""Run with MAMMA Python; --engine points at the installed licensed engine."""
import argparse, ast, sys, time, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.mamma_fit_cache import VisibilityCache, smooth_visibility, install
p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(2)
# Load only actual solver slicing and loss functions, without body/model assets.
scope={'torch':torch,'np':np,'List':list}
tree=ast.parse((args.engine/'optimization/losses/losses.py').read_text())
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
mod=ast.Module(body=funcs,type_ignores=[])
source=ast.unparse(mod).replace('torch.clip(per_point_weight[cam_id][..., None], min=vis_clip_value)','torch.clamp(per_point_weight[cam_id][..., None], min=0., max=1.)')
exec(source,scope)
tree=ast.parse((args.engine/'optimization/utils/optimization.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OptimizeSMPLX')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('process_data','_resolve_frame_slice')]
exec(compile(ast.Module(body=[cls],type_ignores=[]),'solver','exec'),scope)
Optimizer=scope['OptimizeSMPLX']
results=[]
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 torch.manual_seed(27)
 cache=VisibilityCache(); restore=install(Optimizer,cache)
 obj=Optimizer();T,N,C=40,32,3
 obj.K=[torch.eye(3,device=device).repeat(T,1,1) for _ in range(C)]
 obj.extrinsics=[torch.eye(4,device=device).repeat(T,1,1) for _ in range(C)]
 sources=[torch.rand(T,N,device=device)*.2 for _ in range(C)]
 points=[torch.rand(T,N,2,device=device) for _ in range(C)]
 indices=np.arange(0,N,2)
 def run():
  xyz=(torch.rand(T,N,3,device=device)+1).requires_grad_()
  return obj.process_data(xyz,xyz,points,sources,None,None,None,skip_first_t_frames=2,valid_verts_idx=indices)
 for repeat in range(3):
  out=run(); raw=out[6]; old=smooth_visibility(raw); new=cache.weights(raw)
  assert all(torch.equal(a,b) for a,b in zip(old,new))
  for loss in ('mse','huber','tukey','geman_mcclure'):
   a=scope['proj_pts_loss'](out[0],out[3],out[1],out[2],loss,old)
   b=scope['proj_pts_loss'](out[0],out[3],out[1],out[2],loss,new)
   ga=torch.autograd.grad(a,out[3],retain_graph=True)[0];gb=torch.autograd.grad(b,out[3],retain_graph=True)[0]
   assert torch.equal(a,b) and torch.equal(ga,gb),(device,loss)
 assert cache.hits==2
 sources[0].mul_(.5);out=run();assert cache.misses==2
 assert all(torch.equal(a,b) for a,b in zip(smooth_visibility(out[6]),cache.weights(out[6])))
 indices=np.arange(1,N,2);run();assert cache.misses==3
 sources[0]=sources[0].clone();run();assert cache.misses==4
 sources[0].requires_grad_();out=run();assert cache.current is None
 grad=torch.autograd.grad(sum(v.sum() for v in cache.weights(out[6])),sources[0])[0];assert grad is not None
 restore();assert cache.bytes==0 and not cache.entries
 # Isolated visibility preparation benchmark, including slicing on both paths.
 sources=[torch.rand(300,512,device=device)*.2 for _ in range(2)];indices=np.arange(512)
 cache=VisibilityCache()
 def baseline():return smooth_visibility([v[:,indices] for v in sources])
 def cached():
  raw=[v[:,indices] for v in sources];cache.prepare(raw,sources,0,indices);return cache.weights(raw)
 for _ in range(3):baseline();cached()
 def measure(fn):
  if device=='cuda':torch.cuda.synchronize()
  start=time.perf_counter()
  for _ in range(30):fn()
  if device=='cuda':torch.cuda.synchronize()
  return (time.perf_counter()-start)/30*1000
 results.append(dict(device=device,baseline_ms=measure(baseline),cached_ms=measure(cached)))
 # LRU and oversized fallback.
 tiny=VisibilityCache(max_bytes=1);raw=[v[:,indices] for v in sources];tiny.prepare(raw,sources,0,indices);assert not tiny.entries
 bounded=VisibilityCache(max_entries=1)
 bounded.prepare(raw,sources,0,indices);bounded.prepare(raw,sources,1,indices);assert len(bounded.entries)==1
 assert cache.weights(None) is None
print(json.dumps({'passed':True,'exact_loss_and_gradient':True,'visibility_only_benchmark':results},indent=2))
