"""Verify pinned dependencies and actual native CUDA kernels, without weights."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import sys


def verify(lock):
    if tuple(map(int,lock['python'].split('.')))!=sys.version_info[:3]:raise RuntimeError('MAMMA Python version mismatch.')
    for p in lock['packages']:
        if metadata.version(p['name'])!=p['version']:raise RuntimeError('MAMMA dependency mismatch: '+p['name'])
    import torch,numpy,scipy,smplx,timm,torchvision,open_clip,cv2,sam2
    from sam2 import _C
    from pytorch_sdf import BVH,solid_angles_cuda
    if not torch.cuda.is_available():raise RuntimeError('MAMMA requires an NVIDIA CUDA GPU.')
    torch.set_num_threads(4);torch.manual_seed(17)
    # Windows long/int64 and native-build mistakes must fail here, not on a video.
    import trimesh
    mesh=trimesh.creation.box(extents=[1,1,1])
    tri=torch.tensor(numpy.asarray(mesh.triangles).copy(),dtype=torch.float32,device='cuda')[None]
    pts=(torch.rand(1,101,3,device='cuda')*2-1).requires_grad_()
    distance,closest,indices,_=BVH()(tri,pts)
    _,reference,_=trimesh.proximity.closest_point_naive(mesh,pts.detach().cpu().numpy()[0])
    error=float(numpy.max(numpy.abs(distance.detach().cpu().numpy()[0]-reference**2)))
    distance.sum().backward();gradient_error=float((pts.grad-2*(pts.detach()-closest)).abs().max())
    if error>2e-6 or gradient_error>2e-6 or indices.min()<0 or indices.max()>=len(mesh.faces):raise RuntimeError('SDF CUDA validation failed.')
    probes=torch.tensor([[[0.,0.,0.],[2.,0.,0.]]],device='cuda')
    winding=solid_angles_cuda.solid_angles_cuda(probes,tri,1e-8).sum(-1)/(4*torch.pi)
    if abs(abs(float(winding[0,0]))-1)>1e-4 or abs(float(winding[0,1]))>1e-4:raise RuntimeError('SDF winding validation failed.')
    mask=torch.zeros((1,1,8,8),device='cuda',dtype=torch.uint8);mask[:,:,2:4,2:4]=1
    from sam2.utils.misc import get_connected_components
    labels,areas=get_connected_components(mask)
    if int(areas[0,0,2,2])!=4:raise RuntimeError('SAM 2 CUDA components validation failed.')
    checked=0
    for name in [p['name'] for p in lock['packages']]+['SAM-2','pytorch_sdf']:
        distribution=metadata.distribution(name)
        for entry in distribution.files or []:
            if not entry.hash or entry.hash.mode!='sha256' or Path(entry).suffix.lower() in ('.pyc','.pyo','.lib','.exp','.pdb','.obj','.o','.a','.ilk'):continue
            path=Path(distribution.locate_file(entry))
            hasher=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2),b''):hasher.update(block)
            value=hasher.digest()
            if base64.urlsafe_b64encode(value).decode().rstrip('=')!=entry.hash.value:
                raise RuntimeError('MAMMA installed file failed verification: '+str(entry))
            checked+=1
    return dict(python=sys.version.split()[0],packages=len(lock['packages'])+2,verified_files=checked,
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        sdf_error=error,sdf_gradient_error=gradient_error,sam2_components=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();report=verify(json.loads(args.lock.read_text()))
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
