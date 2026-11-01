"""Check exact packages and actual SAM2/torchvision CUDA operations."""
from pathlib import Path
import argparse,base64,hashlib,importlib.metadata as metadata,json,sys


def verify(lock):
    if sys.version.split()[0]!=lock['python']:raise RuntimeError('PromptHMR Python version mismatch.')
    for row in lock['packages']:
        if metadata.version(row['name'])!=row['version']:raise RuntimeError('PromptHMR dependency mismatch: '+row['name'])
    import torch,numpy,scipy,smplx,timm,torchvision,sam2,mmcv,mmengine,joblib,chumpy,xformers
    import os
    folder=Path(sys.executable).parent/'engine'
    sys.path.insert(0,str(folder));os.chdir(folder)
    from pipeline import Pipeline
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from sam2.utils.misc import get_connected_components
    if not torch.cuda.is_available():raise RuntimeError('PromptHMR requires a compatible NVIDIA CUDA GPU.')
    torch.set_num_threads(4)
    mask=torch.zeros((1,1,8,8),device='cuda',dtype=torch.uint8);mask[:,:,2:4,2:4]=1
    labels,areas=get_connected_components(mask)
    if int(areas[0,0,2,2])!=4:raise RuntimeError('SAM2 CUDA verification failed.')
    boxes=torch.tensor([[0.,0.,2.,2.],[.1,.1,2.,2.]],device='cuda');score=torch.tensor([.9,.8],device='cuda')
    if torchvision.ops.nms(boxes,score,.5).tolist()!=[0]:raise RuntimeError('Torchvision CUDA NMS verification failed.')
    q=torch.rand(1,2,32,16,device='cuda');a=torch.nn.functional.scaled_dot_product_attention(q,q,q)
    if not torch.isfinite(a).all():raise RuntimeError('Native attention verification failed.')
    count=0
    for name in [r['name'] for r in lock['packages']]+['SAM-2','chumpy']:
        dist=metadata.distribution(name)
        for entry in dist.files or []:
            if not entry.hash or entry.hash.mode!='sha256' or Path(entry).suffix.lower() in ('.pyc','.lib','.exp','.pdb','.obj','.o','.a','.ilk'):continue
            p=Path(dist.locate_file(entry));h=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(2**20),b''):h.update(block)
            if base64.urlsafe_b64encode(h.digest()).decode().rstrip('=')!=entry.hash.value:raise RuntimeError('Installed file verification failed: '+str(entry))
            count+=1
    return dict(python=sys.version.split()[0],packages=len(lock['packages'])+3,verified_files=count,
                compute_capability=list(torch.cuda.get_device_capability()),cuda=torch.version.cuda,sam2_components=True,torchvision_nms=True,native_attention=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=verify(json.loads(args.lock.read_text()));args.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
