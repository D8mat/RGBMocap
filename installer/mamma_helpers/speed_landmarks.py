"""Bounded masked inference. Existing dataset transforms and output schema retained."""
from pathlib import Path
import os,time,json
import numpy as np
import torch
import cv2
from torch.utils.data._utils.collate import default_collate
from lib.datasets.vitdet_dataset import ViTDetDataset

def process_fast(source,detector,device,model,cfg,out_folder,save_cam_output,masks_path):
    assert detector is None and masks_path is not None and not save_cam_output
    batch_size=1
    if not getattr(ViTDetDataset,'_roi_installed',False):
        from speed_crop import install_exact_roi_crop
        install_exact_roi_crop(ViTDetDataset)
        ViTDetDataset._roi_installed=True
    camera=str(source.cam_name)
    masks=Path(masks_path)/camera/'masks'
    ids=list(range(1,max(int(p.stem.split('_')[2]) for p in masks.glob('*.png'))+1))
    count=len(source)
    arrays={k:np.zeros((count,len(ids),cfg.num_joints,3 if k=='landmarks' else 1),dtype=np.float32) for k in ['landmarks','visibilities','contacts','floor_contacts']}
    queue=[];times={'crop_seconds':0.,'inference_seconds':0.,'decode_seconds':0.,'calls':0,'items':0}
    def flush():
        if not queue:return
        items=default_collate([x[2] for x in queue])
        img=items['img'].to(device);mask=items['mask'].to(device)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            output=model(img,mask)
        torch.cuda.synchronize();times['inference_seconds']+=time.perf_counter()-start
        points=output['joints2d'].float().cpu().numpy()
        if cfg.data['train']['normalize_plus_min_one']:
            points[:,:,:2]=(points[:,:,:2]+1)/2
        # Backproject exactly as upstream, per crop, without image readback.
        centers=items['box_center'].float().numpy();sizes=items['box_size'].float().numpy()
        for dim in (0,1):points[:,:,dim]=points[:,:,dim]*sizes[:,None,dim]-sizes[:,None,dim]/2+centers[:,None,dim]
        signals={k:torch.sigmoid(output[v].squeeze(-1).float()).cpu().numpy() for k,v in [('visibilities','visibility'),('contacts','contact'),('floor_contacts','floor_contact')]}
        for j,(frame,person,_) in enumerate(queue):
            arrays['landmarks'][frame,person]=points[j]
            for key,values in signals.items():arrays[key][frame,person,:,0]=values[j]
        times['calls']+=1;times['items']+=len(queue);queue.clear()
    for frame in range(count):
        start=time.perf_counter();image=cv2.cvtColor(source.read_rgb(frame),cv2.COLOR_RGB2BGR);times['decode_seconds']+=time.perf_counter()-start
        for person,pid in enumerate(ids):
            path=masks/f'mask_{frame:04d}_{pid:02d}.png'
            mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) if path.exists() else None
            if mask is None or np.sum(mask)/mask.size<.01:continue
            ys,xs=np.where(mask);boxes=np.array([[xs.min(),ys.min(),xs.max(),ys.max()]])
            start=time.perf_counter();dataset=ViTDetDataset(cfg,image,mask,boxes=boxes)
            item=dataset[0];times['crop_seconds']+=time.perf_counter()-start
            queue.append((frame,person,item))
            if len(queue)>=batch_size:flush()
        if frame%10==0:print(f'FAST camera={camera} frame={frame}/{count}',flush=True)
    flush()
    # Match upstream squeeze(-1) signal layout.
    for key in ('visibilities','contacts','floor_contacts'):arrays[key]=arrays[key][...,0]
    np.savez(Path(out_folder)/f'{camera}.npz',**arrays,contacts_gt=None,floor_contacts_gt=None)
    (Path(os.environ['MAMMA_SPEED_DIR'])/f'profile-{camera}.json').write_text(json.dumps(times,indent=2))
    print('FAST_PROFILE '+json.dumps(times),flush=True)
