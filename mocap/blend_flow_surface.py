"""Stream the committed FlowHMR surface into a portable Blender PC2 cache."""
import struct
from pathlib import Path
import numpy as np
from .blend_support import disk_check


def bake(project,info,scene,work,inputs,emit=lambda *a,**k:None):
    paths={}
    for key in ('raw_file','raw_asset'):
        name=info.get(key,'')
        if not name or Path(name).name!=name:raise ValueError('Prepare the FlowHMR mesh preview before Blender export')
        paths[key]=inputs.track(Path(project)/name)
    with np.load(paths['raw_asset'],allow_pickle=False) as asset:
        count=int(asset['vertex_count']);faces=asset['faces']
    if count!=6890 or faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer) or not faces.size or faces.min()<0 or faces.max()>=count:
        raise ValueError('Invalid FlowHMR surface topology')
    transform=np.eye(4) if scene is None else np.asarray(scene['transform'],float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid body placement')
    frames=np.load(paths['raw_file'],mmap_mode='r',allow_pickle=False)
    try:
        if frames.shape!=(int(info['frames']),count,3) or frames.dtype!=np.float32:
            raise ValueError('FlowHMR surface does not match the selected animation')
        disk_check(Path(work),len(frames)*count*12+count*12+faces.nbytes)
        with (Path(work)/'body.pc2').open('xb') as out:
            out.write(struct.pack('<12siiffi',b'POINTCACHE2\0',1,count,0.,1.,len(frames)))
            for index,frame in enumerate(frames):
                vertices=np.asarray(frame,dtype=np.float64)@transform[:3,:3].T+transform[:3,3]
                vertices=vertices[:,[0,2,1]].copy();vertices[:,1]*=-1
                if not np.isfinite(vertices).all():raise ValueError('Non-finite FlowHMR surface')
                if index==0:np.savez(Path(work)/'body_geometry.npz',vertices=vertices,faces=faces)
                out.write(vertices.astype('<f4').tobytes())
                if index%30==0:emit('progress',stage='save_blend',current=index,total=len(frames),message='Baking FlowHMR body surface')
    finally:frames._mmap.close()
    return dict(geometry='body_geometry.npz',cache='body.pc2',vertices=count)
