from pathlib import Path
import os,sys,time,json,ast,types
from ..paths import ROOT,MODELS
from ..pear_runtime import activate
from ..pear_models import geometry_folder
ENGINE=activate()
ASSETS=geometry_folder()
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache/pear/yolo')
os.environ['TORCH_HOME']=str(ROOT/'cache/pear/torch')
os.environ['TEMP']=str(ROOT/'temp');os.environ['TMP']=str(ROOT/'temp')
(ROOT/'cache/pear/yolo').mkdir(parents=True,exist_ok=True)
(ROOT/'temp/torch/kernels').mkdir(parents=True,exist_ok=True)

import numpy as np
import inspect
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for _name,_value in {'bool':np.bool_,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
    if _name not in np.__dict__:setattr(np,_name,_value)
import torch
import cv2
from omegaconf import OmegaConf
torch.set_num_threads(4)

def crop_functions():
    # Import ONLY official crop definitions; don't import its renderer/demo dependencies.
    source=ENGINE/'inference_images.py';tree=ast.parse(source.read_text(encoding='utf8'))
    names={'sanitize_bbox','process_bbox','rotate_2d','gen_trans_from_patch_cv','generate_patch_image'}
    mod=ast.Module([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    scope={'np':np,'cv2':cv2};exec(compile(mod,str(source),'exec'),scope);return scope
CROP=crop_functions()

class Pear:
    def __init__(self,mesh=True,detector='s',profile='fast',detect_every=1,progress=lambda text:None,fixed_rig=False):
        if detector not in ('s','x'):raise ValueError('Unsupported PEAR person detector')
        progress('Preparing PEAR inference modules…')
        from .acceleration import PROFILES,Network,trim_unused_geometry,install_cached_fk
        self.profile=profile;self.config=PROFILES[profile];self.geometry_mode=self.config['geometry'];self.detect_every=detect_every;self.frame_counter=0
        if self.config.get('fast_fk'):install_cached_fk()
        start=time.perf_counter();old=os.getcwd();os.chdir(ASSETS.parent)
        try:
            from models.pipeline.ehm_pipeline import Ehm_Pipeline
            cfg=OmegaConf.load(ENGINE/'configs/infer.yaml')
            progress('Constructing PEAR network…')
            self.model=Ehm_Pipeline(cfg)
            progress('Loading PEAR checkpoint on CPU…')
            state=torch.load(MODELS/'pear/pear_model.pt',map_location='cpu',weights_only=True,mmap=True)
            for key in ['backbone','head']:
                missing=getattr(self.model,key).load_state_dict(state[key],strict=True)
            del state
            if self.config['precision']!='fp32':
                self.model.backbone.to(dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[self.config['precision']])
            progress('Moving PEAR network to GPU…')
            self.model=self.model.cuda().eval()
            self.ehm=None
            self.rig=None
            if mesh or fixed_rig:
                from models.modules.ehm.EHM_v2 import EHM_v2
                progress('Loading registered body geometry…')
                self.ehm=EHM_v2('assets/FLAME','assets/SMPLX').eval()
                if profile!='baseline':trim_unused_geometry(self.ehm)
                if fixed_rig:
                    from .fixed_rig import FixedRig
                    self.rig=FixedRig(self.ehm);self.parents=self.rig.native_parents
                    self.ehm=None  # Calibration stays on CPU; never evaluate skin per live frame.
                else:
                    self.ehm=self.ehm.cuda()
                    self.faces=self.ehm.smplx.faces_tensor.detach().cpu().numpy()
                    self.parents=self.ehm.smplx.parents.detach().cpu().numpy()
                    from ..source_surface import layout,endpoints
                    neutral=self.ehm.smplx.v_template.detach().cpu().numpy()
                    self.neutral_joints=(self.ehm.smplx.J_regressor.detach().cpu().numpy()@neutral).astype('float32')
                    ids,lookup=layout('smplx')
                    self.neutral_surface=endpoints(neutral[None,ids],lookup)[0].astype('float32')
            if profile!='baseline':
                self.model=Network(self.model,self.config).eval()
                torch.cuda.empty_cache()
        finally:os.chdir(old)
        progress('Loading person detector…')
        from ultralytics import YOLO
        from ultralytics.nn import tasks
        from ..storage import sha256
        from ..models import BY_KEY
        detector_spec=BY_KEY['pear_yolo' if detector=='s' else 'yolo'];path=detector_spec.path
        if sha256(path)!=detector_spec.expected_sha256:raise ValueError('Invalid official YOLO checkpoint')
        original_loader=tasks.torch_safe_load
        def trusted_loader(weight,*args,**kwargs):
            if Path(weight).resolve()!=path.resolve():return original_loader(weight,*args,**kwargs)
            return torch.load(path,map_location='cpu',weights_only=False),str(path)
        tasks.torch_safe_load=trusted_loader
        self.detector=YOLO(str(path));self.detector_kind=detector
        self.load_seconds=time.perf_counter()-start
        self.previous=None
    @torch.inference_mode()
    def run(self,bgr,box_override=None,parameter_filter=None,timestamp=None,geometry=True):
        start=time.perf_counter();rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        t=time.perf_counter()
        if box_override is not None:boxes=np.asarray(box_override)[None]
        elif self.previous is not None and self.frame_counter%self.detect_every:
            boxes=self.previous[None].copy()
        else:
            result=self.detector.predict(bgr,device=0,classes=[0],conf=.5,imgsz=640,verbose=False)[0]
            boxes=result.boxes.xyxy.detach().cpu().numpy()
        self.frame_counter+=1
        detection=(time.perf_counter()-t)*1000
        if not len(boxes):return None
        if self.previous is None:
            box=max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        else:
            # Stay with the previous person while visible; do not jump to another person.
            p=self.previous;lo=np.maximum(boxes[:,:2],p[:2]);hi=np.minimum(boxes[:,2:],p[2:])
            area=np.maximum(0,hi-lo).prod(1)
            iou=area/np.maximum(1,(boxes[:,2:]-boxes[:,:2]).prod(1)+(p[2:]-p[:2]).prod()-area)
            if iou.max()<.1:self.previous=None;return None
            box=boxes[iou.argmax()]
        self.previous=box.copy();x,y,x2,y2=box
        t=time.perf_counter()
        bbox=CROP['process_bbox'](np.array([x,y,x2-x,y2-y]),rgb.shape[1],rgb.shape[0],[256,256],ratio=1.25)
        if bbox is None:return None
        patch,_,_=CROP['generate_patch_image'](rgb,bbox,1.,0.,False,[256,256])
        inp=torch.from_numpy(patch.astype(np.float32).transpose(2,0,1).copy()).unsqueeze(0).cuda()/255
        preprocess=(time.perf_counter()-t)*1000
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True)
        a.record();out=self.model(inp);b.record()
        raw_body=None;raw_flame=None;filtered_body=None;filtered_flame=None
        body=out['body_param'];flame=out['flame_param']
        if parameter_filter is not None:
            from .parameter_io import filter_parameters
            raw_body,raw_flame,filtered_body,filtered_flame,body,flame=filter_parameters(body,flame,parameter_filter,timestamp)
        geom=None
        if self.ehm is not None and geometry:geom=self.ehm(body,flame,pose_type='aa')
        c.record();c.synchronize()
        packet={'body':filtered_body if filtered_body is not None else {k:v.detach().cpu().numpy() for k,v in out['body_param'].items() if torch.is_tensor(v)},
                'flame':filtered_flame if filtered_flame is not None else {k:v.detach().cpu().numpy() for k,v in out['flame_param'].items() if torch.is_tensor(v)},
                'camera':out['pd_cam'].detach().cpu().numpy()[0],'bbox':box,'crop_bbox':bbox.copy(),
                'image_size':np.array([bgr.shape[1],bgr.shape[0]],dtype=np.int32),'patch':patch.astype(np.uint8)}
        packet['raw_body']=raw_body if raw_body is not None else packet['body']
        packet['raw_flame']=raw_flame if raw_flame is not None else packet['flame']
        packet['smoothing_enabled']=parameter_filter is not None
        if geom is not None:
            packet['vertices']=geom['vertices'].detach().cpu().numpy()[0]
            packet['joints']=geom['joints'].detach().cpu().numpy()[0]
            from .bone_roll import source_rotations
            packet['rotations']=source_rotations(packet['body'],self.parents,packet['camera'])
        packet['timing']={'detection_ms':detection,'preprocess_ms':preprocess,'pear_cuda_ms':a.elapsed_time(b),
                          'geometry_cuda_ms':b.elapsed_time(c),'total_ms':(time.perf_counter()-start)*1000}
        if not all(np.isfinite(v).all() for v in packet['body'].values()):raise RuntimeError('Nonfinite PEAR output')
        return packet
