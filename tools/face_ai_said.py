"""Experimental two-second, audio-guided mouth edit; preserves the raw capture."""
from pathlib import Path
import argparse, json, sys, types, subprocess, tempfile, os
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--frame',type=int,required=True)
p.add_argument('--app',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
root=a.runtime.resolve();sys.path.insert(0,str(a.app))
from mocap.face_extract import ARKIT,source_identity
source=source_identity(a.project);capture=a.project/'face.npz';stamp=capture.stat()
with np.load(capture,allow_pickle=False) as z:
    raw=z['values'].astype(np.float32);valid=z['valid'];fps=float(z['fps'])
    if json.loads(str(z['source']))!=source:raise ValueError('Stale face capture')
if not 0<=a.frame<len(raw):raise ValueError('Invalid frame')
end=min(len(raw),a.frame+round(2*fps));duration=(end-a.frame)/fps
if duration<.5 or not valid[a.frame:end].all():raise ValueError('Select at least half a second with detected faces')
names=['jawForward','jawLeft','jawRight','jawOpen','mouthClose','mouthFunnel','mouthPucker','mouthLeft','mouthRight','mouthSmileLeft','mouthSmileRight','mouthFrownLeft','mouthFrownRight','mouthDimpleLeft','mouthDimpleRight','mouthStretchLeft','mouthStretchRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthPressLeft','mouthPressRight','mouthLowerDownLeft','mouthLowerDownRight','mouthUpperUpLeft','mouthUpperUpRight','cheekPuff','cheekSquintLeft','cheekSquintRight','noseSneerLeft','noseSneerRight']
indices=[ARKIT.index(n) for n in names]
audio_source=a.project/'input.mp4';offset=0.
metadata=a.project/'project.json'
if metadata.is_file():
    import hashlib
    info=json.loads(metadata.read_text());original=Path(info['source']['path'])
    if not original.is_file():raise ValueError('Original video with audio is missing')
    digest=hashlib.sha256()
    with original.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=info.get('source_sha256'):raise ValueError('Original audio/video identity changed')
    audio_source=original;offset=float(info['start'])
command=[str(a.app/'tools/bin/ffmpeg.exe'),'-v','error','-ss',str(offset+a.frame/fps),'-i',str(audio_source),'-t','2','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']
audio=np.frombuffer(subprocess.check_output(command),dtype='<f4').copy()
if len(audio)<8000 or np.max(np.abs(audio))<1e-5:raise ValueError('No usable audio in the selected working-video segment')
audio=np.pad(audio[:32000],(0,max(0,32000-len(audio))))
samples=np.column_stack([np.interp(np.arange(120)/60,np.arange(end-a.frame)/fps,raw[a.frame:end,i]) for i in indices]).astype(np.float32)
package=types.ModuleType('said');package.__path__=[str(root/'SAiD/said')];sys.modules['said']=package
from said.model.diffusion import SAID_UNet1D
from transformers import Wav2Vec2FeatureExtractor
processor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
processor.feature_extractor=processor
model=SAID_UNet1D(audio_processor=processor);model.load_state_dict(torch.load(root/'SAiD.pth',map_location='cpu',weights_only=True));model.eval().cuda()
torch.manual_seed(42)
with torch.inference_mode():
    result=model.inference(model.process_audio([audio]).cuda(),init_samples=torch.from_numpy(samples[None]).cuda(),num_inference_steps=50,strength=.2,show_process=False).result[0].cpu().numpy()
if result.shape!=(120,32) or not np.isfinite(result).all():raise ValueError('Invalid model result')
output=raw.copy();times=np.arange(end-a.frame)/fps
# Blend the candidate gently and fade at both boundaries of the edited interval.
weight=.35*np.minimum(np.minimum(times/.15,(duration-1/fps-times)/.15),1).clip(0,1)
for j,i in enumerate(indices):
    candidate=np.interp(times,np.arange(120)/60,result[:,j]).clip(0,1)
    output[a.frame:end,i]=(1-weight)*raw[a.frame:end,i]+weight*candidate
if source_identity(a.project)!=source or capture.stat().st_mtime_ns!=stamp.st_mtime_ns:raise ValueError('Capture changed during inference')
fd,tmp=tempfile.mkstemp(prefix='face_ai-',suffix='.npz',dir=a.project);os.close(fd)
try:
    np.savez(tmp,values=output,source=json.dumps(source),capture_bytes=stamp.st_size,capture_mtime=stamp.st_mtime_ns)
    os.replace(tmp,a.project/'face_ai.npz')
finally:
    if os.path.exists(tmp):os.unlink(tmp)
print('SAiD candidate ready: frames',a.frame+1,end,flush=True)
