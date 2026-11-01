"""PromptHMR subprocess bootstrap. Dependencies never enter the editor process."""
from pathlib import Path
import sys,os,time,atexit,json
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
os.environ['PATH']=str(root/'tools/bin')+os.pathsep+os.environ.get('PATH','')
cache=Path(os.environ.get('RGB_MOCAP_DATA',root))/'cache/prompthmr';temp=cache/'temp';temp.mkdir(parents=True,exist_ok=True)
import tempfile
tempfile.tempdir=str(temp)
os.environ.update(TEMP=str(temp),TMP=str(temp),HF_HUB_OFFLINE='1',MPLBACKEND='Agg',WANDB_MODE='disabled',NO_ALBUMENTATIONS_UPDATE='1')
os.environ.setdefault('RGB_MOCAP_WORKER_THREADS','4')
from mocap.compute_policy import configure_worker_cpu
configure_worker_cpu(sys.argv[1])
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf8',errors='backslashreplace')
from mocap.prompthmr_runtime import activate
engine=activate();started=time.monotonic()
def resources():
    try:
        import psutil,torch
        info=psutil.Process().memory_info();cuda=torch.cuda.is_initialized()
        row=dict(stage=sys.argv[1],process_seconds=time.monotonic()-started,peak_working_set_bytes=getattr(info,'peak_wset',info.rss),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else 0,cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else 0)
        print('@@MOCAP@@'+json.dumps(dict(type='resource_usage',**row)),flush=True)
        from mocap.prompthmr_pipeline import request,work_for
        from mocap.storage import atomic_json
        job=request(sys.argv[sys.argv.index('--request')+1]);suffix='-cache' if job.get('cached') else ''
        atomic_json(work_for(job)/(row['stage']+suffix+'-resources.json'),row)
    except Exception:pass
atexit.register(resources)
from mocap.prompthmr_pipeline import main
raise SystemExit(main(sys.argv[1:],engine=engine))
