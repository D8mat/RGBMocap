"""Optional MAMMA worker, isolated from the editor's Python dependencies."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
import os
os.environ.setdefault('RGB_MOCAP_WORKER_THREADS','4')
from mocap.compute_policy import configure_worker_cpu
configure_worker_cpu(sys.argv[1])
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')
from mocap.mamma_runtime import activate
engine = activate()
import atexit,time,json
started=time.monotonic()
stage=sys.argv[1]
request_path=sys.argv[sys.argv.index('--request')+1] if '--request' in sys.argv else None
def report_resources():
    try:
        import psutil
        info=psutil.Process().memory_info()
        torch=sys.modules.get('torch')
        cuda=bool(torch is not None and torch.cuda.is_initialized())
        row=dict(stage=stage,process_seconds=time.monotonic()-started,
            peak_working_set_bytes=getattr(info,'peak_wset',info.rss),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else 0)
        print('@@MOCAP@@'+json.dumps(dict(type='resource_usage',**row)),flush=True)
        if request_path:
            from mocap.mamma_pipeline import request,work_for
            job=request(request_path)
            from mocap.storage import atomic_json
            suffix='-cache-resources.json' if job.get('cached') else '-resources.json'
            atomic_json(work_for(job)/(row['stage']+suffix),row)
    except Exception:pass
atexit.register(report_resources)
from mocap.mamma_pipeline import main
raise SystemExit(main(sys.argv[1:], engine=engine))
