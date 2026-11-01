"""Bounded stage queues and request files; safe to import in the GUI."""
from pathlib import Path
import json
import uuid

BATCH_STAGES = {'flow_batch','exmr_batch','gvhmr_batch'}
PHASES = {
    'flow_batch': {'flow_features':'flow_features','flowhmr':'flowhmr','all':'flowhmr'},
    'exmr_batch': {'exmr_preprocess':'exmr_preprocess','body':'exmr_features','hand':'exmr_features','exmr':'exmr','all':'exmr'},
    'gvhmr_batch': {'pose':'features','features':'features','gvhmr':'gvhmr','all':'gvhmr'},
}
MAX_ITEMS = 128
MAX_BYTES = 512 * 1024


def argument(args, name):
    return args[args.index(name)+1]


def grouped_jobs(jobs, root):
    projects = {argument(args,'--project') for _,args in jobs}
    buckets = {}
    for stage,args in jobs:
        buckets.setdefault(stage,[]).append(args)
    # ExMR benefits for both one and multiple actors. FlowHMR's measured gain
    # is for one actor; retain its existing two-phase multi-person scheduling.
    joined = {}
    for stages,batch in [(('exmr_preprocess','exmr_features','exmr'),'exmr_batch'),
                         (('flow_features','flowhmr'),'flow_batch')]:
        if batch=='flow_batch' and len(projects)!=1:continue
        if not all(stage in buckets for stage in stages):continue
        items=[[_item(a) for a in buckets[stage]] for stage in stages]
        if any(item is None for group in items for item in group):continue
        if any(group!=items[0] for group in items[1:]):continue
        joined[stages[0]]=(batch,items[0])
        for stage in stages[1:]:del buckets[stage]
    result = []
    for stage,items in buckets.items():
        if stage in joined:
            batch,requests=joined[stage]
            for begin in range(0,len(requests),MAX_ITEMS):
                result.append((batch,_batch_args(root,'all',requests[begin:begin+MAX_ITEMS])))
            continue
        spec = {'flow_features':('flow_batch',['flow_features']),
                'flowhmr':('flow_batch',['flowhmr']),
                'exmr_preprocess':('exmr_batch',['exmr_preprocess']),
                'exmr_features':('exmr_batch',['body','hand']),
                'exmr':('exmr_batch',['exmr']),
                'gvhmr':('gvhmr_batch',['all'])}.get(stage)
        if spec is None or len(items)<2 or any(_item(a) is None for a in items):
            result.extend((stage,args) for args in items)
            continue
        batch,phases = spec
        for phase in phases:
            for begin in range(0,len(items),MAX_ITEMS):
                chunk = items[begin:begin+MAX_ITEMS]
                # Only explicit actor paths, selected results and force survive.
                requests = [dict(project=argument(a,'--project'),result=argument(a,'--result'),force='--force' in a) for a in chunk]
                result.append((batch,['--batch-root',str(root),'--batch-phase',phase,
                    '--batch-items',json.dumps(requests),'--project',requests[0]['project'],
                    '--result',requests[0]['result']]))
    return result


def _item(args):
    # Do not drop unrecognized processing options when combining jobs.
    allowed = {'--project', '--result', '--force'}
    if any(a.startswith('--') and a not in allowed for a in args):return None
    try:
        return dict(project=argument(args,'--project'),
                    result=argument(args,'--result') if '--result' in args else 'auto',
                    force='--force' in args)
    except (ValueError, IndexError):return None


def _batch_args(root, phase, items):
    return ['--batch-root',str(root),'--batch-phase',phase,
            '--batch-items',json.dumps(items),'--project',items[0]['project'],
            '--result',items[0]['result']]


def join_single_pipeline(jobs, project):
    """Join only the ready queue head, after prepare/track chose its project."""
    if project is None or not jobs:return jobs
    stage = jobs[0][0]
    stages = {'exmr_preprocess':('exmr_preprocess','exmr_features','exmr'),
              'flow_features':('flow_features','flowhmr')}.get(stage)
    if stages is None or tuple(s for s,_ in jobs[:len(stages)])!=stages:return jobs
    items = []
    for _,args in jobs[:len(stages)]:
        args=list(args or [])
        if '--project' not in args:args=['--project',str(project),*args]
        item=_item(args)
        if item is None or Path(item['project'])!=Path(project):return jobs
        items.append(item)
    if any(item!=items[0] for item in items[1:]):return jobs
    batch='exmr_batch' if stage=='exmr_preprocess' else 'flow_batch'
    return [(batch,_batch_args(project,'all',[items[0]])),*jobs[len(stages):]]


def validate(request, stage):
    if stage not in PHASES or request.get('version')!=1:raise ValueError('Invalid batch worker request')
    phase = request.get('phase')
    if phase not in PHASES[stage]:raise ValueError('Unsupported batch phase')
    root = Path(request['root']).resolve(strict=True)
    jobs = request.get('jobs')
    if not isinstance(jobs,list) or not 1<=len(jobs)<=MAX_ITEMS:raise ValueError('Invalid batch size')
    model = {'flow_batch':'flowhmr','exmr_batch':'exmr','gvhmr_batch':'gvhmr'}[stage]
    from .motion_tools import RESULTS
    clean = []
    for job in jobs:
        project = Path(job['project']).resolve(strict=True)
        if not project.is_relative_to(root):raise ValueError('Batch project is outside its scene')
        metadata = project/'project.json'
        if metadata.stat().st_size>1024**2:raise ValueError('Project metadata is too large')
        state = json.loads(metadata.read_text(encoding='utf-8'))
        if state.get('pipeline',{}).get('model','gvhmr')!=model:raise ValueError('Batch motion model does not match its runtime')
        selected = job['result']
        if selected not in ('auto',*RESULTS):raise ValueError('Invalid batch result')
        clean.append(dict(project=project,result=selected,force=bool(job.get('force',False))))
    return phase,clean


def write_request(stage, args):
    from .paths import HOME
    from .storage import atomic_json
    value = dict(version=1,root=argument(args,'--batch-root'),phase=argument(args,'--batch-phase'),
                 jobs=json.loads(argument(args,'--batch-items')))
    if len(json.dumps(value).encode('utf-8'))>MAX_BYTES:raise ValueError('Batch request is too large')
    # Worker validates model metadata off the GUI thread; only serialize here.
    folder = HOME/'cache/worker_jobs';folder.mkdir(parents=True,exist_ok=True)
    path = folder/(uuid.uuid4().hex+'.json')
    atomic_json(path,value)
    return path


def remove_request(path):
    if path is None:return
    from .paths import HOME
    folder = (HOME/'cache/worker_jobs').resolve()
    path = Path(path).resolve()
    if path.parent==folder and path.suffix=='.json':path.unlink(missing_ok=True)
