"""Apply RGB MoCap's Windows adapters to a user's pinned upstream sources.

Upstream source and data are downloaded directly, under their own licenses;
they are never included in this repository. Fail on changed upstream syntax.
"""
from pathlib import Path
import ast
import re
import shutil


def replace(path, old, new, count=1):
    source=path.read_text(encoding='utf-8')
    if source.count(old)!=count:raise ValueError('Unsupported upstream source: '+path.name)
    path.write_text(source.replace(old,new),encoding='utf-8')


def adapt_engine(engine, helpers):
    engine=Path(engine)
    path=engine/'landmarks/run_ma_2d.py'
    source=path.read_text(encoding='utf-8');tree=ast.parse(source)
    # Mask-based inference never needs Detectron2 or its extra model download.
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and
             n.module in ('utils.utils_detectron2','detectron2.config')]
    if len(imports)!=2:raise ValueError('Unexpected upstream detector imports.')
    lines=source.splitlines(keepends=True)
    for node in sorted(imports,key=lambda n:n.lineno,reverse=True):del lines[node.lineno-1:node.end_lineno]
    source=''.join(lines);tree=ast.parse(source);edits=[]
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    process=functions['process_data'];args=[a.arg for a in process.args.args]
    expected=['frame_source','detector','device','model','cfg','out_folder','save_cam_output','masks_path']
    if args!=expected+['downsampled_verts_pth']:raise ValueError('Unexpected upstream landmark interface.')
    start=process.body[1].lineno if isinstance(process.body[0],ast.Expr) and isinstance(process.body[0].value,ast.Constant) else process.body[0].lineno
    edits.append((start,process.end_lineno,
        '    from speed_landmarks import process_fast\n    return process_fast('+', '.join(expected)+')\n'))
    main=functions['main']
    first=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg_path' for t in n.targets))
    last=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='detector' for t in n.targets))
    lines=source.splitlines(keepends=True)
    original=''.join(lines[first.lineno-1:last.end_lineno])
    block='    detector = None\n    if masks_folder is None:\n'
    block+=''.join('        '+ast.unparse(n)+'\n' for n in imports)
    block+=''.join('    '+line for line in original.splitlines(keepends=True))
    edits.append((first.lineno,last.end_lineno,block))
    for begin,end,text in sorted(edits,reverse=True):lines[begin-1:end]=[text]
    path.write_text(''.join(lines),encoding='utf-8')
    replace(path,'torch.load(args.weights)',"torch.load(args.weights, weights_only=False, map_location='cpu')")
    for name in ('optimization/scene_debug/renderer.py','optimization/utils/utils_camera.py'):
        p=engine/name
        replace(p,"os.environ['PYOPENGL_PLATFORM'] = 'egl'", "if os.name != 'nt':\n    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')")
    replace(engine/'optimization/utils/fitting.py','    if parallel:',"    if parallel and os.name != 'nt':")
    for name in ('landmarks/utils/video_utils.py','optimization/utils/video_tools.py'):
        replace(engine/name,'"/usr/bin/ffmpeg"',"__import__('imageio_ffmpeg').get_ffmpeg_exe()")
    replace(engine/'optimization/run_ma_3d.py','    use_cam_names = cam_names is not None',
            '    if cam_names:\n        cam_names = sorted(cam_names)\n    use_cam_names = cam_names is not None')
    path=engine/'optimization/utils/optimization.py'
    replace(path,'import torch\n','import torch\nimport os\n')
    # Keep the validated quiet-fit schedule, including its early-stop cadence.
    replace(path,'            show_log = True',"            show_log = iter % 25 == 0")
    source=path.read_text(encoding='utf-8');lines=source.splitlines(keepends=True);tree=ast.parse(source)
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
             ast.unparse(n.value.func)=='self.errors.update']
    if len(matches)!=1:raise ValueError('Unexpected fitting diagnostics.')
    n=matches[0];lines[n.lineno-1:n.end_lineno]=['                        pass  # Scalar diagnostics omitted; uncertainty updates retained.\n']
    path.write_text(''.join(lines),encoding='utf-8')
    for name in ('speed_crop.py','speed_landmarks.py'):shutil.copy2(Path(helpers)/name,engine/name)
    for name in ('landmarks/run_ma_2d.py','optimization/utils/optimization.py'):
        ast.parse((engine/name).read_text(encoding='utf-8'))


def adapt_sdf(root):
    """MSVC integer-width/API fixes; leave the original SDF algorithm intact."""
    root=Path(root)/'src'
    for name,annotation in [('aabb.hpp','__align__(32)'),('triangle.hpp','__align__(48)')]:
        replace(root/name,annotation,'')
    path=root/'bvh_cuda_op.cu';source=path.read_text(encoding='utf-8')
    source=source.replace('triangles.type(), "bvh_tree_building"','triangles.scalar_type(), "bvh_tree_building"')
    source=re.sub(r'\blong\b','int64_t',source)
    start=source.index('  AT_DISPATCH_FLOATING_TYPES(');head,tail=source[:start],source[start:]
    lines=[];skip=False
    for line in tail.splitlines(keepends=True):
        if re.match(r'\s*#if (PRINT_TIMINGS|DEBUG_PRINT) == 1',line):
            if skip:raise ValueError('Unexpected nested SDF diagnostics.')
            skip=True;continue
        if skip and re.match(r'\s*#endif',line):skip=False;continue
        if not skip:lines.append(line)
    if skip:raise ValueError('Unterminated SDF diagnostics.')
    # Upstream defines BVH_PROFILING=0, but tests it with #ifdef. Retain that
    # behavior while moving preprocessor directives out of the dispatch macro.
    tail=''.join(lines).replace('#ifdef BVH_PROFILING\n','').replace('#endif\n','')
    path.write_text(head+tail,encoding='utf-8')
