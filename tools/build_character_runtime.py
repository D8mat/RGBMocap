"""Stage an independently acquired Blender 4.5 LTS; no AI environment is changed."""
from pathlib import Path
import json,os,re,shutil,subprocess,zipfile
from runtime_support import ROOT,sha

def discover(explicit=None):
    values=[explicit,os.environ.get('RGBMOCAP_BLENDER'),shutil.which('blender')]
    values+=sorted(Path('C:/Program Files/Blender Foundation').glob('Blender 4.5/blender.exe'))
    for value in values:
        if not value:continue
        path=Path(value).resolve()
        executable=path/'blender.exe' if path.is_dir() else path
        if not executable.is_file():
            if value==explicit:raise FileNotFoundError('Blender executable not found: '+str(executable))
            continue
        result=subprocess.run([str(executable),'--version'],capture_output=True,text=True,encoding='utf8',errors='replace',check=True,timeout=30)
        if not re.search(r'^Blender 4\.5\.',result.stdout,re.M):
            raise ValueError('Character processing requires Blender 4.5 LTS. Use --blender-dir.')
        folder=executable.parent
        if not (folder/'4.5/python').is_dir() or not (folder/'license').is_dir():
            raise ValueError('Supply the complete Blender 4.5 Windows distribution, including Python and licenses.')
        return executable,result.stdout.splitlines()[0]
    raise FileNotFoundError('Install/extract Blender 4.5 LTS and pass --blender-dir, or set RGBMOCAP_BLENDER.')

def stage(executable,app):
    executable=Path(executable).resolve();app=Path(app).resolve()
    source=executable.parent;target=app/'runtime_character'
    if target.is_relative_to(source) or source.is_relative_to(target):raise ValueError('Blender source and staging must be separate folders')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'),dirs_exist_ok=False)
    # Retain OpenCC as editable source in Git; reconstruct its add-on wheel only
    # inside the ignored build staging folder. Standalone OpenCC source is also
    # on the add-on search path used by the character worker.
    filename='opencc_python_reimplemented-0.1.7-py2.py3-none-any.whl'
    wheel_source=ROOT/'installer/runtime_sources'/Path(filename).stem
    wheel=app/'third_party/character_import/mmd_tools/wheels'/filename
    if not wheel_source.is_dir():raise FileNotFoundError('Missing OpenCC source archive contents')
    wheel.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(wheel,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted(wheel_source.rglob('*')):
            if not p.is_file() or p.name=='SOURCE_WHEEL.json':continue
            entry=zipfile.ZipInfo(p.relative_to(wheel_source).as_posix(),date_time=(1980,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o644<<16
            archive.writestr(entry,p.read_bytes())
    (target/'RGBMOCAP_RUNTIME.json').write_text(json.dumps(dict(family='Blender 4.5 LTS',executable_sha256=sha(executable),source='https://www.blender.org/download/lts/4-5/',licenses='license/'),indent=2),encoding='utf8')
    return dict(files=sum(p.is_file() for p in target.rglob('*')),bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),executable_sha256=sha(target/'blender.exe'))
