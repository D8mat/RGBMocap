"""Check source checkout composition without installing or running AI models."""
from pathlib import Path
import ast,hashlib,json,os,re,sys
from runtime_support import cache_dependency_files

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.bootstrap','.runtime','.cache','.setup','.venv','.venv-flow','.venv-hand','.venv-htd','.venv-exmr','build','dist','releases','models','projects','cache','character_scenes','.build-inputs','runtime','runtime_gui','runtime_flow','runtime_hand','runtime_htd','runtime_exmr','runtime_character'}
FORBIDDEN={'.exe','.dll','.pyd','.pyc','.whl','.pt','.pth','.ckpt','.safetensors','.onnx','.npz','.npy','.pkl','.fbx','.bvh','.blend','.mp4','.vrm','.pmx','.vmd','.pmd','.vpd','.pmm','.avi','.mov','.mkv','.webm','.zip','.7z'}

def main():
    errors=[];count=0
    try:cache_dependency_files(ROOT)
    except (OSError,ValueError) as exc:errors.append(str(exc))
    resources={r['path']:r for r in json.loads((ROOT/'installer/source-resources.json').read_text())['files']}
    auxiliary={r['path']:r for r in json.loads((ROOT/'installer/external-assets.json').read_text())['files']}
    def files():
        for folder,dirs,names in os.walk(ROOT):
            relative=Path(folder).relative_to(ROOT)
            dirs[:]=[d for d in dirs if d not in {'.git','__pycache__'} and not (relative==Path('.') and d in SKIP) and (relative/d).as_posix() not in {'tools/bin','third_party/KimodoMotionCorrection/bin'}]
            for name in names:yield Path(folder)/name
    for p in files():
        if not p.is_file():continue
        r=p.relative_to(ROOT);count+=1
        if r.as_posix().startswith(('tools/bin/','third_party/KimodoMotionCorrection/bin/')):continue
        if r.as_posix() in auxiliary:
            expected=auxiliary[r.as_posix()]
            if expected['bytes']!=p.stat().st_size or expected['sha256']!=hashlib.sha256(p.read_bytes()).hexdigest():errors.append('Changed auxiliary data: '+str(r))
            continue
        if p.suffix.lower() in FORBIDDEN:
            resource=resources.get(r.as_posix())
            if not resource or resource['sha256']!=hashlib.sha256(p.read_bytes()).hexdigest() or resource['bytes']!=p.stat().st_size:
                errors.append('Generated or model payload: '+str(r))
        if p.name.lower() in {'agents.md','claude.md','codex.md','gemini.md'}:errors.append('AI instruction file: '+str(r))
        if p.suffix=='.py':
            try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(r))
            except (SyntaxError,UnicodeError) as exc:errors.append(str(exc))
    for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md','docs/USER_GUIDE.md','docs/BUILD.md','installer/runtime-lock.json','installer/external-assets.json']:
        if not (ROOT/name).is_file():errors.append('Missing '+name)
    lock=json.loads((ROOT/'installer/runtime-lock.json').read_text())
    if {g['key'] for g in lock['groups']}!={'shared','gui','flow','hand','htd','exmr'}:errors.append('Runtime profile set changed')
    print(json.dumps(dict(files=count,errors=errors),indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
