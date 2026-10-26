"""Prepare model-isolated development environments from reviewed wheel pins."""
from pathlib import Path
import argparse,importlib.util,json,sys
from runtime_support import ROOT,prepared_lock,find_uv,run,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',nargs='+',default=['shared','flow','hand','htd','exmr'],choices=['shared','flow','hand','htd','exmr']);p.add_argument('--plan',action='store_true');p.add_argument('--uv',type=Path);args=p.parse_args()
    work=ROOT/'build/dependencies';lock=prepared_lock(work)
    keys=list(dict.fromkeys((['shared'] if any(k in args.profiles for k in ['flow','hand','htd']) else [])+args.profiles))
    groups=[g for g in lock['groups'] if g['key'] in keys]
    if args.plan:
        print(json.dumps([dict(profile=g['key'],python=g['python'],packages=len(g['packages']),overlay=g.get('base')) for g in groups],indent=2));return
    if sys.platform!='win32':raise RuntimeError('This development workflow requires native 64-bit Windows.')
    uv=find_uv(args.uv)
    if sha(uv)!=lock['uv_sha256']:raise RuntimeError('Unexpected uv helper. Use the pinned Windows uv build.')
    spec=importlib.util.spec_from_file_location('engine',ROOT/'installer/engine.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    folders={'shared':'.venv','flow':'.venv-flow','hand':'.venv-hand','htd':'.venv-htd','exmr':'.venv-exmr'}
    for g in groups:
        target=ROOT/folders[g['key']];python=target/'Scripts/python.exe'
        run([uv,'python','install',g['python'],'--no-bin','--no-registry'])
        if not python.is_file():run([uv,'venv',target,'--python',g['python']])
        files=[];total=sum(x['bytes'] for x in g['packages']);done=0
        for package in g['packages']:
            if package.get('bundled'):wheel=work/package['bundled']
            else:wheel=engine.download(package,ROOT/'.cache/wheels',None,done,total)
            if sha(wheel)!=package['sha256']:raise RuntimeError('Wheel verification failed')
            files.append(wheel);done+=package['bytes']
        for start in range(0,len(files),20):run([uv,'pip','install','--python',python,'--no-deps',*files[start:start+20]])
        if g.get('base'):
            (target/'Lib/site-packages/shared_runtime.pth').write_text('import sys; sys.path.append('+ascii(str(ROOT/'.venv/Lib/site-packages'))+')\n',encoding='ascii')
        print('READY',g['key'],flush=True)
    if 'shared' in keys:
        run([uv,'pip','install','--python',ROOT/'.venv/Scripts/python.exe','-r',ROOT/'requirements-build.txt'])
    print('Development environments ready. Import model files separately through Model Manager.')

if __name__=='__main__':main()
