"""Experimental Git installer: verified profile reuse and transaction-safe promotion."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,stat,sys,urllib.request,uuid,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from runtime_support import cache_dependency_files,prepared_lock,sha
from session_lock import app_lock
from git_transaction import begin,promote,recover,store

def atomic(path,value):
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(temp,path)

def setup():
    parser=argparse.ArgumentParser();parser.add_argument('--assets-root',type=Path,default=None)
    args=parser.parse_args();folder=ROOT/'.setup';resources=folder/'bootstrap-git';resources.mkdir(parents=True,exist_ok=True)
    cache_dependency_files(ROOT)
    from stage_assets import stage as stage_assets
    with app_lock(ROOT):
        pending=folder/'git-sync.pending';pending.write_text('Run install.cmd to complete or retry.\n',encoding='utf8')
        lock=prepared_lock(resources)
        for origin,name in [(ROOT/'installer/engine.py','engine.py'),(ROOT/'installer/verify_runtime.py','verify_runtime.py'),(ROOT/'mocap/runtime_registry.py','runtime_registry.py')]:
            shutil.copy2(origin,resources/name)
        spec=importlib.util.spec_from_file_location('git_setup_engine',resources/'engine.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        if recover(ROOT,engine.remove_owned):print('Recovered an interrupted runtime update.',flush=True)
        available=shutil.disk_usage(ROOT).free
        print('Free disk: %.1f GiB; fresh runtime installation estimate: %.1f GiB' % (available/1024**3,lock['fresh_free_bytes_required']/1024**3),flush=True)
        if not (ROOT/'SETUP_RECEIPT.json').exists() and available<lock['fresh_free_bytes_required']:
            raise OSError('Insufficient free space for the runtime installation.')
        support_path=ROOT/'installer/git-support-lock.json'
        support=json.loads(support_path.read_text(encoding='utf8'))
        roots={'runtime_character','tools/bin','third_party/KimodoMotionCorrection/bin'}
        state_file=folder/'git-runtime-state.json'
        try:
            previous=json.loads(state_file.read_text(encoding='utf8'))
            inventory=previous['support_inventory']
            def unchanged(row):
                path=ROOT/row['path'];st=path.stat()
                return path.is_file() and not (st.st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT) and [st.st_size,st.st_mtime_ns]==row['stat']
            support_ready=(previous['support_sha256']==sha(support_path) and bool(inventory) and all(unchanged(row) for row in inventory))
        except (OSError,ValueError,KeyError,TypeError):support_ready=False
        support_stage=folder/('git-support-'+uuid.uuid4().hex)
        if not support_ready:
            support_stage.mkdir()
            from support_components import stage as stage_support
            stage_support(support_stage, resources/'support-downloads', support)
        if args.assets_root:stage_assets(args.assets_root,ROOT)
        else:stage_assets(None,ROOT,download=True)
        logs=folder/'logs';logs.mkdir(exist_ok=True)
        reused=engine.reusable_groups(ROOT,lock,None,logs)
        stage,reports=engine.install_runtimes(ROOT,resources,lock,None,reused)
        names=[*engine.TARGETS.values(),*sorted(roots)]
        if not (ROOT/'SETUP_RECEIPT.json').is_file():
            conflicts=[name for name in names if (ROOT/name).exists()]
            if conflicts:raise RuntimeError('Existing unmanaged runtime files were preserved: '+', '.join(conflicts))
        for name in names:
            dest=ROOT/name
            if dest.exists() and (dest.is_symlink() or dest.stat().st_file_attributes&stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise RuntimeError('A runtime path is a link; refusing to replace it: '+str(dest))
        receipt=ROOT/'SETUP_RECEIPT.json';state=folder/'git-runtime-state.json'
        journal,transaction=begin(ROOT)
        try:
            for name in names:
                incoming=(stage/name) if name in engine.TARGETS.values() else (support_stage/name)
                if not incoming.exists():continue
                promote(ROOT,journal,transaction,name,incoming)
            atomic(receipt,dict(schema=1,id=lock['id'],version=lock['version'],runtimes=reports,group_ids=engine.group_ids(lock),models_included=False))
            inventory=[]
            for name in sorted(roots):
                for path in sorted((ROOT/name).rglob('*')):
                    if path.is_file():
                        st=path.stat();inventory.append(dict(path=path.relative_to(ROOT).as_posix(),stat=[st.st_size,st.st_mtime_ns]))
            atomic(state,dict(lock_sha256=sha(ROOT/'installer/runtime-lock.json'),registry_sha256=sha(ROOT/'mocap/runtime_registry.py'),support_sha256=sha(support_path),assets_sha256=sha(ROOT/'installer/external-assets.json'),support_inventory=inventory))
            transaction['committed']=True;store(journal,transaction)
        except BaseException:
            recover(ROOT,engine.remove_owned)
            raise
        recover(ROOT,engine.remove_owned)
        if support_stage.exists():engine.remove_owned(folder,support_stage)
        pending.unlink()
        print('Installation complete. Existing models and projects were retained.',flush=True)

if __name__=='__main__':setup()
