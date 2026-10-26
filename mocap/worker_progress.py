"""Presentation-only worker entry; inference/cache identities stay unchanged."""
import json,runpy,sys


def emit(kind,**value):
    print('@@MOCAP@@'+json.dumps(dict(type=kind,**value)),flush=True)


def main():
    stage=sys.argv[1] if len(sys.argv)>1 else ''
    if stage=='character':
        from .retarget.worker import main as character_main
        return character_main(sys.argv[2:])
    if stage=='save_project':
        from .project_save import main as save_main
        sys.argv.pop(1)
        return save_main(emit)
    if stage=='download':
        from . import models
        original=models.download
        models.download=lambda key:original(key,progress=emit)
    if stage.startswith('exmr') or stage=='download':
        from .progress_events import tqdm_events
        with tqdm_events(emit):runpy.run_module('mocap.worker',run_name='__main__')
    else:runpy.run_module('mocap.worker',run_name='__main__')


if __name__=='__main__':main()
