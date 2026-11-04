"""Validated export checkpoints. OS locks release automatically after cancellation."""
import hashlib,json,os
from pathlib import Path
from contextlib import contextmanager
from .storage import atomic_json,sha256


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@contextmanager
def locked(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if folder.is_symlink():raise ValueError('Export cache cannot be a symbolic link')
    with (folder/'lock').open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('This export is already running') from None
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield folder
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def valid(folder,stage):
    try:
        receipt=json.loads((folder/(stage+'.json')).read_text(encoding='utf8'))
        for name,digest in receipt['files'].items():
            if Path(name).name!=name or sha256(folder/name)!=digest:return None
        return receipt['data']
    except (OSError,ValueError,KeyError):return None


def complete(folder,stage,names,data):
    atomic_json(folder/(stage+'.json'),dict(files={n:sha256(folder/n) for n in names},data=data))


def publish(source,target):
    import shutil,tempfile
    target=Path(target)
    fd,temp=tempfile.mkstemp(prefix='.'+target.name+'.',suffix='.partial',dir=target.parent)
    os.close(fd)
    try:shutil.copyfile(source,temp);os.replace(temp,target)
    finally:Path(temp).unlink(missing_ok=True)
