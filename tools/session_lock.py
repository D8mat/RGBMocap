"""Cooperate across app, installation and updates; never break another process's lock."""
from contextlib import contextmanager
from pathlib import Path
import msvcrt
import os

@contextmanager
def app_lock(root):
    folder=Path(root)/'.setup';folder.mkdir(exist_ok=True)
    fd=os.open(folder/'app.lock',os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(fd,'r+b') as stream:
        if os.fstat(stream.fileno()).st_size==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Close RGB MoCap or the other installation/update before continuing.') from None
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
