"""Forward genuine upstream tqdm observations without changing iteration/data."""
from contextlib import contextmanager
import math,time,weakref


@contextmanager
def tqdm_events(emit):
    from tqdm import tqdm
    original=tqdm.display;states=weakref.WeakKeyDictionary()
    def display(bar,*args,**kwargs):
        result=original(bar,*args,**kwargs)
        total=getattr(bar,'total',None);current=getattr(bar,'n',0)
        if total is None or not math.isfinite(total) or total<=0:return result
        now=time.monotonic();last=states.get(bar)
        if last is None or now-last[0]>=.25 or (current>=total and last[1]!=current):
            states[bar]=(now,current)
            emit('progress',current=current,total=total,message=str(bar.desc or 'Processing'),
                 phase='tqdm-'+str(id(bar)),unit=getattr(bar,'unit','it'))
        return result
    tqdm.display=display
    try:yield
    finally:tqdm.display=original;states.clear()
