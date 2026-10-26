"""Small, clock-driven task status. Never reads tensors or changes processing."""
from collections import OrderedDict
import math


def duration(seconds):
    seconds=max(0,math.ceil(seconds));minutes,seconds=divmod(seconds,60);hours,minutes=divmod(minutes,60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes}m {seconds:02d}s' if minutes else f'{seconds}s'


class TaskProgress:
    def __init__(self):
        self.history=OrderedDict();self.reset(0.)

    def reset(self,now):
        self.started=now;self.completed=0;self.active=False;self.phase=None

    def begin(self,now,key,label,estimate=None):
        self.active=True;self.job_started=now;self.key=key;self.label=label
        self.estimate=estimate if estimate is not None else self.history.get(key)
        self.phase=None;self.message=label;self.current=self.total=0;self.samples=[];self.rate=None;self.cached=False

    def update(self,event,now):
        try:current=float(event.get('current',0));total=float(event.get('total',0))
        except (ValueError,TypeError):return
        if not math.isfinite(current) or not math.isfinite(total) or total<0:return
        current=max(0,min(current,total)) if total else 0
        phase=(event.get('phase') or event.get('message','Processing'),total)
        if phase!=self.phase or current<self.current:
            self.phase=phase;self.samples=[];self.rate=None
        self.message=event.get('message',self.label);self.current=current;self.total=total
        # Positive observations exclude model loading from the measured rate.
        # Keep bounded samples; timer ticks never manufacture work or percent.
        if total and current>0 and (not self.samples or current>self.samples[-1][1]):
            self.samples.append((now,current));self.samples=self.samples[-12:]
            dt=now-self.samples[0][0];units=current-self.samples[0][1]
            if dt>=1 and units>0:self.rate=units/dt

    def finish(self,now,success=True):
        if self.active:
            elapsed=max(0,now-self.job_started)
            if success and not self.cached and elapsed>=1:
                self.history[self.key]=elapsed;self.history.move_to_end(self.key)
                while len(self.history)>64:self.history.popitem(last=False)
            if success:self.completed+=1
        self.active=False

    def status(self,now,queued):
        elapsed=max(0,now-self.started)
        count=self.completed+1+len(queued)
        prefix=f'Step {self.completed+1}/{count}'
        measured=self.total>0 and self.current>0
        percent=min(100,round(100*self.current/self.total)) if measured else None
        left=None;scope='Step'
        if measured and self.current>=self.total:
            state='Finalizing';left_text='Remaining: estimating'
        else:
            state=f'{percent}%' if percent is not None else 'Working'
            # An old rate must not keep promising the same deadline in a stall.
            if self.rate and self.total>self.current and now-self.samples[-1][0]<=10:
                left=(self.total-self.current)/self.rate
            # Whole-worker timing remains useful for opaque model inference.
            job_left=max(0,self.estimate-(now-self.job_started)) if self.estimate is not None else None
            if job_left is not None and job_left<1:job_left=None
            estimates=[estimate if estimate is not None else self.history.get(key) for key,estimate in queued]
            if job_left is not None and all(e is not None for e in estimates):
                left=job_left+sum(estimates);scope='Total'
            left_text=f'{scope} left ~{duration(left)}' if left is not None else 'Remaining: estimating'
        return f'{prefix} · {state} · {left_text} · Elapsed {duration(elapsed)}',percent
