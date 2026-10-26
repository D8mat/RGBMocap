"""One persistent writer; coalesce pending edits per project/result."""
from collections import OrderedDict
from threading import Condition
from pathlib import Path
from PySide6.QtCore import QThread,Signal
from .foot_contacts import save_edit


class ContactWriter(QThread):
    saved=Signal(int,object)
    failed=Signal(int,str)
    settled=Signal(int)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.condition=Condition();self.pending=OrderedDict();self.stopping=False;self.sequence=0

    def request(self,project,base,ranges,context):
        with self.condition:
            self.sequence+=1
            key=(str(Path(project).resolve()),base)
            if key not in self.pending and len(self.pending)>=8:raise RuntimeError('Foot-lock save queue is full; retry after saving completes')
            self.pending[key]=(self.sequence,ranges,dict(context))
            self.condition.notify()
            return self.sequence

    def stop(self):
        with self.condition:self.stopping=True;self.condition.notify()

    def run(self):
        while True:
            with self.condition:
                while not self.pending and not self.stopping:self.condition.wait()
                if not self.pending and self.stopping:return
                (project,base),(sequence,ranges,context)=self.pending.popitem(last=False)
            try:
                state=save_edit(project,base,ranges,context)
                self.saved.emit(sequence,{'project':project,'base':base,'state':state,'ranges':ranges})
            except Exception as error:self.failed.emit(sequence,str(error))
            with self.condition:
                if not self.pending:self.settled.emit(self.sequence)
