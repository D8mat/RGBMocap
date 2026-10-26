"""Bounded, sequential video crops for temporal inference (compute process only)."""
from collections import OrderedDict
import queue
import threading
import traceback


class CropPrefetch:
    """One CPU producer, bounded queue, explicit decoder ownership and shutdown."""
    def __init__(self,factory,maximum=2):
        self.factory=factory
        self.queue=queue.Queue(maxsize=max(1,maximum))
        self.stop=threading.Event()
        self.finished=False
        self.thread=threading.Thread(target=self._produce,name='gvhmr-crops',daemon=True)
        self.thread.start()

    def _put(self,value):
        while not self.stop.is_set():
            try:
                self.queue.put(value,timeout=.1)
                return
            except queue.Full:pass

    def _produce(self):
        iterator=None
        try:
            iterator=self.factory()
            for item in iterator:
                if self.stop.is_set():break
                self._put(('value',item))
        except Exception:
            # A traceback object can retain whole crop batches. Send text only.
            self._put(('error',traceback.format_exc()))
        finally:
            try:
                if iterator is not None:iterator.close()
            except Exception:
                self._put(('error',traceback.format_exc()))
            self._put(('done',None))

    def __iter__(self):return self

    def __next__(self):
        if self.finished:raise StopIteration
        try:kind,value=self.queue.get(timeout=60)
        except queue.Empty:raise RuntimeError('Video crop preparation did not respond for 60 seconds') from None
        if kind=='done':
            self.finished=True
            raise StopIteration
        if kind=='error':raise RuntimeError('Video crop preparation failed:\n'+value)
        return value

    def close(self):
        self.stop.set()
        self.finished=True
        self.thread.join(timeout=10)
        while True:
            try:self.queue.get_nowait()
            except queue.Empty:break
        self.factory=None
        if self.thread.is_alive():raise RuntimeError('Video crop preparation failed to stop')


def bounded_spatial_encoder(model, batch, observer=None):
    """Batch only independent image encoding; preserve the full temporal window."""
    from types import MethodType
    import torch
    original = model.encode_visual_features
    def encode(self, images):
        if images.shape[1] <= batch:
            if observer:observer(images.shape[1])
            return original(images)
        output = None
        for first in range(0,images.shape[1],batch):
            if observer:observer(min(batch,images.shape[1]-first))
            part = original(images[:,first:first+batch])
            if output is None:
                output = part.new_empty((part.shape[0],images.shape[1],*part.shape[2:]))
            output[:,first:first+len(part[0])] = part
        return output
    model.encode_visual_features = MethodType(encode,model)


class VideoCrops:
    def __init__(self, path, boxes, crop, maximum=120):
        import cv2
        self.capture = cv2.VideoCapture(str(path))
        self.boxes, self.crop, self.maximum = boxes, crop, maximum
        self.cache = OrderedDict()
        self.position = 0
        self.shape = (1,len(boxes),3,256,256)

    def __getitem__(self, item):
        import cv2
        import torch
        rows, frames = item
        indices = range(*frames.indices(len(self.boxes)))
        if len(indices) > self.maximum:
            raise ValueError("PVA crop requests must fit the bounded inference window")
        output = []
        for index in indices:
            if index not in self.cache:
                if index != self.position:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read PVA frame {index}")
                self.position = index+1
                image = self.crop(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),self.boxes[index].numpy())
                self.cache[index] = torch.from_numpy(image).permute(2,0,1).float().div_(255)
            self.cache.move_to_end(index)
            output.append(self.cache[index])
            while len(self.cache) > self.maximum:
                self.cache.popitem(last=False)
        return torch.stack(output)[None][rows]

    def close(self):
        self.capture.release()
        self.cache.clear()


def crop_batches(project,boxes,batch):
    """CPU-only producer; batch is a getter so OOM backoff also reduces crops."""
    import imageio.v3 as iio
    import numpy as np
    import torch
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    reader=iio.imiter(str(project/'input.mp4'),plugin='pyav',
        filter_sequence=[('scale','iw*0.5:ih*0.5')],thread_count=2)
    first=0
    try:
        with torch.inference_mode():
            while first<len(boxes):
                frames=[]
                for index in range(min(batch(),len(boxes)-first)):
                    try:frames.append(next(reader))
                    except StopIteration:raise RuntimeError(f'Cannot decode GVHMR frame {first+index}') from None
                crops,adjusted=get_batch(np.stack(frames),boxes[first:first+len(frames)]*.5,img_ds=1.,path_type='np')
                del frames
                adjusted*=2
                yield first,crops,adjusted
                first+=len(crops)
    finally:reader.close()


def streamed_features(project,boxes,predict,batch=16,prefetch=0,progress=None):
    """Bounded CPU crop overlap; GPU inference and OOM retries stay on consumer."""
    import torch
    from .compute_policy import retry_cuda
    results=[];current=batch
    factory=lambda:crop_batches(project,boxes,lambda:current)
    iterator=CropPrefetch(factory,prefetch) if prefetch else factory()
    try:
        with torch.inference_mode():
            for first,crops,adjusted in iterator:
                offset=0
                def reduced(value):
                    nonlocal current
                    current=value
                while offset<len(crops):
                    def infer(size):
                        return predict(crops[offset:offset+size],adjusted[offset:offset+size])
                    part=retry_cuda(infer,min(current,len(crops)-offset),reduced)
                    results.append(part.cpu());offset+=len(part)
                    del part
                if progress:progress(first+len(crops),len(boxes))
    finally:iterator.close()
    return torch.cat(results)
