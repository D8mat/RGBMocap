"""Exact, bounded Gaussian crop preprocessing in original image coordinates."""
import math
import numpy as np

def install_exact_roi_crop(dataset_type):
    """Keep original global affine coordinates; localize only Gaussian work."""
    import threading,sys
    module=sys.modules[dataset_type.__module__]
    original_get=dataset_type.__getitem__;original_gaussian=module.gaussian
    context=threading.local()
    def gaussian(image,*args,**kwargs):
        bounds=getattr(context,'bounds',None)
        if bounds is None:return original_gaussian(image,*args,**kwargs)
        x0,y0,x1,y1=bounds
        part=original_gaussian(image[y0:y1,x0:x1],*args,**kwargs)
        # Same full-frame coordinate system as the original warp. Pixels outside
        # this halo cannot contribute to the output patch (rotation is zero).
        full=np.zeros(image.shape,dtype=part.dtype)
        full[y0:y1,x0:x1]=part
        return full
    def get(self,idx):
        from lib.datasets.utils_eval import expand_to_aspect_ratio
        box=expand_to_aspect_ratio(self.scale[idx]*200*1.2,target_aspect_ratio=self.cfg.data_cfg['image_size'])
        cx,cy=self.center[idx];h,w=self.img_cv2.shape[:2]
        downsampling=(box.max()/self.img_size[1])/2
        sigma=max(0.,(downsampling-1)/2) if downsampling>1.1 else 0.
        halo=math.ceil(4*sigma)+8
        context.bounds=(max(0,math.floor(cx-box[0]/2)-halo),max(0,math.floor(cy-box[1]/2)-halo),
            min(w,math.ceil(cx+box[0]/2)+halo),min(h,math.ceil(cy+box[1]/2)+halo))
        try:return original_get(self,idx)
        finally:context.bounds=None
    module.gaussian=gaussian;dataset_type.__getitem__=get
