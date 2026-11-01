"""Preserve the evaluated camera pixels while using ExMR's public SimpleVO.

The input is RGB MoCap's already rotated, constant-frame-rate working video,
not the user's original file. The estimator and its matching settings are
unchanged. The small upstream frame_reader extension is explicit in provenance.
"""
import numpy as np

def rotations(video_path, frame_count):
    import imageio.v3 as iio
    from exmr.preprocessing.visual_odometry import SimpleVO
    def pixels(indices):
        wanted=list(indices)
        if not wanted:return
        iterator=iio.imiter(video_path,plugin='pyav',thread_type='FRAME',
            filter_sequence=[('scale','iw*0.5:ih*0.5')])
        position=0
        try:
            for index,image in enumerate(iterator):
                if index==wanted[position]:
                    yield np.ascontiguousarray(image[:,:,::-1])
                    position+=1
                    if position==len(wanted):return
            raise ValueError('Canonical camera video has incomplete frames')
        finally:
            close=getattr(iterator,'close',None)
            if close:close()
    return SimpleVO(scale=1.,progress=False,frame_reader=pixels)(frame_count=frame_count)
