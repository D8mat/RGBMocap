"""AprilTag 36h11 observations in original image pixels; no implicit calibration."""
import hashlib
import cv2
import numpy as np


def dictionary():
    if not hasattr(cv2,'aruco') or not hasattr(cv2.aruco,'ArucoDetector'):
        raise RuntimeError('The installed OpenCV has no AprilTag dictionary detector')
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def detect(image,tag_size_mm=150.):
    if image is None or image.ndim not in (2,3) or image.dtype!=np.uint8:raise ValueError('Expected an 8-bit image')
    if not np.isfinite(tag_size_mm) or not 1<=tag_size_mm<=5000:raise ValueError('Marker size must be 1-5000 mm')
    d=dictionary();params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    corners,ids,rejected=cv2.aruco.ArucoDetector(d,params).detectMarkers(gray)
    observations=[]
    if ids is not None:
        counts={int(i):int(np.count_nonzero(ids==i)) for i in ids.ravel()}
        for points,ident in zip(corners,ids.ravel()):
            p=points.reshape(4,2)
            observations.append(dict(id=int(ident),corners_px=p.astype(float).tolist(),center_px=p.mean(0).astype(float).tolist(),duplicate_id=counts[int(ident)]>1))
    result=dict(schema='rgbmocap-apriltag-observations-v1',family='tag36h11',detector='OpenCV ArucoDetector',
        opencv_version=cv2.__version__,dictionary_sha256=hashlib.sha256(d.bytesList.tobytes()).hexdigest(),
        image_size=[image.shape[1],image.shape[0]],tag_size_m=float(tag_size_mm)/1000,
        size_definition='Outer black square edge, excluding white margin; user supplied, not measured',
        corner_order='OpenCV detector native order, keyed by dictionary ID; coordinates in original image pixels',
        calibration_applied=False,markers=observations,rejected_candidates=len(rejected))
    # Keep preview allocation bounded; coordinates in the result stay full resolution.
    factor=min(1.,1280/image.shape[1],720/image.shape[0])
    preview=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR) if image.ndim==2 else image.copy()
    if factor<1:preview=cv2.resize(preview,None,fx=factor,fy=factor)
    if ids is not None:cv2.aruco.drawDetectedMarkers(preview,[p*factor for p in corners],ids)
    for marker in observations:
        for index,point in enumerate(marker['corners_px']):
            xy=tuple(np.rint(np.array(point)*factor).astype(int))
            cv2.putText(preview,str(index),xy,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
    return result,preview
