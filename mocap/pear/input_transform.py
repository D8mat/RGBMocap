"""Rotate input pixels before display, detection and pose estimation."""
import math
import cv2

def rotate_frame(frame, clockwise_degrees):
    angle=float(clockwise_degrees)
    if not math.isfinite(angle):raise ValueError('Input rotation must be finite')
    angle=angle%360
    if angle==0:return frame
    if angle==90:return cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
    if angle==180:return cv2.rotate(frame,cv2.ROTATE_180)
    if angle==270:return cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
    h,w=frame.shape[:2]
    matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),-angle,1.)
    # Expand rather than cut off the head or feet at arbitrary angles.
    out_w=math.ceil(w*abs(matrix[0,0])+h*abs(matrix[0,1]))
    out_h=math.ceil(h*abs(matrix[0,0])+w*abs(matrix[0,1]))
    matrix[0,2]+=(out_w-w)/2;matrix[1,2]+=(out_h-h)/2
    return cv2.warpAffine(frame,matrix,(out_w,out_h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
