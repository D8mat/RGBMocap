"""Fixed cameras relative to a stationary, face-up floor checkerboard."""
import cv2
import numpy as np
from .calibration_solver import array


def solve_board(corners, lens, size, cols, rows, mm):
    if not 3 <= cols <= 30 or not 3 <= rows <= 30 or cols == rows or not 1 <= mm <= 1000:
        raise ValueError('Use a rectangular board, 3-30 inner corners and 1-1000 mm squares')
    p = array(corners, (cols*rows,2))
    if list(size) != lens['image_size']:raise ValueError('Photo and lens calibration resolutions differ')
    k = array(lens['intrinsic_matrix'],(3,3)); d=np.asarray(lens['distortion_coeffs'],float)
    if d.shape not in ((4,),(5,)) or not np.isfinite(d).all():raise ValueError('Invalid lens distortion')
    if min(k[0,0],k[1,1]) <= 0 or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0]) > 1e-8:raise ValueError('Invalid lens intrinsics')
    obj=np.zeros((cols*rows,3),float)
    obj[:,:2]=np.mgrid[:cols,:rows].T.reshape(-1,2)*[mm/1000,-mm/1000]
    if cv2.contourArea(cv2.convexHull(p.astype(np.float32))) < size[0]*size[1]*.005:
        raise ValueError('Board is too small in the image; move it closer or use a larger board')
    result=cv2.solvePnPGeneric(obj,p,k,d,flags=cv2.SOLVEPNP_IPPE)
    candidates=[]
    for r,t in zip(result[1],result[2]):
        r,t=cv2.solvePnPRefineLM(obj,p,k,d,r.copy(),t.copy())
        rot=cv2.Rodrigues(r)[0]; center=-rot.T@t.ravel()
        if not np.isfinite(rot).all() or not np.isfinite(center).all():continue
        if center[2]<=0 or np.min((obj@rot.T+t.ravel())[:,2])<=0:continue
        err=np.linalg.norm(cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2)-p,axis=1)
        candidates.append((float(np.sqrt(np.mean(err**2))),rot,t,center,float(err.max())))
    candidates.sort(key=lambda c:c[0])
    if not candidates or candidates[0][0]>2:raise ValueError('No valid above-floor pose with RMS <= 2 px')
    best=candidates[0]
    if len(candidates)>1:
        other=candidates[1]
        if other[0]-best[0]<.1 and (np.linalg.norm(other[3]-best[3])>.01 or np.linalg.norm(other[1]-best[1])>.02):
            raise ValueError('Ambiguous planar pose; use a larger board or a more oblique view')
    return dict(intrinsic_matrix=k.tolist(),distortion_coeffs=d.tolist(),image_size=list(size),
                extrinsics_matrix=np.column_stack((best[1],best[2])).tolist(),
                quality=dict(rms_px=best[0],max_error_px=best[4],camera_center_m=best[3].tolist()))


def calibration(cameras):
    if len(cameras)<2:raise ValueError('Add at least two different camera IDs')
    return dict(schema='rgbmocap-camera-calibration-v1',units='metres',world_up='Z',
                extrinsics_convention='world_to_camera',cameras=cameras,
                requires_undistortion=any(np.any(np.abs(c['distortion_coeffs'])>1e-8) for c in cameras.values()),
                note='Stationary face-up floor board; user verified identical origin and X direction in all views. Origin is first inner corner, X along first row, Y opposite row progression. Board surface is Z=0.')
