"""Metric AprilTag board and video-only camera calibration (no inference)."""
import math
from pathlib import Path
import cv2
import numpy as np
from .apriltag_detection import detect
from .calibration_solver import array
from .checkerboard_pose import calibration

BOARD = dict(name='RGBMoCap AprilTag Board 3x2 v1', family='tag36h11',
             columns=3, rows=2, ids=[0,1,2,3,4,5], tag_mm=150,
             gap_mm=30, margin_mm=30, width_mm=570, height_mm=390,
             origin='printed face center', x='right', y='towards top row', z='out of printed face')


def layout():
    markers = {}
    for ident in BOARD['ids']:
        row, col = divmod(ident, 3)
        x, y = -.255 + col*.18, .165 - row*.18
        markers[str(ident)] = [[x,y,0],[x+.15,y,0],[x+.15,y-.15,0],[x,y-.15,0]]
    return dict(units='metres', world_up='Z', family='tag36h11', markers=markers, board=dict(BOARD))


def correspondences(observation):
    if observation.get('family') != 'tag36h11':raise ValueError('Expected AprilTag 36h11')
    world = layout()['markers']; obj, img, ids = [], [], []
    for marker in sorted(observation['markers'], key=lambda m:int(m['id'])):
        ident = int(marker['id'])
        if ident in ids or marker.get('duplicate_id'):raise ValueError('Duplicate tag ID: use only one board')
        if str(ident) not in world:raise ValueError('Unexpected tag ID: use only the supplied 3x2 board')
        ids.append(ident);obj.extend(world[str(ident)]);img.extend(array(marker['corners_px'],(4,2)))
    if len(ids)<3:raise ValueError('Need at least three visible board tags')
    obj, img = np.asarray(obj,np.float32), np.asarray(img,np.float32)
    size = array(observation['image_size'],(2,))
    if np.any(size<=0):raise ValueError('Invalid image size')
    if cv2.contourArea(cv2.convexHull(img)) < np.prod(size)*.005:
        raise ValueError('Board too small: move closer or use higher resolution')
    return obj, img, ids


def fit_lens(observations, pinhole=False, cancel=lambda:False):
    if len(observations)<12:raise ValueError('Need 12+ distinct views; move AND tilt the board in several directions')
    size=tuple(int(v) for v in observations[0]['image_size'])
    if any(tuple(o['image_size'])!=size for o in observations):raise ValueError('Video resolution changed')
    pairs=[correspondences(o) for o in observations]
    all_pixels=np.concatenate([p[1] for p in pairs])
    if cv2.contourArea(cv2.convexHull(all_pixels)) < size[0]*size[1]*.15:
        raise ValueError('Insufficient image coverage; move the board across the image and towards the edges')
    # Reserve every fifth view for an independent reprojection check.
    train=[p for i,p in enumerate(pairs) if i%5!=0]; hold=[p for i,p in enumerate(pairs) if i%5==0]
    flags=0
    if pinhole:flags=cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3
    if cancel():raise InterruptedError('Cancelled')
    rms,k,d,rs,ts,std,_,errors=cv2.calibrateCameraExtended(
        [p[0] for p in train],[p[1] for p in train],size,None,None,flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-8))
    if cancel():raise InterruptedError('Cancelled')
    if not np.isfinite(k).all() or not np.isfinite(d).all() or not np.isfinite(rms) or rms>2:
        raise ValueError('Lens fit failed or RMS exceeds 2 px; use sharper, varied views')
    if not (.1*size[0]<k[0,0]<20*size[0] and .1*size[1]<k[1,1]<20*size[1] and 0<k[0,2]<size[0] and 0<k[1,2]<size[1]):
        raise ValueError('Implausible lens estimate; add varied tilted board views')
    normals=np.array([cv2.Rodrigues(r)[0][:,2] for r in rs])
    variation=np.linalg.svd(normals-normals.mean(0),compute_uv=False)/np.sqrt(len(normals))
    if variation[0]<.1 or variation[1]<.035:
        raise ValueError('Insufficient tilt diversity: tilt the board about BOTH horizontal and vertical axes')
    if not np.isfinite(std).all() or max(float(std[0,0]/k[0,0]),float(std[1,0]/k[1,1]))>.15:
        raise ValueError('Focal length is poorly constrained; add more varied board views')
    validation=[]
    for obj,img,_ in hold:
        if cancel():raise InterruptedError('Cancelled')
        ok,r,t=cv2.solvePnP(obj,img,k,d,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:raise ValueError('Held-out view pose failed')
        error=cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2)-img
        validation.append(float(np.sqrt(np.mean(np.sum(error**2,axis=1)))))
    if max(validation)>2.5:raise ValueError('Held-out view error exceeds 2.5 px; recapture sharper, varied views')
    return dict(intrinsic_matrix=k.tolist(),distortion_coeffs=d.ravel().tolist(),image_size=list(size),
                board=dict(BOARD),lens_model='pinhole_zero_distortion' if pinhole else 'pinhole_radtan5',
                quality=dict(rms_px=float(rms),accepted_views=len(pairs),fitted_views=len(train),
                             per_view_rms_px=errors.ravel().tolist(),held_out_rms_px=validation,
                             focal_std_px=std[:2].ravel().tolist(),
                             warning='Low residual is not ground truth; validate known distances.'))


class Video:
    def __init__(self,path):
        self.path=Path(path);self.stamp=self.signature()
        self.cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1,
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,2000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000])
        self.fps=self.cap.get(cv2.CAP_PROP_FPS);self.count=self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self.cap.isOpened() or not math.isfinite(self.fps) or self.fps<=0 or not math.isfinite(self.count) or self.count<1:
            self.cap.release();raise ValueError('Cannot read video: '+str(path))
    def signature(self):
        stat=self.path.stat();return [stat.st_size,stat.st_mtime_ns]
    def read(self,seconds):
        if not math.isfinite(seconds) or seconds<0 or seconds>=self.count/self.fps:
            raise ValueError('Requested time is outside video: '+self.path.name)
        frame=min(int(self.count)-1,round(seconds*self.fps))
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame):raise ValueError('Cannot seek video')
        ok,image=self.cap.read()
        if not ok:raise ValueError('Cannot decode frame '+str(frame))
        obs,_=detect(image,BOARD['tag_mm'])
        obs.update(frame_index=frame,actual_seconds=frame/self.fps)
        return obs
    def verify(self):
        if self.signature()!=self.stamp:raise ValueError('Video changed during calibration')
    def close(self):self.cap.release()


def video_lens(path,start,end,interval,pinhole=False,cancel=lambda:False,progress=lambda text:None):
    if not all(math.isfinite(v) for v in (start,end,interval)) or start<0 or end<0 or interval<.1:
        raise ValueError('Invalid sampling settings')
    video=Video(path);observations=[];skipped=[];seen=[]
    try:
        stop=video.count/video.fps if end==0 else min(end,video.count/video.fps)
        if start>=stop:raise ValueError('Start must precede end')
        count=math.ceil((stop-start)/interval)
        if count>300:raise ValueError('Maximum 300 samples; shorten range or increase sample interval')
        for i in range(count):
            if cancel():raise InterruptedError('Cancelled')
            obs=video.read(start+i*interval)
            progress(f'{Path(path).name}: sample {i+1}/{count}, {len(observations)} accepted')
            try:obj,img,ids=correspondences(obs)
            except ValueError as exc:skipped.append(dict(frame=obs['frame_index'],reason=str(exc)));continue
            norm=img/np.array(obs['image_size'])
            if any(ids==old_ids and np.sqrt(np.mean(np.sum((norm-old)**2,axis=1)))<.01 for old_ids,old in seen):
                skipped.append(dict(frame=obs['frame_index'],reason='near-duplicate view'));continue
            seen.append((ids,norm));observations.append(obs)
        video.verify()
        progress('Fitting lens; Stop takes effect after the current OpenCV operation...')
        result=fit_lens(observations,pinhole,cancel)
        result['video_sampling']=dict(source_file=video.path.name,source_stamp=video.stamp,
            fps=video.fps,start_seconds=start,end_seconds=stop,interval_seconds=interval,
            accepted_frame_indices=[o['frame_index'] for o in observations],skipped_frames=skipped)
        return result
    finally:video.close()


def floor_pose(obs,lens):
    obj,img,ids=correspondences(obs)
    if lens.get('board')!=BOARD:raise ValueError('Lens was calibrated using a different board')
    if obs['image_size']!=lens['image_size']:raise ValueError('Lens and pose video resolutions differ')
    k=array(lens['intrinsic_matrix'],(3,3));d=array(lens['distortion_coeffs'],(5,))
    if min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0])>1e-8:
        raise ValueError('Invalid lens intrinsics')
    candidates=[]
    result=cv2.solvePnPGeneric(obj,img,k,d,flags=cv2.SOLVEPNP_IPPE)
    for r,t in zip(result[1],result[2]):
        r,t=cv2.solvePnPRefineLM(obj,img,k,d,r.copy(),t.copy())
        rot=cv2.Rodrigues(r)[0];center=-rot.T@t.ravel()
        if not np.isfinite(center).all() or center[2]<=0 or np.min((obj@rot.T+t.ravel())[:,2])<=0:continue
        error=np.linalg.norm(cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2)-img,axis=1)
        candidates.append((float(np.sqrt(np.mean(error**2))),rot,t,center,float(error.max())))
    candidates.sort(key=lambda c:c[0])
    if not candidates or candidates[0][0]>2 or candidates[0][4]>4:
        raise ValueError('Floor pose failed: ensure a sharp, sufficiently large, face-up board (RMS <=2 px)')
    best=candidates[0]
    if len(candidates)>1:
        other=candidates[1]
        if other[0]-best[0]<.1 and (np.linalg.norm(other[3]-best[3])>.01 or np.linalg.norm(other[1]-best[1])>.02):
            raise ValueError('Ambiguous planar pose; use a closer or more oblique board view')
    return dict(intrinsic_matrix=k.tolist(),distortion_coeffs=d.tolist(),image_size=obs['image_size'],
        extrinsics_matrix=np.column_stack((best[1],best[2])).tolist(),
        quality=dict(rms_px=best[0],max_error_px=best[4],camera_center_m=best[3].tolist(),tag_ids=ids,
                     frame_index=obs.get('frame_index'),lens_quality=lens['quality']))


def camera_positions(entries,lenses,seconds,cancel=lambda:False,progress=lambda text:None):
    ids=[e['id'] for e in entries]
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Need at least two distinct camera IDs')
    cameras={}
    for entry in entries:
        if cancel():raise InterruptedError('Cancelled')
        name=entry['id'];progress('Solving floor board pose: '+name)
        if name not in lenses:raise ValueError('Calibrate the lens for '+name+' first')
        video=Video(entry['path'])
        try:
            obs=video.read(seconds);cameras[name]=floor_pose(obs,lenses[name]);video.verify()
            cameras[name]['source']=dict(file=video.path.name,stamp=video.stamp,
                requested_seconds=seconds,actual_seconds=obs['actual_seconds'],fps=video.fps)
        finally:video.close()
    result=calibration(cameras)
    result.update(board=dict(BOARD),board_layout=layout(),lenses=lenses,
        note='User confirmed a stationary face-up floor board at the common original-video time. '
             'Origin is printed board center; +X right, +Y towards IDs 0-2, +Z up. '
             'Camera positions are relative to the board, not an external Unreal scene origin.')
    return result
