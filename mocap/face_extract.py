"""Offline ARKit capture adapted from Chamiseul_RGBMocap f5eccc5 (MIT).

CPU-only optional MediaPipe worker; never modifies body motion.
"""
import argparse
import hashlib
import tempfile
import os
import json
import sys
from pathlib import Path

import numpy as np

ARKIT = (
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft',
    'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight',
    'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight', 'jawForward',
    'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
    'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 'browDownRight',
    'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
)
FILE = 'face.npz'
CROP = 512


def emit(kind, **value):
    print('@@MOCAP@@' + json.dumps(dict(type=kind, **value)), flush=True)


def model_path(root):
    lock = json.loads((root / 'installer' / 'face-runtime-lock.json').read_text(encoding='utf-8'))
    path = root / lock['model']['path']
    if not path.is_file():
        raise RuntimeError('The face model is missing. Run Setup_Face.cmd.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock['model']['sha256']:
        raise RuntimeError('Face model checksum mismatch. Run Setup_Face.cmd.')
    return path


def head_box(box, width, height):
    """Square around the head: the top of the person box, a third of its height."""
    x0, y0, x1, y1 = (float(v) for v in box)
    tall = max(y1 - y0, 1.0)
    side = max(tall * 0.32, 64.0)
    cx, cy = (x0 + x1) / 2, y0 + tall * 0.09
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return a, b, int(min(width, a + side)), int(min(height, b + side))


def extract(take, root, progress=lambda done, total: None, selection=None):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    take = Path(take)
    video = take / 'input.mp4'
    boxes_file = take / 'tracking.npz'
    if not video.is_file():
        raise RuntimeError('This take has no prepared input.mp4.')
    identity = source_identity(take)
    boxes = observed = None
    if boxes_file.is_file():
        with np.load(boxes_file, allow_pickle=False) as data:
            boxes = np.asarray(data['boxes'], float)
            observed = np.asarray(data['observed'], bool) if 'observed' in data else np.ones(len(boxes), bool)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not 0 < len(boxes) <= 36000 or observed.shape != (len(boxes),):
            raise ValueError('Invalid person tracking data.')
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(root))),
        running_mode=vision.RunningMode.IMAGE, num_faces=4, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or not 0 < fps <= 240:
        capture.release()
        raise ValueError('Cannot read the take video or its frame rate.')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not 0 < total <= 36000 or (boxes is not None and total != len(boxes)):
        capture.release()
        raise ValueError('Video and tracking frame counts differ; reconstruct this take first.')
    values = np.zeros((total, len(ARKIT)), np.float32)
    valid = np.zeros(total, bool)
    regions = np.full((total, 4), np.nan, np.float32)
    landmarks = np.full((total,478,2),np.nan,np.float16)
    index = 0
    preferred = None
    anchor=0 if selection is None else int(selection[0])
    if selection is not None and (not 0<=anchor<total or not all(np.isfinite(selection[1:])) or not all(0<=v<=1 for v in selection[1:])):
        capture.release();raise ValueError('Invalid selected face/frame')
    order=list(range(anchor,total))+list(range(anchor-1,-1,-1))
    target=None if selection is None else np.array([selection[1],selection[2],selection[1],selection[2]])
    initial_target=target;anchor_region=None;missing=0;processed=0
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            for index in order:
                if selection is not None:
                    if index==anchor-1:target=anchor_region;missing=0
                    if int(capture.get(cv2.CAP_PROP_POS_FRAMES))!=index:capture.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok, frame = capture.read()
                if not ok:
                    break
                if (selection is not None and missing<=round(fps*.5)) or (selection is None and (boxes is None or (observed[index] and np.isfinite(boxes[index]).all()))):
                    box = None if selection is not None or boxes is None else boxes[index]
                    result, region, points, preferred = detect_face(frame, box, landmarker, mp, cv2, preferred, target)
                    if result is not None:
                        values[index] = [result.get(name, 0.0) for name in ARKIT]
                        valid[index] = True
                        regions[index] = region
                        landmarks[index] = points
                        if selection is not None:
                            target=region;missing=0
                            if index==anchor:anchor_region=region
                    elif selection is not None:
                        missing+=1
                        if index==anchor:raise ValueError('No unambiguous face at the clicked position; choose a clearer frame.')
                processed += 1
                if processed % 15 == 0 or processed == total:progress(processed, total)
    finally:
        capture.release()
    index=processed
    if index != total:
        raise ValueError("Video decode ended early; previous face capture was preserved.")
    if source_identity(take) != identity:
        raise ValueError("Video or tracking changed during capture.")
    values, valid = values[:index], valid[:index]
    _hold(values, valid, max_gap=max(1, round(fps * .2)))
    out = take / FILE
    handle, name = tempfile.mkstemp(prefix='face-', suffix='.partial.npz', dir=take)
    os.close(handle)
    partial = Path(name)
    try:
        np.savez_compressed(partial, values=values.astype(np.float16), valid=valid,
                            names=np.array(ARKIT), fps=np.float32(fps), regions=regions, landmarks=landmarks, version=np.int32(2),
                            source=np.array(json.dumps(identity, sort_keys=True)))
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)
    return dict(take=str(take), frames=int(index), found=int(valid.sum()))


def detection_crops(box, width, height, preferred=None):
    """Keep aspect ratio and actor scope; use a head crop only as fallback."""
    if box is None:
        return [('frame', (0, 0, width, height))]
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return []
    person = (max(0,int(x0)),max(0,int(y0)),min(width,int(np.ceil(x1))),min(height,int(np.ceil(y1))))
    head = head_box(box, width, height)
    rows = [('person',person),('head',head)]
    if preferred == 'head':rows.reverse()
    return [(name, crop) for name, crop in rows if crop[2]>crop[0] and crop[3]>crop[1]]


def detect_face(frame, box, landmarker, mp, cv2, preferred=None, target=None):
    height, width = frame.shape[:2]
    for name, (a,b,c,d) in detection_crops(box,width,height,preferred):
        crop = frame[b:d,a:c]
        scale = min(1.,768/max(crop.shape[:2]))
        if scale < 1:crop = cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        result = landmarker.detect(image)
        candidates = []
        for landmarks, blendshapes in zip(result.face_landmarks,result.face_blendshapes):
            points = np.array([(p.x,p.y) for p in landmarks],float)
            if not np.isfinite(points).all():continue
            low,high=points.min(axis=0),points.max(axis=0)
            center=(low+high)/2
            source_center=np.array([a,b])+center*np.array([c-a,d-b])
            if box is not None and not (box[0]<=source_center[0]<=box[2] and box[1]<=source_center[1]<=box[3]):continue
            margin=(high-low)*.12
            region=np.r_[np.maximum(0,low-margin),np.minimum(1,high+margin)]
            region=(region*np.array([c-a,d-b,c-a,d-b])+np.array([a,b,a,b]))/np.array([width,height,width,height])
            source_points=(points*np.array([c-a,d-b])+np.array([a,b]))/np.array([width,height])
            if source_points.shape != (478,2):continue
            candidates.append((float(np.sum((center-[.5,.15])**2)),blendshapes,region,source_points))
        if target is not None:
            center=(target[:2]+target[2:])/2;size=max(target[2]-target[0],target[3]-target[1])
            ranked=[]
            for _,shapes,region,points in candidates:
                distance=float(np.linalg.norm((region[:2]+region[2:])/2-center))
                if size==0:
                    if not (region[0]<=center[0]<=region[2] and region[1]<=center[1]<=region[3]):continue
                elif distance>max(.04,size*.65):continue
                ranked.append((distance,shapes,region,points))
            ranked.sort(key=lambda row:row[0])
            if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>max(.015,size*.15)):
                _,shapes,region,points=ranked[0]
                return {c.category_name:float(np.clip(c.score,0,1)) for c in shapes},region,points,name
            continue
        # Without person tracking, refuse ambiguous multi-face frames.
        if candidates and (box is not None or len(candidates)==1):
            _,shapes,region,points=min(candidates,key=lambda row:row[0])
            return {c.category_name:float(np.clip(c.score,0,1)) for c in shapes},region,points,name
    return None,None,None,preferred


def source_identity(take):
    result = {}
    for name in ('input.mp4', 'tracking.npz'):
        path = Path(take) / name
        if name == 'tracking.npz' and not path.exists():
            result[name] = None
        else:
            stat = path.stat()
            result[name] = [stat.st_size, stat.st_mtime_ns]
    return result


def _hold(values, valid, max_gap=6):
    """Fill only short, bounded interior gaps; retain neutral for long losses."""
    found = np.flatnonzero(valid)
    for left, right in zip(found[:-1], found[1:]):
        if right - left - 1 <= max_gap:
            for index in range(left + 1, right):
                values[index] = values[left if index-left <= right-index else right]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--take', '--project', dest='take', action='append', required=True)
    parser.add_argument('--face-point',type=float,nargs=2)
    parser.add_argument('--face-frame',type=int,default=0)
    parser.add_argument('--face-time',type=float)
    args = parser.parse_args(sys.argv[2:])
    root = Path(__file__).resolve().parents[1]
    takes = [Path(take) for take in args.take]
    if not takes:
        emit('failed', stage='face', message='No take was given.')
        return 1
    summary = []
    for number, take in enumerate(takes):
        def progress(done, total, number=number):
            emit('progress', stage='face', current=number * 1000 + int(1000 * done / max(total, 1)),
                 total=len(takes) * 1000, message='Face capture %d/%d' % (number + 1, len(takes)))
        try:
            frame=args.face_frame
            if args.face_time is not None:
                state=json.loads((take/'project.json').read_text(encoding='utf-8'))
                frame=round((args.face_time-float(state.get('start',0)))*float(state['fps']))
            summary.append(extract(take, root, progress, (frame,*args.face_point) if args.face_point else None))
        except Exception as error:  # report per take, keep going
            summary.append(dict(take=str(take), error=str(error)))
    emit('face_done', stage='face', takes=summary)
    for row in summary:
        if 'error' in row:
            emit('failed', stage='face', message=row['error'])
    return 0 if all('error' not in row for row in summary) else 1


if __name__ == '__main__':
    sys.exit(main())
