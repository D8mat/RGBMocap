"""CPU calibration: measured tag map + pinhole/radtan lens calibration."""
import hashlib
import cv2
import numpy as np
from .apriltag_detection import dictionary


def array(value, shape):
    a = np.asarray(value, np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Expected finite array of shape {shape}')
    return a


def lens_points(points, size, columns=9, rows=6, square_mm=25.):
    """Fit independent views of the same printed checkerboard; units metres."""
    if not 3 <= columns <= 30 or not 3 <= rows <= 30 or not 1 <= square_mm <= 1000:
        raise ValueError('Invalid board dimensions')
    size = tuple(int(x) for x in array(size, (2,)))
    if min(size) <= 0 or len(points) < 10:
        raise ValueError('Need at least 10 accepted views of one resolution')
    obj = np.zeros((columns * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_mm / 1000
    images = [array(p, (columns * rows, 2)).astype(np.float32) for p in points]
    normalized = np.array(images) / np.array(size)
    if np.max(np.std(normalized, axis=0)) < .015:
        raise ValueError('Board views are too similar; vary position, distance and tilt')
    rms, k, d, rv, tv = cv2.calibrateCamera([obj] * len(images), images, size, None, None)
    if not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Lens solution is invalid')
    errors = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)-p)**2, axis=1)))) for r, t, p in zip(rv, tv, images)]
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rv])
    if np.max(np.linalg.norm(normals - normals[0], axis=1)) < .1:
        raise ValueError('Insufficient board tilt diversity; capture more angled views')
    if not np.isfinite(rms) or rms > 2.:
        raise ValueError(f'Lens RMS {rms:.3f} px exceeds 2 px; improve captures')
    return dict(intrinsic_matrix=k.tolist(), distortion_coeffs=d.ravel().tolist(), image_size=list(size),
                quality=dict(rms_px=float(rms), per_view_rms_px=errors, accepted_views=len(images),
                             warning='Low residual alone does not prove accuracy; validate on independent views.'))


def solve(observations, lenses, layout, cancelled=lambda: False):
    """Each map marker has four measured XYZ corners in detector-native order."""
    if layout.get('units') != 'metres' or layout.get('world_up') != 'Z':
        raise ValueError('Marker map requires units=metres and world_up=Z')
    if layout.get('family') != 'tag36h11':
        raise ValueError('Marker map must use tag36h11')
    expected = hashlib.sha256(dictionary().bytesList.tobytes()).hexdigest()
    markers = layout['markers']; world = {}
    for key, value in markers.items():
        p = array(value, (4, 3)); edges = np.roll(p, -1, axis=0)-p
        lengths = np.linalg.norm(edges, axis=1)
        if min(lengths) <= 0 or np.max(np.abs(lengths-lengths.mean())) > lengths.mean()*.02:
            raise ValueError(f'Marker {key}: corners must form a square')
        if max(abs(np.dot(edges[i], edges[(i+1)%4])) for i in range(4)) > lengths.mean()**2*.02 or np.linalg.norm(p[0]+p[2]-p[1]-p[3]) > lengths.mean()*.02:
            raise ValueError(f'Marker {key}: invalid square corner order')
        world[int(key)] = p
    cameras = {}
    for obs in observations:
        if cancelled():raise InterruptedError('Calibration cancelled')
        name = obs['camera_id']
        if name in cameras:raise ValueError('Use one fixed-scene observation per camera')
        if obs.get('dictionary_sha256') != expected or obs.get('family') != 'tag36h11':
            raise ValueError('Observation dictionary differs from the installed detector')
        lens = lenses[name]; k = array(lens['intrinsic_matrix'], (3, 3))
        d = np.asarray(lens['distortion_coeffs'], np.float64)
        if d.shape not in ((4,), (5,)) or not np.isfinite(d).all():raise ValueError('Invalid distortion')
        if min(k[0,0], k[1,1]) <= 0 or not np.allclose(k[2], [0,0,1]) or abs(k[0,1])+abs(k[1,0]) > 1e-8:raise ValueError('Invalid pinhole intrinsics')
        if lens['image_size'] != obs['image_size']:raise ValueError(name+': lens/image resolution mismatch')
        obj, img, seen = [], [], set()
        for m in obs['markers']:
            ident = int(m['id'])
            if ident in seen or m.get('duplicate_id'):raise ValueError(name+': duplicate tag ID')
            seen.add(ident)
            if ident not in world:continue
            p = world[ident]
            if not np.isclose(np.linalg.norm(p[1]-p[0]), obs['tag_size_m'], rtol=.02):raise ValueError('Printed tag size differs from marker map')
            obj.extend(p); img.extend(array(m['corners_px'], (4,2)))
        if len(obj) < 12:raise ValueError(name+': need at least 3 mapped tags')
        obj = np.array(obj); img = np.array(img)
        if np.linalg.matrix_rank(obj-obj.mean(0), tol=1e-7) < 2:raise ValueError('Degenerate marker layout')
        ok, r, t, inliers = cv2.solvePnPRansac(obj, img, k, d, iterationsCount=200, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 12 or len(inliers) < .8*len(obj):raise ValueError(name+': insufficient consistent corners')
        ix = inliers.ravel(); r, t = cv2.solvePnPRefineLM(obj[ix], img[ix], k, d, r, t)
        rot = cv2.Rodrigues(r)[0]
        if np.min((obj@rot.T+t.ravel())[:,2]) <= 0:raise ValueError(name+': markers behind camera')
        error = np.linalg.norm(cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2)-img, axis=1)
        rms = float(np.sqrt(np.mean(error[ix]**2)))
        if not np.isfinite(rms) or rms > 2.:raise ValueError(name+': reprojection RMS exceeds 2 px')
        cameras[name] = dict(intrinsic_matrix=k.tolist(), distortion_coeffs=d.tolist(), image_size=obs['image_size'],
                             extrinsics_matrix=np.column_stack((rot,t)).tolist(),
                             quality=dict(rms_px=rms, max_error_px=float(error.max()), inlier_corners=len(ix), total_corners=len(obj)))
    if not cameras:raise ValueError('No camera observations')
    return dict(schema='rgbmocap-camera-calibration-v1', units='metres', world_up='Z',
                extrinsics_convention='world_to_camera', cameras=cameras,
                requires_undistortion=any(np.any(np.abs(c['distortion_coeffs']) > 1e-8) for c in cameras.values()),
                note='Measured map defines world frame. Nonzero distortion requires undistorted videos before MAMMA import.')
