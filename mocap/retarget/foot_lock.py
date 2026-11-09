"""Contact-only target IK: fixed bone lengths, no source or asset edits."""
import numpy as np
from .solver import fk, local_from_world, align


def runs(mask):
    edges=np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def solve_goals(base, goals, pelvis=None):
    n=len(base['root'])
    world=base['world'].copy();root=base['root'].copy() if pelvis is None else pelvis.copy();rest=base['rest']
    weights=np.ones((n,2));active=weights.astype(bool)
    lengths=[(np.linalg.norm(rest[k]-rest[h]),np.linalg.norm(rest[a]-rest[k])) for h,k,a in ((1,4,7),(2,5,8))]
    for i in np.flatnonzero(weights.any(1)):
        # Only compensate pelvis if the target's shorter legs cannot reach.
        # Project onto both reach spheres; never stretch the imported bones.
        hip_offsets=base['points'][i,[1,2]]-base['root'][i]
        for _ in range(32):
            largest=0.
            for side in (0,1):
                if not weights[i,side]:continue
                u,v=lengths[side];d=root[i]+hip_offsets[side]-goals[i,side];dist=np.linalg.norm(d)
                excess=max(0.,dist-(u+v-1e-7));largest=max(largest,excess)
                if excess:root[i]-=d*(excess/max(dist,1e-12))
            if largest<1e-8:break
        shift=root[i]-base['root'][i]
        for side,(h,k,a) in enumerate(((1,4,7),(2,5,8))):
            if not weights[i,side]:continue
            hip=base['points'][i,h]+shift;old_knee=base['points'][i,k]+shift
            vector=goals[i,side]-hip;distance=np.linalg.norm(vector)
            direction=vector/max(distance,1e-12);u,v=lengths[side]
            d=np.clip(distance,abs(u-v)+1e-7,u+v-1e-7)
            pole=old_knee-hip;pole-=direction*np.dot(pole,direction)
            if np.linalg.norm(pole)<1e-8:
                pole=base['world'][i,h]@np.array([0.,0.,1.]);pole-=direction*np.dot(pole,direction)
            if np.linalg.norm(pole)<1e-8:
                pole=np.eye(3)[np.argmin(abs(direction))];pole-=direction*np.dot(pole,direction)
            pole/=max(np.linalg.norm(pole),1e-12)
            along=(u*u-v*v+d*d)/(2*d)
            knee=hip+direction*along+pole*np.sqrt(max(0.,u*u-along*along))
            end=hip+direction*d
            world[i,h]=align(base['points'][i,k]-base['points'][i,h],knee-hip)@base['world'][i,h]
            world[i,k]=align(base['points'][i,a]-base['points'][i,k],end-knee)@base['world'][i,k]
    local=local_from_world(world);points,world=fk(local,rest,root)
    errors=np.linalg.norm(points[:,[7,8]]-goals,axis=-1)
    report=dict(contact_ik=True,contact_frames=active.sum(axis=0).tolist(),
                ankle_error_max_mm=float(errors[active].max(initial=0)*1000),
                pelvis_compensation_max_mm=float(np.linalg.norm(root-base['root'],axis=-1).max(initial=0)*1000))
    if report['ankle_error_max_mm']>2.:
        raise ValueError('Target proportions cannot reach source foot trajectories without stretching.')
    return dict(base,local=local,world=world,points=points,root=root),report


def sole_heights(data, geo):
    from .solver import skin
    result=np.empty((len(data['root']),2))
    for a in range(0,len(result),64):
        part=dict(data,points=data['points'][a:a+64],world=data['world'][a:a+64])
        for side in (0,1):
            result[a:a+64,side]=skin(part,geo[f'foot_v{side}'],geo[f'foot_w{side}'])[:,:,1].min(axis=1)
    return result


def solve(base, source, geo, active):
    """Match source sole height and travel, including lifts inside contact intervals."""
    heights=np.asarray(source['foot_surface_height'])
    n=len(base['root'])
    if heights.shape!=(n,2) or not np.isfinite(heights).all():
        raise ValueError('Prepare fresh retarget source foot-surface evidence.')
    source_ankles=source['points'][:,[7,8]]+source['origin']
    goals=base['points'][:,[7,8]].copy()
    # Retain the target's stance-width offset while following source travel.
    for axis in (0,2):
        goals[:,:,axis]=source_ankles[:,:,axis]+(goals[0,:,axis]-source_ankles[0,:,axis])
    current=base
    for iteration in range(8):
        error=heights-sole_heights(current,geo)
        goals[:,:,1]+=error
        current,report=solve_goals(base,goals,pelvis_for_legs(base,source,goals))
        if abs(heights-sole_heights(current,geo)).max()<.0001:break
    error=sole_heights(current,geo)-heights
    report.update(mode_detail='source_surface_trajectory',contact_frames=np.asarray(active)[:,:2].sum(axis=0).tolist(),
                  sole_height_error_max_mm=float(abs(error).max()*1000),surface_iterations=iteration+1)
    if abs(error).max()>.002:raise ValueError('Retarget foot-surface alignment did not converge within 2 mm.')
    return current,report


def pelvis_for_legs(base, source, goals):
    """Fit one pelvis to source knee flexion using unchanged target leg lengths."""
    rest=base['rest'];p=source['points'];n=len(p)
    hip_offsets=base['points'][:,[1,2]]-base['root'][:,None]
    centres=goals-hip_offsets
    candidates=[];radii=[]
    for side,(h,k,a) in enumerate(((1,4,7),(2,5,8))):
        upper=p[:,k]-p[:,h];lower=p[:,a]-p[:,k]
        upper/=np.maximum(np.linalg.norm(upper,axis=-1,keepdims=True),1e-12)
        lower/=np.maximum(np.linalg.norm(lower,axis=-1,keepdims=True),1e-12)
        u=np.linalg.norm(rest[k]-rest[h]);v=np.linalg.norm(rest[a]-rest[k])
        vector=upper*u+lower*v
        radii.append(np.clip(np.linalg.norm(vector,axis=-1),abs(u-v)+1e-7,u+v-1e-7))
        candidates.append(centres[:,side]-vector)
    # Exact sphere intersection can jump between a circle and the fallback when
    # the two proportions cannot satisfy both source knee angles simultaneously.
    # Use the continuous least-squares compromise of the two directional roots.
    prior=(candidates[0]+candidates[1])*.5
    from scipy.optimize import least_squares
    from scipy.ndimage import gaussian_filter1d
    # Soft length agreement avoids the discontinuity of exact sphere solutions.
    for i in range(n):
        def residual(x):
            return np.r_[np.linalg.norm(x-centres[i],axis=-1)-np.array([radii[0][i],radii[1][i]]), .02*(x-prior[i])]
        prior[i]=least_squares(residual,prior[i],max_nfev=20).x
    # Filter only proportion compensation, never the source pelvis trajectory.
    fps=float(source.get('fps',30.))
    offset=prior-base['root']
    return base['root']+gaussian_filter1d(offset,max(.5,fps*.12),axis=0,mode='nearest')


