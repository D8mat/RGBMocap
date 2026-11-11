"""Blender subprocess: extract relative ARKit morphs without modifying the FBX."""
import argparse
import sys
from pathlib import Path
import json
import bpy
import numpy as np


def export(source, target):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mocap.face_extract import ARKIT
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(source), use_anim=False)
    objects = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    vertices, morphs, triangles, colors, rows = [], [], [], [], []
    offset = 0
    for obj in objects:
        keys = obj.data.shape_keys
        if keys is None or not any(name in keys.key_blocks for name in ARKIT):
            continue
        if not keys.use_relative:
            raise ValueError('Absolute shape keys are not supported: ' + obj.name)
        n = len(obj.data.vertices)
        if offset + n > 200000:
            raise ValueError('Face preview supports up to 200000 vertices.')
        transform = np.array(obj.matrix_world, np.float32)
        base = np.empty((n, 3), np.float32)
        keys.reference_key.data.foreach_get('co', base.ravel())
        world = base @ transform[:3, :3].T + transform[:3, 3]
        delta = np.zeros((52, n, 3), np.float32)
        matched = []
        for index, name in enumerate(ARKIT):
            key = keys.key_blocks.get(name)
            if key is None:
                continue
            if key.vertex_group:
                raise ValueError('Masked shape keys are not supported: ' + name)
            value = np.empty_like(base); relative = np.empty_like(base)
            key.data.foreach_get('co', value.ravel())
            key.relative_key.data.foreach_get('co', relative.ravel())
            delta[index] = (value - relative) @ transform[:3, :3].T
            matched.append(name)
        obj.data.calc_loop_triangles()
        faces = np.array([tuple(t.vertices) for t in obj.data.loop_triangles], np.uint32) + offset
        # Neutral studio colors; original textures are not imported into the app.
        color = (.72, .47, .36) if len(matched) > 20 else (.88, .88, .82)
        vertices.append(world);morphs.append(delta);triangles.append(faces)
        colors.append(np.tile(color,(n,1)).astype(np.float32))
        rows.append(dict(object=obj.name, channels=matched, vertices=n))
        offset += n
    if not vertices:
        raise ValueError('No meshes with exact ARKit shape-key names found.')
    base = np.concatenate(vertices); delta = np.concatenate(morphs, axis=1)
    center = (base.min(axis=0)+base.max(axis=0))/2
    scale = float(np.max(np.ptp(base,axis=0)))
    if not np.isfinite(base).all() or not np.isfinite(delta).all() or scale <= 0:
        raise ValueError('Invalid face geometry.')
    # Preserve Z-up axes, normalize only the preview camera's geometry.
    base = (base-center)/scale; delta /= scale
    faces=np.concatenate(triangles)
    # Incident triangle corners for smooth normal recomputation on the GPU.
    a=faces.ravel();b=np.roll(faces,-1,axis=1).ravel();c=np.roll(faces,-2,axis=1).ravel()
    order=np.argsort(a,kind='stable')
    offsets=np.r_[0,np.cumsum(np.bincount(a,minlength=len(base)))].astype(np.uint32)
    neighbors=np.column_stack((b[order],c[order])).astype(np.uint32).ravel()
    with open(target,'wb') as handle:
        np.savez(handle, packed=np.concatenate((base[None],delta)).astype(np.float32),
                 faces=faces, offsets=offsets, neighbors=neighbors, colors=np.concatenate(colors),
                 vertex_count=np.array(len(base)), report=np.array(json.dumps(rows)))
    print('FACE_MODEL_READY '+json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    export(Path(args.source),Path(args.output))
