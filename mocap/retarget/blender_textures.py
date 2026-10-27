"""Original-size base-color images and per-corner UVs, independent of motion."""
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import bpy
import numpy as np

from .file_paths import io_path

CACHE = 'preview-textures-v1'


def extract(out):
    out = Path(io_path(out))
    if (out / CACHE / 'manifest.json').is_file():
        return
    temp = out / ('.textures-' + uuid.uuid4().hex)
    temp.mkdir()
    try:
        images, sizes, batches, corners, faces = {}, {}, [], [], []
        offset = total = 0
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            mesh.calc_loop_triangles()
            uv = mesh.uv_layers.active
            faces.extend([[offset + i for i in t.vertices] for t in mesh.loop_triangles])
            for mi in sorted({t.material_index for t in mesh.loop_triangles}):
                mat = mesh.materials[mi] if mi < len(mesh.materials) else None
                color = list(mat.diffuse_color) if mat else [.65, .65, .65, 1.]
                tex = None
                if mat and mat.use_nodes:
                    nodes = mat.node_tree.nodes
                    bs = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                    inp = bs.inputs.get('Base Color') if bs else None
                    if inp is not None:
                        color = list(inp.default_value)
                        if inp.is_linked:
                            node = inp.links[0].from_node
                            if node.type == 'TEX_IMAGE':
                                tex = node.image
                    if tex is None:
                        node = next((n for n in nodes if n.type == 'TEX_IMAGE' and n.image and 'shade' not in n.name.lower()), None)
                        if node:
                            tex = node.image
                key = None
                if tex and uv and tex.size[0] and tex.size[1]:
                    key = images.get(tex.name)
                    if key is None:
                        w, h = tex.size
                        total += w * h * 4
                        if w * h * 4 > 256 * 1024**2 or total > 512 * 1024**2:
                            raise ValueError('Original character textures exceed the 256 MB/image or 512 MB/character preview limit.')
                        key = 'image' + str(len(images))
                        images[tex.name] = key
                        sizes[key] = [w, h]
                        pixels = np.empty(w * h * 4, np.float32)
                        tex.pixels.foreach_get(pixels)
                        np.clip(pixels, 0, 1, out=pixels)
                        pixels *= 255
                        np.save(temp / (key + '.npy'), pixels.reshape(h, w, 4).astype(np.uint8))
                        del pixels
                start = len(corners)
                for tri in mesh.loop_triangles:
                    if tri.material_index == mi:
                        for li in tri.loops:
                            corners.append([offset + mesh.loops[li].vertex_index, *(uv.data[li].uv if uv else (0, 0))])
                batches.append(dict(start=start, count=len(corners)-start, image=key, color=color))
            offset += len(mesh.vertices)
        faces = np.asarray(faces, np.uint32)
        with np.load(out / 'preview.npz', allow_pickle=False) as asset:
            if len(asset['vertices']) != offset or not np.array_equal(asset['faces'], faces):
                raise ValueError('Stored character geometry differs from the texture source. Reimport the character.')
        np.save(temp / 'corners.npy', np.asarray(corners, np.float32))
        metadata = dict(version=1, vertices=offset, faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
                        batches=batches, images=sizes)
        (temp / 'manifest.json').write_text(json.dumps(metadata), encoding='utf8')
        temp.rename(out / CACHE)
    finally:
        # Only this invocation's temporary directory, never a user asset.
        if temp.exists():
            shutil.rmtree(temp)
