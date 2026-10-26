import numpy as np


def checkerboard_geometry(
    length=12.0,
    color0=(0.8, 0.9, 0.9),
    color1=(0.6, 0.7, 0.7),
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    if up not in {"y", "z"}:
        raise ValueError(f"unsupported up axis {up!r}")
    colors = (np.array((*color0, alpha)), np.array((*color1, alpha)))
    size = max(2, int(length / tile_width))
    radius = size * tile_width / 2.0
    vertices = []
    faces = []
    vertex_colors = []
    face_template = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    for row in range(size):
        for column in range(size):
            u0 = column * tile_width - radius
            v0 = row * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zeros = np.zeros(4)
            if up == "y":
                tile_vertices = np.stack([us + c1, zeros, vs + c2], axis=-1)
            else:
                tile_vertices = np.stack([us + c1, vs + c2, zeros], axis=-1)
            tile_index = row * size + column
            tile_color = colors[(row + column) % 2]
            vertices.append(tile_vertices)
            faces.append(face_template + 4 * tile_index)
            vertex_colors.append(np.repeat(tile_color[None], 4, axis=0))
    vertices = np.concatenate(vertices).astype(np.float32)
    faces = np.concatenate(faces).astype(np.float32)
    vertex_colors = np.concatenate(vertex_colors).astype(np.float32)
    return vertices, faces, vertex_colors, vertex_colors.copy()
