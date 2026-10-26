"""Small posed cuboid avatar. Finger articulation is a neutral display pose."""
from __future__ import annotations

import numpy as np

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * .5
FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 4, 7, 3],
                  [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2]])
BODY = (229, 237, 245)
LEFT = (128, 211, 196)
RIGHT = (239, 183, 143)


def unit(vector, fallback=(0, 1, 0)):
    length = float(np.linalg.norm(vector))
    return np.asarray(vector) / length if length > 1e-7 else np.asarray(fallback, dtype=float)


def body_frame(joints):
    up = unit(joints[12] - joints[0])
    left = unit(joints[16] - joints[17], (1, 0, 0))
    front = unit(np.cross(left, up), (0, 0, 1))
    left = unit(np.cross(up, front), (1, 0, 0))
    return np.column_stack((left, up, front))


def global_rotations(local_rotvec):
    """SMPL-X body rotations, composed down the original joint hierarchy."""
    from scipy.spatial.transform import Rotation
    local = Rotation.from_rotvec(np.asarray(local_rotvec).reshape(-1, 3)).as_matrix()
    local = local.reshape(-1, 22, 3, 3)
    world = local.copy()
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            world[:, joint] = world[:, parent] @ local[:, joint]
    return world.astype(np.float32)


def build_avatar(joints, rotations=None, limb_mode="smpl"):
    """Create a small display rig without changing the source skeleton.

    ``smpl`` retains proximal SMPL-X roll. ``hinge`` retargets ONLY box
    cross-sections to the three-point bend plane; wrist/ankle motion is retained.
    The latter is an explicitly different display rig, not inferred rotations.
    Both modes keep every original joint center, translation and bend angle.
    """
    joints = np.asarray(joints, dtype=float)
    frame = body_frame(joints)
    # Bone lengths are invariant under articulation. A neck-to-pelvis chord
    # changed all widths and hand/foot sizes when the spine bent.
    scale = np.clip(sum(np.linalg.norm(joints[b] - joints[a]) for a, b in
                        ((0, 3), (3, 6), (6, 9), (9, 12))) / .49, .6, 1.5)
    centers, sizes, frames, colors, names = [], [], [], [], []
    custom_vertices, hidden_faces = {}, []

    def rotation(joint):
        return rotations[joint] if rotations is not None else frame

    def box(name, center, size, axes, color=BODY):
        centers.append(center)
        sizes.append(np.asarray(size) * scale)
        frames.append(axes)
        colors.append(color)
        names.append(name)
        return len(names) - 1

    def segment(name, start, end, width, depth, hint, color=BODY, hinge_axis=None):
        delta = end - start
        length = np.linalg.norm(delta)
        if length < 1e-6:
            return
        up = delta / length
        side = (hinge_axis - up * np.dot(up, hinge_axis)) if hinge_axis is not None else np.cross(up, hint)
        if np.linalg.norm(side) < 1e-5:
            side = np.cross(up, (1, 0, 0) if abs(up[0]) < .8 else (0, 0, 1))
        side = unit(side)
        axes = np.column_stack((side, up, np.cross(side, up)))
        # Small overlap covers the joint instead of opening a black gap.
        box(name, (start + end) * .5, (width, length / scale + .008, depth), axes, color)

    pelvis = rotation(0)
    # A continuous box-section torso, with a shared ring at EVERY spine joint.
    # Adjacent sections share their vertices, like a very small skinned mesh.
    # Former rigid boxes skipped spine3 and left disconnected, tilting seams.
    knots = [(joints[1] + joints[2]) * .5 - pelvis[:, 1] * .035 * scale,
             joints[0], joints[3], joints[6], joints[9], joints[12]]
    widths = (.225, .235, .19, .22, .275, .28)
    depths = (.15, .165, .14, .155, .17, .17)
    ring_frames = []
    for k, (parent, current) in enumerate(((0, 0), (0, 0), (0, 3), (3, 6), (6, 9), (9, 9))):
        tangent = unit(knots[min(k + 1, 5)] - knots[max(k - 1, 0)])
        front = rotation(parent)[:, 2] + rotation(current)[:, 2]
        side = unit(np.cross(tangent, front), frame[:, 0])
        ring_frames.append(np.column_stack((side, tangent, np.cross(side, tangent))))
    for k, name in enumerate(("pelvis", "waist", "abdomen", "ribs", "chest")):
        index = box(name, (knots[k] + knots[k + 1]) * .5, (1, 1, 1), frame)
        vertices = []
        for x, y, z in CORNERS:
            ring = k + int(y > 0)
            axes = ring_frames[ring]
            vertices.append(knots[ring] + scale * (x * widths[ring] * axes[:, 0] + z * depths[ring] * axes[:, 2]))
        custom_vertices[index] = vertices
        # FACES[4/5] cap the Y ends. Internal coincident caps are not surfaces.
        if k > 0:
            hidden_faces.append(index * 6 + 4)
        if k < 4:
            hidden_faces.append(index * 6 + 5)
    head = rotation(15)
    head_center = joints[15] + head[:, 1] * .045 * scale
    # The neck ends at the head JOINT. Head rotation cannot move this endpoint.
    segment("neck", joints[12], joints[15], .075, .075, rotation(12)[:, 2])
    box("head", head_center, (.18, .23, .175), head)

    for label, hip, knee, ankle, toe, shoulder, elbow, wrist, sign, tint in (
        ("left", 1, 4, 7, 10, 16, 18, 20, 1, LEFT),
        ("right", 2, 5, 8, 11, 17, 19, 21, -1, RIGHT)):
        def bend_axis(a, b, c):
            incoming = unit(joints[b] - joints[a])
            outgoing = unit(joints[c] - joints[b])
            reference = unit(np.cross(incoming, rotation(a)[:, 2]), frame[:, 0])
            normal = np.cross(incoming, outgoing)
            sine = np.linalg.norm(normal)
            if sine < 1e-7:
                return reference
            normal /= sine
            if np.dot(normal, reference) < 0:
                normal = -normal
            # A nearly straight chain has no reliable bend plane. Smoothly
            # approach the proximal frame below 6 degrees; no previous-frame
            # state, so arbitrary seeks produce exactly the same geometry.
            weight = np.clip(sine / .105, 0, 1)
            weight = weight * weight * (3 - 2 * weight)
            return unit(reference * (1 - weight) + normal * weight)

        leg_axis = bend_axis(hip, knee, ankle) if limb_mode == "hinge" else None
        arm_axis = bend_axis(shoulder, elbow, wrist) if limb_mode == "hinge" else None
        for name, a, b, width, depth, hinge in (("thigh", hip, knee, .10, .115, leg_axis),
                ("shin", knee, ankle, .078, .085, leg_axis),
                ("upper_arm", shoulder, elbow, .075, .082, arm_axis),
                ("forearm", elbow, wrist, .058, .063, arm_axis)):
            # A joint rotates its outgoing bone. Distal (knee/elbow/wrist/ankle)
            # rotation must never determine the incoming segment's axial roll.
            segment(label + name, joints[a], joints[b], width, depth, rotation(a)[:, 2], tint, hinge)
        # Foot direction is visible even for legacy position-only results.
        forward = unit(joints[toe] - joints[ankle], frame[:, 2])
        up = unit(rotation(ankle)[:, 1])
        lateral = unit(np.cross(up, forward), frame[:, 0])
        foot_frame = np.column_stack((lateral, np.cross(forward, lateral), forward))
        foot_center = joints[ankle] + forward * .028 * scale - foot_frame[:, 1] * .018 * scale
        box(label + "foot", foot_center, (.080, .045, .17), foot_frame, tint)
        if rotations is not None:
            distal = unit(rotation(wrist)[:, 0] * sign)
            across = unit(rotation(wrist)[:, 1])
        else:
            distal = unit(joints[wrist] - joints[elbow])
            across = unit(np.cross(frame[:, 2], distal), frame[:, 0])
        normal = unit(np.cross(across, distal), frame[:, 2])
        across = unit(np.cross(distal, normal))
        hand_frame = np.column_stack((across, distal, normal))
        wrist_position = joints[wrist]
        palm = wrist_position + distal * .043 * scale
        box(label + "palm", palm, (.077, .082, .028), hand_frame, tint)
        for finger, offset, length in (("little", -.028, .052), ("ring", -.0095, .070),
                                       ("middle", .0095, .077), ("index", .028, .065)):
            root = wrist_position + (distal * .086 + across * offset) * scale
            direction = unit(distal + across * offset * 2)
            for phalanx, fraction in enumerate((.43, .32, .25)):
                end = root + direction * length * fraction * scale
                segment(label + finger + str(phalanx), root, end, .011, .013, normal, tint)
                root = end
        root = wrist_position + (distal * .026 + across * .041) * scale
        direction = unit(distal * .67 + across * .74)
        for phalanx in range(2):
            end = root + direction * .028 * scale
            segment(label + "thumb" + str(phalanx), root, end, .014, .016, normal, tint)
            root = end

    vertices = np.einsum("nij,nkj->nki", np.asarray(frames), CORNERS[None] * np.asarray(sizes)[:, None])
    vertices += np.asarray(centers)[:, None]
    for index, points in custom_vertices.items():
        vertices[index] = points
    faces = vertices[:, FACES].reshape(-1, 4, 3)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    colors = np.repeat(np.asarray(colors), 6, axis=0)
    light = unit(np.array([-.35, .8, .5]))
    shading = .78 + .22 * np.maximum(0, normals @ light)
    enabled = np.ones(len(faces), dtype=bool)
    enabled[hidden_faces] = False
    return {"faces": faces, "normals": normals, "colors": np.clip(colors * shading[:, None], 0, 255).astype(np.uint8),
            "forward": pelvis[:, 2], "names": names, "enabled_faces": enabled}
