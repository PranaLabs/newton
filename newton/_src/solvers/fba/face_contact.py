# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Face contacts: the deepest point of a deformable triangle inside a rigid shape.

Particle contacts test only vertices, so a box edge can pass through a triangle between them. The
point minimizes the shape's signed distance over the triangle with the Frank-Wolfe iteration of
RealSim's face query (``SDFCollisionDevice.cuh``), from the deepest vertex. Vertex optima are left to
the particle contacts; unselected vertices keep exactly zero weight, which marks edge optima.
"""

import warp as wp

from ...geometry import kernels as sdf
from ...geometry.types import Axis, GeoType


@wp.func
def shape_sdf(geo_type: int, scale: wp.vec3, x: wp.vec3) -> wp.vec4:
    """Signed distance [m] and its gradient at ``x`` in the shape frame; ``1e6`` for unsupported shapes."""
    d = float(1.0e6)
    g = wp.vec3(0.0, 0.0, 1.0)
    if geo_type == GeoType.SPHERE:
        d = sdf.sdf_sphere(x, scale[0])
        g = sdf.sdf_sphere_grad(x, scale[0])
    elif geo_type == GeoType.BOX:
        d = sdf.sdf_box(x, scale[0], scale[1], scale[2])
        g = sdf.sdf_box_grad(x, scale[0], scale[1], scale[2])
    elif geo_type == GeoType.CAPSULE:
        d = sdf.sdf_capsule(x, scale[0], scale[1], int(Axis.Z))
        g = sdf.sdf_capsule_grad(x, scale[0], scale[1], int(Axis.Z))
    elif geo_type == GeoType.CYLINDER:
        d = sdf.sdf_cylinder(x, scale[0], scale[1], int(Axis.Z))
        g = sdf.sdf_cylinder_grad(x, scale[0], scale[1], int(Axis.Z))
    elif geo_type == GeoType.CONE:
        d = sdf.sdf_cone(x, scale[0], scale[1], int(Axis.Z))
        g = sdf.sdf_cone_grad(x, scale[0], scale[1], int(Axis.Z))
    elif geo_type == GeoType.ELLIPSOID:
        d = sdf.sdf_ellipsoid(x, scale)
        g = sdf.sdf_ellipsoid_grad(x, scale)
    return wp.vec4(d, g[0], g[1], g[2])


@wp.func
def min_index(a: wp.vec3) -> int:
    """Index of the smallest component, the first on ties."""
    j = int(0)
    if a[1] < a[j]:
        j = 1
    if a[2] < a[j]:
        j = 2
    return j


@wp.kernel
def face_contact_kernel(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    pair_face: wp.array[wp.int32],
    pair_shape: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    iterations: int,
    depth_ratio: float,
    count: wp.array[wp.int32],
    pair: wp.array[wp.int32],
    bary: wp.array[wp.vec3],
    depth: wp.array[wp.float32],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    """Deepest point of each candidate ``(triangle, shape)`` pair, emitted if it lies inside the triangle
    or on an edge and ``depth_ratio`` times the triangle's size below every vertex. ``body_pos`` is the
    closest surface point in the body frame, as ``Contacts.soft_contact_body_pos``.
    """
    i = wp.tid()
    f = pair_face[i]
    s = pair_shape[i]
    X_bs = shape_transform[s]
    X_ws = X_bs
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_bs)
    X_sw = wp.transform_inverse(X_ws)
    P = wp.matrix_from_rows(
        wp.transform_point(X_sw, particle_q[tri_indices[f, 0]]),
        wp.transform_point(X_sw, particle_q[tri_indices[f, 1]]),
        wp.transform_point(X_sw, particle_q[tri_indices[f, 2]]),
    )
    geo = shape_type[s]
    scale = shape_scale[s]
    d = wp.vec3(shape_sdf(geo, scale, P[0])[0], shape_sdf(geo, scale, P[1])[0], shape_sdf(geo, scale, P[2])[0])
    w = wp.vec3(0.0)
    w[min_index(d)] = 1.0
    for k in range(iterations):
        sd = shape_sdf(geo, scale, w @ P)
        # Step toward the vertex with the lowest directional derivative until the Frank-Wolfe gap closes.
        a = P @ wp.vec3(sd[1], sd[2], sd[3])
        j = min_index(a)
        if wp.dot(w, a) - a[j] <= 0.0:
            break
        step = 2.0 / float(k + 2)
        w = (1.0 - step) * w
        w[j] += step
    selected = int(0)
    for k in range(3):
        if w[k] > 0.0:
            selected += 1
    x = w @ P
    sd = shape_sdf(geo, scale, x)
    size = wp.max(wp.length(P[1] - P[0]), wp.max(wp.length(P[2] - P[1]), wp.length(P[0] - P[2])))
    if selected < 2 or sd[0] >= wp.min(d) - depth_ratio * size:
        return
    n = wp.normalize(wp.vec3(sd[1], sd[2], sd[3]))
    idx = wp.atomic_add(count, 0, 1)
    pair[idx] = i
    bary[idx] = w
    depth[idx] = sd[0]
    body_pos[idx] = wp.transform_point(X_bs, x - sd[0] * n)
    normal[idx] = wp.transform_vector(X_ws, n)
