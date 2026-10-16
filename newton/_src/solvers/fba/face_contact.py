# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Face contacts: the deepest point of a deformable triangle inside a rigid shape.

Particle contacts test only vertices, so a box edge can pass through a triangle between them. The
point minimizes the shape's signed distance over the triangle with the Frank-Wolfe iteration of
RealSim's face query (``SDFCollisionDevice.cuh``), from the deepest vertex. Vertex optima are left to
the particle contacts; unselected vertices keep exactly zero weight, which marks edge optima.
"""

import numpy as np
import warp as wp

from ...geometry import kernels as sdf
from ...geometry.types import Axis, GeoType


@wp.func
def shape_sdf(geo_type: int, scale: wp.vec3, x: wp.vec3, edge_radius: float) -> wp.vec4:
    """Signed distance [m] and its gradient at ``x`` in the shape frame; ``1e6`` for unsupported shapes.

    Box and cylinder edges are rounded by ``edge_radius``: inside a sharp edge the gradient jumps
    between the two faces, so a point that sinks into the edge would flip its normal.
    """
    d = float(1.0e6)
    g = wp.vec3(0.0, 0.0, 1.0)
    if geo_type == GeoType.SPHERE:
        d = sdf.sdf_sphere(x, scale[0])
        g = sdf.sdf_sphere_grad(x, scale[0])
    elif geo_type == GeoType.BOX:
        r = wp.min(edge_radius, 0.5 * wp.min(scale[0], wp.min(scale[1], scale[2])))
        d = sdf.sdf_box(x, scale[0] - r, scale[1] - r, scale[2] - r) - r
        g = sdf.sdf_box_grad(x, scale[0] - r, scale[1] - r, scale[2] - r)
    elif geo_type == GeoType.CAPSULE:
        d = sdf.sdf_capsule(x, scale[0], scale[1], int(Axis.Z))
        g = sdf.sdf_capsule_grad(x, scale[0], scale[1], int(Axis.Z))
    elif geo_type == GeoType.CYLINDER:
        r = wp.min(edge_radius, 0.5 * wp.min(scale[0], scale[1]))
        d = sdf.sdf_cylinder(x, scale[0] - r, scale[1] - r, int(Axis.Z)) - r
        g = sdf.sdf_cylinder_grad(x, scale[0] - r, scale[1] - r, int(Axis.Z))
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
    edge_radius: float,
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
    r = edge_radius
    d = wp.vec3(shape_sdf(geo, scale, P[0], r)[0], shape_sdf(geo, scale, P[1], r)[0], shape_sdf(geo, scale, P[2], r)[0])
    w = wp.vec3(0.0)
    w[min_index(d)] = 1.0
    for k in range(iterations):
        sd = shape_sdf(geo, scale, w @ P, r)
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
    sd = shape_sdf(geo, scale, x, r)
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


class FaceContactQuery:
    """Face contacts of the triangles around a step's particle contacts.

    A triangle is tested against a shape when one of its particles touches
    the shape, so the particle contact margin also bounds this search.
    """

    def __init__(self, tri_indices: np.ndarray, particle_count: int, device):
        self.tri = np.asarray(tri_indices, dtype=np.int32).reshape(-1, 3)
        order = np.argsort(self.tri.ravel(), kind="stable")
        self.offsets = np.searchsorted(self.tri.ravel()[order], np.arange(particle_count + 1))
        self.incident = (order // 3).astype(np.int64)
        self.tri_d = wp.array(self.tri, dtype=wp.int32, device=device)
        self.no_bodies = wp.empty(1, dtype=wp.transform, device=device)
        self.device = device

    def query(
        self,
        model,
        particle_q,
        body_q,
        particle,
        shape,
        edge_radius: float,
        iterations: int = 32,
        depth_ratio: float = 0.02,
    ):
        """Face contacts as ``(face, shape, bary, body_pos, normal)`` arrays, ordered by shape and triangle.

        ``particle`` and ``shape`` list the particle contacts; ``edge_radius``
        rounds box and cylinder edges as in :func:`shape_sdf`. When two
        triangles report the same edge optimum, the deeper contact remains.
        """
        start, stop = self.offsets[particle], self.offsets[particle + 1]
        counts, n_tri, device = stop - start, len(self.tri), self.device
        faces = self.incident[np.repeat(start - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())]
        candidates = np.unique(np.repeat(np.asarray(shape, dtype=np.int64), counts) * n_tri + faces)
        cand_shape, cand_face = np.divmod(candidates, n_tri)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        out = [
            wp.empty(len(candidates), dtype=t, device=device) for t in (wp.int32, wp.vec3, wp.float32, wp.vec3, wp.vec3)
        ]
        if len(candidates):
            wp.launch(
                face_contact_kernel,
                dim=len(candidates),
                inputs=[
                    *(particle_q, self.tri_d),
                    *(wp.array(a.astype(np.int32), dtype=wp.int32, device=device) for a in (cand_face, cand_shape)),
                    self.no_bodies if body_q is None or not body_q.shape[0] else body_q,
                    *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                    *(iterations, depth_ratio, edge_radius),
                ],
                outputs=[count, *out],
                device=device,
            )
        m = int(count.numpy()[0])
        # The kernel emits in thread order; sort first so equal depths below break ties the same way every run.
        order = np.argsort(out[0].numpy()[:m])
        pair, bary, depth, body_pos, normal = (a.numpy()[:m][order] for a in out)
        face, shp = cand_face[pair], cand_shape[pair]
        # An edge optimum keeps one zero weight: key it by its shape and sorted endpoints.
        zero, rows = np.argmax(bary == 0.0, axis=1), np.arange(m)
        ends = np.sort(np.stack([self.tri[face, (zero + 1) % 3], self.tri[face, (zero + 2) % 3]], axis=1), axis=1)
        n = len(self.offsets)
        key = np.where((bary == 0.0).any(axis=1), (shp * n + ends[:, 0]) * n + ends[:, 1], -1 - rows)
        by_key = np.lexsort((depth, key))
        deepest = by_key[np.diff(key[by_key], prepend=key[by_key][:1] - 1) != 0]
        sel = np.sort(deepest)
        return face[sel], shp[sel], bary[sel], body_pos[sel], normal[sel]
