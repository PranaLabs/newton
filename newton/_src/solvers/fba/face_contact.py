# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Face contacts: the deepest point of a deformable triangle inside a rigid shape.

Particle contacts test only vertices, so a box edge can pass through a triangle between them. The
point minimizes the shape's signed distance over the triangle with the Frank-Wolfe iteration of
RealSim's face query (``SDFCollisionDevice.cuh``), from the deepest vertex. Vertex optima are left to
the particle contacts; unselected vertices keep exactly zero weight, which marks edge optima. A face
contact appears as soon as the optimum leaves the vertices, where it coincides with that vertex's
particle contact, so switching it on or off moves no constraint beyond the Frank-Wolfe accuracy.
"""

import numpy as np
import warp as wp

from ...geometry import kernels as sdf
from ...geometry.flags import ShapeFlags
from ...geometry.types import Axis, GeoType


@wp.func
def shape_sdf(geo_type: int, scale: wp.vec3, mesh: wp.uint64, x: wp.vec3, edge_radius: float) -> wp.vec4:
    """Signed distance [m] and its gradient at ``x`` in the shape frame; ``1e6`` for unsupported shapes.

    Box and cylinder edges are rounded by ``edge_radius``: inside a sharp edge the gradient jumps
    between the two faces, so a point that sinks into the edge would flip its normal. Meshes use the
    closest point and sign of Warp's query, scaled as in Newton's particle contacts.
    """
    d = float(1.0e6)
    g = wp.vec3(0.0, 0.0, 1.0)
    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        q = wp.mesh_query_point_sign_normal(mesh, wp.cw_div(x, scale), 1.0e6)
        if q.result:
            delta = x - wp.cw_mul(wp.mesh_eval_position(mesh, q.face, q.u, q.v), scale)
            d = q.sign * wp.length(delta)
            g = wp.mesh_eval_face_normal(mesh, q.face)
            if wp.length(delta) > 0.0:
                g = q.sign * wp.normalize(delta)
    elif geo_type == GeoType.SPHERE:
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
    shape_source_ptr: wp.array[wp.uint64],
    iterations: int,
    edge_radius: float,
    count: wp.array[wp.int32],
    pair: wp.array[wp.int32],
    bary: wp.array[wp.vec3],
    depth: wp.array[wp.float32],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    """Deepest point of each candidate ``(triangle, shape)`` pair, emitted if it lies inside the triangle
    or on an edge, below every vertex. ``body_pos`` is the closest surface point in the body frame, as
    ``Contacts.soft_contact_body_pos``.
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
    geo, scale, mesh, r = shape_type[s], shape_scale[s], shape_source_ptr[s], edge_radius
    d = wp.vec3(
        shape_sdf(geo, scale, mesh, P[0], r)[0],
        shape_sdf(geo, scale, mesh, P[1], r)[0],
        shape_sdf(geo, scale, mesh, P[2], r)[0],
    )
    w = wp.vec3(0.0)
    w[min_index(d)] = 1.0
    for k in range(iterations):
        sd = shape_sdf(geo, scale, mesh, w @ P, r)
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
    sd = shape_sdf(geo, scale, mesh, x, r)
    if selected < 2 or sd[0] >= wp.min(d):
        return
    n = wp.normalize(wp.vec3(sd[1], sd[2], sd[3]))
    idx = wp.atomic_add(count, 0, 1)
    pair[idx] = i
    bary[idx] = w
    depth[idx] = sd[0]
    body_pos[idx] = wp.transform_point(X_bs, x - sd[0] * n)
    normal[idx] = wp.transform_vector(X_ws, n)


@wp.kernel
def near_face_kernel(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    margin: float,
    edge_radius: float,
    count: wp.array[wp.int32],
    pair: wp.array[wp.int64],
):
    """``shape * tri_count + face`` of each pair whose triangle may come within ``margin`` of the shape: the
    signed distance changes no faster than the point moves, so no triangle point is closer than the
    centroid's distance minus the triangle's radius about the centroid."""
    f, s = wp.tid()
    world = shape_world[s]
    if (shape_flags[s] & ShapeFlags.COLLIDE_PARTICLES) == 0 or (
        world >= 0 and world != particle_world[tri_indices[f, 0]]
    ):
        return
    X_ws = shape_transform[s]
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_ws)
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, particle_q[tri_indices[f, 0]])
    b = wp.transform_point(X_sw, particle_q[tri_indices[f, 1]])
    c = wp.transform_point(X_sw, particle_q[tri_indices[f, 2]])
    o = (a + b + c) / 3.0
    radius = wp.max(wp.length(a - o), wp.max(wp.length(b - o), wp.length(c - o)))
    d = shape_sdf(shape_type[s], shape_scale[s], shape_source_ptr[s], o, edge_radius)[0]
    if d - radius <= margin:
        pair[wp.atomic_add(count, 0, 1)] = wp.int64(s) * wp.int64(tri_indices.shape[0]) + wp.int64(f)


@wp.func
def mesh_triangle(mesh_id: wp.uint64, t: int, scale: wp.vec3) -> wp.mat33:
    """Vertices of triangle ``t`` of a mesh in the shape frame, as rows."""
    mesh = wp.mesh_get(mesh_id)
    return wp.matrix_from_rows(
        wp.cw_mul(mesh.points[mesh.indices[3 * t]], scale),
        wp.cw_mul(mesh.points[mesh.indices[3 * t + 1]], scale),
        wp.cw_mul(mesh.points[mesh.indices[3 * t + 2]], scale),
    )


@wp.kernel
def mesh_minima_kernel(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[wp.float32],
    contact_particle: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[wp.float32],
    neighbor_start: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    alarm: float,
    found: wp.array2d[wp.int32],
    body_pos: wp.array2d[wp.vec3],
    normal: wp.array2d[wp.vec3],
):
    """The other local minima of each mesh contact's particle distance to the mesh, within the particle radius
    plus the shape margin plus ``alarm``: projections into faces whose outer side the particle is on, as
    RealSim's vertex-face contacts, and closest points on an edge that the triangle across it shares. A
    contact's closest point holds its particle against one side of a concave corner; these hold it against the
    others. Minima at vertices are left out. ``neighbors[neighbor_start[s] + t]`` lists the triangles across
    the edges ``ab``, ``ac`` and ``bc`` of triangle ``t`` of shape ``s``. ``body_pos`` and ``normal`` are as in
    ``Contacts``; ``found`` marks the filled slots of each contact.
    """
    c = wp.tid()
    s = contact_shape[c]
    if s < 0 or shape_type[s] != GeoType.MESH:
        return
    X_bs = shape_transform[s]
    X_ws = X_bs
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_bs)
    X_sw = wp.transform_inverse(X_ws)
    p = contact_particle[c]
    x = wp.transform_point(X_sw, particle_q[p])
    closest = wp.transform_point(wp.transform_inverse(X_bs), contact_body_pos[c])
    # Inside the mesh, the faces around the particle would push it out of opposite sides.
    if wp.dot(wp.transform_vector(X_sw, contact_normal[c]), x - closest) <= 0.0:
        return
    reach = particle_radius[p] + shape_margin[s] + alarm
    tol = 1.0e-3 * reach
    mesh_id = shape_source_ptr[s]
    scale = shape_scale[s]
    parity = wp.sign(scale[0] * scale[1] * scale[2])
    lo = wp.cw_div(x - wp.vec3(reach), scale)
    hi = wp.cw_div(x + wp.vec3(reach), scale)
    query = wp.mesh_query_aabb(mesh_id, wp.min(lo, hi), wp.max(lo, hi))
    t = int(0)
    emitted = int(0)
    while wp.mesh_query_aabb_next(query, t):
        T = mesh_triangle(mesh_id, t, scale)
        point, _bary, feature = sdf.triangle_closest_point(T[0], T[1], T[2], x)
        delta = x - point
        dist = wp.length(delta)
        outside = parity * wp.dot(wp.cross(T[1] - T[0], T[2] - T[0]), delta) > 0.0
        if not outside or dist > reach or wp.length(point - closest) <= tol:
            continue
        # On an edge, a minimum only if the triangle across it comes no closer; at a vertex, none.
        if feature >= sdf.TRI_CONTACT_FEATURE_EDGE_AB and feature <= sdf.TRI_CONTACT_FEATURE_EDGE_BC:
            other = neighbors[neighbor_start[s] + t, feature - sdf.TRI_CONTACT_FEATURE_EDGE_AB]
            if other < 0:
                continue
            U = mesh_triangle(mesh_id, other, scale)
            across, _bary_across, _feature_across = sdf.triangle_closest_point(U[0], U[1], U[2], x)
            if wp.length(across - point) > tol:
                continue
        elif feature != sdf.TRI_CONTACT_FEATURE_FACE_INTERIOR:
            continue
        duplicate = int(0)
        for e in range(emitted):
            if wp.length(body_pos[c, e] - point) <= tol:
                duplicate = 1
        if duplicate == 0 and emitted < found.shape[1]:
            found[c, emitted] = 1
            body_pos[c, emitted] = point
            normal[c, emitted] = delta / dist
            emitted += 1
    for e in range(emitted):
        body_pos[c, e] = wp.transform_point(X_bs, body_pos[c, e])
        normal[c, e] = wp.transform_vector(X_ws, normal[c, e])


def _mesh_neighbors(model, device) -> tuple[wp.array, wp.array]:
    """Per shape, the first row of its triangles in the neighbor table, and the table: for each triangle of each
    mesh shape, the triangles across its edges ``ab``, ``ac`` and ``bc``, ``-1`` on a boundary."""
    shape_type = model.shape_type.numpy()
    start, tables, rows = np.zeros(max(len(shape_type), 1), dtype=np.int32), [np.zeros((0, 3), np.int32)], 0
    for s in np.flatnonzero(shape_type == int(GeoType.MESH)):
        tri = np.asarray(model.shape_source[s].indices, dtype=np.int64).reshape(-1, 3)
        edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]]), axis=1)
        owner, slot = np.tile(np.arange(len(tri)), 3), np.repeat(np.arange(3), len(tri))
        order = np.lexsort((owner, edges[:, 1], edges[:, 0]))
        pair = np.flatnonzero((edges[order][1:] == edges[order][:-1]).all(axis=1))
        table = np.full((len(tri), 3), -1, dtype=np.int32)
        a, b = order[pair], order[pair + 1]
        table[owner[a], slot[a]], table[owner[b], slot[b]] = owner[b], owner[a]
        start[s], rows = rows, rows + len(tri)
        tables.append(table)
    table = np.concatenate(tables) if rows else np.full((1, 3), -1, dtype=np.int32)
    return wp.array(start, dtype=wp.int32, device=device), wp.array(table, dtype=wp.int32, device=device)


class FaceContactQuery:
    """Face contacts of the triangles around a step's particle contacts.

    A triangle is tested against a shape when one of its particles touches the shape or, given a margin,
    when the triangle may come within that margin of the shape; a shape tip or edge can poke into a
    triangle whose particles all lie beyond the particle contact margin.
    """

    def __init__(self, tri_indices: np.ndarray, particle_count: int, device):
        self._mesh_neighbors = None
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
        margin: float | None = None,
        iterations: int = 128,
    ):
        """Face contacts as ``(face, shape, bary, body_pos, normal)`` arrays, ordered by shape and triangle.

        ``particle`` and ``shape`` list the particle contacts; ``edge_radius``
        rounds box and cylinder edges as in :func:`shape_sdf`. A ``margin``
        [m] also tests every triangle that may come that close to a shape.
        When two triangles report the same edge optimum, the deeper contact
        remains.
        """
        start, stop = self.offsets[particle], self.offsets[particle + 1]
        counts, n_tri, device = stop - start, len(self.tri), self.device
        faces = self.incident[np.repeat(start - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())]
        candidates = np.unique(np.repeat(np.asarray(shape, dtype=np.int64), counts) * n_tri + faces)
        bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
        if margin is not None and model.shape_count:
            near_count = wp.zeros(1, dtype=wp.int32, device=device)
            near = wp.empty(n_tri * model.shape_count, dtype=wp.int64, device=device)
            wp.launch(
                near_face_kernel,
                dim=(n_tri, model.shape_count),
                inputs=[
                    *(particle_q, self.tri_d, model.particle_world, bodies),
                    *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                    *(model.shape_source_ptr, model.shape_flags, model.shape_world, margin, edge_radius),
                ],
                outputs=[near_count, near],
                device=device,
            )
            candidates = np.union1d(candidates, near.numpy()[: int(near_count.numpy()[0])])
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
                    bodies,
                    *(model.shape_transform, model.shape_body, model.shape_type),
                    *(model.shape_scale, model.shape_source_ptr),
                    *(iterations, edge_radius),
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

    def mesh_minima(self, model, particle_q, body_q, particle, shape, body_pos, normal, alarm: float, slots: int = 3):
        """The other local minima of each particle contact's distance to a mesh, within the particle radius plus
        the shape margin plus ``alarm`` [m], as ``(particle, shape, body_pos, normal, contact)`` arrays, where
        ``contact`` indexes the particle contact each comes from.

        ``particle``, ``shape``, ``body_pos`` and ``normal`` list the particle contacts as ``Contacts`` holds
        them; each contact gets at most ``slots`` more, in the order of the contacts.
        """
        m, device = len(particle), self.device
        if self._mesh_neighbors is None:
            self._mesh_neighbors = _mesh_neighbors(model, device)
        found = wp.zeros((max(m, 1), slots), dtype=wp.int32, device=device)
        out = [wp.zeros((max(m, 1), slots), dtype=wp.vec3, device=device) for _ in range(2)]
        if m:
            bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
            contact = [
                wp.array(np.asarray(a, dtype=np.int32), dtype=wp.int32, device=device) for a in (particle, shape)
            ]
            contact += [
                wp.array(np.asarray(a, dtype=np.float32), dtype=wp.vec3, device=device) for a in (body_pos, normal)
            ]
            wp.launch(
                mesh_minima_kernel,
                dim=m,
                inputs=[
                    *(particle_q, model.particle_radius, *contact, bodies),
                    *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                    *(model.shape_source_ptr, model.shape_margin, *self._mesh_neighbors, alarm),
                ],
                outputs=[found, *out],
                device=device,
            )
        hit = found.numpy()[:m].astype(bool)
        rows = np.nonzero(hit)[0]
        minima = np.asarray(particle)[rows], np.asarray(shape)[rows], out[0].numpy()[:m][hit], out[1].numpy()[:m][hit]
        return (*minima, rows)
