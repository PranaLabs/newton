# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Face contacts: the deepest point of a deformable triangle inside a rigid shape.

Particle contacts test only vertices, so a box edge can pass through a triangle between them. The
point minimizes the shape's signed distance over the triangle with a Frank-Wolfe iteration from the
deepest vertex. Vertex optima are left to
the particle contacts; unselected vertices keep exactly zero weight, which marks edge optima. A face
contact appears as soon as the optimum leaves the vertices, where it coincides with that vertex's
particle contact, so switching it on or off moves no constraint beyond the Frank-Wolfe accuracy, except at the
box and cylinder edges that ``edge_radius`` rounds, where the face contact's distance differs from the particle
contact's.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp

from ...geometry import kernels as sdf
from ...geometry.flags import MeshSignMethod, ShapeFlags
from ...geometry.types import Axis, GeoType
from .kernels import host_head

wp.set_module_options({"enable_backward": False})


@wp.func
def shape_sdf(geo_type: int, scale: wp.vec3, mesh: wp.uint64, x: wp.vec3, edge_radius: float) -> wp.vec4:
    """Signed distance [m] and its gradient at ``x`` in the shape frame (:func:`shape_sdf_within`)."""
    return shape_sdf_within(geo_type, scale, mesh, x, edge_radius, 1.0e6)


@wp.func
def shape_sdf_within(
    geo_type: int, scale: wp.vec3, mesh: wp.uint64, x: wp.vec3, edge_radius: float, reach: float
) -> wp.vec4:
    """Signed distance [m] and its gradient at ``x`` in the shape frame; ``1e6`` for unsupported shapes.

    A mesh's query looks for its closest point within ``reach`` [mesh units] of ``x`` first, which prunes its
    bounding volumes sooner, and past it only if it finds none there.

    Box and cylinder edges are rounded by ``edge_radius``: inside a sharp edge the gradient jumps
    between the two faces, so a point that sinks into the edge would flip its normal. Barrel cylinders
    (``scale[2]``, the barrel radius, positive) keep their edges. Meshes use the
    closest point and sign of Warp's query, scaled as in Newton's particle contacts.
    """
    d = float(1.0e6)
    g = wp.vec3(0.0, 0.0, 1.0)
    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        local = wp.cw_div(x, scale)
        q = wp.mesh_query_point_sign_normal(mesh, local, reach)
        if not q.result and reach < 1.0e6:
            q = wp.mesh_query_point_sign_normal(mesh, local, 1.0e6)
        if q.result:
            delta = x - wp.cw_mul(wp.mesh_eval_position(mesh, q.face, q.u, q.v), scale)
            d = q.sign * wp.length(delta)
            g = sdf.mesh_contact_normal(
                mesh, q.face, wp.cw_div(x, scale), delta, q.sign, scale, int(MeshSignMethod.NORMAL)
            )
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
        if scale[2] > 0.0:
            # A barrel, of radius scale[2] at its middle, as Newton's particle contacts take it; its edges are not
            # rounded.
            d = sdf.sdf_cylinder(x, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
            g = sdf.sdf_cylinder_grad(x, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
        else:
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
def box_distance(lower: wp.vec3, upper: wp.vec3, x: wp.vec3) -> float:
    """Distance [m] from ``x`` to the box from ``lower`` to ``upper``, 0 inside: a lower bound of the signed
    distance to any shape inside the box, such as a shape within its collision bounds."""
    return wp.length(wp.max(wp.max(lower - x, x - upper), wp.vec3(0.0)))


@wp.func
def min_index(a: wp.vec3) -> int:
    """Index of the smallest component, the first on ties."""
    j = int(0)
    if a[1] < a[j]:
        j = 1
    if a[2] < a[j]:
        j = 2
    return j


@wp.func
def shape_frame_triangle(
    particle_q: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32], f: int, X_sw: wp.transform
) -> wp.mat33:
    """The vertices of triangle ``f`` in the shape frame, as rows."""
    return wp.matrix_from_rows(
        wp.transform_point(X_sw, particle_q[tri_indices[f, 0]]),
        wp.transform_point(X_sw, particle_q[tri_indices[f, 1]]),
        wp.transform_point(X_sw, particle_q[tri_indices[f, 2]]),
    )


@wp.func
def deepest_point(P: wp.mat33, geo: int, scale: wp.vec3, mesh: wp.uint64, r: float, iterations: int):
    """The Frank-Wolfe iteration of :func:`face_contact_kernel` over the triangle ``P`` (rows, shape frame): the
    count of vertices it selects, the least signed distance of the vertices, the barycentric weights, the point
    and its signed distance and gradient (:func:`shape_sdf`)."""
    d = wp.vec3(
        shape_sdf(geo, scale, mesh, P[0], r)[0],
        shape_sdf(geo, scale, mesh, P[1], r)[0],
        shape_sdf(geo, scale, mesh, P[2], r)[0],
    )
    w = wp.vec3(0.0)
    w[min_index(d)] = 1.0
    # The distance changes by at most the length moved, so no point of the triangle is farther from a mesh than its
    # nearest vertex and the triangle's longest edge; in mesh units, divided by the least scale.
    longest = wp.max(wp.length(P[1] - P[0]), wp.max(wp.length(P[2] - P[1]), wp.length(P[0] - P[2])))
    least_scale = wp.min(scale[0], wp.min(scale[1], scale[2]))
    reach = float(1.0e6)
    if least_scale > 0.0:
        reach = wp.min(reach, 1.001 * (wp.min(wp.abs(d)) + longest) / least_scale + 1.0e-6)
    for k in range(iterations):
        sd = shape_sdf_within(geo, scale, mesh, w @ P, r, reach)
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
    return selected, wp.min(d), w, x, sd


@wp.func
def near_face(
    f: int,
    s: int,
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
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    gap: float,
    edge_radius: float,
) -> bool:
    """Whether triangle ``f`` may come within ``gap`` of shape ``s`` (:func:`near_face_kernel`)."""
    world = shape_world[s]
    # A local shape and a particle of another world do not touch; global particles touch every shape.
    particle_world_f = particle_world[tri_indices[f, 0]]
    if (shape_flags[s] & ShapeFlags.COLLIDE_PARTICLES) == 0 or (
        world >= 0 and particle_world_f >= 0 and world != particle_world_f
    ):
        return False
    X_ws = shape_transform[s]
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_ws)
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, particle_q[tri_indices[f, 0]])
    b = wp.transform_point(X_sw, particle_q[tri_indices[f, 1]])
    c = wp.transform_point(X_sw, particle_q[tri_indices[f, 2]])
    o = (a + b + c) / 3.0
    radius = wp.max(wp.length(a - o), wp.max(wp.length(b - o), wp.length(c - o)))
    if box_distance(shape_lower[s], shape_upper[s], o) - radius > gap:
        return False
    d = shape_sdf(shape_type[s], shape_scale[s], shape_source_ptr[s], o, edge_radius)[0]
    return d - radius <= gap


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
    P = shape_frame_triangle(particle_q, tri_indices, f, wp.transform_inverse(X_ws))
    selected, least, w, x, sd = deepest_point(
        P, shape_type[s], shape_scale[s], shape_source_ptr[s], edge_radius, iterations
    )
    if selected < 2 or sd[0] >= least:
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
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    gap: float,
    edge_radius: float,
    count: wp.array[wp.int32],
    pair: wp.array[wp.int64],
):
    """``shape * tri_count + face`` of each pair whose triangle may come within ``gap`` of the shape: the
    signed distance changes no faster than the point moves, so no triangle point is closer than the
    centroid's distance minus the triangle's radius about the centroid. Triangles that far from the shape's
    collision bounds, ``shape_lower`` to ``shape_upper`` in the shape frame, skip the signed distance."""
    f, s = wp.tid()
    if near_face(
        f,
        s,
        particle_q,
        tri_indices,
        particle_world,
        body_q,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_flags,
        shape_world,
        shape_lower,
        shape_upper,
        gap,
        edge_radius,
    ):
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
    count: wp.array[wp.int32],  # the stored contacts of a contact buffer's whole capacity, or empty for all
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
    plus the shape margin plus ``alarm``: projections into faces whose outer side the particle is on, and
    closest points on an edge that the triangle across it shares. A
    contact's closest point holds its particle against one side of a concave corner; these hold it against the
    others. Minima at vertices are left out. ``neighbors[neighbor_start[s] + t]`` lists the triangles across
    the edges ``ab``, ``ac`` and ``bc`` of triangle ``t`` of shape ``s``. ``body_pos`` and ``normal`` are as in
    ``Contacts``; ``found`` marks the filled slots of each contact. Given ``count``, the contacts past ``count[0]``
    and the sentinel ones (particle -1) have none.
    """
    c = wp.tid()
    if count.shape[0] > 0:
        if c >= count[0] or contact_particle[c] < 0:
            return
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


def _edge_neighbors(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The other triangle edges with the endpoints of each triangle edge: those of the edge of triangle ``t`` opposite
    its vertex ``z``, at ``t * 3 + z``, are the entries ``start[t * 3 + z]`` to ``start[t * 3 + z + 1]`` of ``other``,
    as ``triangle * 3 + vertex``."""
    n = len(tri)
    t, z = np.divmod(np.arange(3 * n), 3)
    a, b = tri[t, (z + 1) % 3].astype(np.int64), tri[t, (z + 2) % 3].astype(np.int64)
    key = np.minimum(a, b) * (int(tri.max(initial=0)) + 1) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    first = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
    size = np.diff(np.r_[first, 3 * n])
    others = np.zeros(3 * n, dtype=np.int64)
    others[order] = np.repeat(size - 1, size)
    start = np.zeros(3 * n + 1, dtype=np.int32)
    start[1:] = np.cumsum(others)
    other = np.zeros(max(int(start[-1]), 1), dtype=np.int32)
    pair = first[size == 2]
    m0, m1 = order[pair], order[pair + 1]
    other[start[m0]], other[start[m1]] = m1, m0
    for g in np.flatnonzero(size > 2):
        members = order[first[g] : first[g] + size[g]]
        for m in members:
            other[start[m] : start[m + 1]] = members[members != m]
    return start, other


@wp.kernel
def mark_contact_faces_kernel(
    count: wp.array[wp.int32],
    contact_particle: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    offsets: wp.array[wp.int32],  # (N + 1,) the incident triangles of each particle in ``incident``
    incident: wp.array[wp.int32],
    tri_count: int,
    # outputs
    flags: wp.array[wp.int32],  # (S * tri_count,)
):
    """Flag ``shape * tri_count + face`` for each triangle around the particle of each stored contact of a contact
    buffer: the candidates of :meth:`FaceContactQuery.query` that the particle contacts give."""
    c = wp.tid()
    p = contact_particle[c]
    if c >= count[0] or p < 0:
        return
    base = contact_shape[c] * tri_count
    for k in range(offsets[p], offsets[p + 1]):
        flags[base + incident[k]] = 1


@wp.kernel
def near_face_flags_kernel(
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
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    gap: float,
    edge_radius: float,
    # outputs
    flags: wp.array[wp.int32],
):
    """Flag the pairs of :func:`near_face_kernel` at ``shape * tri_count + face``."""
    f, s = wp.tid()
    if near_face(
        f,
        s,
        particle_q,
        tri_indices,
        particle_world,
        body_q,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_flags,
        shape_world,
        shape_lower,
        shape_upper,
        gap,
        edge_radius,
    ):
        flags[s * tri_indices.shape[0] + f] = 1


@wp.kernel
def face_candidates_kernel(
    flags: wp.array[wp.int32],
    rank: wp.array[wp.int32],  # the exclusive scan of ``flags``
    tri_count: int,
    # outputs
    count: wp.array[wp.int32],
    cand_face: wp.array[wp.int32],
    cand_shape: wp.array[wp.int32],
):
    """The flagged pairs at their ranks, ascending in ``shape * tri_count + face`` as :func:`numpy.unique` orders the
    candidates of :meth:`FaceContactQuery.query`, and their count in ``count[0]``."""
    i = wp.tid()
    if flags[i] != 0:
        cand_face[rank[i]] = i % tri_count
        cand_shape[rank[i]] = i // tri_count
    if i == flags.shape[0] - 1:
        count[0] = rank[i] + flags[i]


@wp.kernel
def face_contact_capacity_kernel(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    cand_face: wp.array[wp.int32],
    cand_shape: wp.array[wp.int32],
    count: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    iterations: int,
    edge_radius: float,
    # outputs, at the candidate's index
    emitted: wp.array[wp.int32],
    bary: wp.array[wp.vec3],
    depth: wp.array[wp.float32],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    """:func:`face_contact_kernel` for the first ``count[0]`` candidates of a launch over a capacity, each written at
    its own index and marked in ``emitted``."""
    i = wp.tid()
    emitted[i] = 0
    if i >= count[0]:
        return
    f = cand_face[i]
    s = cand_shape[i]
    X_bs = shape_transform[s]
    X_ws = X_bs
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_bs)
    P = shape_frame_triangle(particle_q, tri_indices, f, wp.transform_inverse(X_ws))
    selected, least, w, x, sd = deepest_point(
        P, shape_type[s], shape_scale[s], shape_source_ptr[s], edge_radius, iterations
    )
    if selected < 2 or sd[0] >= least:
        return
    n = wp.normalize(wp.vec3(sd[1], sd[2], sd[3]))
    emitted[i] = 1
    bary[i] = w
    depth[i] = sd[0]
    body_pos[i] = wp.transform_point(X_bs, x - sd[0] * n)
    normal[i] = wp.transform_vector(X_ws, n)


@wp.func
def first_zero(w: wp.vec3) -> int:
    """The index of the first zero weight, -1 for none."""
    z = int(-1)
    for k in range(3):
        if z < 0 and w[k] == 0.0:
            z = k
    return z


@wp.kernel
def face_edge_keep_kernel(
    count: wp.array[wp.int32],
    emitted: wp.array[wp.int32],
    bary: wp.array[wp.vec3],
    depth: wp.array[wp.float32],
    cand_face: wp.array[wp.int32],
    cand_shape: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    edge_start: wp.array[wp.int32],  # (3 tri_count + 1,) the other triangle edges with each edge's endpoints
    edge_other: wp.array[wp.int32],  # triangle * 3 + the vertex opposite its edge
    pinned: wp.array[wp.int32],  # 1 for each particle the solver holds, or empty
    pushed: wp.array[wp.int32],  # 1 for each shape on a body the solver integrates, or empty
    # outputs
    found: wp.array[wp.int32],
    keep: wp.array[wp.int32],
):
    """Keep each emitted face contact but an edge optimum that another triangle on the edge reaches deeper, or as deep
    at a lower candidate, as :meth:`FaceContactQuery.query` keeps the deepest of each edge, then drop those whose
    particles of nonzero weight the solver all holds, but on the shapes of the bodies it integrates (``pushed``), as
    :meth:`SolverFBA._ingest_contacts` drops them; ``found[0]``
    counts the contacts kept before that."""
    i = wp.tid()
    keep[i] = 0
    if i >= count[0] or emitted[i] == 0:
        return
    w = bary[i]
    f = cand_face[i]
    z = first_zero(w)
    if z >= 0:
        e = f * 3 + z
        for q in range(edge_start[e], edge_start[e + 1]):
            other = edge_other[q]
            j = cand_shape[i] * tri_indices.shape[0] + other // 3
            if flags[j] != 0:
                r = rank[j]
                if emitted[r] != 0 and first_zero(bary[r]) == other % 3:
                    if depth[r] < depth[i] or (depth[r] == depth[i] and r < i):
                        return
    wp.atomic_add(found, 0, 1)
    if pinned.shape[0] > 0:
        free = int(0)
        for k in range(3):
            if w[k] != 0.0 and pinned[tri_indices[f, k]] == 0:
                free = 1
        # A row on held particles alone still pushes a body the solver integrates.
        if free == 0 and pushed.shape[0] > 0:
            free = pushed[cand_shape[i]]
        if free == 0:
            return
    keep[i] = 1


@wp.kernel
def face_rows_kernel(
    count: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    position: wp.array[wp.int32],  # the exclusive scan of ``keep``
    cand_face: wp.array[wp.int32],
    cand_shape: wp.array[wp.int32],
    bary: wp.array[wp.vec3],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    first: wp.array[wp.int32],
    limit: int,
    # outputs: the contact rows
    particle: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    alpha: wp.array[wp.float32],
    shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    record: wp.array[wp.int32],
    extra_particle: wp.array2d[wp.int32],
    extra_alpha: wp.array2d[wp.float32],
):
    """Write each kept face contact into the contact rows from row ``first[0]`` on, in candidate order, up to row
    ``limit``: its triangle's first particle at the first weight, the others in the extra slots, record -1."""
    i = wp.tid()
    if i >= count[0] or keep[i] == 0:
        return
    c = first[0] + position[i]
    if c >= limit:
        return
    f = cand_face[i]
    w = bary[i]
    particle[c] = tri_indices[f, 0]
    alpha[c] = w[0]
    extra_particle[c, 0] = tri_indices[f, 1]
    extra_particle[c, 1] = tri_indices[f, 2]
    extra_alpha[c, 0] = w[1]
    extra_alpha[c, 1] = w[2]
    shape[c] = cand_shape[i]
    contact_body_pos[c] = body_pos[i]
    contact_normal[c] = normal[i]
    record[c] = -1


@wp.kernel
def face_count_kernel(
    first: wp.array[wp.int32],
    position: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    found: wp.array[wp.int32],
    gain: wp.float64,
    # outputs
    active: wp.array[wp.int32],
    contact_gain: wp.array[wp.float64],
):
    """The rows' total, ``first[0]`` plus the kept face contacts, in ``active[0]``, and each contact's gain of the
    diagonal of its rows: ``gain`` when the step found face contacts, as the host path regularizes them, 1 otherwise."""
    c = wp.tid()
    last = keep.shape[0] - 1
    if c == 0:
        active[0] = first[0] + position[last] + keep[last]
    contact_gain[c] = wp.where(found[0] > 0, gain, wp.float64(1.0))


class FaceContactQuery:
    """Face contacts of the triangles around a step's particle contacts.

    A triangle is tested against a shape when one of its particles touches the shape or, given a gap,
    when the triangle may come within that gap of the shape; a shape tip or edge can poke into a
    triangle whose particles all lie beyond the reach of the particle contacts.
    """

    def __init__(self, tri_indices: np.ndarray, particle_count: int, device):
        self._mesh_neighbors = None
        self.tri = np.asarray(tri_indices, dtype=np.int32).reshape(-1, 3)
        order = np.argsort(self.tri.ravel(), kind="stable")
        self.offsets = np.searchsorted(self.tri.ravel()[order], np.arange(particle_count + 1))
        self.incident = (order // 3).astype(np.int64)
        self.tri_d = wp.array(self.tri, dtype=wp.int32, device=device)
        self.no_bodies = wp.empty(1, dtype=wp.transform, device=device)
        self.no_count = wp.empty(0, dtype=wp.int32, device=device)
        self.device = device
        self._cap = None

    def query(
        self,
        model,
        particle_q,
        body_q,
        particle,
        shape,
        edge_radius: float,
        gap: float | None = None,
        iterations: int = 128,
    ):
        """Face contacts as ``(face, shape, bary, body_pos, normal)`` arrays, ordered by shape and triangle.

        ``particle`` and ``shape`` list the particle contacts; ``edge_radius``
        rounds box and cylinder edges as in :func:`shape_sdf`. A ``gap``
        [m] also tests every triangle that may come that close to a shape.
        When two triangles report the same edge optimum, the deeper contact
        remains.
        """
        start, stop = self.offsets[particle], self.offsets[particle + 1]
        counts, n_tri, device = stop - start, len(self.tri), self.device
        faces = self.incident[np.repeat(start - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())]
        candidates = np.unique(np.repeat(np.asarray(shape, dtype=np.int64), counts) * n_tri + faces)
        bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
        if gap is not None and model.shape_count:
            near_count = wp.zeros(1, dtype=wp.int32, device=device)
            near = wp.empty(n_tri * model.shape_count, dtype=wp.int64, device=device)
            wp.launch(
                near_face_kernel,
                dim=(n_tri, model.shape_count),
                inputs=[
                    *(particle_q, self.tri_d, model.particle_world, bodies),
                    *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                    *(model.shape_source_ptr, model.shape_flags, model.shape_world),
                    *(model.shape_collision_aabb_lower, model.shape_collision_aabb_upper, gap, edge_radius),
                ],
                outputs=[near_count, near],
                device=device,
            )
            candidates = np.union1d(candidates, host_head(near, int(near_count.numpy()[0])))
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
        order = np.argsort(host_head(out[0], m))
        pair, bary, depth, body_pos, normal = (host_head(a, m)[order] for a in out)
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
                    *(self.no_count, particle_q, model.particle_radius, *contact, bodies),
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

    def reserve_capacity(self, model) -> None:
        """Allocate the buffers of :meth:`capacity_contacts` and :meth:`capacity_minima`, so that a graph capture of
        them allocates nothing: one candidate slot per pair of a shape and a triangle."""
        F, device = max(model.shape_count * len(self.tri), 1), self.device
        if self._cap is not None and self._cap.flags.shape[0] == F:
            return
        if self._mesh_neighbors is None:
            self._mesh_neighbors = _mesh_neighbors(model, device)
        start, other = _edge_neighbors(self.tri)

        def ints():
            return wp.zeros(F, dtype=wp.int32, device=device)

        self._cap = SimpleNamespace(
            offsets=wp.array(self.offsets.astype(np.int32), dtype=wp.int32, device=device),
            incident=wp.array(np.r_[self.incident, 0].astype(np.int32), dtype=wp.int32, device=device),
            edge_start=wp.array(start, dtype=wp.int32, device=device),
            edge_other=wp.array(other, dtype=wp.int32, device=device),
            flags=ints(),
            rank=ints(),
            face=ints(),
            shape=ints(),
            emitted=ints(),
            keep=ints(),
            position=ints(),
            count=wp.zeros(1, dtype=wp.int32, device=device),
            found=wp.zeros(1, dtype=wp.int32, device=device),
            bary=wp.zeros(F, dtype=wp.vec3, device=device),
            depth=wp.zeros(F, dtype=wp.float32, device=device),
            body_pos=wp.zeros(F, dtype=wp.vec3, device=device),
            normal=wp.zeros(F, dtype=wp.vec3, device=device),
        )

    def capacity_minima(self, model, particle_q, body_q, contacts, alarm: float, found, body_pos, normal) -> None:
        """:meth:`mesh_minima` on the device alone, for every slot of the contact buffer of ``contacts``: the minima of
        slot ``c`` in row ``c`` of ``found``, ``body_pos`` and ``normal`` (shape ``(C, 3)``), those past the stored
        count none."""
        found.zero_()
        bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
        wp.launch(
            mesh_minima_kernel,
            dim=contacts.soft_contact_particle.shape[0],
            inputs=[
                *(contacts.soft_contact_count, particle_q, model.particle_radius, contacts.soft_contact_particle),
                *(contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_normal, bodies),
                *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                *(model.shape_source_ptr, model.shape_margin, *self._mesh_neighbors, alarm),
            ],
            outputs=[found, body_pos, normal],
            device=self.device,
        )

    def capacity_contacts(
        self,
        model,
        particle_q,
        body_q,
        contacts,
        pinned,
        pushed,
        edge_radius: float,
        gap: float | None,
        first,
        limit: int,
        rows: list,
        extra: list,
        active,
        gain: float,
        contact_gain,
        iterations: int = 128,
    ) -> None:
        """The face contacts of :meth:`query` on the device alone, nothing read back, for the particle contacts of the
        whole capacity of ``contacts``: the candidates flagged and ranked by a scan, each one's optimum at its own
        index, the deepest of each edge kept, and those on held particles alone (``pinned``) dropped but on the shapes of
        integrated bodies (``pushed``). They are written,
        in the order of :meth:`query`, into the contact rows ``rows`` (particle, normal, alpha, shape, body_pos,
        record) and ``extra`` (particles and weights) from row ``first[0]`` on, up to row ``limit``; ``active[0]``
        takes the rows' total, which may pass ``limit``, and ``contact_gain`` each contact's gain of its rows'
        diagonal, ``gain`` when the step finds face contacts, 1 otherwise. :meth:`reserve_capacity` sizes the
        buffers."""
        cap, device = self._cap, self.device
        n_tri, S, F = len(self.tri), model.shape_count, self._cap.flags.shape[0]
        bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
        cap.flags.zero_()
        inputs = [contacts.soft_contact_count, contacts.soft_contact_particle, contacts.soft_contact_shape]
        inputs += [cap.offsets, cap.incident, n_tri]
        wp.launch(
            mark_contact_faces_kernel, contacts.soft_contact_particle.shape[0], inputs, [cap.flags], device=device
        )
        if gap is not None and S:
            wp.launch(
                near_face_flags_kernel,
                dim=(n_tri, S),
                inputs=[
                    *(particle_q, self.tri_d, model.particle_world, bodies),
                    *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                    *(model.shape_source_ptr, model.shape_flags, model.shape_world),
                    *(model.shape_collision_aabb_lower, model.shape_collision_aabb_upper, gap, edge_radius),
                ],
                outputs=[cap.flags],
                device=device,
            )
        wp.utils.array_scan(cap.flags, cap.rank, inclusive=False)
        wp.launch(
            face_candidates_kernel, F, [cap.flags, cap.rank, n_tri], [cap.count, cap.face, cap.shape], device=device
        )
        wp.launch(
            face_contact_capacity_kernel,
            dim=F,
            inputs=[
                *(particle_q, self.tri_d, cap.face, cap.shape, cap.count, bodies),
                *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                *(model.shape_source_ptr, iterations, edge_radius),
            ],
            outputs=[cap.emitted, cap.bary, cap.depth, cap.body_pos, cap.normal],
            device=device,
        )
        cap.found.zero_()
        inputs = [cap.count, cap.emitted, cap.bary, cap.depth, cap.face, cap.shape, cap.flags, cap.rank, self.tri_d]
        inputs += [cap.edge_start, cap.edge_other, pinned, pushed]
        wp.launch(face_edge_keep_kernel, F, inputs, [cap.found, cap.keep], device=device)
        wp.utils.array_scan(cap.keep, cap.position, inclusive=False)
        inputs = [cap.count, cap.keep, cap.position, cap.face, cap.shape, cap.bary, cap.body_pos, cap.normal]
        inputs += [self.tri_d, first, int(limit)]
        wp.launch(face_rows_kernel, F, inputs, [*rows, *extra], device=device)
        inputs = [first, cap.position, cap.keep, cap.found, wp.float64(gain)]
        wp.launch(face_count_kernel, contact_gain.shape[0], inputs, [active, contact_gain], device=device)


@wp.func
def near_segment(
    e: int,
    s: int,
    particle_q: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    particle_radius: wp.array[wp.float32],
    particle_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    gap: float,
    edge_radius: float,
) -> bool:
    """Whether segment ``e``, a capsule of its particles' radius, may come within ``gap`` of shape ``s``: no point of
    the segment is closer than its midpoint's distance minus half its length. Shapes without ``COLLIDE_PARTICLES`` at
    the step are skipped, and segments that far from a shape's collision bounds skip the signed distance."""
    world = shape_world[s]
    i, j = segments[e, 0], segments[e, 1]
    if (shape_flags[s] & ShapeFlags.COLLIDE_PARTICLES) == 0 or (
        world >= 0 and particle_world[i] >= 0 and world != particle_world[i]
    ):
        return False
    X_ws = shape_transform[s]
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_ws)
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, particle_q[i])
    b = wp.transform_point(X_sw, particle_q[j])
    o = 0.5 * (a + b)
    reach = wp.max(particle_radius[i], particle_radius[j]) + shape_margin[s] + gap
    if box_distance(shape_lower[s], shape_upper[s], o) - 0.5 * wp.length(b - a) > reach:
        return False
    d = shape_sdf(shape_type[s], shape_scale[s], shape_source_ptr[s], o, edge_radius)[0]
    return d - 0.5 * wp.length(b - a) <= reach


@wp.kernel
def near_segment_kernel(
    particle_q: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    particle_radius: wp.array[wp.float32],
    particle_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    gap: float,
    edge_radius: float,
    shapes: wp.array[wp.int32],
    count: wp.array[wp.int32],
    pair: wp.array[wp.int64],
):
    """``shape * segment_count + segment`` of each pair of :func:`near_segment`; the shapes are ``shapes``."""
    e, k = wp.tid()
    s = shapes[k]
    if near_segment(
        e,
        s,
        particle_q,
        segments,
        particle_radius,
        particle_world,
        body_q,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_flags,
        shape_world,
        shape_margin,
        shape_lower,
        shape_upper,
        gap,
        edge_radius,
    ):
        pair[wp.atomic_add(count, 0, 1)] = wp.int64(s) * wp.int64(segments.shape[0]) + wp.int64(e)


@wp.kernel
def near_segment_flags_kernel(
    particle_q: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    particle_radius: wp.array[wp.float32],
    particle_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    gap: float,
    edge_radius: float,
    # outputs
    flags: wp.array[wp.int32],
):
    """Flag the pairs of :func:`near_segment` at ``shape * segment_count + segment``, over all shapes."""
    e, s = wp.tid()
    if near_segment(
        e,
        s,
        particle_q,
        segments,
        particle_radius,
        particle_world,
        body_q,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_flags,
        shape_world,
        shape_margin,
        shape_lower,
        shape_upper,
        gap,
        edge_radius,
    ):
        flags[s * segments.shape[0] + e] = 1


@wp.func
def segment_contact(
    e: int,
    s: int,
    particle_q: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    particle_radius: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[wp.float32],
    gap: float,
    iterations: int,
    edge_radius: float,
):
    """The deepest point of segment ``e`` in shape ``s``: 1 if it lies strictly inside the segment, deeper than both
    of its particles and within the capsule radius plus the shape margin plus ``gap`` of the shape, else 0, then its
    weight ``t``, its depth, the closest surface point in the body frame, as ``Contacts.soft_contact_body_pos``, and
    the normal. The point is ``(1 - t) a + t b`` at the root of the signed distance's derivative along the segment,
    found by bisection."""
    X_bs = shape_transform[s]
    X_ws = X_bs
    if shape_body[s] >= 0:
        X_ws = wp.transform_multiply(body_q[shape_body[s]], X_bs)
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, particle_q[segments[e, 0]])
    b = wp.transform_point(X_sw, particle_q[segments[e, 1]])
    geo, scale, mesh, r = shape_type[s], shape_scale[s], shape_source_ptr[s], edge_radius
    ab = b - a
    sa = shape_sdf(geo, scale, mesh, a, r)
    sb = shape_sdf(geo, scale, mesh, b, r)
    # The distance falls into the segment from both ends only if its derivative is negative at a and positive at b.
    if wp.dot(wp.vec3(sa[1], sa[2], sa[3]), ab) >= 0.0 or wp.dot(wp.vec3(sb[1], sb[2], sb[3]), ab) <= 0.0:
        return 0, 0.0, 0.0, wp.vec3(0.0), wp.vec3(0.0)
    lo = float(0.0)
    hi = float(1.0)
    for _ in range(iterations):
        t = 0.5 * (lo + hi)
        g = shape_sdf(geo, scale, mesh, a + t * ab, r)
        if wp.dot(wp.vec3(g[1], g[2], g[3]), ab) < 0.0:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    x = a + t * ab
    sd = shape_sdf(geo, scale, mesh, x, r)
    radius = (1.0 - t) * particle_radius[segments[e, 0]] + t * particle_radius[segments[e, 1]]
    if sd[0] >= wp.min(sa[0], sb[0]) or sd[0] > radius + shape_margin[s] + gap:
        return 0, 0.0, 0.0, wp.vec3(0.0), wp.vec3(0.0)
    n = wp.normalize(wp.vec3(sd[1], sd[2], sd[3]))
    return 1, t, sd[0], wp.transform_point(X_bs, x - sd[0] * n), wp.transform_vector(X_ws, n)


@wp.kernel
def segment_contact_kernel(
    particle_q: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    particle_radius: wp.array[wp.float32],
    pair_segment: wp.array[wp.int32],
    pair_shape: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[wp.float32],
    gap: float,
    iterations: int,
    edge_radius: float,
    count: wp.array[wp.int32],
    pair: wp.array[wp.int32],
    weight: wp.array[wp.float32],
    depth: wp.array[wp.float32],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    """The contact of :func:`segment_contact` of each candidate ``(segment, shape)`` pair that has one, emitted with
    the candidate's index ``pair`` and its weight ``t`` in ``weight``."""
    k = wp.tid()
    emitted, t, d, x, n = segment_contact(
        pair_segment[k],
        pair_shape[k],
        particle_q,
        segments,
        particle_radius,
        body_q,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        gap,
        iterations,
        edge_radius,
    )
    if emitted == 0:
        return
    idx = wp.atomic_add(count, 0, 1)
    pair[idx] = k
    weight[idx] = t
    depth[idx] = d
    body_pos[idx] = x
    normal[idx] = n


@wp.kernel
def segment_contact_capacity_kernel(
    particle_q: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    particle_radius: wp.array[wp.float32],
    cand_segment: wp.array[wp.int32],
    cand_shape: wp.array[wp.int32],
    count: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[wp.float32],
    gap: float,
    iterations: int,
    edge_radius: float,
    pinned: wp.array[wp.int32],  # 1 for each particle the solver holds, or empty
    pushed: wp.array[wp.int32],  # 1 for each shape on a body the solver integrates, or empty
    # outputs, at the candidate's index
    found: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    weight: wp.array[wp.float32],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    """:func:`segment_contact_kernel` for the first ``count[0]`` candidates of a launch over a capacity, each contact
    written at its candidate's index and marked in ``keep``, but those whose particles of nonzero weight the solver
    all holds, unless on the shape of a body it integrates (``pushed``), as :meth:`SolverFBA._ingest_contacts` drops
    them; ``found[0]`` counts the contacts kept."""
    i = wp.tid()
    keep[i] = 0
    if i >= count[0]:
        return
    e = cand_segment[i]
    s = cand_shape[i]
    emitted, t, _depth, x, n = segment_contact(
        e,
        s,
        particle_q,
        segments,
        particle_radius,
        body_q,
        shape_transform,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        gap,
        iterations,
        edge_radius,
    )
    if emitted == 0:
        return
    if pinned.shape[0] > 0:
        free = int(0)
        if 1.0 - t != 0.0 and pinned[segments[e, 0]] == 0:
            free = 1
        if t != 0.0 and pinned[segments[e, 1]] == 0:
            free = 1
        # A row on held particles alone still pushes a body the solver integrates.
        if free == 0 and pushed.shape[0] > 0:
            free = pushed[s]
        if free == 0:
            return
    wp.atomic_add(found, 0, 1)
    keep[i] = 1
    weight[i] = t
    body_pos[i] = x
    normal[i] = n


@wp.kernel
def segment_rows_kernel(
    count: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    position: wp.array[wp.int32],  # the exclusive scan of ``keep``
    cand_segment: wp.array[wp.int32],
    cand_shape: wp.array[wp.int32],
    weight: wp.array[wp.float32],
    body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    first: wp.array[wp.int32],
    limit: int,
    # outputs: the contact rows
    particle: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    alpha: wp.array[wp.float32],
    shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    record: wp.array[wp.int32],
    extra_particle: wp.array2d[wp.int32],
    extra_alpha: wp.array2d[wp.float32],
):
    """Write each kept segment contact into the contact rows from row ``first[0]`` on, in candidate order, up to row
    ``limit``: the segment's first particle at weight ``1 - t``, its second in the first extra slot at ``t``, record
    -1."""
    i = wp.tid()
    if i >= count[0] or keep[i] == 0:
        return
    c = first[0] + position[i]
    if c >= limit:
        return
    e = cand_segment[i]
    t = weight[i]
    particle[c] = segments[e, 0]
    alpha[c] = 1.0 - t
    extra_particle[c, 0] = segments[e, 1]
    extra_alpha[c, 0] = t
    shape[c] = cand_shape[i]
    contact_body_pos[c] = body_pos[i]
    contact_normal[c] = normal[i]
    record[c] = -1


@wp.kernel
def segment_count_kernel(
    position: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    found: wp.array[wp.int32],
    gain: wp.float64,
    # outputs
    active: wp.array[wp.int32],
    contact_gain: wp.array[wp.float64],
):
    """The rows' total in ``active[0]``, which held the rows before the segment contacts, with the kept segment
    contacts after them, and each contact's gain of the diagonal of its rows set to ``gain`` when the step kept segment
    contacts, as the host path regularizes the rows of steps with contacts on further particles."""
    c = wp.tid()
    last = keep.shape[0] - 1
    if c == 0:
        active[0] = active[0] + position[last] + keep[last]
    if found[0] > 0:
        contact_gain[c] = gain


class SegmentContactQuery:
    """Segment contacts of rods: the deepest point of each spring segment inside a shape, between its particles.

    Particle contacts hold the particles of a rod off a shape, but a thin shape, or a pulley's curve, can still slip
    between two particles. A segment contact holds the capsule of a segment, as the segments of ``rod_contact`` hold
    each other: a row on its two particles with the weights of the deepest point. Its optimum lies strictly inside the
    segment, deeper than both particles; optima at the particles are the particle contacts'.
    """

    def __init__(self, segments: np.ndarray, device):
        self.seg = np.asarray(segments, dtype=np.int32).reshape(-1, 2)
        self.seg_d = wp.array(self.seg, dtype=wp.int32, device=device)
        self.no_bodies = wp.empty(1, dtype=wp.transform, device=device)
        self.no_count = wp.empty(0, dtype=wp.int32, device=device)
        self.device = device
        self._cap = None

    def query(self, model, particle_q, body_q, edge_radius: float, gap: float, iterations: int = 40):
        """Segment contacts as ``(segment, shape, weights (n, 2), body_pos, normal)`` arrays, ordered by shape and
        segment; the weights of the segment's two particles sum to one."""
        n_seg, device = len(self.seg), self.device
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2), np.float32))
        empty += (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
        if not n_seg or not model.shape_count:
            return empty
        bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
        if getattr(self, "shapes", None) is None or self.shapes.shape[0] != model.shape_count:
            # Every shape: the kernel reads the shapes' flags at each step, which a scene may switch, as Newton's
            # particle contacts do.
            self.shapes = wp.array(np.arange(model.shape_count, dtype=np.int32), dtype=wp.int32, device=device)
        if not self.shapes.shape[0]:
            return empty
        near_count = wp.zeros(1, dtype=wp.int32, device=device)
        near = wp.empty(n_seg * self.shapes.shape[0], dtype=wp.int64, device=device)
        wp.launch(
            near_segment_kernel,
            dim=(n_seg, self.shapes.shape[0]),
            inputs=[
                *(particle_q, self.seg_d, model.particle_radius, model.particle_world, bodies),
                *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                *(model.shape_source_ptr, model.shape_flags, model.shape_world, model.shape_margin),
                *(model.shape_collision_aabb_lower, model.shape_collision_aabb_upper, gap, edge_radius),
                self.shapes,
            ],
            outputs=[near_count, near],
            device=device,
        )
        m_near = int(near_count.numpy()[0])
        if not m_near:
            return empty
        candidates = np.sort(host_head(near, m_near))
        cand_shape, cand_seg = np.divmod(candidates, n_seg)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        out = [wp.empty(m_near, dtype=t, device=device) for t in (wp.int32, wp.float32, wp.float32, wp.vec3, wp.vec3)]
        wp.launch(
            segment_contact_kernel,
            dim=m_near,
            inputs=[
                *(particle_q, self.seg_d, model.particle_radius),
                *(wp.array(a.astype(np.int32), dtype=wp.int32, device=device) for a in (cand_seg, cand_shape)),
                bodies,
                *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                *(model.shape_source_ptr, model.shape_margin, gap, iterations, edge_radius),
            ],
            outputs=[count, *out],
            device=device,
        )
        m = int(count.numpy()[0])
        # The kernel emits in thread order: sort by candidate, which orders by shape and segment, for determinism.
        order = np.argsort(host_head(out[0], m))
        pair, t, _depth, body_pos, normal = (host_head(a, m)[order] for a in out)
        weights = np.stack([1.0 - t, t], axis=1).astype(np.float32)
        return cand_seg[pair], cand_shape[pair], weights, body_pos, normal

    def capacity(self, model) -> int:
        """The most segment contacts of a step: one per pair of a shape and a segment."""
        return model.shape_count * len(self.seg)

    def reserve_capacity(self, model) -> None:
        """Allocate the buffers of :meth:`capacity_contacts`, so that a graph capture of it allocates nothing: one
        candidate slot per pair of a shape and a segment."""
        F, device = max(self.capacity(model), 1), self.device
        if self._cap is not None and self._cap.flags.shape[0] == F:
            return

        def ints():
            return wp.zeros(F, dtype=wp.int32, device=device)

        self._cap = SimpleNamespace(
            flags=ints(),
            rank=ints(),
            segment=ints(),
            shape=ints(),
            keep=ints(),
            position=ints(),
            count=wp.zeros(1, dtype=wp.int32, device=device),
            found=wp.zeros(1, dtype=wp.int32, device=device),
            weight=wp.zeros(F, dtype=wp.float32, device=device),
            body_pos=wp.zeros(F, dtype=wp.vec3, device=device),
            normal=wp.zeros(F, dtype=wp.vec3, device=device),
        )

    def capacity_contacts(
        self,
        model,
        particle_q,
        body_q,
        pinned,
        pushed,
        edge_radius: float,
        gap: float,
        limit: int,
        rows: list,
        extra: list,
        active,
        gain: float,
        contact_gain,
        iterations: int = 40,
    ) -> None:
        """The segment contacts of :meth:`query` on the device alone, nothing read back: the candidates flagged and
        ranked by a scan, each one's contact at its own index, and those on held particles alone (``pinned``) dropped
        but on the shapes of integrated bodies (``pushed``). They are written, in the order of :meth:`query`, into the
        contact rows ``rows`` (particle, normal, alpha, shape, body_pos, record) and ``extra`` (particles and weights)
        from row ``active[0]`` on, up to row ``limit``; ``active[0]`` then counts them too, which may pass ``limit``,
        and ``contact_gain`` takes ``gain`` for each contact when the step keeps segment contacts.
        :meth:`reserve_capacity` sizes the buffers."""
        cap, device = self._cap, self.device
        n_seg, S, F = len(self.seg), model.shape_count, self._cap.flags.shape[0]
        bodies = self.no_bodies if body_q is None or not body_q.shape[0] else body_q
        cap.flags.zero_()
        if n_seg and S:
            wp.launch(
                near_segment_flags_kernel,
                dim=(n_seg, S),
                inputs=[
                    *(particle_q, self.seg_d, model.particle_radius, model.particle_world, bodies),
                    *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                    *(model.shape_source_ptr, model.shape_flags, model.shape_world, model.shape_margin),
                    *(model.shape_collision_aabb_lower, model.shape_collision_aabb_upper, gap, edge_radius),
                ],
                outputs=[cap.flags],
                device=device,
            )
        wp.utils.array_scan(cap.flags, cap.rank, inclusive=False)
        # The flagged pairs at their ranks, as face_candidates_kernel ranks the pairs of a shape and a triangle.
        inputs = [cap.flags, cap.rank, max(n_seg, 1)]
        wp.launch(face_candidates_kernel, F, inputs, [cap.count, cap.segment, cap.shape], device=device)
        cap.found.zero_()
        wp.launch(
            segment_contact_capacity_kernel,
            dim=F,
            inputs=[
                *(particle_q, self.seg_d, model.particle_radius, cap.segment, cap.shape, cap.count, bodies),
                *(model.shape_transform, model.shape_body, model.shape_type, model.shape_scale),
                *(model.shape_source_ptr, model.shape_margin, gap, iterations, edge_radius, pinned, pushed),
            ],
            outputs=[cap.found, cap.keep, cap.weight, cap.body_pos, cap.normal],
            device=device,
        )
        wp.utils.array_scan(cap.keep, cap.position, inclusive=False)
        inputs = [cap.count, cap.keep, cap.position, cap.segment, cap.shape, cap.weight, cap.body_pos, cap.normal]
        inputs += [self.seg_d, active, int(limit)]
        wp.launch(segment_rows_kernel, F, inputs, [*rows, *extra], device=device)
        inputs = [cap.position, cap.keep, cap.found, wp.float64(gain)]
        wp.launch(segment_count_kernel, contact_gain.shape[0], inputs, [active, contact_gain], device=device)
