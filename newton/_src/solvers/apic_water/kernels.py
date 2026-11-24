# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of :class:`~newton.solvers.SolverAPICWater`.

The grid is a MAC grid of ``n`` cells of size ``dx`` from ``lo``: velocity component ``a`` lives on the faces normal to
axis ``a`` (shape ``n + e_a``), the solids' signed distances on the nodes (shape ``n + 1``), the liquid's level set,
the cell types and the pressure on the cells. Cells outside the grid are air: water leaves through the box's sides.
"""

from __future__ import annotations

import warp as wp

from ...geometry.flags import ParticleFlags
from ...geometry.kernels import (
    mesh_query_point_sign,
    sdf_box,
    sdf_capsule,
    sdf_cone,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_plane,
    sdf_sphere,
)
from ...geometry.types import Axis, GeoType

wp.set_module_options({"enable_backward": False})

AIR = wp.constant(0)
FLUID = wp.constant(1)
SOLID = wp.constant(2)

ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))

# Rows of each block of the tiled reductions.
BLOCK = wp.constant(256)

# The node distances past this many cells from a solid are clamped to it: solids reach the water through a band of
# that width only.
FAR_CELLS = 4.0

# Nodes within this part of a cell of a solid's surface lie on it, as on a wall in a node plane: they count as just
# outside for the faces' fractions, so that a face that touches the wall along an edge stays open, while a face whose
# corners all lie on or inside the solid is closed and a cell whose corners all do is solid.
ON_SURFACE = wp.constant(1.0e-4)


# ----------------------------------------------------------------------------------------------------------------------
# Transfers


@wp.func
def quadratic_weights(f: float) -> wp.vec3:
    """Quadratic B-spline weights of the three samples from the base sample, for a fraction ``f`` in [0.5, 1.5)."""
    return wp.vec3(0.5 * (1.5 - f) * (1.5 - f), 0.75 - (f - 1.0) * (f - 1.0), 0.5 * (f - 0.5) * (f - 0.5))


@wp.func
def stencil(x: wp.vec3, lo: wp.vec3, inv_dx: float, offset: wp.vec3):
    """The base sample and the per-axis weights of a point on a grid whose samples sit at ``lo + (index + offset) dx``."""
    g = (x - lo) * inv_dx - offset
    base = wp.vec3i(int(wp.floor(g[0] - 0.5)), int(wp.floor(g[1] - 0.5)), int(wp.floor(g[2] - 0.5)))
    wx = quadratic_weights(g[0] - float(base[0]))
    wy = quadratic_weights(g[1] - float(base[1]))
    wz = quadratic_weights(g[2] - float(base[2]))
    return base, wx, wy, wz


@wp.func
def p2g_component(
    x: wp.vec3,
    v: float,
    c: wp.vec3,
    lo: wp.vec3,
    dx: float,
    offset: wp.vec3,
    m: wp.array3d[float],
    mv: wp.array3d[float],
):
    """Spread a particle's weight and its affine velocity component (APIC) onto one face grid."""
    base, wx, wy, wz = stencil(x, lo, 1.0 / dx, offset)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = base[0] + i
                b = base[1] + j
                d = base[2] + k
                if a >= 0 and a < m.shape[0] and b >= 0 and b < m.shape[1] and d >= 0 and d < m.shape[2]:
                    w = wx[i] * wy[j] * wz[k]
                    xf = lo + dx * (wp.vec3(float(a), float(b), float(d)) + offset)
                    wp.atomic_add(m, a, b, d, w)
                    wp.atomic_add(mv, a, b, d, w * (v + wp.dot(c, xf - x)))


@wp.kernel
def p2g_kernel(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    qd_grad: wp.array[wp.mat33],
    flags: wp.array[wp.int32],
    lo: wp.vec3,
    dx: float,
    # outputs
    um: wp.array3d[float],
    umv: wp.array3d[float],
    vm: wp.array3d[float],
    vmv: wp.array3d[float],
    wm: wp.array3d[float],
    wmv: wp.array3d[float],
    cell_count: wp.array3d[wp.int32],
):
    """The active particles' APIC transfer to the faces, each component through its row of ``qd_grad``, and the
    particle count of each cell."""
    p = wp.tid()
    if (flags[p] & ACTIVE) == 0:
        return
    xp = q[p]
    vp = qd[p]
    C = qd_grad[p]
    p2g_component(xp, vp[0], C[0], lo, dx, wp.vec3(0.0, 0.5, 0.5), um, umv)
    p2g_component(xp, vp[1], C[1], lo, dx, wp.vec3(0.5, 0.0, 0.5), vm, vmv)
    p2g_component(xp, vp[2], C[2], lo, dx, wp.vec3(0.5, 0.5, 0.0), wm, wmv)
    g = (xp - lo) / dx
    i = int(wp.floor(g[0]))
    j = int(wp.floor(g[1]))
    k = int(wp.floor(g[2]))
    n = cell_count.shape
    if i >= 0 and i < n[0] and j >= 0 and j < n[1] and k >= 0 and k < n[2]:
        wp.atomic_add(cell_count, i, j, k, 1)


@wp.kernel
def normalize_kernel(
    m: wp.array3d[float],
    mv: wp.array3d[float],
    gravity: wp.array[wp.vec3],
    axis: int,
    dt: float,
    # outputs
    vel: wp.array3d[float],
):
    """Face velocities from the transferred momentum, with gravity (the global world's, the last entry) over the
    step; faces without weight hold zero until the extrapolation."""
    i, j, k = wp.tid()
    mass = m[i, j, k]
    if mass > 1.0e-9:
        g = gravity[gravity.shape[0] - 1]
        vel[i, j, k] = mv[i, j, k] / mass + g[axis] * dt
    else:
        vel[i, j, k] = 0.0


@wp.func
def g2p_component(x: wp.vec3, lo: wp.vec3, dx: float, offset: wp.vec3, vel: wp.array3d[float]):
    """A velocity component at a particle and its affine vector (APIC) from one face grid."""
    base, wx, wy, wz = stencil(x, lo, 1.0 / dx, offset)
    v = float(0.0)
    c = wp.vec3(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = base[0] + i
                b = base[1] + j
                d = base[2] + k
                if a >= 0 and a < vel.shape[0] and b >= 0 and b < vel.shape[1] and d >= 0 and d < vel.shape[2]:
                    w = wx[i] * wy[j] * wz[k]
                    xf = lo + dx * (wp.vec3(float(a), float(b), float(d)) + offset)
                    value = vel[a, b, d]
                    v += w * value
                    c += (w * value * 4.0 / (dx * dx)) * (xf - x)
    return v, c


# ----------------------------------------------------------------------------------------------------------------------
# Solids


@wp.func
def shape_distance(
    geo: int,
    scale: wp.vec3,
    mesh: wp.uint64,
    sign_method: int,
    x: wp.vec3,
    max_dist: float,
) -> float:
    """The signed distance [m] of a shape at ``x`` in its frame, negative inside, ``max_dist`` past a mesh's reach."""
    if geo == GeoType.SPHERE:
        return sdf_sphere(x, scale[0])
    if geo == GeoType.BOX:
        return sdf_box(x, scale[0], scale[1], scale[2])
    if geo == GeoType.CAPSULE:
        return sdf_capsule(x, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.CYLINDER:
        return sdf_cylinder(x, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
    if geo == GeoType.CONE:
        return sdf_cone(x, scale[0], scale[1], int(Axis.Z))
    if geo == GeoType.ELLIPSOID:
        return sdf_ellipsoid(x, scale)
    if geo == GeoType.PLANE:
        return sdf_plane(x, scale[0] * 0.5, scale[1] * 0.5)
    if geo == GeoType.MESH or geo == GeoType.CONVEX_MESH:
        min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
        query = mesh_query_point_sign(mesh, wp.cw_div(x, scale), max_dist / min_scale, sign_method)
        if query.result:
            closest = wp.cw_mul(wp.mesh_eval_position(mesh, query.face, query.u, query.v), scale)
            return wp.length(x - closest) * query.sign
    return max_dist


@wp.kernel
def static_distance_kernel(
    shapes: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source: wp.array[wp.uint64],
    shape_sign: wp.array[wp.int32],
    shape_radius: wp.array[wp.float32],
    lo: wp.vec3,
    dx: float,
    far: float,
    # outputs
    phi: wp.array3d[float],
):
    """The nodes' signed distance to the union of the static shapes ``shapes``, clamped to ``far``."""
    i, j, k = wp.tid()
    x = lo + dx * wp.vec3(float(i), float(j), float(k))
    d = far
    for s in range(shapes.shape[0]):
        shape = shapes[s]
        X = shape_transform[shape]
        geo = shape_type[shape]
        reach = float(shape_radius[shape]) + far
        if geo != GeoType.PLANE and wp.length(x - wp.transform_get_translation(X)) > reach:
            continue
        local = wp.transform_point(wp.transform_inverse(X), x)
        d = wp.min(
            d, shape_distance(geo, shape_scale[shape], shape_source[shape], shape_sign[shape], local, 2.0 * reach)
        )
    phi[i, j, k] = d


@wp.kernel
def static_sub_distance_kernel(
    shapes: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source: wp.array[wp.uint64],
    shape_sign: wp.array[wp.int32],
    shape_radius: wp.array[wp.float32],
    lo: wp.vec3,
    dx: float,
    far: float,
    # outputs
    phi: wp.array4d[float],
):
    """The signed distance to the union of the static shapes ``shapes`` at each cell's eight sub-cell centers,
    clamped to ``far``: the cells' parts outside the static solids, exact where interpolating the nodes' distances
    would cut a concave corner."""
    i, j, k, sub = wp.tid()
    x = lo + dx * wp.vec3(
        float(i) + 0.25 + 0.5 * float(sub // 4),
        float(j) + 0.25 + 0.5 * float((sub // 2) % 2),
        float(k) + 0.25 + 0.5 * float(sub % 2),
    )
    d = far
    for s in range(shapes.shape[0]):
        shape = shapes[s]
        X = shape_transform[shape]
        geo = shape_type[shape]
        reach = float(shape_radius[shape]) + far
        if geo != GeoType.PLANE and wp.length(x - wp.transform_get_translation(X)) > reach:
            continue
        local = wp.transform_point(wp.transform_inverse(X), x)
        d = wp.min(
            d, shape_distance(geo, shape_scale[shape], shape_source[shape], shape_sign[shape], local, 2.0 * reach)
        )
    phi[i, j, k, sub] = d


@wp.kernel
def body_volume_kernel(
    shapes: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source: wp.array[wp.uint64],
    shape_sign: wp.array[wp.int32],
    shape_radius: wp.array[wp.float32],
    origin: wp.vec3,
    spacing: float,
    dims: wp.vec3i,
    far: float,
    offset: int,
    # outputs
    volume: wp.array[float],
):
    """A body's signed distance volume: the distance to the union of its shapes ``shapes`` at the nodes of a grid of
    ``dims`` nodes ``spacing`` apart from ``origin`` in the body's frame, clamped to ``far``, from ``offset`` in
    ``volume``."""
    i, j, k = wp.tid()
    x = origin + spacing * wp.vec3(float(i), float(j), float(k))
    d = far
    for s in range(shapes.shape[0]):
        shape = shapes[s]
        X = shape_transform[shape]
        reach = float(shape_radius[shape]) + far
        local = wp.transform_point(wp.transform_inverse(X), x)
        e = shape_distance(
            shape_type[shape], shape_scale[shape], shape_source[shape], shape_sign[shape], local, 2.0 * reach
        )
        d = wp.min(d, e)
    volume[offset + (i * dims[1] + j) * dims[2] + k] = d


@wp.func
def sample_volume(
    volume: wp.array[float], offset: int, dims: wp.vec3i, origin: wp.vec3, spacing: float, x: wp.vec3, far: float
) -> float:
    """The trilinear value of a body's distance volume at ``x`` in the body's frame, ``far`` outside it."""
    g = (x - origin) / spacing
    if g[0] < 0.0 or g[1] < 0.0 or g[2] < 0.0:
        return far
    if g[0] > float(dims[0] - 1) or g[1] > float(dims[1] - 1) or g[2] > float(dims[2] - 1):
        return far
    i = wp.min(int(g[0]), dims[0] - 2)
    j = wp.min(int(g[1]), dims[1] - 2)
    k = wp.min(int(g[2]), dims[2] - 2)
    fx = g[0] - float(i)
    fy = g[1] - float(j)
    fz = g[2] - float(k)
    value = float(0.0)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - fx, fx)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - fy, fy)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - fz, fz)
                value += wa * wb * wc * volume[offset + ((i + a) * dims[1] + j + b) * dims[2] + k + c]
    return value


@wp.kernel
def body_distance_kernel(
    volume_body: wp.array[wp.int32],
    volume_offset: wp.array[wp.int32],
    volume_dims: wp.array[wp.vec3i],
    volume_origin: wp.array[wp.vec3],
    spacing: float,
    volume: wp.array[float],
    body_q: wp.array[wp.transform],
    lo: wp.vec3,
    dx: float,
    far: float,
    # outputs
    phi: wp.array3d[float],
    owner: wp.array3d[wp.int32],
):
    """The nodes' signed distance to the bodies at their poses, from each body's distance volume, clamped to ``far``,
    and the nearest body (-1 past ``far``)."""
    i, j, k = wp.tid()
    x = lo + dx * wp.vec3(float(i), float(j), float(k))
    d = far
    body = int(-1)
    for v in range(volume_body.shape[0]):
        b = volume_body[v]
        local = wp.transform_point(wp.transform_inverse(body_q[b]), x)
        e = sample_volume(volume, volume_offset[v], volume_dims[v], volume_origin[v], spacing, local, far)
        if e < d:
            d = e
            body = b
    phi[i, j, k] = d
    owner[i, j, k] = body


@wp.func
def body_point_velocity(
    body: int,
    x: wp.vec3,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
) -> wp.vec3:
    """The velocity of body ``body`` at the world point ``x``."""
    com = wp.transform_point(body_q[body], body_com[body])
    twist = body_qd[body]
    return wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), x - com)


@wp.func
def fraction_inside_1d(a: float, b: float) -> float:
    """The fraction of a segment with end distances ``a`` and ``b`` that lies inside (negative)."""
    if a < 0.0 and b < 0.0:
        return 1.0
    if a < 0.0 and b >= 0.0:
        return a / (a - b)
    if a >= 0.0 and b < 0.0:
        return b / (b - a)
    return 0.0


@wp.func
def fraction_inside_2d(p0: float, p1: float, p2: float, p3: float) -> float:
    """The fraction of a square inside, from its corner distances in cyclic order (Batty, Bertails and Bridson 2007)."""
    v = wp.vec4(p0, p1, p2, p3)
    n = int(0)
    for k in range(4):
        if v[k] < 0.0:
            n += 1
    if n == 4:
        return 1.0
    if n == 0:
        return 0.0
    if n == 3:
        # The one outside corner first.
        for _ in range(4):
            if v[0] >= 0.0:
                break
            v = wp.vec4(v[1], v[2], v[3], v[0])
        side0 = 1.0 - fraction_inside_1d(v[0], v[3])
        side1 = 1.0 - fraction_inside_1d(v[0], v[1])
        return 1.0 - 0.5 * side0 * side1
    if n == 1:
        for _ in range(4):
            if v[0] < 0.0:
                break
            v = wp.vec4(v[1], v[2], v[3], v[0])
        side0 = fraction_inside_1d(v[0], v[3])
        side1 = fraction_inside_1d(v[0], v[1])
        return 0.5 * side0 * side1
    # Two inside: adjacent corners, or diagonal ones.
    for _ in range(4):
        if v[0] < 0.0 and v[1] < 0.0:
            break
        v = wp.vec4(v[1], v[2], v[3], v[0])
    if v[0] < 0.0 and v[1] < 0.0:
        left = fraction_inside_1d(v[0], v[3])
        right = fraction_inside_1d(v[1], v[2])
        return 0.5 * (left + right)
    # Diagonal: the center decides whether the inside corners connect.
    center = 0.25 * (p0 + p1 + p2 + p3)
    if center < 0.0:
        area = float(0.0)
        for k in range(4):
            if v[k] >= 0.0:
                a = 1.0 - fraction_inside_1d(v[k], v[(k + 3) % 4])
                b = 1.0 - fraction_inside_1d(v[k], v[(k + 1) % 4])
                area += 0.5 * a * b
        return 1.0 - area
    area = float(0.0)
    for k in range(4):
        if v[k] < 0.0:
            a = fraction_inside_1d(v[k], v[(k + 3) % 4])
            b = fraction_inside_1d(v[k], v[(k + 1) % 4])
            area += 0.5 * a * b
    return area


@wp.func
def off_surface(phi: float, eps: float) -> float:
    """A node distance with the nodes on the surface (within ``eps``) moved just outside."""
    if phi < -eps:
        return phi
    return wp.max(phi, eps)


@wp.func
def face_inside(c: wp.vec4, eps: float) -> float:
    """The fraction of a face inside a solid from its corner distances in cyclic order: closed when every corner lies
    on or inside the solid, else Batty's estimate with the corners on the surface just outside."""
    if wp.max(wp.max(c[0], c[1]), wp.max(c[2], c[3])) <= eps:
        return 1.0
    return fraction_inside_2d(
        off_surface(c[0], eps), off_surface(c[1], eps), off_surface(c[2], eps), off_surface(c[3], eps)
    )


@wp.func
def face_corner(axis: int, corner: int, i: int, j: int, k: int) -> wp.vec3i:
    """Node ``corner`` (0..3, in cyclic order) of face ``(i, j, k)`` normal to ``axis``."""
    a = int(corner == 1 or corner == 2)
    b = int(corner >= 2)
    if axis == 0:
        return wp.vec3i(i, j + a, k + b)
    if axis == 1:
        return wp.vec3i(i + a, j, k + b)
    return wp.vec3i(i + a, j + b, k)


@wp.func
def face_corners(phi: wp.array3d[float], axis: int, i: int, j: int, k: int) -> wp.vec4:
    """The node values at the four corners of face ``(i, j, k)`` normal to ``axis``, in cyclic order."""
    out = wp.vec4()
    for c in range(4):
        node = face_corner(axis, c, i, j, k)
        out[c] = phi[node[0], node[1], node[2]]
    return out


@wp.func
def face_body(phi_body: wp.array3d[float], owner: wp.array3d[wp.int32], axis: int, i: int, j: int, k: int) -> int:
    """The body nearest to face ``(i, j, k)``: the owner of its corner nearest to a body."""
    best = float(1.0e30)
    body = int(-1)
    for c in range(4):
        node = face_corner(axis, c, i, j, k)
        d = phi_body[node[0], node[1], node[2]]
        if d < best:
            best = d
            body = owner[node[0], node[1], node[2]]
    return body


@wp.func
def face_center(lo: wp.vec3, dx: float, axis: int, i: int, j: int, k: int) -> wp.vec3:
    c = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    c[axis] = float(wp.vec3i(i, j, k)[axis])
    return lo + dx * c


@wp.kernel
def face_fractions_kernel(
    phi_static: wp.array3d[float],
    phi_body: wp.array3d[float],
    owner: wp.array3d[wp.int32],
    axis: int,
    lo: wp.vec3,
    dx: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # outputs
    open_frac: wp.array3d[float],
    body_frac: wp.array3d[float],
    solid_vel: wp.array3d[float],
):
    """The open fraction of each face (outside every solid), the bodies' fraction of it, and the normal velocity of
    the solid that closes it: the nearest body's rigid motion where a body is nearer than the static solids, zero
    for them."""
    i, j, k = wp.tid()
    cs = face_corners(phi_static, axis, i, j, k)
    cb = face_corners(phi_body, axis, i, j, k)
    cu = wp.vec4(wp.min(cs[0], cb[0]), wp.min(cs[1], cb[1]), wp.min(cs[2], cb[2]), wp.min(cs[3], cb[3]))
    eps = ON_SURFACE * dx
    open_frac[i, j, k] = 1.0 - face_inside(cu, eps)
    body_frac[i, j, k] = face_inside(cb, eps)
    vs = float(0.0)
    mb = 0.25 * (cb[0] + cb[1] + cb[2] + cb[3])
    ms = 0.25 * (cs[0] + cs[1] + cs[2] + cs[3])
    if mb < ms:
        body = face_body(phi_body, owner, axis, i, j, k)
        if body >= 0:
            vel = body_point_velocity(body, face_center(lo, dx, axis, i, j, k), body_q, body_qd, body_com)
            vs = vel[axis]
    solid_vel[i, j, k] = vs


@wp.func
def trilinear(phi: wp.array3d[float], g: wp.vec3):
    """Trilinear value and gradient (per grid unit) of a node field at grid coordinate ``g``, clamped to the grid."""
    n = phi.shape
    i = wp.clamp(int(wp.floor(g[0])), 0, n[0] - 2)
    j = wp.clamp(int(wp.floor(g[1])), 0, n[1] - 2)
    k = wp.clamp(int(wp.floor(g[2])), 0, n[2] - 2)
    fx = wp.clamp(g[0] - float(i), 0.0, 1.0)
    fy = wp.clamp(g[1] - float(j), 0.0, 1.0)
    fz = wp.clamp(g[2] - float(k), 0.0, 1.0)
    value = float(0.0)
    grad = wp.vec3(0.0)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - fx, fx)
        da = wp.where(a == 0, -1.0, 1.0)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - fy, fy)
            db = wp.where(b == 0, -1.0, 1.0)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - fz, fz)
                dc = wp.where(c == 0, -1.0, 1.0)
                h = phi[i + a, j + b, k + c]
                value += wa * wb * wc * h
                grad += h * wp.vec3(da * wb * wc, wa * db * wc, wa * wb * dc)
    return value, grad


@wp.func
def nearest_owner(owner: wp.array3d[wp.int32], g: wp.vec3) -> int:
    """The body owning the node nearest to grid coordinate ``g``."""
    n = owner.shape
    i = wp.clamp(int(wp.round(g[0])), 0, n[0] - 1)
    j = wp.clamp(int(wp.round(g[1])), 0, n[1] - 1)
    k = wp.clamp(int(wp.round(g[2])), 0, n[2] - 1)
    return owner[i, j, k]


# ----------------------------------------------------------------------------------------------------------------------
# The liquid and the cells


@wp.kernel
def liquid_splat_kernel(
    q: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    lo: wp.vec3,
    dx: float,
    radius: float,
    # outputs
    weight: wp.array3d[float],
    center: wp.array3d[wp.vec3],
):
    """Each active particle's kernel weights ``(1 - (d / R)^2)^3`` and weighted positions at the cell centers within
    ``R``."""
    p = wp.tid()
    if (flags[p] & ACTIVE) == 0:
        return
    xp = q[p]
    g = (xp - lo) / dx - wp.vec3(0.5)
    reach = int(wp.ceil(radius / dx))
    i0 = int(wp.floor(g[0])) - reach + 1
    j0 = int(wp.floor(g[1])) - reach + 1
    k0 = int(wp.floor(g[2])) - reach + 1
    n = weight.shape
    for a in range(2 * reach):
        for b in range(2 * reach):
            for c in range(2 * reach):
                i = i0 + a
                j = j0 + b
                k = k0 + c
                if i >= 0 and i < n[0] and j >= 0 and j < n[1] and k >= 0 and k < n[2]:
                    xc = lo + dx * wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
                    s = wp.length(xc - xp) / radius
                    if s < 1.0:
                        t = 1.0 - s * s
                        w = t * t * t
                        wp.atomic_add(weight, i, j, k, w)
                        wp.atomic_add(center, i, j, k, w * xp)


@wp.kernel
def liquid_phi_kernel(
    weight: wp.array3d[float],
    center: wp.array3d[wp.vec3],
    phi_static: wp.array3d[float],
    phi_body: wp.array3d[float],
    lo: wp.vec3,
    dx: float,
    particle_radius: float,
    far: float,
    # outputs
    phi: wp.array3d[float],
):
    """The liquid's signed distance at each cell center: from the weighted mean of the nearby particles less their
    radius (Zhu and Bridson 2005), ``far`` without particles in reach. A cell whose center lies inside a solid within
    half a cell of the liquid becomes liquid (Batty's variational solver), so that the open part of a cut cell under
    the water holds water rather than air."""
    i, j, k = wp.tid()
    w = weight[i, j, k]
    d = far
    if w > 0.0:
        xc = lo + dx * wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
        d = wp.min(wp.length(xc - center[i, j, k] / w) - particle_radius, far)
    if d < 0.5 * dx:
        s = float(0.0)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    s += wp.min(phi_static[i + a, j + b, k + c], phi_body[i + a, j + b, k + c])
        if s < 0.0:
            d = -0.5 * dx
    phi[i, j, k] = d


@wp.kernel
def classify_kernel(
    liquid_phi: wp.array3d[float],
    cell_count: wp.array3d[wp.int32],
    phi_static: wp.array3d[float],
    phi_body: wp.array3d[float],
    dx: float,
    # outputs
    cell_type: wp.array3d[wp.int32],
    fluid_flags: wp.array[wp.int32],
):
    """Solid where the cell's eight corners all lie on or inside a solid; else fluid where particles are or where the
    liquid's level set is negative at the cell's center, so that the thin open parts of cells cut by a wall under the
    water count too; else air. A cell with particles never takes air's pressure, which would drain the water around
    into it. ``fluid_flags`` marks the fluid cells in flat order."""
    i, j, k = wp.tid()
    inside = int(1)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                if wp.min(phi_static[i + a, j + b, k + c], phi_body[i + a, j + b, k + c]) > ON_SURFACE * dx:
                    inside = 0
    t = AIR
    if inside != 0:
        t = SOLID
    elif liquid_phi[i, j, k] < 0.0 or cell_count[i, j, k] > 0:
        t = FLUID
    cell_type[i, j, k] = t
    n = cell_type.shape
    fluid_flags[(i * n[1] + j) * n[2] + k] = wp.where(t == FLUID, 1, 0)


@wp.kernel
def compact_cells_kernel(
    flags: wp.array[wp.int32],
    position: wp.array[wp.int32],
    # outputs
    index: wp.array[wp.int32],
    cells: wp.array[wp.int32],
    count: wp.array[wp.int32],
    dropped: wp.array[wp.int32],
):
    """The fluid cells in flat order, the row of each cell (-1 off them), their count and the most fluid cells past
    the capacity, which stay out of the solve as air."""
    c = wp.tid()
    if flags[c] != 0 and position[c] < cells.shape[0]:
        index[c] = position[c]
        cells[position[c]] = c
    else:
        index[c] = -1
    if c == flags.shape[0] - 1:
        total = position[c] + flags[c]
        count[0] = wp.min(total, cells.shape[0])
        wp.atomic_max(dropped, 0, total - cells.shape[0])


@wp.func
def unflatten(c: int, ny: int, nz: int) -> wp.vec3i:
    return wp.vec3i(c // (ny * nz), (c // nz) % ny, c % nz)


@wp.func
def kind(cell_type: wp.array3d[wp.int32], i: int, j: int, k: int) -> int:
    """The type of cell ``(i, j, k)``; cells outside the grid are air."""
    n = cell_type.shape
    if i < 0 or i >= n[0] or j < 0 or j >= n[1] or k < 0 or k >= n[2]:
        return AIR
    return cell_type[i, j, k]


@wp.func
def level(liquid_phi: wp.array3d[float], i: int, j: int, k: int, far: float) -> float:
    """The liquid's level set at cell ``(i, j, k)``; ``far`` outside the grid."""
    n = liquid_phi.shape
    if i < 0 or i >= n[0] or j < 0 or j >= n[1] or k < 0 or k >= n[2]:
        return far
    return liquid_phi[i, j, k]


@wp.func
def ghost_theta(phi_in: float, phi_out: float) -> float:
    """The fraction of the way from a fluid cell's center to an air neighbor's at which the surface lies (ghost
    fluid), at least 0.05."""
    if phi_in >= 0.0:
        return 0.05
    return wp.clamp(phi_in / (phi_in - wp.max(phi_out, 0.0)), 0.05, 1.0)


@wp.func
def face_gain(
    cell_type: wp.array3d[wp.int32], liquid_phi: wp.array3d[float], me: float, i: int, j: int, k: int, far: float
) -> float:
    """The factor of a fluid cell's face weight toward neighbor ``(i, j, k)``: ``1 / theta`` to air, 1 otherwise."""
    if kind(cell_type, i, j, k) == AIR:
        return 1.0 / ghost_theta(me, level(liquid_phi, i, j, k, far))
    return 1.0


@wp.func
def q_at(index: wp.array[wp.int32], q: wp.array[float], i: int, j: int, k: int, n: wp.vec3i) -> float:
    """The pressure unknown of cell ``(i, j, k)``, zero off the fluid cells."""
    if i < 0 or i >= n[0] or j < 0 or j >= n[1] or k < 0 or k >= n[2]:
        return 0.0
    r = index[(i * n[1] + j) * n[2] + k]
    if r < 0:
        return 0.0
    return q[r]


# ----------------------------------------------------------------------------------------------------------------------
# Deterministic sums: per-block tile sums, then one block sums the blocks.


@wp.func
def open_volume(static_sub: wp.array4d[float], phi_body: wp.array3d[float], dx: float, i: int, j: int, k: int) -> float:
    """The part of cell ``(i, j, k)`` outside the solids: over its eight half-size sub-cells, each sub-cell's part
    beyond the solids' surface from the distance at its center, the static solids' exact (``static_sub``) and the
    bodies' interpolated, so that a wall in the cell's face plane leaves it whole and a plane through it the part on
    its side."""
    total = float(0.0)
    for sub in range(8):
        g = wp.vec3(
            float(i) + 0.25 + 0.5 * float(sub // 4),
            float(j) + 0.25 + 0.5 * float((sub // 2) % 2),
            float(k) + 0.25 + 0.5 * float(sub % 2),
        )
        db, _gb = trilinear(phi_body, g)
        total += wp.clamp(0.5 + wp.min(static_sub[i, j, k, sub], db) / (0.5 * dx), 0.0, 1.0)
    return 0.125 * total


@wp.kernel
def liquid_volume_kernel(
    cells: wp.array[wp.int32],
    count: wp.array[wp.int32],
    cell_count: wp.array3d[wp.int32],
    rest_count: float,
    static_sub: wp.array4d[float],
    phi_body: wp.array3d[float],
    dx: float,
    cell_type: wp.array3d[wp.int32],
    # outputs
    liquid_part: wp.array[float],
    partials: wp.array[float],
):
    """Each fluid cell's liquid part of a cell as the pressure solve takes it: a fluid cell without air beside it
    holds the part of it outside the solids, one at the surface the particles it holds, at most that part (its
    particle count over ``rest_count``, the particles of a full cell); and the blocks' sums of the liquid parts
    (launched tiled)."""
    b, lane = wp.tid()
    r = b * BLOCK + lane
    part = float(0.0)
    if r < count[0]:
        n = cell_type.shape
        ijk = unflatten(cells[r], n[1], n[2])
        i = ijk[0]
        j = ijk[1]
        k = ijk[2]
        part = open_volume(static_sub, phi_body, dx, i, j, k)
        surface = kind(cell_type, i - 1, j, k) == AIR or kind(cell_type, i + 1, j, k) == AIR
        surface = surface or kind(cell_type, i, j - 1, k) == AIR or kind(cell_type, i, j + 1, k) == AIR
        surface = surface or kind(cell_type, i, j, k - 1) == AIR or kind(cell_type, i, j, k + 1) == AIR
        if surface:
            part = wp.min(part, float(cell_count[i, j, k]) / rest_count)
    liquid_part[r] = part
    wp.tile_store(partials, wp.tile_sum(wp.tile(part)), offset=(b,))


@wp.kernel
def particle_volume_kernel(
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    inv_density: float,
    # outputs
    partials: wp.array[float],
):
    """The blocks' sums of the active particles' volumes ``mass / density`` (launched tiled)."""
    b, lane = wp.tid()
    p = b * BLOCK + lane
    v = float(0.0)
    if p < mass.shape[0]:
        if (flags[p] & ACTIVE) != 0:
            v = mass[p] * inv_density
    wp.tile_store(partials, wp.tile_sum(wp.tile(v)), offset=(b,))


_TOTAL_KERNELS = {}


def total_kernel(size: int):
    """A kernel that writes the sum of ``size`` per-block partial sums into ``out[0]`` (one block, launched tiled),
    in a fixed order."""
    if size not in _TOTAL_KERNELS:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(partials: wp.array[float], out: wp.array[float]):
            total = wp.tile_sum(wp.tile_load(partials, shape=(size,)))
            wp.tile_store(out, total)

        _TOTAL_KERNELS[size] = kernel
    return _TOTAL_KERNELS[size]


# ----------------------------------------------------------------------------------------------------------------------
# The pressure solve


@wp.kernel
def divergence_kernel(
    cells: wp.array[wp.int32],
    count: wp.array[wp.int32],
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    fu: wp.array3d[float],
    fv: wp.array3d[float],
    fw: wp.array3d[float],
    su: wp.array3d[float],
    sv: wp.array3d[float],
    sw: wp.array3d[float],
    cell_type: wp.array3d[wp.int32],
    liquid_phi: wp.array3d[float],
    far: float,
    liquid_volume: wp.array[float],
    particle_volume: wp.array[float],
    cell_volume: float,
    liquid_part: wp.array[float],
    volume_rate: float,
    # outputs
    rhs: wp.array[float],
    diag: wp.array[float],
):
    """``rhs = -div`` of each fluid cell (times ``dx``), the fluid velocity on the open part of each face and the
    solid's on the closed part, less the global volume correction: every fluid cell takes the part ``volume_rate`` of
    the liquid's relative volume error against the particles' volume, times its liquid part, as outflow (inflow when
    the liquid shrank). The diagonal sums the faces' open fractions, those toward air over ``theta`` (ghost fluid)."""
    r = wp.tid()
    if r >= count[0]:
        rhs[r] = 0.0
        diag[r] = 1.0
        return
    n = cell_type.shape
    ijk = unflatten(cells[r], n[1], n[2])
    i = ijk[0]
    j = ijk[1]
    k = ijk[2]
    flux = float(0.0)
    flux += fu[i + 1, j, k] * u[i + 1, j, k] + (1.0 - fu[i + 1, j, k]) * su[i + 1, j, k]
    flux -= fu[i, j, k] * u[i, j, k] + (1.0 - fu[i, j, k]) * su[i, j, k]
    flux += fv[i, j + 1, k] * v[i, j + 1, k] + (1.0 - fv[i, j + 1, k]) * sv[i, j + 1, k]
    flux -= fv[i, j, k] * v[i, j, k] + (1.0 - fv[i, j, k]) * sv[i, j, k]
    flux += fw[i, j, k + 1] * w[i, j, k + 1] + (1.0 - fw[i, j, k + 1]) * sw[i, j, k + 1]
    flux -= fw[i, j, k] * w[i, j, k] + (1.0 - fw[i, j, k]) * sw[i, j, k]
    error = float(0.0)
    if particle_volume[0] > 0.0:
        error = liquid_volume[0] * cell_volume / particle_volume[0] - 1.0
    rhs[r] = -flux - volume_rate * error * liquid_part[r]
    me = liquid_phi[i, j, k]
    d = float(0.0)
    d += fu[i + 1, j, k] * face_gain(cell_type, liquid_phi, me, i + 1, j, k, far)
    d += fu[i, j, k] * face_gain(cell_type, liquid_phi, me, i - 1, j, k, far)
    d += fv[i, j + 1, k] * face_gain(cell_type, liquid_phi, me, i, j + 1, k, far)
    d += fv[i, j, k] * face_gain(cell_type, liquid_phi, me, i, j - 1, k, far)
    d += fw[i, j, k + 1] * face_gain(cell_type, liquid_phi, me, i, j, k + 1, far)
    d += fw[i, j, k] * face_gain(cell_type, liquid_phi, me, i, j, k - 1, far)
    diag[r] = wp.max(d, 1.0e-3)


@wp.kernel
def precondition_kernel(r: wp.array[float], diag: wp.array[float], z: wp.array[float]):
    i = wp.tid()
    z[i] = r[i] / diag[i]


@wp.kernel
def cg_stencil_kernel(
    cells: wp.array[wp.int32],
    count: wp.array[wp.int32],
    index: wp.array[wp.int32],
    diag: wp.array[float],
    fu: wp.array3d[float],
    fv: wp.array3d[float],
    fw: wp.array3d[float],
    u: wp.array[float],
    r: wp.array[float],
    n: wp.vec3i,
    # outputs
    w: wp.array[float],
    partials: wp.array[wp.vec2],
):
    """``w = A u`` of the cut-cell Laplacian and each block's sums of ``r . u`` and ``w . u`` (launched tiled)."""
    b, lane = wp.tid()
    row = b * BLOCK + lane
    wv = float(0.0)
    ru = float(0.0)
    wu = float(0.0)
    if row < count[0]:
        ijk = unflatten(cells[row], n[1], n[2])
        i = ijk[0]
        j = ijk[1]
        k = ijk[2]
        s = fu[i, j, k] * q_at(index, u, i - 1, j, k, n) + fu[i + 1, j, k] * q_at(index, u, i + 1, j, k, n)
        s += fv[i, j, k] * q_at(index, u, i, j - 1, k, n) + fv[i, j + 1, k] * q_at(index, u, i, j + 1, k, n)
        s += fw[i, j, k] * q_at(index, u, i, j, k - 1, n) + fw[i, j, k + 1] * q_at(index, u, i, j, k + 1, n)
        ui = u[row]
        wv = diag[row] * ui - s
        ru = r[row] * ui
        wu = wv * ui
    w[row] = wv
    wp.tile_store(partials, wp.tile_sum(wp.tile(wp.vec2(ru, wu), preserve_type=True)), offset=(b,))


_CG_STEP_KERNELS = {}


def cg_step_kernel(size: int):
    """One step of the single-reduction conjugate gradient (Chronopoulos and Gear) over ``size`` slots of block
    partials (launched tiled, one block of BLOCK rows per launch index)."""
    if size in _CG_STEP_KERNELS:
        return _CG_STEP_KERNELS[size]

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        partials: wp.array[wp.vec2],
        scalars: wp.array[wp.vec3],
        slot: int,
        first: int,
        diag: wp.array[float],
        w: wp.array[float],
        u: wp.array[float],
        p: wp.array[float],
        s: wp.array[float],
        x: wp.array[float],
        r: wp.array[float],
    ):
        """From the blocks' sums ``gamma = r . u`` and ``delta = (A u) . u``: ``beta = gamma / gamma_old`` and
        ``alpha = gamma / (delta - beta gamma / alpha_old)``; ``p = u + beta p``, ``s = A p = w + beta s``,
        ``x += alpha p``, ``r -= alpha s`` and ``u = r / diag``. Every block sums the partials alike; the first
        thread keeps ``(gamma, alpha, stopped)`` in the other slot. A step whose curvature is not positive stops the
        iteration."""
        b, lane = wp.tid()
        total = wp.tile_extract(wp.tile_sum(wp.tile_load(partials, shape=(size,))), 0)
        gamma = total[0]
        delta = total[1]
        prev = scalars[slot]
        stopped = prev[2] != 0.0
        alpha = float(0.0)
        beta = float(0.0)
        if first != 0:
            stopped = bool(False)
            if delta > 0.0:
                alpha = gamma / delta
            else:
                stopped = True
        elif not stopped:
            if prev[0] > 0.0 and prev[1] > 0.0:
                beta = gamma / prev[0]
                den = delta - beta * gamma / prev[1]
                if den > 0.0:
                    alpha = gamma / den
                else:
                    stopped = True
            else:
                stopped = True
        if b == 0 and lane == 0:
            flag = float(0.0)
            if stopped:
                flag = 1.0
            scalars[1 - slot] = wp.vec3(gamma, alpha, flag)
        row = b * BLOCK + lane
        if not stopped:
            pi = u[row]
            si = w[row]
            if first == 0:
                pi = pi + beta * p[row]
                si = si + beta * s[row]
            p[row] = pi
            s[row] = si
            x[row] = x[row] + alpha * pi
            ri = r[row] - alpha * si
            r[row] = ri
            u[row] = ri / diag[row]

    _CG_STEP_KERNELS[size] = kernel
    return kernel


@wp.kernel
def body_impulse_kernel(
    cells: wp.array[wp.int32],
    count: wp.array[wp.int32],
    q: wp.array[float],
    fbu: wp.array3d[float],
    fbv: wp.array3d[float],
    fbw: wp.array3d[float],
    phi_body: wp.array3d[float],
    owner: wp.array3d[wp.int32],
    lo: wp.vec3,
    dx: float,
    scale: float,
    gravity: wp.array[wp.vec3],
    hydrostatic: float,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    impulse: wp.array[wp.spatial_vector],
):
    """The pressure impulse of each fluid cell on the bodies' parts of its faces, pushing out of the cell, and its
    moment about each body's center of mass: ``scale q`` is the pressure times a face's area times the step. The
    pressure is taken at the body's wall within the cell rather than at the cell's center, which lies up to half a
    cell deeper inside a cut cell: moved there along the hydrostatic gradient, gravity times ``hydrostatic`` (the
    step over ``dx``) per meter in units of ``q``."""
    r = wp.tid()
    if r >= count[0]:
        return
    n = wp.vec3i(phi_body.shape[0] - 1, phi_body.shape[1] - 1, phi_body.shape[2] - 1)
    ijk = unflatten(cells[r], n[1], n[2])
    i = ijk[0]
    j = ijk[1]
    k = ijk[2]
    center = lo + dx * wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    pr = q[r]
    db, gb = trilinear(phi_body, wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5))
    if wp.length(gb) > 0.0:
        wall = center - wp.clamp(db, -dx, dx) * wp.normalize(gb)
        pr = pr + hydrostatic * wp.dot(gravity[gravity.shape[0] - 1], wall - center)
    p = scale * pr
    for side in range(6):
        axis = side // 2
        upper = side % 2
        a = i
        b = j
        c = k
        frac = float(0.0)
        if axis == 0:
            a = i + upper
            frac = fbu[a, b, c]
        elif axis == 1:
            b = j + upper
            frac = fbv[a, b, c]
        else:
            c = k + upper
            frac = fbw[a, b, c]
        if frac > 0.0:
            body = face_body(phi_body, owner, axis, a, b, c)
            if body >= 0:
                normal = wp.vec3(0.0)
                normal[axis] = float(2 * upper - 1)
                x = center + 0.5 * dx * normal
                f = (p * frac) * normal
                com = wp.transform_point(body_q[body], body_com[body])
                wp.atomic_add(impulse, body, wp.spatial_vector(f, wp.cross(x - com, f)))


@wp.kernel
def project_kernel(
    cell_type: wp.array3d[wp.int32],
    index: wp.array[wp.int32],
    q: wp.array[float],
    axis: int,
    open_frac: wp.array3d[float],
    solid_vel: wp.array3d[float],
    liquid_phi: wp.array3d[float],
    far: float,
    # outputs
    vel: wp.array3d[float],
    valid: wp.array3d[wp.int32],
):
    """Subtract the pressure gradient from the open faces next to fluid, an air neighbor's ghost pressure vanishing
    at the surface; closed faces take the solid's velocity. ``valid`` marks the faces that hold a velocity."""
    i, j, k = wp.tid()
    n = wp.vec3i(cell_type.shape[0], cell_type.shape[1], cell_type.shape[2])
    lower = wp.vec3i(i, j, k)
    lower[axis] = lower[axis] - 1
    il = lower[0]
    jl = lower[1]
    kl = lower[2]
    a = kind(cell_type, i, j, k)
    b = kind(cell_type, il, jl, kl)
    if open_frac[i, j, k] <= 0.0:
        vel[i, j, k] = solid_vel[i, j, k]
        valid[i, j, k] = 1
        return
    if a == FLUID and b == FLUID:
        vel[i, j, k] = vel[i, j, k] - (q_at(index, q, i, j, k, n) - q_at(index, q, il, jl, kl, n))
        valid[i, j, k] = 1
    elif a == FLUID:
        theta = ghost_theta(level(liquid_phi, i, j, k, far), level(liquid_phi, il, jl, kl, far))
        vel[i, j, k] = vel[i, j, k] - q_at(index, q, i, j, k, n) / theta
        valid[i, j, k] = 1
    elif b == FLUID:
        theta = ghost_theta(level(liquid_phi, il, jl, kl, far), level(liquid_phi, i, j, k, far))
        vel[i, j, k] = vel[i, j, k] + q_at(index, q, il, jl, kl, n) / theta
        valid[i, j, k] = 1
    else:
        valid[i, j, k] = 0


@wp.kernel
def extrapolate_kernel(
    src: wp.array3d[float], valid: wp.array3d[wp.int32], dst: wp.array3d[float], valid_out: wp.array3d[wp.int32]
):
    """One layer of extrapolation: a face without a velocity takes the mean of its neighbors' on the same grid."""
    i, j, k = wp.tid()
    if valid[i, j, k] != 0:
        dst[i, j, k] = src[i, j, k]
        valid_out[i, j, k] = 1
        return
    s = float(0.0)
    c = int(0)
    n = src.shape
    for d in range(6):
        e = wp.vec3i(i, j, k)
        e[d // 2] = e[d // 2] + 2 * (d % 2) - 1
        if e[0] >= 0 and e[0] < n[0] and e[1] >= 0 and e[1] < n[1] and e[2] >= 0 and e[2] < n[2]:
            if valid[e[0], e[1], e[2]] != 0:
                s += src[e[0], e[1], e[2]]
                c += 1
    if c > 0:
        dst[i, j, k] = s / float(c)
        valid_out[i, j, k] = 1
    else:
        dst[i, j, k] = src[i, j, k]
        valid_out[i, j, k] = 0


# ----------------------------------------------------------------------------------------------------------------------
# Back to the particles


@wp.func
def push_out(
    x: wp.vec3,
    phi_static: wp.array3d[float],
    phi_body: wp.array3d[float],
    owner: wp.array3d[wp.int32],
    lo: wp.vec3,
    dx: float,
    margin: float,
):
    """The point ``x`` moved out of the solids to ``margin`` outside, the normal of the nearer solid, whether it
    moved, and the body nearest to it (-1 for the static solids)."""
    g = (x - lo) / dx
    ds, gs = trilinear(phi_static, g)
    db, gb = trilinear(phi_body, g)
    d = ds
    grad = gs
    body = int(-1)
    if db < ds:
        d = db
        grad = gb
        body = nearest_owner(owner, g)
    moved = bool(False)
    normal = wp.vec3(0.0)
    if d < margin and wp.length(grad) > 0.0:
        normal = wp.normalize(grad)
        x = x + (margin - d) * normal
        moved = True
    return x, normal, moved, body


@wp.func
def park(p: int, lo: wp.vec3, dx: float) -> wp.vec3:
    """Where an inactive particle waits: far below the grid, each at its own point."""
    return lo - wp.vec3(1.0e3 + dx * float(p), 1.0e3, 1.0e3)


@wp.kernel
def g2p_kernel(
    u: wp.array3d[float],
    v: wp.array3d[float],
    w: wp.array3d[float],
    phi_static: wp.array3d[float],
    phi_body: wp.array3d[float],
    owner: wp.array3d[wp.int32],
    q_in: wp.array[wp.vec3],
    qd_in: wp.array[wp.vec3],
    qd_grad_in: wp.array[wp.mat33],
    lo: wp.vec3,
    hi: wp.vec3,
    dx: float,
    dt: float,
    margin: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # outputs
    flags: wp.array[wp.int32],
    q_out: wp.array[wp.vec3],
    qd_out: wp.array[wp.vec3],
    qd_grad_out: wp.array[wp.mat33],
):
    """APIC transfer back, advection, and a push out of the solids to ``margin`` outside, the velocity into the solid
    removed relative to it. A particle that leaves the grid's box, less a stencil's reach, or turns non-finite leaves
    the water: inactive, parked, at rest."""
    p = wp.tid()
    if (flags[p] & ACTIVE) == 0:
        q_out[p] = q_in[p]
        qd_out[p] = qd_in[p]
        qd_grad_out[p] = qd_grad_in[p]
        return
    xp = q_in[p]
    a, ca = g2p_component(xp, lo, dx, wp.vec3(0.0, 0.5, 0.5), u)
    b, cb = g2p_component(xp, lo, dx, wp.vec3(0.5, 0.0, 0.5), v)
    c, cc = g2p_component(xp, lo, dx, wp.vec3(0.5, 0.5, 0.0), w)
    vp = wp.vec3(a, b, c)
    xn, normal, moved, body = push_out(xp + dt * vp, phi_static, phi_body, owner, lo, dx, margin)
    if moved:
        solid_v = wp.vec3(0.0)
        if body >= 0:
            solid_v = body_point_velocity(body, xn, body_q, body_qd, body_com)
        rel = wp.dot(vp - solid_v, normal)
        if rel < 0.0:
            vp = vp - rel * normal
    reach = 1.5 * dx
    inside = xn[0] > lo[0] + reach and xn[1] > lo[1] + reach and xn[2] > lo[2] + reach
    inside = inside and xn[0] < hi[0] - reach and xn[1] < hi[1] - reach and xn[2] < hi[2] - reach
    finite = wp.isfinite(xn[0]) and wp.isfinite(xn[1]) and wp.isfinite(xn[2])
    finite = finite and wp.isfinite(vp[0]) and wp.isfinite(vp[1]) and wp.isfinite(vp[2])
    if inside and finite:
        q_out[p] = xn
        qd_out[p] = vp
        qd_grad_out[p] = wp.mat33(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2], cc[0], cc[1], cc[2])
    else:
        flags[p] = flags[p] & ~ACTIVE
        q_out[p] = park(p, lo, dx)
        qd_out[p] = wp.vec3(0.0)
        qd_grad_out[p] = wp.mat33(0.0)


@wp.kernel
def grid_points_kernel(q: wp.array[wp.vec3], flags: wp.array[wp.int32], lo: wp.vec3, dx: float, out: wp.array[wp.vec3]):
    """The points of the separation's hash grid: the active particles, the inactive ones each at its own far point."""
    p = wp.tid()
    if (flags[p] & ACTIVE) != 0:
        out[p] = q[p]
    else:
        out[p] = park(p, lo, dx)


@wp.kernel
def separate_kernel(
    grid: wp.uint64,
    q: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    spacing: float,
    strength: float,
    # outputs
    shift: wp.array[wp.vec3],
):
    """Push each pair of active particles closer than ``spacing`` apart, each by ``strength`` times half the overlap,
    so that the water cannot pack past its rest spacing (particle separation)."""
    p = wp.tid()
    s = wp.vec3(0.0)
    if (flags[p] & ACTIVE) != 0:
        xp = q[p]
        query = wp.hash_grid_query(grid, xp, spacing)
        other = int(0)
        while wp.hash_grid_query_next(query, other):
            if other != p and (flags[other] & ACTIVE) != 0:
                d = xp - q[other]
                r = wp.length(d)
                if r < spacing and r > 1.0e-9:
                    s += (0.5 * strength * (spacing - r) / r) * d
    shift[p] = s


@wp.kernel
def apply_separation_kernel(
    shift: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    phi_static: wp.array3d[float],
    phi_body: wp.array3d[float],
    owner: wp.array3d[wp.int32],
    lo: wp.vec3,
    dx: float,
    margin: float,
    # outputs
    q: wp.array[wp.vec3],
):
    """Move the active particles by their separation and keep them out of the solids."""
    p = wp.tid()
    if (flags[p] & ACTIVE) == 0:
        return
    xn, _normal, _moved, _body = push_out(q[p] + shift[p], phi_static, phi_body, owner, lo, dx, margin)
    q[p] = xn


# ----------------------------------------------------------------------------------------------------------------------
# Streams


@wp.kernel
def free_flags_kernel(flags: wp.array[wp.int32], free: wp.array[wp.int32]):
    """1 for each inactive particle."""
    p = wp.tid()
    free[p] = wp.where((flags[p] & ACTIVE) == 0, 1, 0)


@wp.kernel
def free_list_kernel(
    free: wp.array[wp.int32],
    position: wp.array[wp.int32],
    # outputs
    free_list: wp.array[wp.int32],
    free_count: wp.array[wp.int32],
):
    """The inactive particles in index order, and their count; the streams of a step take them from the front."""
    p = wp.tid()
    if free[p] != 0:
        free_list[position[p]] = p
    if p == free.shape[0] - 1:
        free_count[0] = position[p] + free[p]
        free_count[1] = 0


@wp.kernel
def stream_schedule_kernel(
    clock: wp.array[wp.float64],
    dt: wp.float64,
    start: wp.float64,
    stop: wp.float64,
    interval: wp.float64,
    layer_size: int,
    max_layers: int,
    next_layer: wp.array[wp.int64],
    free_count: wp.array[wp.int32],
    # outputs
    plan: wp.array[wp.int64],
):
    """The layers of a stream due over the step from ``clock[0]``, as many whole layers as the free particles hold:
    ``plan = (first layer, layers, first free particle)``."""
    t = clock[0]
    first = next_layer[0]
    # The layers L >= first leaving at start + L interval before the step's end and before the stop.
    end = wp.min(t + dt, stop)
    last = wp.int64(-1)
    if end > start:
        last = wp.int64(wp.ceil((end - start) / interval)) - wp.int64(1)
    due = wp.int64(0)
    if last >= first:
        due = last - first + wp.int64(1)
    due = wp.min(due, wp.int64(max_layers))
    free = wp.int64(free_count[0] - free_count[1]) // wp.int64(layer_size)
    due = wp.min(due, wp.max(free, wp.int64(0)))
    plan[0] = first
    plan[1] = due
    plan[2] = wp.int64(free_count[1])
    next_layer[0] = first + due
    free_count[1] = free_count[1] + int(due) * layer_size


@wp.kernel
def stream_emit_kernel(
    plan: wp.array[wp.int64],
    clock: wp.array[wp.float64],
    dt: wp.float64,
    start: wp.float64,
    interval: wp.float64,
    layer: wp.array[wp.vec3],
    origin: wp.vec3,
    velocity: wp.vec3,
    gravity: wp.array[wp.vec3],
    free_list: wp.array[wp.int32],
    # outputs
    flags: wp.array[wp.int32],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    qd_grad: wp.array[wp.mat33],
):
    """The planned layers join the water at the step's end: each particle where its flight from its layer's point
    puts it, moving as the flight does."""
    t = wp.tid()
    m = layer.shape[0]
    k = wp.int64(t // m)
    if k >= plan[1]:
        return
    p = free_list[int(plan[2]) + t]
    tau = float(clock[0] + dt - (start + wp.float64(plan[0] + k) * interval))
    g = gravity[gravity.shape[0] - 1]
    q[p] = origin + layer[t % m] + tau * velocity + (0.5 * tau * tau) * g
    qd[p] = velocity + tau * g
    qd_grad[p] = wp.mat33(0.0)
    flags[p] = flags[p] | ACTIVE


@wp.kernel
def advance_clock_kernel(clock: wp.array[wp.float64], dt: wp.float64):
    clock[0] = clock[0] + dt


@wp.kernel
def add_body_forces_kernel(impulse: wp.array[wp.spatial_vector], inv_dt: float, body_f: wp.array[wp.spatial_vector]):
    """``body_f += impulse / dt``: the water's last impulse on each body as a constant wrench over the next step."""
    b = wp.tid()
    body_f[b] = body_f[b] + impulse[b] * inv_dt
