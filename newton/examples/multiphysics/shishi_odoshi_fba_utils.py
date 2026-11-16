# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometry of the ``shishi_odoshi_fba`` example: the bamboo tube as a signed distance field, its mass properties, a
closed triangle mesh of it (surface nets) for the MPM collider and the viewer, convex parts for the rigid contacts,
and the static scenery (basin, stone, spout).

Frame: z up, the hinge pin on the world y axis through the origin. The tube lies in the xz plane with its mouth at -x
and its tail at +x; a positive rotation about +y lowers the tail. All lengths in meters.
"""

import numpy as np

REST = np.radians(23.0)  # tail-down rest angle, the tail on the stone
LENGTH, MOUTH = 0.90, 0.39  # tube length; hinge distance from the mouth end
SPEAR = 0.035  # the lower lip runs this far past the mouth end
R_OUT, R_IN = 0.043, 0.023  # outer and inner radius
X_TIP = -MOUTH - SPEAR  # mouth tip (local x)
X_TAIL = LENGTH - MOUTH  # tail end (local x)
X_NODE = -0.03  # bottom of the mouth cavity, the node wall spans [X_NODE, 0]
TAIL_CAVITY = (0.0, X_TAIL - 0.015)  # sealed tail cavity
TUBE_DENSITY = 90.0  # [kg/m^3], dry bamboo wall

# The spear cut: the half-space n . (p - lip) > 0 is removed; it is level at the rest angle.
CUT_N = np.array([-np.sin(REST), 0.0, np.cos(REST)])
CUT_LIP = np.array([X_TIP, 0.0, -R_OUT + 0.004])

GROUND_Z = -0.30
TILT = np.radians(-22.0)  # the mouth-down angle at which the tube lands on the basin's back rim
BASIN_INNER = (-0.46, -0.16, -0.11, 0.11)  # inner x and y range of the basin
BASIN_FLOOR = -0.27  # top of the basin floor

# The spout: a thin pipe sloping down toward +x, its open end above the mouth just behind the spear tip.
SPOUT_END = np.array([-0.355, 0.0, 0.171])
SPOUT_SLOPE = np.radians(22.0)
SPOUT_RADIUS = 0.016


def _capped_cylinder_x(p, radius, x0, x1):
    """Exact signed distance to a capped cylinder along x."""
    r = np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    dx = np.abs(p[..., 0] - 0.5 * (x0 + x1)) - 0.5 * (x1 - x0)
    dr = r - radius
    outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dr, 0.0) ** 2)
    inside = np.minimum(np.maximum(dx, dr), 0.0)
    return outside + inside


def tube_sdf(p, hollow_tail=False):
    """Signed distance (approximate away from the surface) to the tube in its frame; with ``hollow_tail`` the sealed
    tail cavity is carved out, for the mass properties (no water reaches it, so the collider keeps it solid)."""
    d = _capped_cylinder_x(p, R_OUT, X_TIP, X_TAIL)
    d = np.maximum(d, -_capped_cylinder_x(p, R_IN, X_TIP - 0.05, X_NODE))
    if hollow_tail:
        d = np.maximum(d, -_capped_cylinder_x(p, R_IN, *TAIL_CAVITY))
    return np.maximum(d, (p - CUT_LIP) @ CUT_N)


def mouth_cavity_sdf(p):
    """Signed distance to the open mouth cavity, where water can sit, below the cut plane."""
    d = _capped_cylinder_x(p, R_IN, X_TIP - 0.05, X_NODE)
    return np.maximum(d, (p - CUT_LIP) @ CUT_N)


def rot_y(angle):
    """Rotation matrix about +y."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tube_mass_properties(h=0.5e-3):
    """Mass [kg], center of mass and inertia about it [kg m^2] of the tube in its frame, by voxel integration."""
    xs = np.arange(X_TIP + 0.5 * h, X_TAIL, h)
    ys = np.arange(-R_OUT + 0.5 * h, R_OUT, h)
    yy, zz = np.meshgrid(ys, ys, indexing="ij")
    mass, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    dm = TUBE_DENSITY * h**3
    for x0 in range(0, len(xs), 64):
        x = xs[x0 : x0 + 64]
        p = np.stack(np.broadcast_arrays(x[:, None, None], yy[None], zz[None]), axis=-1)
        q = p[tube_sdf(p, hollow_tail=True) < 0.0]
        mass += dm * len(q)
        first += dm * q.sum(axis=0)
        second += dm * q.T @ q
    com = first / mass
    second -= mass * np.outer(com, com)  # sum m (r - c)(r - c)^T
    return mass, com, np.trace(second) * np.eye(3) - second


def surface_nets(sdf, xs, ys, zs):
    """Closed triangle mesh of the zero level set of ``sdf`` sampled on the grid ``xs x ys x zs``, its normals
    pointing out of the negative region: a vertex per grid cell with a sign change, at the mean of its edges'
    crossings, and a quad per grid edge with a sign change, joining its four cells. The field must be positive on
    the grid boundary."""
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    f = sdf(P)
    acc = np.zeros((*(n - 1 for n in f.shape), 3))
    count = np.zeros(acc.shape[:3])

    def edges(axis):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        f0, f1, p0, p1 = f[tuple(lo)], f[tuple(hi)], P[tuple(lo)], P[tuple(hi)]
        change = (f0 < 0.0) != (f1 < 0.0)
        t = np.where(change, f0 / np.where(change, f0 - f1, 1.0), 0.0)
        return change, p0 + t[..., None] * (p1 - p0), f0 < 0.0

    edge_data = [edges(axis) for axis in range(3)]
    for axis, (change, cross, _) in enumerate(edge_data):
        # An edge along ``axis`` at grid index e borders the cells e - d on the two other axes, d in {0, 1}.
        others = [a for a in range(3) if a != axis]
        for da in (0, 1):
            for db in (0, 1):
                src = [slice(None)] * 3
                for a, d in zip(others, (da, db), strict=True):
                    src[a] = slice(d, f.shape[a] - 1 + d)
                c = change[tuple(src)]
                acc += np.where(c[..., None], cross[tuple(src)], 0.0)
                count += c
    has_vertex = count > 0
    vertex_index = -np.ones(count.shape, dtype=np.int64)
    vertex_index[has_vertex] = np.arange(np.count_nonzero(has_vertex))
    vertices = acc[has_vertex] / count[has_vertex][:, None]

    quads = []
    for axis, (change, _, lower_inside) in enumerate(edge_data):
        a, b = [ax for ax in range(3) if ax != axis]
        idx = np.argwhere(change)
        idx = idx[(idx[:, a] >= 1) & (idx[:, a] <= f.shape[a] - 2) & (idx[:, b] >= 1) & (idx[:, b] <= f.shape[b] - 2)]

        def cell(da, db, idx=idx, a=a, b=b):
            c = idx.copy()
            c[:, a] -= da
            c[:, b] -= db
            return vertex_index[tuple(c.T)]

        q = np.stack([cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)], axis=1)
        # (a, b) follows ``axis`` right-handed for the x and z axes, left-handed for y.
        inside = lower_inside[tuple(idx.T)]
        outward = inside if axis != 1 else ~inside
        q[~outward] = q[~outward][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads)
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return vertices.astype(np.float32), triangles.astype(np.int32)


def tube_mesh(h=1.6e-3, h_tail=12e-3):
    """Closed mesh of the tube, its tail solid, in its frame: ``h`` across and along the mouth, coarser along the
    plain tail."""
    pad = 3 * h
    xs = np.concatenate(
        [
            np.arange(X_TIP - pad, 0.03, h),
            np.arange(0.03, X_TAIL - 0.02, h_tail),
            np.arange(X_TAIL - 0.02, X_TAIL + pad, h),
        ]
    )
    ys = np.arange(-R_OUT - pad, R_OUT + pad + 1e-9, h)
    return surface_nets(tube_sdf, xs, ys, ys)


def tube_convex_parts():
    """Point clouds of two convex parts covering the tube's outside: the mouth below the cut, and node and tail."""
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    ring = np.stack([np.zeros_like(phi), R_OUT * np.cos(phi), R_OUT * np.sin(phi)], axis=1)
    mouth = []
    for x in np.linspace(X_TIP, X_NODE, 40):
        points = ring + np.array([x, 0.0, 0.0])
        mouth.append(points[(points - CUT_LIP) @ CUT_N <= 0.0])
    # where the cut plane meets the outer cylinder
    z = R_OUT * np.sin(phi)
    x_cut = X_TIP + (z - CUT_LIP[2]) / np.tan(REST)
    on = (x_cut >= X_TIP) & (x_cut <= X_NODE)
    mouth.append(np.stack([x_cut[on], R_OUT * np.cos(phi[on]), z[on]], axis=1))
    tail = np.concatenate([ring + np.array([x, 0.0, 0.0]) for x in (X_NODE - 0.01, X_TAIL)])
    return np.concatenate(mouth).astype(np.float32), tail.astype(np.float32)


def stone():
    """Center and points of the stone under the tail, an ellipsoid whose top touches the tail's lower corner at the
    rest angle: 162 points spread over it, of which its convex hull is made."""
    tail_x = X_TAIL * np.cos(REST) - R_OUT * np.sin(REST)
    tail_corner_z = -(X_TAIL * np.sin(REST) + R_OUT * np.cos(REST))
    radii = np.array([0.12, 0.09, 0.06])
    center = np.array([tail_x + 0.03, 0.0, tail_corner_z - radii[2]])
    k = np.arange(162) + 0.5
    polar, azimuth = np.arccos(1.0 - 2.0 * k / 162), np.pi * (1.0 + 5.0**0.5) * k
    sphere = np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=1)
    # the top point exactly, where the tail rests
    sphere = np.vstack([sphere, (0.0, 0.0, 1.0)])
    return center, (sphere * radii).astype(np.float32)


def basin_boxes():
    """The basin under the mouth as boxes ``(center, half extents)``: its floor and four walls, the back wall (+x) a
    rim that stops the tube at the tilt angle."""
    x0, x1, y0, y1 = BASIN_INNER
    t = 0.03  # wall thickness
    rim = -0.20  # top of the front and side walls
    # The back wall's inner top edge (x = x1) meets the tube's underside at the tilt angle.
    s = (x1 + R_OUT * np.sin(TILT)) / np.cos(TILT)
    back_top = -s * np.sin(TILT) - R_OUT * np.cos(TILT)
    xm, ym, wx, wy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    floor = 0.5 * (GROUND_Z + BASIN_FLOOR), 0.5 * (BASIN_FLOOR - GROUND_Z)
    wall = 0.5 * (GROUND_Z + rim), 0.5 * (rim - GROUND_Z)
    back = 0.5 * (GROUND_Z + back_top), 0.5 * (back_top - GROUND_Z)
    return [
        ((xm, ym, floor[0]), (wx + t, wy + t, floor[1])),
        ((x0 - 0.5 * t, ym, wall[0]), (0.5 * t, wy + t, wall[1])),
        ((x1 + 0.5 * t, ym, back[0]), (0.5 * t, wy + t, back[1])),
        ((xm, y0 - 0.5 * t, wall[0]), (wx, 0.5 * t, wall[1])),
        ((xm, y1 + 0.5 * t, wall[0]), (wx, 0.5 * t, wall[1])),
    ]


def prefill_points(spacing, level, rng):
    """Water particles filling the mouth cavity at rest below the world height ``level``, a ``spacing`` apart and
    kept half a spacing off the walls, in the world frame with the tube at its rest angle."""
    s = spacing
    xs = np.arange(X_TIP + 0.5 * s, X_NODE, s)
    ys = np.arange(-R_IN + 0.5 * s, R_IN, s)
    p = np.stack(np.meshgrid(xs, ys, ys, indexing="ij"), axis=-1).reshape(-1, 3)
    p = p + rng.uniform(-0.15 * s, 0.15 * s, p.shape)
    p = p[mouth_cavity_sdf(p) < -0.5 * s]
    world = p @ rot_y(REST).T
    return world[world[:, 2] < level]
