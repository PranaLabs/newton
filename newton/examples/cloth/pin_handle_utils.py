# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pin handles and a surface intersection check of the ``shirt_pins_fba`` and ``bunny_ears_fba`` examples."""

from itertools import pairwise

import numpy as np
import warp as wp


def keyframed_handles(time, keys):
    """Handle positions [m] at ``time`` [s] between the keys ``(time, positions, name)``, eased so that the handles
    start and stop at rest at every key."""
    for (t0, p0, _), (t1, p1, _) in pairwise(keys):
        if time <= t1:
            u = np.clip((time - t0) / (t1 - t0), 0.0, 1.0)
            s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
            return (1.0 - s) * np.asarray(p0) + s * np.asarray(p1)
    return np.asarray(keys[-1][1])


class PinHandles:
    """Groups of particles that follow two handles: three rows along the axes hold each particle at its rest offset
    from its group's handle, through :meth:`~newton.solvers.SolverFBA.set_particle_constraints`."""

    def __init__(self, rest, groups):
        self.indices = np.concatenate(groups).astype(np.int32)
        if len(self.indices) > 21 or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("the groups hold at most 21 distinct particles, the 64 exact rows of the lite NSN")
        self.centers = np.array([rest[g].mean(0) for g in groups])
        self.offsets = np.concatenate([rest[g] - c for g, c in zip(groups, self.centers, strict=True)])
        self.which = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        self.directions = np.tile(np.eye(3), (len(self.indices), 1))
        self.particles = np.repeat(self.indices, 3)[:, None]
        self.coefficients = np.ones((3 * len(self.indices), 1))
        self.targets = rest[self.indices].copy()
        self.positions = self.centers.copy()
        self.colors = wp.array([[0.12, 0.85, 1.0], [1.0, 0.35, 0.12]], dtype=wp.vec3)
        self.points = wp.array(self.positions, dtype=wp.vec3)
        self.trail = []

    def apply(self, solver, centers):
        self.positions = np.asarray(centers)
        self.targets = self.positions[self.which] + self.offsets
        solver.set_particle_constraints(
            self.particles, self.coefficients, self.targets.reshape(-1), directions=self.directions
        )

    def render(self, viewer, radius):
        self.points.assign(self.positions.astype(np.float32))
        viewer.log_points("/pin_handles", self.points, radii=wp.full(2, radius, dtype=wp.float32), colors=self.colors)
        self.trail.append(self.positions.copy())
        self.trail = self.trail[-90:]
        if len(self.trail) > 1:
            p = np.asarray(self.trail)
            viewer.log_lines(
                "/pin_trails",
                wp.array(p[:-1].reshape(-1, 3), dtype=wp.vec3),
                wp.array(p[1:].reshape(-1, 3), dtype=wp.vec3),
                colors=(0.45, 0.65, 0.72),
            )

    def error(self, positions):
        return float(np.linalg.norm(positions[self.indices] - self.targets, axis=1).max())


@wp.kernel
def _count_surface_intersections(
    mesh: wp.uint64,
    positions: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    count: wp.array[wp.int32],
):
    """Count unique intersecting triangle pairs, excluding triangles sharing a vertex."""
    i = wp.tid()
    a, b, c = indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]
    p, q, r = positions[a], positions[b], positions[c]
    lo = wp.min(wp.min(p, q), r)
    hi = wp.max(wp.max(p, q), r)
    query = wp.mesh_query_aabb(mesh, lo, hi)
    for j in query:
        if j <= i:
            continue
        d, e, f = indices[3 * j], indices[3 * j + 1], indices[3 * j + 2]
        if a == d or a == e or a == f or b == d or b == e or b == f or c == d or c == e or c == f:
            continue
        # Warp's predicate uses a fixed epsilon on unnormalized plane distances.
        # Center and normalize the pair to avoid false hits on centimeter triangles.
        u, v, w = positions[d], positions[e], positions[f]
        scale = wp.float64(1.0) / wp.float64(
            wp.max(wp.max(wp.length(q - p), wp.length(r - p)), wp.max(wp.length(v - u), wp.length(w - u)))
        )
        origin = wp.vec3d(p)
        if wp.intersect_tri_tri(
            wp.vec3d(0.0),
            (wp.vec3d(q) - origin) * scale,
            (wp.vec3d(r) - origin) * scale,
            (wp.vec3d(u) - origin) * scale,
            (wp.vec3d(v) - origin) * scale,
            (wp.vec3d(w) - origin) * scale,
        ):
            wp.atomic_add(count, 0, 1)


class SurfaceIntersectionCheck:
    """Geometric validation independent of the solver's self-contact settings and candidate count."""

    def __init__(self, model):
        self.positions = wp.clone(model.particle_q)
        self.indices = model.tri_indices.flatten()
        self.mesh = wp.Mesh(points=self.positions, indices=self.indices)
        self.count = wp.zeros(1, dtype=wp.int32, device=model.device)
        self.triangle_count = model.tri_count

    def intersections(self, positions):
        wp.copy(self.positions, positions)
        self.mesh.refit()
        self.count.zero_()
        wp.launch(
            _count_surface_intersections,
            dim=self.triangle_count,
            inputs=[self.mesh.id, self.positions, self.indices],
            outputs=[self.count],
            device=self.positions.device,
        )
        return int(self.count.numpy()[0])
