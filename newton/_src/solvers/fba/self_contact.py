# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contacts of a cloth with itself for :class:`~newton.solvers.SolverFBA`."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry.kernels import compute_tri_aabbs, triangle_closest_point
from ...sim import Model


@wp.func
def _in_ring(ring_offsets: wp.array[wp.int32], ring: wp.array[wp.int32], v: wp.int32, u: wp.int32) -> bool:
    """Whether ``u`` is among the sorted particles within the rings of ``v``."""
    lo = ring_offsets[v]
    hi = ring_offsets[v + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        w = ring[mid]
        if w == u:
            return True
        if w < u:
            lo = mid + 1
        else:
            hi = mid
    return False


@wp.kernel
def vertex_face_contacts_kernel(
    q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    bvh: wp.uint64,
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    rest_radius: float,
    margin: float,
    count: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
):
    """The nearest triangle of each particle within ``margin``, outside its rings and ``rest_radius`` from it
    at rest."""
    v = wp.tid()
    p = q[v]
    query = wp.bvh_query_aabb(bvh, p - wp.vec3(margin), p + wp.vec3(margin))
    t = wp.int32(0)
    best = margin
    best_t = wp.int32(-1)
    best_bary = wp.vec3(0.0)
    best_point = wp.vec3(0.0)
    while wp.bvh_query_next(query, t):
        a = tri_indices[t, 0]
        b = tri_indices[t, 1]
        c = tri_indices[t, 2]
        if (
            _in_ring(ring_offsets, ring, v, a)
            or _in_ring(ring_offsets, ring, v, b)
            or _in_ring(ring_offsets, ring, v, c)
        ):
            continue
        if rest_radius > 0.0:
            rest_point, _rest_bary, _rest_feature = triangle_closest_point(rest[a], rest[b], rest[c], rest[v])
            if wp.length(rest[v] - rest_point) < rest_radius:
                continue
        point, bary, _feature = triangle_closest_point(q[a], q[b], q[c], p)
        d = wp.length(p - point)
        if d < best:
            best = d
            best_t = t
            best_bary = bary
            best_point = point
    if best_t < 0 or best < 1.0e-6 * margin:
        return
    k = wp.atomic_add(count, 0, 1)
    if k < particles.shape[0]:
        particles[k, 0] = v
        alpha[k, 0] = 1.0
        for i in range(3):
            particles[k, i + 1] = tri_indices[best_t, i]
            alpha[k, i + 1] = -best_bary[i]
        normal[k] = (p - best_point) / best


@wp.kernel
def vertex_vertex_contacts_kernel(
    q: wp.array[wp.vec3],
    grid: wp.uint64,
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    rest_radius: float,
    margin: float,
    count: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
):
    """Each pair of particles within ``margin``, outside each other's rings and ``rest_radius`` apart at rest,
    once."""
    a = wp.tid()
    p = q[a]
    query = wp.hash_grid_query(grid, p, margin)
    b = int(0)
    while wp.hash_grid_query_next(query, b):
        if b <= a or _in_ring(ring_offsets, ring, a, b) or wp.length(rest[a] - rest[b]) < rest_radius:
            continue
        d = wp.length(p - q[b])
        if d >= margin or d < 1.0e-6 * margin:
            continue
        k = wp.atomic_add(count, 0, 1)
        if k < particles.shape[0]:
            particles[k, 0] = a
            particles[k, 1] = b
            particles[k, 2] = -1
            particles[k, 3] = -1
            alpha[k, 0] = 1.0
            alpha[k, 1] = -1.0
            alpha[k, 2] = 0.0
            alpha[k, 3] = 0.0
            normal[k] = (p - q[b]) / d


def particle_rings(tri_indices: np.ndarray, particle_count: int, rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted CSR of the particles within ``rings`` mesh edges of each particle, the particle included."""
    neighbors = [set() for _ in range(particle_count)]
    for a, b, c in tri_indices:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    reached = []
    for v in range(particle_count):
        reach, front = {v}, {v}
        for _ in range(rings):
            front = set().union(*(neighbors[u] for u in front)) - reach
            reach |= front
        reached.append(sorted(reach))
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in reached])]).astype(np.int32)
    return offsets, np.concatenate(reached).astype(np.int32)


class SelfContact:
    """Contacts of the model's triangles with themselves, found on the positions at the start of a step.

    With ``mode="vertex_face"``, each particle takes the nearest triangle within ``margin``; with
    ``mode="vertex_vertex"``, every pair of particles within ``margin`` is a contact. A contact acts along
    the direction between its two points at detection: the particle and its closest point on the triangle,
    or the two particles. Particles within ``rings`` mesh edges of each other, and triangles with such a
    particle, never collide, nor do a particle and a triangle or particle closer than ``rest_radius`` in the
    model's particle positions.

    Args:
        model: Model with the triangles.
        mode: ``"vertex_face"`` or ``"vertex_vertex"``.
        margin: Detection distance [m].
        rings: Mesh edges within which particles do not collide.
        rest_radius: Distance in the model's particle positions within which nothing collides [m].
    """

    def __init__(self, model: Model, mode: str, margin: float, rings: int = 2, rest_radius: float = 0.0):
        if mode not in ("vertex_face", "vertex_vertex"):
            raise ValueError(f"self contact mode {mode!r} is not 'vertex_face' or 'vertex_vertex'")
        self.mode, self.margin, self.device = mode, float(margin), model.device
        self.rest, self.rest_radius = wp.clone(model.particle_q), float(rest_radius)
        rings_csr = particle_rings(model.tri_indices.numpy(), model.particle_count, rings)
        self.ring_offsets, self.ring = (wp.array(a, dtype=wp.int32, device=self.device) for a in rings_csr)
        if mode == "vertex_face":
            self.tri_indices = model.tri_indices
            self.lower = wp.zeros(model.tri_count, dtype=wp.vec3, device=self.device)
            self.upper = wp.zeros(model.tri_count, dtype=wp.vec3, device=self.device)
            self._bounds(model.particle_q)
            self.bvh = wp.Bvh(self.lower, self.upper)
        else:
            self.grid = wp.HashGrid(128, 128, 128, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._allocate(model.particle_count * (1 if mode == "vertex_face" else 8))

    def _bounds(self, q: wp.array) -> None:
        inputs = [q, self.tri_indices]
        wp.launch(compute_tri_aabbs, self.tri_indices.shape[0], inputs, [self.lower, self.upper], device=self.device)

    def _allocate(self, capacity: int) -> None:
        self.particles = wp.empty((capacity, 4), dtype=wp.int32, device=self.device)
        self.alpha = wp.empty((capacity, 4), dtype=wp.float32, device=self.device)
        self.normal = wp.empty(capacity, dtype=wp.vec3, device=self.device)

    def detect(self, q: wp.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Particles and coefficients, shape ``(C, 4)`` with particle ``-1`` in unused slots, and normals,
        shape ``(C, 3)``, of the contacts at ``q``, sorted by their particles."""
        if self.mode == "vertex_face":
            self._bounds(q)
            self.bvh.refit()
            kernel, scene = vertex_face_contacts_kernel, [self.tri_indices, self.bvh.id]
        else:
            self.grid.build(q, self.margin)
            kernel, scene = vertex_vertex_contacts_kernel, [self.grid.id]
        while True:
            self.count.zero_()
            outputs = [self.count, self.particles, self.alpha, self.normal]
            inputs = [q, *scene, self.ring_offsets, self.ring, self.rest, self.rest_radius, self.margin]
            wp.launch(kernel, q.shape[0], inputs, outputs, device=self.device)
            n = int(self.count.numpy()[0])
            if n <= self.particles.shape[0]:
                break
            self._allocate(2 * n)
        particles, alpha, normal = self.particles.numpy()[:n], self.alpha.numpy()[:n], self.normal.numpy()[:n]
        order = np.lexsort((particles[:, 1], particles[:, 0]))
        return particles[order], alpha[order], normal[order]
