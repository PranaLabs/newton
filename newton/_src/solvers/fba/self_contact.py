# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contacts of cloth and solid surfaces with themselves for :class:`~newton.solvers.SolverFBA`."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry.kernels import compute_tri_aabbs, triangle_closest_point
from ...sim import Model
from .kernels import host_head

# Threads that share the candidate triangles of one particle.
_LANES = wp.constant(32)


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
def triangle_candidates_kernel(
    q: wp.array[wp.vec3],
    query_mask: wp.array[wp.int32],
    bvh: wp.uint64,
    query_radius: float,
    candidates: wp.array2d[wp.int32],
    candidate_count: wp.array[wp.int32],
    count: wp.array[wp.int32],
):
    """The triangles whose bounds come within ``query_radius`` of each particle along every axis, in the order of the
    query, as many as fit; ``count[1]`` takes the most of any particle, and a particle outside ``query_mask`` none.
    The search runs apart from the tests of :func:`vertex_face_contacts_kernel`, whose registers would otherwise spill
    in its loop."""
    v = wp.tid()
    if query_mask[v] == 0:
        candidate_count[v] = 0
        return
    p = q[v]
    query = wp.bvh_query_aabb(bvh, p - wp.vec3(query_radius), p + wp.vec3(query_radius))
    t = wp.int32(0)
    k = wp.int32(0)
    while wp.bvh_query_next(query, t):
        if k < candidates.shape[1]:
            candidates[v, k] = t
        k += 1
    candidate_count[v] = k
    wp.atomic_max(count, 1, k)


@wp.kernel
def vertex_face_contacts_kernel(
    q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    candidates: wp.array2d[wp.int32],
    candidate_count: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    rest_radius: float,
    query_radius: float,
    count: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    found: wp.array[wp.int32],
):
    """The nearest triangle of each particle within ``query_radius``, outside its rings and ``rest_radius`` from it
    at rest, the first of equals among its candidates, in the particle's slot; ``found`` marks the particles
    that have one. The threads of a block, ``_LANES`` on CUDA and one on the CPU, take the candidates
    of one particle in turn, and test the rings and rest distance of only those nearer than their nearest yet."""
    v, lane = wp.tid()
    p = q[v]
    best = query_radius
    best_k = wp.int32(2147483647)
    best_t = wp.int32(-1)
    best_bary = wp.vec3(0.0)
    best_point = wp.vec3(0.0)
    for k in range(lane, wp.min(candidate_count[v], candidates.shape[1]), wp.block_dim()):
        t = candidates[v, k]
        a = tri_indices[t, 0]
        b = tri_indices[t, 1]
        c = tri_indices[t, 2]
        point, bary, _feature = triangle_closest_point(q[a], q[b], q[c], p)
        d = wp.length(p - point)
        if d >= best:
            continue
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
        best = d
        best_k = k
        best_t = t
        best_bary = bary
        best_point = point
    nearest = wp.tile_min(wp.tile(best))[0]
    first = wp.tile_min(wp.tile(wp.where(best == nearest, best_k, wp.int32(2147483647))))[0]
    hit = first != wp.int32(2147483647) and nearest >= 1.0e-6 * query_radius
    if lane == 0:
        found[v] = wp.where(hit, 1, 0)
    if not hit or best_k != first:
        return
    wp.atomic_add(count, 0, 1)
    particles[v, 0] = v
    alpha[v, 0] = 1.0
    for i in range(3):
        particles[v, i + 1] = tri_indices[best_t, i]
        alpha[v, i + 1] = -best_bary[i]
    normal[v] = (p - best_point) / best


@wp.kernel
def compact_contacts_kernel(
    found: wp.array[wp.int32],
    position: wp.array[wp.int32],  # exclusive scan of ``found``
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    # outputs
    out_particles: wp.array2d[wp.int32],
    out_alpha: wp.array2d[wp.float32],
    out_normal: wp.array[wp.vec3],
):
    """Gather the contacts in the slots of the particles that have one, in the order of the particles."""
    v = wp.tid()
    if found[v] == 0:
        return
    k = position[v]
    for i in range(4):
        out_particles[k, i] = particles[v, i]
        out_alpha[k, i] = alpha[v, i]
    out_normal[k] = normal[v]


@wp.kernel
def vertex_vertex_contacts_kernel(
    q: wp.array[wp.vec3],
    grid: wp.uint64,
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    rest_radius: float,
    query_radius: float,
    count: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
):
    """Each pair of particles within ``query_radius``, outside each other's rings and ``rest_radius`` apart at rest,
    once."""
    a = wp.tid()
    p = q[a]
    query = wp.hash_grid_query(grid, p, query_radius)
    b = int(0)
    while wp.hash_grid_query_next(query, b):
        if b <= a or _in_ring(ring_offsets, ring, a, b) or wp.length(rest[a] - rest[b]) < rest_radius:
            continue
        d = wp.length(p - q[b])
        if d >= query_radius or d < 1.0e-6 * query_radius:
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


def particle_rings(
    tri_indices: np.ndarray, particle_count: int, rings: int, tet_indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted CSR of the particles within ``rings`` mesh edges of each particle, the particle included; the edges of
    ``tet_indices``, when given, count as mesh edges too."""
    neighbors = [set() for _ in range(particle_count)]
    for a, b, c in tri_indices:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    if tet_indices is not None:
        for tet in np.asarray(tet_indices).reshape(-1, 4).tolist():
            for v in tet:
                neighbors[v].update(tet)
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

    With ``mode="vertex_face"``, each particle takes the nearest triangle within ``query_radius``; with
    ``mode="vertex_vertex"``, every pair of particles within ``query_radius`` is a contact. A contact acts along
    the direction between its two points at detection: the particle and its closest point on the triangle,
    or the two particles. Particles within ``rings`` mesh edges of each other, and triangles with such a
    particle, never collide, nor do a particle and a triangle or particle closer than ``rest_radius`` in the
    model's particle positions.

    Solids collide through their surface triangles: with ``mode="vertex_face"``, the particles of tetrahedra that
    lie on no triangle take no contact. In both modes the tetrahedron edges count as mesh edges for the rings, since
    the two faces of a thin solid share its tetrahedra while lying far apart on its surface.

    Args:
        model: Model with the triangles.
        mode: ``"vertex_face"`` or ``"vertex_vertex"``.
        query_radius: Detection distance [m], the contact margin plus the gap.
        rings: Mesh edges within which particles do not collide.
        rest_radius: Distance in the model's particle positions within which nothing collides [m].
    """

    def __init__(self, model: Model, mode: str, query_radius: float, rings: int = 2, rest_radius: float = 0.0):
        if mode not in ("vertex_face", "vertex_vertex"):
            raise ValueError(f"self contact mode {mode!r} is not 'vertex_face' or 'vertex_vertex'")
        self.mode, self.query_radius, self.device = mode, float(query_radius), model.device
        self.rest, self.rest_radius = wp.clone(model.particle_q), float(rest_radius)
        triangles = model.tri_indices.numpy()
        tets = model.tet_indices.numpy().reshape(-1, 4) if model.tet_count else None
        rings_csr = particle_rings(triangles, model.particle_count, rings, tets)
        self.ring_offsets, self.ring = (wp.array(a, dtype=wp.int32, device=self.device) for a in rings_csr)
        if mode == "vertex_face":
            # The particles that look for a nearest triangle: all but the interior particles of solids.
            query_mask = np.ones(model.particle_count, dtype=np.int32)
            if tets is not None:
                query_mask[tets.ravel()] = 0
                query_mask[triangles.ravel()] = 1
            self.query_mask = wp.array(query_mask, dtype=wp.int32, device=self.device)
            self.tri_indices = model.tri_indices
            self.lower = wp.zeros(model.tri_count, dtype=wp.vec3, device=self.device)
            self.upper = wp.zeros(model.tri_count, dtype=wp.vec3, device=self.device)
            self._bounds(model.particle_q)
            self.bvh = wp.Bvh(self.lower, self.upper)
            self.candidates = wp.empty((model.particle_count, 64), dtype=wp.int32, device=self.device)
            self.candidate_count = wp.empty(model.particle_count, dtype=wp.int32, device=self.device)
            # Particles with a contact, their places among the contacts, and the contacts in that order.
            self.found = wp.empty(model.particle_count, dtype=wp.int32, device=self.device)
            self.position = wp.empty(model.particle_count, dtype=wp.int32, device=self.device)
            self.found_particles = wp.empty((model.particle_count, 4), dtype=wp.int32, device=self.device)
            self.found_alpha = wp.empty((model.particle_count, 4), dtype=wp.float32, device=self.device)
            self.found_normal = wp.empty(model.particle_count, dtype=wp.vec3, device=self.device)
        else:
            self.grid = wp.HashGrid(128, 128, 128, device=self.device)
        # Contacts found, and the most candidate triangles of a particle.
        self.count = wp.zeros(2, dtype=wp.int32, device=self.device)
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
        n = self.find(q)
        if self.mode == "vertex_face":
            return host_head(self.found_particles, n), host_head(self.found_alpha, n), host_head(self.found_normal, n)
        particles, alpha, normal = host_head(self.particles, n), host_head(self.alpha, n), host_head(self.normal, n)
        order = np.lexsort((particles[:, 1], particles[:, 0]))
        return particles[order], alpha[order], normal[order]

    def find(self, q: wp.array) -> int:
        """Find the contacts at ``q`` and return their number. With ``mode="vertex_face"`` they stay on the
        device in ``found_particles``, ``found_alpha`` and ``found_normal``, in the order of their particles."""
        if self.mode == "vertex_face":
            self._bounds(q)
            self.bvh.refit()
        else:
            self.grid.build(q, self.query_radius)
        while True:
            self.count.zero_()
            outputs = [self.count, self.particles, self.alpha, self.normal]
            filters = [self.ring_offsets, self.ring, self.rest, self.rest_radius, self.query_radius]
            if self.mode == "vertex_face":
                found = [self.candidates, self.candidate_count]
                inputs = [q, self.query_mask, self.bvh.id, self.query_radius]
                wp.launch(triangle_candidates_kernel, q.shape[0], inputs, [*found, self.count], device=self.device)
                inputs = [q, self.tri_indices, *found, *filters]
                kernel, outputs = vertex_face_contacts_kernel, [*outputs, self.found]
                wp.launch_tiled(
                    kernel, dim=[q.shape[0]], inputs=inputs, outputs=outputs, block_dim=_LANES, device=self.device
                )
                wp.utils.array_scan(self.found, self.position, inclusive=False)
                inputs = [self.found, self.position, self.particles, self.alpha, self.normal]
                outputs = [self.found_particles, self.found_alpha, self.found_normal]
                wp.launch(compact_contacts_kernel, q.shape[0], inputs, outputs, device=self.device)
            else:
                inputs = [q, self.grid.id, *filters]
                wp.launch(vertex_vertex_contacts_kernel, q.shape[0], inputs, outputs, device=self.device)
            n, most = (int(x) for x in self.count.numpy())
            if self.mode == "vertex_face" and most > self.candidates.shape[1]:
                self.candidates = wp.empty((q.shape[0], 2 * most), dtype=wp.int32, device=self.device)
            elif n > self.particles.shape[0]:
                self._allocate(2 * n)
            else:
                return n
