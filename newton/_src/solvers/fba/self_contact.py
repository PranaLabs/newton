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
    groups: wp.array[wp.int32],
    group_margins: wp.array[wp.float32],
    margin: float,
    gap: float,
    has_groups: int,
    count: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    found: wp.array[wp.int32],
    nearest_distance: wp.array[float],
):
    """The nearest triangle of each particle within ``query_radius``, outside its rings and ``rest_radius`` from it
    at rest, the first of equals among its candidates, in the particle's slot; ``found`` marks the particles
    that have one. The threads of a block, ``_LANES`` on CUDA and one on the CPU, take the candidates
    of one particle in turn, and test the rings and rest distance of only those nearer than their nearest yet.

    With ``has_groups``, a particle and a triangle of two groups keep ``margin`` apart, and of one group the group's
    ``group_margins``, or do not collide where it is 0: the nearest triangle is the one whose distance less that margin
    is least, below ``gap``."""
    v, lane = wp.tid()
    p = q[v]
    best = query_radius
    if has_groups:
        best = gap
    best_k = wp.int32(2147483647)
    best_t = wp.int32(-1)
    best_bary = wp.vec3(0.0)
    best_point = wp.vec3(0.0)
    best_distance = query_radius
    for k in range(lane, wp.min(candidate_count[v], candidates.shape[1]), wp.block_dim()):
        t = candidates[v, k]
        a = tri_indices[t, 0]
        b = tri_indices[t, 1]
        c = tri_indices[t, 2]
        point, bary, _feature = triangle_closest_point(q[a], q[b], q[c], p)
        d = wp.length(p - point)
        key = d
        if has_groups:
            pair = margin
            if groups[v] == groups[a]:
                pair = group_margins[groups[v]]
                if pair <= 0.0:
                    continue
            key = d - pair
        if key >= best:
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
        best = key
        best_k = k
        best_t = t
        best_bary = bary
        best_point = point
        best_distance = d
    nearest = wp.tile_min(wp.tile(best))[0]
    first = wp.tile_min(wp.tile(wp.where(best == nearest, best_k, wp.int32(2147483647))))[0]
    distance = nearest
    if has_groups:
        distance = wp.tile_min(wp.tile(wp.where(best_k == first, best_distance, query_radius)))[0]
    hit = first != wp.int32(2147483647) and distance >= 1.0e-6 * query_radius
    if lane == 0:
        found[v] = wp.where(hit, 1, 0)
        nearest_distance[v] = wp.where(hit, distance, query_radius)
    if not hit or best_k != first:
        return
    wp.atomic_add(count, 0, 1)
    particles[v, 0] = v
    alpha[v, 0] = 1.0
    for i in range(3):
        particles[v, i + 1] = tri_indices[best_t, i]
        alpha[v, i + 1] = -best_bary[i]
    normal[v] = (p - best_point) / best_distance


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


@wp.kernel
def edge_bounds_kernel(
    q: wp.array[wp.vec3],
    edges: wp.array[wp.vec2i],
    # outputs
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    """The bounds of each edge."""
    e = wp.tid()
    a = q[edges[e][0]]
    b = q[edges[e][1]]
    lower[e] = wp.min(a, b)
    upper[e] = wp.max(a, b)


@wp.kernel
def edge_edge_contacts_kernel(
    q: wp.array[wp.vec3],
    edges: wp.array[wp.vec2i],
    bvh: wp.uint64,
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    rest_radius: float,
    query_radius: float,
    nearest_distance: wp.array[float],
    nearer: float,
    count: wp.array[wp.int32],
    keys: wp.array[wp.int64],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
):
    """Each pair of edges within ``query_radius`` whose closest points lie inside both edges, nearer than ``nearer``
    times the nearest triangles of their particles, with no particles within each other's rings and ``rest_radius``
    apart at rest, once: a contact on their four particles with the weights of the two closest points, keyed by the
    pair. Closest points at a particle are the vertex-face contacts'."""
    ea = wp.tid()
    a0 = edges[ea][0]
    a1 = edges[ea][1]
    pa0 = q[a0]
    pa1 = q[a1]
    reach_a = wp.min(nearest_distance[a0], nearest_distance[a1])
    # Edges shorter than a thousandth of the query radius count as points.
    eps = 1.0e-6 * query_radius * query_radius
    query = wp.bvh_query_aabb(bvh, wp.min(pa0, pa1) - wp.vec3(query_radius), wp.max(pa0, pa1) + wp.vec3(query_radius))
    eb = wp.int32(0)
    # The cheap geometric tests reject most of the candidates first; the ring and rest tests run on the few left.
    while wp.bvh_query_next(query, eb):
        if eb <= ea:
            continue
        b0 = edges[eb][0]
        b1 = edges[eb][1]
        pb0 = q[b0]
        pb1 = q[b1]
        st = wp.closest_point_edge_edge(pa0, pa1, pb0, pb1, eps)
        s = st[0]
        u = st[1]
        d = st[2]
        if d >= query_radius or d < 1.0e-6 * query_radius:
            continue
        if s <= 1.0e-3 or s >= 1.0 - 1.0e-3 or u <= 1.0e-3 or u >= 1.0 - 1.0e-3:
            continue
        # Where the four particles keep from the other side at least as far as the edges keep from each other, the
        # vertex-face contacts already hold the edges apart.
        reach = wp.min(reach_a, wp.min(nearest_distance[b0], nearest_distance[b1]))
        if d > nearer * reach:
            continue
        if (
            _in_ring(ring_offsets, ring, a0, b0)
            or _in_ring(ring_offsets, ring, a0, b1)
            or _in_ring(ring_offsets, ring, a1, b0)
            or _in_ring(ring_offsets, ring, a1, b1)
        ):
            continue
        if rest_radius > 0.0:
            rest_st = wp.closest_point_edge_edge(rest[a0], rest[a1], rest[b0], rest[b1], eps)
            if rest_st[2] < rest_radius:
                continue
        k = wp.atomic_add(count, 0, 1)
        if k < particles.shape[0]:
            keys[k] = wp.int64(ea) * wp.int64(edges.shape[0]) + wp.int64(eb)
            particles[k, 0] = a0
            particles[k, 1] = a1
            particles[k, 2] = b0
            particles[k, 3] = b1
            alpha[k, 0] = 1.0 - s
            alpha[k, 1] = s
            alpha[k, 2] = u - 1.0
            alpha[k, 3] = -u
            normal[k] = (pa0 + s * (pa1 - pa0) - (pb0 + u * (pb1 - pb0))) / d


@wp.kernel
def iota_kernel(values: wp.array[wp.int32]):
    """``values[i] = i``."""
    i = wp.tid()
    values[i] = i


@wp.kernel
def gather_edge_contacts_kernel(
    order: wp.array[wp.int32],  # the edge contacts in the order of their keys
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    start: int,
    # outputs
    out_particles: wp.array2d[wp.int32],
    out_alpha: wp.array2d[wp.float32],
    out_normal: wp.array[wp.vec3],
):
    """Append the edge-edge contacts after the vertex-face contacts, in the order of their keys."""
    k = wp.tid()
    i = order[k]
    for j in range(4):
        out_particles[start + k, j] = particles[i, j]
        out_alpha[start + k, j] = alpha[i, j]
    out_normal[start + k] = normal[i]


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

    With ``mode="vertex_face"``, each particle takes the nearest triangle within ``query_radius``, and with
    ``edge_edge`` each pair of edges within it whose closest points lie inside both edges is a contact too; with
    ``mode="vertex_vertex"``, every pair of particles within ``query_radius`` is a contact. A contact acts along
    the direction between its two points at detection: the particle and its closest point on the triangle,
    the closest points of the two edges, or the two particles. Particles within ``rings`` mesh edges of each
    other, and triangles with such a
    particle, never collide, nor do a particle and a triangle or particle closer than ``rest_radius`` in the
    model's particle positions.

    Solids collide through their surface triangles: with ``mode="vertex_face"``, the particles of tetrahedra that
    lie on no triangle take no contact. In both modes the tetrahedron edges count as mesh edges for the rings, since
    the two faces of a thin solid share its tetrahedra while lying far apart on its surface.

    :meth:`set_groups` gives the vertex-face contacts a margin within each group of particles, or none.

    Args:
        model: Model with the triangles.
        mode: ``"vertex_face"`` or ``"vertex_vertex"``.
        query_radius: Detection distance [m], the contact margin plus the gap.
        rings: Mesh edges within which particles do not collide.
        rest_radius: Distance in the model's particle positions within which nothing collides [m].
        edge_edge: With ``mode="vertex_face"``, also find edge-edge contacts.
        edge_edge_ratio: An edge pair takes a contact only where its edges come nearer than this times the nearest
            triangles of its particles: between flat layers the vertex-face contacts hold the edges apart, and only
            ridges, such as creases, bring edges nearer than their particles.
    """

    def __init__(
        self,
        model: Model,
        mode: str,
        query_radius: float,
        rings: int = 2,
        rest_radius: float = 0.0,
        edge_edge: bool = False,
        edge_edge_ratio: float = 0.9,
    ):
        if mode not in ("vertex_face", "vertex_vertex"):
            raise ValueError(f"self contact mode {mode!r} is not 'vertex_face' or 'vertex_vertex'")
        if edge_edge and mode != "vertex_face":
            raise ValueError("edge-edge self contacts need mode='vertex_face'")
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
            # Each particle's distance to its nearest triangle, the query radius when it has none.
            self.nearest_distance = wp.empty(model.particle_count, dtype=float, device=self.device)
            # The groups of the particles and their margins, from set_groups.
            self.groups, self.group_margins, self.margin, self.gap = None, None, 0.0, 0.0
            self.within_groups = True
        else:
            self.grid = wp.HashGrid(128, 128, 128, device=self.device)
        # Contacts found, and the most candidate triangles of a particle.
        self.count = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._allocate(model.particle_count * (1 if mode == "vertex_face" else 8))
        self.edge_edge = bool(edge_edge)
        if self.edge_edge:
            tris = model.tri_indices.numpy()
            edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
            self.edges = wp.array(np.unique(edges, axis=0).astype(np.int32), dtype=wp.vec2i, device=self.device)
            n_edges = self.edges.shape[0]
            self.edge_lower = wp.zeros(n_edges, dtype=wp.vec3, device=self.device)
            self.edge_upper = wp.zeros(n_edges, dtype=wp.vec3, device=self.device)
            inputs = [model.particle_q, self.edges]
            wp.launch(edge_bounds_kernel, n_edges, inputs, [self.edge_lower, self.edge_upper], device=self.device)
            self.edge_bvh = wp.Bvh(self.edge_lower, self.edge_upper)
            self.edge_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            self.edge_edge_ratio = float(edge_edge_ratio)
            self._allocate_edge_contacts(n_edges)

    def set_groups(self, groups: np.ndarray, group_margins: np.ndarray | None, margin: float, gap: float) -> None:
        """Keep a particle and a triangle of two groups ``margin`` apart and of one group ``group_margins[group]``,
        or let them pass where it is 0 or without ``group_margins``, finding the contacts within the margin plus
        ``gap``.

        Args:
            groups: Group of each particle, shape ``(particle_count,)``; the particles of a triangle share one.
            group_margins: Margin [m] within each group, 0 for none, or ``None`` for none in any group.
            margin: Margin [m] between groups.
            gap: Detection distance [m] beyond the margin.
        """
        if self.mode != "vertex_face" or self.edge_edge:
            raise ValueError("self-contact groups need vertex-face contacts without edge-edge contacts")
        self._groups_h = np.asarray(groups, dtype=np.int32)
        if group_margins is None:
            group_margins = np.zeros(int(self._groups_h.max()) + 1)
        self._group_margins_h = np.asarray(group_margins, dtype=np.float32)
        # Whether any group collides with itself.
        self.within_groups = bool((self._group_margins_h > 0.0).any())
        self.groups = wp.array(self._groups_h, dtype=wp.int32, device=self.device)
        self.group_margins = wp.array(self._group_margins_h, dtype=wp.float32, device=self.device)
        self.set_margin(margin, gap)

    def set_margin(self, margin: float, gap: float) -> None:
        """Change the margin between groups and the detection gap of :meth:`set_groups`."""
        self.margin, self.gap = float(margin), float(gap)
        self.query_radius = max(self.margin, float(self._group_margins_h.max())) + self.gap

    def contact_margins(self, particles: np.ndarray) -> np.ndarray:
        """Margins [m] of vertex-face contacts on ``particles``, shape ``(C, 4)``, under :meth:`set_groups`; float32."""
        group = self._groups_h[particles[:, 0]]
        same = group == self._groups_h[particles[:, 1]]
        return np.where(same, self._group_margins_h[group], np.float32(self.margin)).astype(np.float32)

    def _bounds(self, q: wp.array) -> None:
        inputs = [q, self.tri_indices]
        wp.launch(compute_tri_aabbs, self.tri_indices.shape[0], inputs, [self.lower, self.upper], device=self.device)

    def _allocate(self, capacity: int) -> None:
        self.particles = wp.empty((capacity, 4), dtype=wp.int32, device=self.device)
        self.alpha = wp.empty((capacity, 4), dtype=wp.float32, device=self.device)
        self.normal = wp.empty(capacity, dtype=wp.vec3, device=self.device)

    def _allocate_edge_contacts(self, capacity: int) -> None:
        # The sort takes twice the keys it sorts.
        self.edge_keys = wp.empty(2 * capacity, dtype=wp.int64, device=self.device)
        self.edge_order = wp.empty(2 * capacity, dtype=wp.int32, device=self.device)
        self.edge_particles = wp.empty((capacity, 4), dtype=wp.int32, device=self.device)
        self.edge_alpha = wp.empty((capacity, 4), dtype=wp.float32, device=self.device)
        self.edge_normal = wp.empty(capacity, dtype=wp.vec3, device=self.device)

    def _find_edge_contacts(self, q: wp.array, start: int) -> int:
        """Find the edge-edge contacts at ``q``, append them to the vertex-face contacts from ``start`` in the order
        of their edge pairs, and return their number."""
        n_edges = self.edges.shape[0]
        outputs = [self.edge_lower, self.edge_upper]
        wp.launch(edge_bounds_kernel, n_edges, [q, self.edges], outputs, device=self.device)
        self.edge_bvh.refit()
        while True:
            self.edge_contact_count.zero_()
            inputs = [q, self.edges, self.edge_bvh.id, self.ring_offsets, self.ring, self.rest, self.rest_radius]
            inputs += [self.query_radius, self.nearest_distance, self.edge_edge_ratio]
            outputs = [self.edge_contact_count, self.edge_keys, self.edge_particles, self.edge_alpha, self.edge_normal]
            wp.launch(edge_edge_contacts_kernel, n_edges, inputs, outputs, device=self.device)
            n = int(self.edge_contact_count.numpy()[0])
            if n <= self.edge_particles.shape[0]:
                break
            self._allocate_edge_contacts(2 * n)
        if n == 0:
            return 0
        wp.launch(iota_kernel, n, outputs=[self.edge_order], device=self.device)
        wp.utils.radix_sort_pairs(self.edge_keys, self.edge_order, n)
        if start + n > self.found_particles.shape[0]:
            capacity = 2 * (start + n)
            grown = [wp.empty((capacity, 4), dtype=d, device=self.device) for d in (wp.int32, wp.float32)]
            grown.append(wp.empty(capacity, dtype=wp.vec3, device=self.device))
            for dst, src in zip(grown, (self.found_particles, self.found_alpha, self.found_normal), strict=True):
                wp.copy(dst, src, count=start * (4 if src.ndim == 2 else 1))
            self.found_particles, self.found_alpha, self.found_normal = grown
        inputs = [self.edge_order, self.edge_particles, self.edge_alpha, self.edge_normal, start]
        outputs = [self.found_particles, self.found_alpha, self.found_normal]
        wp.launch(gather_edge_contacts_kernel, n, inputs, outputs, device=self.device)
        return n

    def detect(self, q: wp.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Particles and coefficients, shape ``(C, 4)`` with particle ``-1`` in unused slots, and normals,
        shape ``(C, 3)``, of the contacts at ``q``, sorted by their particles; edge-edge contacts follow, sorted by
        their edge pairs."""
        n = self.find(q)
        if self.mode == "vertex_face":
            return host_head(self.found_particles, n), host_head(self.found_alpha, n), host_head(self.found_normal, n)
        particles, alpha, normal = host_head(self.particles, n), host_head(self.alpha, n), host_head(self.normal, n)
        order = np.lexsort((particles[:, 1], particles[:, 0]))
        return particles[order], alpha[order], normal[order]

    def find(self, q: wp.array) -> int:
        """Find the contacts at ``q`` and return their number. With ``mode="vertex_face"`` they stay on the
        device in ``found_particles``, ``found_alpha`` and ``found_normal``, in the order of their particles, and the
        edge-edge contacts follow them."""
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
                groups = [self.groups, self.group_margins, self.margin, self.gap, 1]
                if self.groups is None:
                    empty = (wp.empty(1, dtype=t, device=self.device) for t in (wp.int32, wp.float32))
                    groups = [*empty, 0.0, 0.0, 0]
                inputs = [q, self.tri_indices, *found, *filters, *groups]
                kernel, outputs = vertex_face_contacts_kernel, [*outputs, self.found, self.nearest_distance]
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
            elif self.edge_edge:
                return n + self._find_edge_contacts(q, n)
            else:
                return n
