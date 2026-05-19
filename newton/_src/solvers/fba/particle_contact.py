# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Particle-particle contact broadphase + filter for SolverFBA.

``ParticleContactBroadphase`` enumerates **all** particle pairs within
``2·particle_radius + margin`` of each other — both intra-mesh
(traditional cloth self-contact) and inter-mesh (e.g. two separate cloth
instances stacking on the same obstacle).  Topology + rest-pose filters
only ever exclude pairs that share connectivity within a single cloth's
triangle / edge list, so cross-mesh pairs always pass through to the
solver.  This matches RealSim's particle-particle contact model, where
``self-contact`` is a misnomer in the strict physics sense but happens
to be the field name the rest of the engine uses.

Components:

* The ``wp.HashGrid`` used to enumerate particle pairs within
  ``2·particle_radius + margin`` of each other.
* Per-particle topology adjacency (1- and 2-ring neighbours) used to discard
  candidate pairs whose particles are already constrained by mesh edges.
* Rest-pose pair exclusion: pairs whose initial separation is below
  ``rest_exclusion_radius`` (e.g. T-shirt double-layer collar, hem) are
  permanently filtered.

Per-step output (device-resident, sorted by ``(a, b)`` for determinism):

* ``pair_particle_a, pair_particle_b`` — int32 arrays of length ``pair_count``
* ``pair_normal``                     — vec3 (from b to a, unit) per pair
* ``pair_dist``                       — float32 distance per pair (≤ threshold)
* ``pair_count``                      — int32[1] atomic counter for the call
"""

from __future__ import annotations

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def find_particle_contact_pairs_kernel(
    grid_id: wp.uint64,
    positions: wp.array[wp.vec3],
    threshold: wp.float32,
    # Topology/rest filter (CSR over forbidden ``j`` indices per ``i``).
    excl_offsets: wp.array[wp.int32],   # (N+1,) start of forbidden list per i
    excl_indices: wp.array[wp.int32],   # (E,)   sorted forbidden j's
    # Output buffers (atomic-append).
    max_pairs: wp.int32,
    pair_count: wp.array[wp.int32],     # (1,)
    pair_a: wp.array[wp.int32],         # (max_pairs,)
    pair_b: wp.array[wp.int32],         # (max_pairs,)
    pair_normal: wp.array[wp.vec3],     # (max_pairs,) unit normal from b → a
    pair_dist: wp.array[wp.float32],    # (max_pairs,)
):
    """Per-particle thread: query the hash grid, filter, atomic-append pairs.

    Filter order (cheapest first):

    1. ``j > i`` — emit each pair exactly once (i < j convention).
    2. ``dist < threshold`` — geometric proximity test.
    3. Binary-search ``j`` in this particle's exclusion CSR — covers both the
       topology filter (1-/2-ring neighbours) and the rest-pose filter
       (initially-close particles).  Built once at setup time so the per-step
       cost is one ``O(log k)`` lookup per candidate, where ``k`` is the
       per-particle exclusion-list length (~10-30 for typical cloth meshes).

    The atomic counter ``pair_count`` is incremented before writing so output
    is densely packed.  Caller is responsible for sorting by ``(a, b)`` if
    determinism beyond ``a < b`` is required.
    """
    i = wp.tid()
    p_i = positions[i]

    excl_start = excl_offsets[i]
    excl_end = excl_offsets[i + 1]

    query = wp.hash_grid_query(grid_id, p_i, threshold)
    j = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        if j <= i:
            continue
        p_j = positions[j]
        d_vec = p_i - p_j
        d = wp.length(d_vec)
        if d >= threshold or d < wp.float32(1.0e-9):
            continue

        # Binary search excl_indices[excl_start:excl_end] for j.
        lo = excl_start
        hi = excl_end
        found = wp.int32(0)
        while lo < hi:
            mid = (lo + hi) // 2
            v = excl_indices[mid]
            if v == j:
                found = wp.int32(1)
                break
            if v < j:
                lo = mid + 1
            else:
                hi = mid
        if found == wp.int32(1):
            continue

        slot = wp.atomic_add(pair_count, 0, 1)
        if slot >= max_pairs:
            # Out of capacity — atomic over-write would corrupt; just leave.
            continue
        pair_a[slot] = i
        pair_b[slot] = j
        inv_d = wp.float32(1.0) / d
        pair_normal[slot] = wp.vec3(d_vec[0] * inv_d, d_vec[1] * inv_d, d_vec[2] * inv_d)
        pair_dist[slot] = d


# ---------------------------------------------------------------------------
# Host-side broadphase manager
# ---------------------------------------------------------------------------


class ParticleContactBroadphase:
    """Particle-particle proximity enumerator with topology + rest filters.

    Args:
        model: :class:`newton.Model` — used at setup time to read particle rest
            positions and cloth topology (``model.tri_indices``,
            ``model.edge_indices``).
        threshold: Distance below which a particle pair becomes a self-contact
            candidate.  Typically ``2 * particle_radius + particle_contact_margin``.
        topology_ring: Neighbour-ring depth to exclude (1 = direct mesh
            neighbours, 2 = neighbours of neighbours, ...).  VBD reference uses
            2; we match that.
        rest_exclusion_radius: Particles initially closer than this in their
            rest pose are permanently excluded.  Mirrors VBD's
            ``particle_rest_shape_contact_exclusion_radius``.
        max_pairs: Hard cap on per-step pair count.  Allocated once.  Exceeded
            entries are silently dropped (sentinel — caller checks
            ``pair_count > max_pairs`` to know).
    """

    def __init__(
        self,
        model,
        *,
        threshold: float,
        topology_ring: int = 2,
        rest_exclusion_radius: float = 0.5,
        max_pairs: int | None = None,
    ) -> None:
        if model.particle_count == 0:
            raise ValueError("ParticleContactBroadphase requires a non-zero particle_count")
        self.model = model
        self.device = model.device
        self.n = int(model.particle_count)
        self.threshold = float(threshold)
        self.topology_ring = int(topology_ring)
        self.rest_exclusion_radius = float(rest_exclusion_radius)
        self.max_pairs = int(max_pairs) if max_pairs is not None else max(1024, 8 * self.n)

        # ---- Build per-particle exclusion list -----------------------------
        # 1. Topology N-ring neighbours from tri + edge connectivity.
        # 2. Rest-pose proximity (anything initially within rest_exclusion_radius).
        # Both unioned into a single sorted CSR per particle.
        excl = [set() for _ in range(self.n)]

        # ---- Topology: from triangle connectivity ----
        adj = [set() for _ in range(self.n)]
        if hasattr(model, "tri_indices") and model.tri_indices is not None and model.tri_indices.size > 0:
            tri_flat = model.tri_indices.numpy().astype(np.int32)
            tri = tri_flat.reshape(-1, 3)
            for a, b, c in tri:
                a, b, c = int(a), int(b), int(c)
                adj[a].update([b, c])
                adj[b].update([a, c])
                adj[c].update([a, b])

        # Also pull explicit bending edges (which include all interior edges)
        # so isolated edges without a triangle stencil don't slip past.
        if hasattr(model, "edge_indices") and model.edge_indices is not None and model.edge_indices.size > 0:
            ei = model.edge_indices.numpy().astype(np.int32).reshape(-1, 4)
            for row in ei:
                v1, v2 = int(row[2]), int(row[3])
                if 0 <= v1 < self.n and 0 <= v2 < self.n:
                    adj[v1].add(v2)
                    adj[v2].add(v1)

        # BFS to ``topology_ring`` depth.
        for p in range(self.n):
            frontier = {p}
            visited = {p}
            for _ in range(self.topology_ring):
                nxt = set()
                for q in frontier:
                    nxt.update(adj[q] - visited)
                visited |= nxt
                frontier = nxt
            visited.discard(p)  # particle itself isn't a candidate
            excl[p].update(visited)

        # ---- Rest-pose proximity ----
        # O(n²) brute force is fine at n ≤ 5e4 (T-shirt scale); switch to
        # a spatial hash if we ever target larger meshes.
        rest_q = model.particle_q.numpy().astype(np.float64)
        thr_rest = self.rest_exclusion_radius
        if thr_rest > 0.0:
            # Vectorised pairwise distance with masking — much faster than
            # the per-particle Python loop at n ~ 1k.
            n = self.n
            chunk = 1024
            for i_start in range(0, n, chunk):
                i_end = min(i_start + chunk, n)
                d = np.linalg.norm(
                    rest_q[i_start:i_end, None, :] - rest_q[None, :, :], axis=-1
                )
                close = d < thr_rest
                # Don't exclude self.
                for ii in range(i_end - i_start):
                    p = i_start + ii
                    row = np.where(close[ii])[0]
                    row = row[row != p]
                    excl[p].update(int(j) for j in row)

        # ---- Pack into sorted CSR ----
        offsets = np.zeros(self.n + 1, dtype=np.int32)
        for p in range(self.n):
            offsets[p + 1] = offsets[p] + len(excl[p])
        excl_indices_np = np.empty(int(offsets[-1]), dtype=np.int32)
        for p in range(self.n):
            sorted_excl = np.sort(np.fromiter(excl[p], dtype=np.int32))
            excl_indices_np[offsets[p]: offsets[p + 1]] = sorted_excl

        self._excl_offsets_d = wp.array(offsets, dtype=wp.int32, device=self.device)
        self._excl_indices_d = wp.array(excl_indices_np, dtype=wp.int32, device=self.device)
        self._excl_total = int(offsets[-1])

        # ---- HashGrid sized for the rest-pose bounding box (+ a little
        # headroom since particles can drift during sim).  HashGrid cell
        # size = threshold is the rule of thumb (one query inspects one
        # 3x3x3 supercell).
        bb_min = rest_q.min(axis=0)
        bb_max = rest_q.max(axis=0)
        bb_extent = (bb_max - bb_min).max()
        cell = max(self.threshold, 1.0e-3)
        dim = max(8, int(np.ceil(bb_extent / cell)) + 2)
        # Cap dim so we don't allocate a 1000³ cell grid for an outlier mesh.
        dim = min(dim, 256)
        self._grid = wp.HashGrid(dim_x=dim, dim_y=dim, dim_z=dim, device=self.device)
        self._cell_size = cell

        # ---- Output buffers (allocated once, reused per step). ----
        self._pair_count_d = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._pair_a_d = wp.empty(self.max_pairs, dtype=wp.int32, device=self.device)
        self._pair_b_d = wp.empty(self.max_pairs, dtype=wp.int32, device=self.device)
        self._pair_normal_d = wp.empty(self.max_pairs, dtype=wp.vec3, device=self.device)
        self._pair_dist_d = wp.empty(self.max_pairs, dtype=wp.float32, device=self.device)

    # ------------------------------------------------------------------
    # Per-step API
    # ------------------------------------------------------------------

    def find_pairs(self, particle_q: wp.array) -> int:
        """Run broadphase + filters on ``particle_q``.  Returns the host-side
        ``pair_count`` (truncated to ``max_pairs``).

        After this call, ``self._pair_*_d[:count]`` hold the surviving pairs.
        """
        self._pair_count_d.zero_()
        self._grid.build(points=particle_q, radius=self._cell_size)
        wp.launch(
            find_particle_contact_pairs_kernel,
            dim=self.n,
            inputs=[
                self._grid.id,
                particle_q,
                wp.float32(self.threshold),
                self._excl_offsets_d,
                self._excl_indices_d,
                wp.int32(self.max_pairs),
            ],
            outputs=[
                self._pair_count_d,
                self._pair_a_d,
                self._pair_b_d,
                self._pair_normal_d,
                self._pair_dist_d,
            ],
            device=self.device,
        )
        count = int(self._pair_count_d.numpy()[0])
        return min(count, self.max_pairs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def exclusion_total(self) -> int:
        """Total excluded directed pairs (i → j) in the precomputed CSR."""
        return self._excl_total

    @property
    def stats(self) -> dict:
        return {
            "n_particles": self.n,
            "threshold": self.threshold,
            "topology_ring": self.topology_ring,
            "rest_exclusion_radius": self.rest_exclusion_radius,
            "max_pairs": self.max_pairs,
            "excl_total_directed": self._excl_total,
            "grid_cell_size": self._cell_size,
        }
