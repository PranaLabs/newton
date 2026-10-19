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
