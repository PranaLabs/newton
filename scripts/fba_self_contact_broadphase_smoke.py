# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.1 smoke: ParticleContactBroadphase on the cloth-on-bar scene.

Builds the cloth-on-bar baseline (which now produces a wrapped drape with
two halves dangling below the bar), runs broadphase per frame, and prints
the candidate-pair count.  Expectation: ~0 pairs at frame 0 (cloth flat),
ramping up as the two halves swing together below the bar.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.particle_contact import ParticleContactBroadphase
from newton.solvers import SolverFBA

# Reuse the baseline scene builder.
from scripts.fba_cloth_on_bar_smoke import (
    build_scene,
    _DT,
    _PD_ITER,
    _NSN_ITER,
    _FRICTION_MU,
    _PIN_STIFFNESS,
    _OBJ_MASS,
    _PARTICLE_RADIUS,
    _CONTACT_MARGIN,
)


def main() -> int:
    wp.init()
    b = build_scene()
    model = b.finalize()
    uniform_mass = _OBJ_MASS / model.particle_count
    model.particle_mass.assign(np.full(model.particle_count, uniform_mass, np.float32))
    model.particle_inv_mass.assign(np.full(model.particle_count, 1.0 / uniform_mass, np.float32))
    mu_o = np.full(model.particle_count, _FRICTION_MU, np.float64)
    solver = SolverFBA(
        model, iterations=_PD_ITER, nsn_iterations=_NSN_ITER, friction=True,
        stretching_model="arap", mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS, nsn_schur_mode="lite",
    )
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=_CONTACT_MARGIN)
    contacts = pipeline.contacts()
    state_0 = model.state(); state_1 = model.state()

    # ---- Broadphase setup ----
    # Threshold = 2 · particle_radius + a small margin (matches the
    # ``soft_contact_margin`` convention used for rigid contact).
    self_threshold = 2 * _PARTICLE_RADIUS + 0.04
    bp = ParticleContactBroadphase(
        model,
        threshold=self_threshold,
        topology_ring=2,
        rest_exclusion_radius=0.1,
        max_pairs=64 * 1024,
    )
    print(f"broadphase stats: {bp.stats}")

    print(f"\n{'frame':>5}  {'y_min':>7}  {'y_max':>7}  {'n_rigid':>8}  {'n_self':>7}")
    for f in range(500):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, _DT)
        state_0, state_1 = state_1, state_0
        n_self = bp.find_pairs(state_0.particle_q)
        if f % 50 == 0 or f == 499:
            q = state_0.particle_q.numpy()
            nc = int(contacts.soft_contact_count.numpy()[0]) if hasattr(contacts, "soft_contact_count") else 0
            print(f"{f:5d}  {q[:,1].min():7.3f}  {q[:,1].max():7.3f}  {nc:8d}  {n_self:7d}")

    # Dump last-frame pair sample for inspection.
    n_last = bp.find_pairs(state_0.particle_q)
    if n_last > 0:
        pa = bp._pair_a_d.numpy()[:n_last]
        pb = bp._pair_b_d.numpy()[:n_last]
        pd = bp._pair_dist_d.numpy()[:n_last]
        pn = bp._pair_normal_d.numpy()[:n_last]
        print(f"\nLast frame: {n_last} self-contact pairs.")
        print(f"  dist range: [{pd.min():.4f}, {pd.max():.4f}] m  (threshold {self_threshold:.4f})")
        print("  first 5 pairs (a, b, dist, normal):")
        for k in range(min(5, n_last)):
            print(f"    ({pa[k]:4d}, {pb[k]:4d})  d={pd[k]:.4f}  n=({pn[k,0]:+.3f}, {pn[k,1]:+.3f}, {pn[k,2]:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
