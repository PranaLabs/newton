# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Smoke: run the BSR-based LiteNSN inner driver on rigid-only contacts and
confirm it produces the same physics as the fused per-particle / per-contact
kernel paths.

This is the precursor for self-contact: if BSR fails for rigid-only (where
``particle_b = -1`` everywhere) it'll certainly fail with self-contact rows
mixed in.
"""

from __future__ import annotations

import sys
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA
from newton.examples.cloth.example_cloth_on_sphere_fba import (
    _DT, _PD_ITER, _NSN_ITER, _FRICTION_MU, _PIN_STIFFNESS, _OBJ_MASS,
    build_cloth_on_sphere_builder,
)


def setup(extra_overrides=None):
    b = build_cloth_on_sphere_builder()
    m = b.finalize()
    um = _OBJ_MASS / m.particle_count
    m.particle_mass.assign(np.full(m.particle_count, um, np.float32))
    m.particle_inv_mass.assign(np.full(m.particle_count, 1.0 / um, np.float32))
    mu_o = np.full(m.particle_count, _FRICTION_MU, np.float64)
    s = SolverFBA(
        m, iterations=_PD_ITER, nsn_iterations=_NSN_ITER, friction=True,
        stretching_model="arap", mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS, nsn_schur_mode="lite",
    )
    pipe = newton.CollisionPipeline(m, soft_contact_margin=0.05)
    return m, s, pipe


def main():
    wp.init()
    # Run cloth-on-sphere with the existing per-contact path (k_p=1 for all
    # particles) to get the baseline trajectory.
    m, s, pipe = setup()
    c = pipe.contacts()
    s0, s1 = m.state(), m.state()
    target_frame = 100
    for f in range(target_frame):
        s0.clear_forces()
        pipe.collide(s0, c)
        s.step(s0, s1, None, c, _DT)
        s0, s1 = s1, s0
    q_baseline = s0.particle_q.numpy().copy()
    print(f"baseline @ frame {target_frame}: y range = [{q_baseline[:,1].min():.4f}, {q_baseline[:,1].max():.4f}]")

    # Now run with the BSR driver instead — fake "self-contact present" so
    # the dispatcher routes to BSR.  We'll manually call the BSR driver from
    # Python with rigid-only data (particle_b = -1 everywhere) and verify
    # the result matches.
    #
    # (Once we plumb the auto-dispatch this will all happen inside step().)
    print("\nBSR rigid-only test: TODO — invoke _solve_nsn_coulomb_lite_bsr manually")
    print("Requires building combined row arrays from the live contact set —")
    print("this is the next integration step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
