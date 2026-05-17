# CudaTests Phase 2 (SqueezingBall) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce CudaTests/SqueezingBall — a 7129-particle Neo-Hookean tet ball squeezed between 4 rolling cylinders (mu=0.5, angular speed ±3 rad/s) and a plane (mu=0.5) under gravity. Match the RealSim binary baseline (54 ms/step).

**Architecture:** Cylinders stay `body=-1` (static geometry from FBA's view); their kinematic rotation enters the simulation only through the Stage B friction tangent residual. Concretely, each shape can have an associated scalar angular speed `ω` about its local +Z (the cylinder's primitive axis). At `update_contacts` time, for every contact whose shape has nonzero `ω`, FBA computes `v_anchor = ω·axis_world × (world_anchor − shape_center)` and adds `v_anchor·dt` to that contact's tangential offsets. Normal contact handling is unchanged. Phase 1's `set_pin_targets` and Schur-cache infrastructure is reused as-is.

**Tech Stack:** Python 3, `uv`, `warp`, `numpy`, Newton (`/home/ziqiu/work/newton`), Medit `.mesh` parser (`scripts/fba_cudatest_bench/medit.py` from Phase 1), `matplotlib` (Agg) for snapshot strips, `unittest` (NOT pytest — see `AGENTS.md`).

---

## Pre-flight Checks

- [ ] **Confirm branch.**
```bash
cd /home/ziqiu/work/newton
git status
```
Expected: `On branch ziqiu/fba-solver-design`. Working tree clean (Phase 1 wrapped at commit `b29044cb`).

- [ ] **Confirm ball mesh exists.**
```bash
ls -la /home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/ball_volume_7129P.mesh
```
Expected: ~few-MB Medit text file. First line `MeshVersionFormatted 2`.

- [ ] **Confirm RealSim baseline exists.**
```bash
python3 -c "import json; b=json.load(open('/home/ziqiu/work/newton/scripts/cudatest_baselines.json')); print(b['SqueezingBall'])"
```
Expected: `{"demo": "SqueezingBall", "status": "ok", "mean_ms": 53.986, ...}` (from Phase 0 / Task C).

- [ ] **Confirm Phase 1 head is in place.**
```bash
cd /home/ziqiu/work/newton
git log --oneline -3
```
Expected: top commit is `b29044cb Document PullingWooper demo and Phase 1 completion`.

---

## Demo parameters (from `simulation/config/CudaTests/SqueezingBall/`)

`scene.json`:
- `gravity = [0.0, -10.0, 0.0]` (Y-down)
- `LocalGlobal_CUDA: 5` → PD iterations = 5
- `timestep = 0.01`, `stop = 600` frames (offline)
- `constraintsolver.iterations: 10` → FBA `nsn_iterations = 1` (Phase 1 default — Task M showed RealSim's `iterations` is a Newton-iter cap, not GS-sweep count; 1 is sufficient)
- `linearsolver: SPARSE_INVERSE_CUDA` → matches FBA

`ball_7k.json`:
- mesh `ball_volume_7129P.mesh` (7129 particles, tetrahedra)
- `transformation`: `trans=(0, 2.6, 0)`, `rotation=(0,0,0)`, `scale=(3,3,3)`, `center=(0,0,0)`
- `mechanical_props`: `obj_mass=1000`, `young=1e4`, `poisson=0.4`, `constitutive=TET_PD_NEOHOOKEAN`

Cylinders (all `mu=0.5`, `radius=1`):
- `cyl0`: `base=(0, -1, -1.5)`, `axis=(1,0,0)`, `rollingvel=-3.0`
- `cyl1`: `base=(0, -1, +1.5)`, `axis=(1,0,0)`, `rollingvel=+3.0`
- `cyl2`: `base=(-1.5, -3.5, 0)`, `axis=(0,0,1)`, `rollingvel=+3.0`
- `cyl3`: `base=(+1.5, -3.5, 0)`, `axis=(0,0,1)`, `rollingvel=-3.0`

Plane: `base=(0, -10, 0)`, `normal=(0, 1, 0)`, `mu=0.5`.

Layout: ball drops onto cyl pair 0/1 (rolling-X, would carry the ball in the ±Z direction), then onto cyl pair 2/3 (rolling-Z, would carry the ball in the ±X direction), eventually onto the plane at y=-10. Friction (mu=0.5) couples the rotation into ball motion.

Lamé params from `young=1e4`, `poisson=0.4`:
- `mu = E/(2·(1+ν)) = 1e4 / 2.8 ≈ 3.571e3`
- `lam = E·ν/((1+ν)·(1−2·ν)) = 1e4 · 0.4 / (1.4 · 0.2) ≈ 1.429e4`

---

## File map

- **Modify** `newton/_src/solvers/fba/solver_fba.py`
  - Add `shape_angular_velocity: dict[int, float] | None = None` constructor param
  - Add storage: per-shape `_shape_omega_h: np.ndarray` (size = `num_shapes`, default 0)
  - In `update_contacts`: after world_anchor is computed for each contact, if `_shape_omega_h[s_idx] != 0`, compute `axis_world = quat_rotate(shape_q, (0,0,1))`, `v_anchor = ω · axis_world × (world_anchor − shape_p)`, and add `dt · v_anchor` to the tangential offsets

- **Modify** `newton/tests/test_solver_fba.py`
  - Add `SolverFBAKinematicCylinderTests` with regression tests

- **Create** `scripts/fba_demo5_squeezing_ball.py`
  - Demo driver using the kinematic API

- **Modify** `scripts/contact_demos_out/README.md`
  - Add Demo 5 section

- **Modify** `docs/superpowers/fba-dev-progress.md`
  - Add 2026-05-XX P2 completion entry

---

## Task A: Kinematic shape angular velocity infrastructure

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py:149-300` (constructor), `:710-870` (`update_contacts`)
- Modify: `newton/tests/test_solver_fba.py` (append a new TestCase)

### Step A.1: Append the failing unit test

Append this class to `newton/tests/test_solver_fba.py`:

```python
class SolverFBAKinematicCylinderTests(unittest.TestCase):
    """Per-shape angular velocity surfaces in Stage B tangent offsets."""

    def _ball_and_cylinder(self):
        """Tiny tet ball + 1 cylinder (axis +X, world frame). Returns (model, pipeline, contacts)."""
        import newton

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-10.0)
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.5, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.2,
            cell_y=0.2,
            cell_z=0.2,
            density=500.0,
            k_mu=5.0e3,
            k_lambda=5.0e3,
            k_damp=0.0,
        )
        # Z-aligned cylinder mapped to +X axis via xform rotation.
        # Quaternion that rotates +Z to +X: 90° about -Y.
        import math
        half = math.pi / 4.0
        qy = -math.sin(half)
        qw = math.cos(half)
        q = wp.quat(0.0, qy, 0.0, qw)
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, -0.2, 0.0), q),
            radius=0.3,
            half_height=2.0,
        )
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
        contacts = pipeline.contacts()
        return model, pipeline, contacts

    def test_default_no_kinematic_motion(self) -> None:
        """Default behaviour matches Phase 1: no per-shape kinematic motion."""
        from newton.solvers import SolverFBA
        model, _, _ = self._ball_and_cylinder()
        solver = SolverFBA(model, friction=True)
        # No shape_angular_velocity passed → all shapes have ω = 0.
        self.assertTrue(np.all(solver._shape_omega_h == 0.0))

    def test_kinematic_motion_stored_per_shape(self) -> None:
        """Constructor stores ω per shape index."""
        from newton.solvers import SolverFBA
        model, _, _ = self._ball_and_cylinder()
        solver = SolverFBA(
            model,
            friction=True,
            shape_angular_velocity={0: -3.0},  # shape 0 is the cylinder
        )
        self.assertEqual(float(solver._shape_omega_h[0]), -3.0)

    def test_kinematic_anchor_velocity_shifts_tangent_offset(self) -> None:
        """A nonzero ω advances the contact anchor in the tangential direction.

        Without any spinning, the tangent offset = dot(t, anchor).  With ω ≠ 0
        the offset shifts by dt · dot(t, v_anchor), where
        v_anchor = ω · axis_world × (anchor − cyl_center).
        """
        import newton
        from newton.solvers import SolverFBA

        # Drop the ball onto the cylinder once so we have real contacts.
        model, pipeline, contacts = self._ball_and_cylinder()
        s_in = model.state()
        s_out = model.state()

        # Two solver instances: ω = 0 vs ω = -3, same scene.
        solver_static = SolverFBA(model, friction=True, mu_per_pair_override=np.full(model.particle_count, 0.5))
        solver_spin = SolverFBA(
            model,
            friction=True,
            mu_per_pair_override=np.full(model.particle_count, 0.5),
            shape_angular_velocity={0: -3.0},
        )

        # Run a few steps to land contacts.
        for _ in range(20):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver_static.step(s_in, s_out, None, contacts, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
            if solver_static._contact_count > 0:
                break
        self.assertGreater(solver_static._contact_count, 0, "Need at least one contact to exercise the kinematic path.")

        # Solver_spin sees the same contact set right now; only ω differs.
        # Compare its tangent offsets vs solver_static's at the same world_anchor.
        solver_spin.update_contacts(contacts, s_in)
        M_static = solver_static._contact_count
        M_spin = solver_spin._contact_count
        self.assertEqual(M_static, M_spin)
        # update_contacts caches base offsets in _contact_tangent{1,2}_offset_h;
        # step() applies dt·dot(t, v_anchor) on top.  Drive one step to force
        # both solvers through the dt-shift path, then compare the *effective*
        # offsets that the residual uses (rebuilt in step()).
        s2_in = model.state(); s2_out = model.state()
        solver_static.step(s_in, s2_out, None, contacts, 1.0 / 60.0)
        solver_spin.step(s2_in, s_out, None, contacts, 1.0 / 60.0)
        # The base offsets should be identical (same anchors); the v_anchor
        # cache should differ.
        v_static = solver_static._contact_v_anchor_h
        v_spin = solver_spin._contact_v_anchor_h
        self.assertEqual(v_static.shape, v_spin.shape)
        np.testing.assert_allclose(v_static, 0.0)
        self.assertGreater(float(np.linalg.norm(v_spin)), 1e-6,
                           "Expected ω≠0 to produce nonzero v_anchor for at least one contact.")
```

### Step A.2: Run the failing tests

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k SolverFBAKinematicCylinder
```

Expected: 3 FAILs (all reference attributes / params that don't exist yet: `_shape_omega_h`, `shape_angular_velocity=`, `_tangent1_offset_h`).

### Step A.3: Add constructor parameter + storage

In `newton/_src/solvers/fba/solver_fba.py`, find the `SolverFBA.__init__` signature. Add `shape_angular_velocity: dict[int, float] | None = None` as the last parameter (after `use_isodof`):

```python
        use_isodof: bool = True,
        shape_angular_velocity: dict[int, float] | None = None,
    ) -> None:
```

In the docstring `Args:` section, add:
```
            shape_angular_velocity: Optional ``{shape_index: ω_rad_s}`` mapping.
                Each listed shape rotates about its local +Z axis (Newton's
                cylinder primitive axis) at the given angular speed.  Surfaces
                with nonzero ω contribute a tangential anchor velocity
                ``v_anchor = ω · axis_world × (world_anchor − shape_center)``
                into the Stage B friction residual; the shape itself stays
                static geometry.  Defaults to no kinematic motion on any shape.
                Mirrors RealSim's ``cylindercollisions[i].rollingvel`` config.
```

In `__init__` body, after `self.use_isodof = bool(use_isodof)`, add:
```python
        # Per-shape kinematic angular velocity (rad/s about local +Z).  RealSim
        # parity for ``cylindercollisions.rollingvel``.
        n_shapes = int(model.shape_count) if hasattr(model, "shape_count") and model.shape_count else 0
        self._shape_omega_h = np.zeros(max(n_shapes, 1), dtype=np.float64)
        if shape_angular_velocity is not None:
            for s_idx, omega in shape_angular_velocity.items():
                if 0 <= int(s_idx) < n_shapes:
                    self._shape_omega_h[int(s_idx)] = float(omega)
```

### Step A.4: Inject `v_anchor·dt` into the tangent offsets

In `update_contacts` (around line 829-863 currently), inside the `if self.friction:` block, BEFORE the existing loop that fills `tangent1_offset_h` / `tangent2_offset_h`, capture the per-shape spin axis in world frame. Then in the existing loop body, after `world_anchor` is computed for each contact, compute `v_anchor` and add `dt · dot(t, v_anchor)` to each tangent offset.

Concretely, modify the friction block as follows. Locate this code (around line 830-863):

```python
        if self.friction:
            t1_h = np.zeros((M, 3), dtype=np.float32)
            t2_h = np.zeros((M, 3), dtype=np.float32)
            tangent1_offset_h = np.zeros(M, dtype=np.float64)
            tangent2_offset_h = np.zeros(M, dtype=np.float64)
            for c in range(M):
                t1, t2 = compute_tangent_basis(normal_h[c])
                ...
                tangent1_offset_h[c] = float(np.dot(t1, world_anchor))
                tangent2_offset_h[c] = float(np.dot(t2, world_anchor))
```

Replace with (inserting the per-shape axis precomputation and the v_anchor branch):

```python
        if self.friction:
            t1_h = np.zeros((M, 3), dtype=np.float32)
            t2_h = np.zeros((M, 3), dtype=np.float32)
            tangent1_offset_h = np.zeros(M, dtype=np.float64)
            tangent2_offset_h = np.zeros(M, dtype=np.float64)

            # Precompute per-shape world-frame axis (Newton's cylinder extends
            # along local +Z; the world axis is the shape's quaternion applied
            # to (0,0,1)).
            shape_transform_np = (
                model.shape_transform.numpy()
                if hasattr(model, "shape_transform") and model.shape_transform is not None
                else None
            )
            # _kinematic_dt is the most recent step's dt; needed below.  We
            # don't have dt at update_contacts time, so we cache the anchor
            # offset in a buffer and lazily apply ω·dt at the start of step().
            # Simpler approach: cache (axis_world, r_local) per contact for the
            # spinning case, then apply during step() once dt is known.

            self._contact_v_anchor_h = np.zeros((M, 3), dtype=np.float64)
            for c in range(M):
                t1, t2 = compute_tangent_basis(normal_h[c])
                t1_h[c] = t1.astype(np.float32)
                t2_h[c] = t2.astype(np.float32)
                s_idx = int(shape_h[c])
                bpos = body_pos_h[c]
                world_anchor = bpos.copy()
                if shape_body_np is not None and s_idx >= 0:
                    b_idx = int(shape_body_np[s_idx])
                    if b_idx >= 0 and body_q_np is not None:
                        bq = body_q_np[b_idx]
                        pos_b = bq[:3]
                        quat_b = bq[3:]
                        world_anchor = _transform_point(pos_b, quat_b, bpos)
                tangent1_offset_h[c] = float(np.dot(t1, world_anchor))
                tangent2_offset_h[c] = float(np.dot(t2, world_anchor))

                # Kinematic anchor velocity for spinning shapes.
                if (
                    s_idx >= 0
                    and s_idx < self._shape_omega_h.shape[0]
                    and self._shape_omega_h[s_idx] != 0.0
                    and shape_transform_np is not None
                ):
                    xf = shape_transform_np[s_idx]
                    shape_p = np.asarray(xf[:3], dtype=np.float64)
                    shape_q = np.asarray(xf[3:], dtype=np.float64)
                    # axis_world = rotate (0,0,1) by shape_q.
                    axis_world = _quat_rotate_z_axis(shape_q)
                    r_local = world_anchor - shape_p
                    omega = float(self._shape_omega_h[s_idx])
                    v_anchor = omega * np.cross(axis_world, r_local)
                    self._contact_v_anchor_h[c] = v_anchor

            self._contact_tangent1_d.assign(t1_h)
            self._contact_tangent2_d.assign(t2_h)
            # Stage B residual reads these host arrays (no device-side offset
            # buffer exists today).  Stash the BASE offsets (no dt-shift); the
            # dt-shift from v_anchor is applied at step() entry.
            self._contact_tangent1_offset_h_base = tangent1_offset_h.copy()
            self._contact_tangent2_offset_h_base = tangent2_offset_h.copy()
            self._contact_tangent1_offset_h = tangent1_offset_h
            self._contact_tangent2_offset_h = tangent2_offset_h
```

(The non-kinematic case: `_contact_v_anchor_h` is left at zeros, so the dt-shift in Step A.5 adds zero and the offsets equal the base.)

Add a small helper near the top of the file (next to `compute_tangent_basis`):

```python
def _quat_rotate_z_axis(q: np.ndarray) -> np.ndarray:
    """Apply quaternion ``q = (qx, qy, qz, qw)`` to the local +Z unit vector.

    Returns the world-frame direction of a shape's local +Z, used as the spin
    axis for Newton's cylinder primitive (extends along local +Z).
    """
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Quaternion · z_unit via Rodrigues' formula (specialized for z=(0,0,1)).
    return np.array(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dtype=np.float64,
    )
```

### Step A.5: Apply `v_anchor·dt` at step entry

The tangent offsets above are *purely* `dot(t, world_anchor)`. To make the anchor "advance" by `v_anchor·dt` per step, we need to add `dt · dot(t, v_anchor)` to the offsets that the NSN actually reads.

Locate the `step()` method in `solver_fba.py` (the `dt` parameter is in scope). At the entry of `step()` — AFTER `update_contacts` runs (which it does before the PD outer loop) and BEFORE the PD outer loop — add:

```python
        # Apply kinematic anchor advancement: tangent_offset = base + dt · dot(t, v_anchor).
        # update_contacts cached the base offsets and v_anchor; we apply the
        # dt-dependent shift here so the residual computation reads the right
        # values.  When ω = 0 everywhere, v_anchor is all zeros and this is a
        # no-op (offsets equal base).
        if (
            self.friction
            and self._contact_count > 0
            and hasattr(self, "_contact_v_anchor_h")
        ):
            M = self._contact_count
            v_anchor = self._contact_v_anchor_h[:M]
            t1 = self._contact_tangent1_d.numpy()[:M].astype(np.float64)
            t2 = self._contact_tangent2_d.numpy()[:M].astype(np.float64)
            shift1 = dt * np.einsum("ij,ij->i", t1, v_anchor)
            shift2 = dt * np.einsum("ij,ij->i", t2, v_anchor)
            self._contact_tangent1_offset_h = self._contact_tangent1_offset_h_base + shift1
            self._contact_tangent2_offset_h = self._contact_tangent2_offset_h_base + shift2
```

This way the *base* offsets in `_contact_tangent{1,2}_offset_h_base` stay pure `dot(t, anchor)`, and the dt-dependent shift is applied only at step entry. If `dt` changes between steps, the shift is re-applied with the new dt.  The residual reader at `_compute_contact_residual_friction` (around line 926-927) reads `_contact_tangent1_offset_h` / `_contact_tangent2_offset_h` unchanged — no signature changes required there.

### Step A.6: Re-run the tests

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k SolverFBAKinematicCylinder
```
Expected: 3 PASS.

### Step A.7: Full FBA suite regression

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba
```
Expected: 83 prior + 3 new = 86 PASS. If any prior friction test fails, the new kinematic branch is gating things it shouldn't — verify the `if self._shape_omega_h[s_idx] != 0.0` guard.

### Step A.8: Commit

```bash
cd /home/ziqiu/work/newton
git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Add per-shape kinematic angular velocity for SolverFBA friction"
```

---

## Task B: SqueezingBall demo

**Files:**
- Create: `scripts/fba_demo5_squeezing_ball.py`

### Step B.1: Create the demo script

Create `/home/ziqiu/work/newton/scripts/fba_demo5_squeezing_ball.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Demo 5: SqueezingBall — CudaTests reproduction.

A 7129-particle Neo-Hookean tet ball drops between 2 pairs of rolling
cylinders (mu=0.5, |ω|=3 rad/s) onto a static plane (mu=0.5).  The rolling
cylinders should friction-drag the ball laterally as it passes through.

Outputs PNGs + perf row to ``scripts/contact_demos_out/demo5/``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA
from scripts.fba_cudatest_bench.medit import load_medit_mesh
from scripts.fba_cudatest_bench.perf import record_row, stats_from_times_ms

# ---------------------------------------------------------------------------
# Constants from RealSim CudaTests/SqueezingBall
# ---------------------------------------------------------------------------
MESH_PATH = Path(
    "/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/ball_volume_7129P.mesh"
)
DT = 0.01
TOTAL_FRAMES = 600
PD_ITERATIONS = 5
NSN_ITERATIONS = 1  # Phase 1 default; RealSim "iterations: 10" is a cap
GRAVITY = -10.0  # Y-down

YOUNG = 1.0e4
POISSON = 0.4
OBJ_MASS = 1000.0

# Ball transform from ball_7k.json: scale=(3,3,3), trans=(0, 2.6, 0)
BALL_SCALE = 3.0
BALL_TRANS = np.array([0.0, 2.6, 0.0])

# Cylinder geometry (radius=1, half_height=3 — cosmetic; collision is along axis).
CYL_RADIUS = 1.0
CYL_HALF_HEIGHT = 3.0
FRICTION_MU = 0.5

# (base, axis_world, omega_rad_s) per cylinder.
CYLINDERS: list[tuple[np.ndarray, np.ndarray, float]] = [
    (np.array([0.0, -1.0, -1.5]), np.array([1.0, 0.0, 0.0]), -3.0),  # cyl0
    (np.array([0.0, -1.0, +1.5]), np.array([1.0, 0.0, 0.0]), +3.0),  # cyl1
    (np.array([-1.5, -3.5, 0.0]), np.array([0.0, 0.0, 1.0]), +3.0),  # cyl2
    (np.array([+1.5, -3.5, 0.0]), np.array([0.0, 0.0, 1.0]), -3.0),  # cyl3
]

# Plane at y = -10.
PLANE_BASE = np.array([0.0, -10.0, 0.0])

OUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo5"
SNAPSHOT_FRAMES = [0, 50, 150, 300, 450, 599]


def transform_verts(verts_local: np.ndarray) -> np.ndarray:
    return verts_local * BALL_SCALE + BALL_TRANS


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def cyl_xform(base: np.ndarray, axis: np.ndarray) -> wp.transform:
    """Build a Newton transform that orients a Z-aligned cylinder along ``axis``."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(z, a)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, a))
    if s < 1e-9:
        if c > 0:
            q = wp.quat_identity()
        else:
            q = wp.quat(1.0, 0.0, 0.0, 0.0)
    else:
        ang = math.atan2(s, c)
        axis_n = v / s
        half = ang * 0.5
        sh = math.sin(half)
        q = wp.quat(axis_n[0] * sh, axis_n[1] * sh, axis_n[2] * sh, math.cos(half))
    return wp.transform(wp.vec3(*base.tolist()), q)


def build_model() -> tuple:
    verts_local, tets = load_medit_mesh(MESH_PATH)
    verts_world = transform_verts(verts_local)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)

    total_vol = 0.0
    for t in tets:
        v0, v1, v2, v3 = (verts_world[i] for i in t)
        total_vol += abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
    density = OBJ_MASS / max(total_vol, 1e-12)

    mu, lam = lame_from_young_poisson(YOUNG, POISSON)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_world],
        indices=tets.reshape(-1).tolist(),
        density=density,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
        add_surface_mesh_edges=False,
    )

    # 4 cylinders + 1 plane (in order — shape indices 0..4).
    for base, axis, _omega in CYLINDERS:
        builder.add_shape_cylinder(
            body=-1,
            xform=cyl_xform(base, axis),
            radius=CYL_RADIUS,
            half_height=CYL_HALF_HEIGHT,
        )
    # Plane at y = -10 (Y-up world).  Plane equation `0·x + 1·y + 0·z + 10 = 0`
    # gives y = -10 with normal (0, 1, 0).
    builder.add_shape_plane(
        plane=(0.0, 1.0, 0.0, 10.0),
        body=-1,
        width=0.0,   # 0 = infinite plane
        length=0.0,
    )

    model = builder.finalize()

    # Set friction mu on every shape (0..4).
    if hasattr(model, "shape_material_mu") and model.shape_material_mu is not None:
        mu_arr = model.shape_material_mu.numpy()
        mu_arr[:] = FRICTION_MU
        model.shape_material_mu.assign(mu_arr.astype(np.float32))

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()

    return model, pipeline, contacts, verts_world, mu, lam


def render_frame(ax, q: np.ndarray, frame: int) -> None:
    ax.clear()
    step = max(1, len(q) // 500)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 1], cmap="viridis")
    ax.set_title(f"SqueezingBall — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-11, 6)
    ax.set_zlim(-4, 4)


def run() -> dict:
    model, pipeline, contacts, verts_world, mu, lam = build_model()
    n_tets = len(model.tet_indices.numpy()) // 4
    print(f"  particles={model.particle_count}  tets={n_tets}  shapes={model.shape_count}")

    # Cylinders are shape indices 0..3 (plane is 4).
    shape_omega = {i: CYLINDERS[i][2] for i in range(4)}

    n_max_contacts = model.particle_count  # upper bound
    mu_override = np.full(n_max_contacts, FRICTION_MU, dtype=np.float64)

    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=True,
        stretching_model="neohookean",
        mu=mu,
        lam=lam,
        mu_per_pair_override=mu_override,
        shape_angular_velocity=shape_omega,
    )
    s_in = model.state()
    s_out = model.state()

    snapshots: dict[int, np.ndarray] = {}
    step_times: list[float] = []

    for frame in range(TOTAL_FRAMES + 1):
        if frame in SNAPSHOT_FRAMES:
            snapshots[frame] = s_in.particle_q.numpy().copy()
        if frame == TOTAL_FRAMES:
            break

        wp.synchronize_device()
        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        s_in, s_out = s_out, s_in
        wp.synchronize_device()
        step_times.append(1000.0 * (time.perf_counter() - t0))

    q_final = s_in.particle_q.numpy()
    finite_ok = bool(np.all(np.isfinite(q_final)))
    stats = stats_from_times_ms(step_times)
    stats["stable"] = finite_ok
    stats["min_y"] = float(q_final[:, 1].min())
    stats["x_drift"] = float(q_final[:, 0].mean())  # ball lateral drift from friction-drag
    stats["z_drift"] = float(q_final[:, 2].mean())
    return {"snapshots": snapshots, "stats": stats}


def main() -> None:
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Demo 5: SqueezingBall")
    print("=" * 60)

    result = run()
    stats = result["stats"]
    print(
        f"  mean_ms={stats['mean_ms']:.2f}  median={stats['median_ms']:.2f}  p95={stats['p95_ms']:.2f}"
    )
    print(
        f"  stable={stats['stable']}  min_y={stats['min_y']:.3f}  "
        f"x_drift={stats['x_drift']:.3f}  z_drift={stats['z_drift']:.3f}"
    )

    fig = plt.figure(figsize=(4 * len(result["snapshots"]), 4))
    fig.suptitle("Demo 5: SqueezingBall")
    for col, (frame, q) in enumerate(sorted(result["snapshots"].items())):
        ax = fig.add_subplot(1, len(result["snapshots"]), col + 1, projection="3d")
        render_frame(ax, q, frame)
    plt.tight_layout()
    out = OUT_DIR / "squeezing_ball_strip.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  snapshot strip: {out}")

    record_row("SqueezingBall", stats)
    print("  perf row saved to scripts/fba_cudatest_rows.json")


if __name__ == "__main__":
    main()
```

### Step B.2: Run the demo

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo5_squeezing_ball.py
```

Expected output:
- `particles=7129 tets=<some_count> shapes=5` (4 cyls + 1 plane)
- `mean_ms=<X.XX>` (target ≤ 54 ms)
- `stable=True`
- `min_y` close to -10 (ball sits on plane) at final frame
- `x_drift`, `z_drift` nontrivial (friction from rolling cylinders carries the ball laterally)
- Writes `scripts/contact_demos_out/demo5/squeezing_ball_strip.png`
- Updates `scripts/fba_cudatest_rows.json`

### Step B.3: Visual sanity check

Open `scripts/contact_demos_out/demo5/squeezing_ball_strip.png`. Confirm:
- Frame 0: ball is a sphere-blob at y ≈ 2.6
- Frame 50-150: ball squeezed by first cyl pair (cyl0/cyl1 at y=-1, axis ±X); ball compressed in Z direction
- Frame 300: ball threading through to second cyl pair (cyl2/cyl3 at y=-3.5, axis ±Z); compressed in X
- Frame 450-599: ball rests on plane at y ≈ -10, possibly drifted laterally from cylinder friction
- No NaN, no fragmentation

If the ball passes straight through (no deformation), friction or contact is wrong — debug. If the ball ricochets out, the kinematic offset sign is likely wrong — flip sign in the cross product.

### Step B.4: Commit

```bash
cd /home/ziqiu/work/newton
git add scripts/fba_demo5_squeezing_ball.py
git commit -m "Add CudaTests SqueezingBall demo for SolverFBA"
```

---

## Task C: Perf gate + final docs

**Files:**
- Modify: `scripts/contact_demos_out/README.md`
- Modify: `docs/superpowers/fba-dev-progress.md`

### Step C.1: Perf gate

```bash
cd /home/ziqiu/work/newton
python3 -c "
import json
b = json.load(open('scripts/cudatest_baselines.json'))['SqueezingBall']
f = json.load(open('scripts/fba_cudatest_rows.json'))['SqueezingBall']
print(f'RealSim mean_ms = {b[\"mean_ms\"]:.2f}')
print(f'FBA     mean_ms = {f[\"mean_ms\"]:.2f}')
print(f'ratio (FBA/RealSim) = {f[\"mean_ms\"]/b[\"mean_ms\"]:.2f}')
"
```

**Acceptance**: ratio ≤ 1.0. If ratio > 1.0, profile (synchronous wp.synchronize_device() around each phase) to find the bottleneck:
- `update_contacts` host-loop (now iterates over per-contact shape lookups + cross products)
- isodof Schur build (should be cached from Phase 1)
- Anything new

If `update_contacts` is the bottleneck, vectorize the per-contact host loop with NumPy (precompute `shape_q[:, None] @ z_axis` for all contact shapes in one shot).

If ratio is ≤ 1.0, proceed. If 1.0 < ratio ≤ 1.5, document the gap and proceed (Phase 2 acceptance is qualitative; perf is bonus). If ratio > 1.5, STOP and diagnose.

### Step C.2: Update Demo 5 section in `scripts/contact_demos_out/README.md`

Append after Demo 4:

```markdown
## Demo 5 — SqueezingBall (CudaTests reproduction)

`scripts/fba_demo5_squeezing_ball.py`

7129-particle tet ball (Neo-Hookean, E=1e4, ν=0.4, mass=1000),
600 frames at dt=0.01, gravity=-10 Y.  Two pairs of rolling cylinders
(r=1, mu=0.5, |ω|=3 rad/s — first pair X-axis, second pair Z-axis) drag
the ball laterally; plane at y=-10 catches it.

| Solver        | mean ms | median ms | p95 ms | stable |
|---------------|---:|---:|---:|:---:|
| RealSim NSN_CUDA (baseline) | 53.99 | – | – | ✓ |
| SolverFBA (isodof + kinematic) | <FBA_MS_HERE> | <MED> | <P95> | ✓ |

Acceptance:
- Visual: ball squeezed by first cyl pair (compressed in Z), then second pair (compressed in X), settles on plane with lateral drift from friction-drag
- Stability: `stable=True`, `min_y ≈ -10` at final frame
- Perf: ms/step ≤ RealSim baseline

### New infrastructure

`SolverFBA(shape_angular_velocity={shape_idx: ω})` — per-shape spin about local
+Z axis. Static geometry stays at `body=-1`; the rotation appears only in the
Stage B friction tangent offsets as `v_anchor·dt` where
`v_anchor = ω · axis_world × (anchor − shape_center)`. Mirrors RealSim's
`cylindercollisions[i].rollingvel` config.

Snapshots: `demo5/squeezing_ball_strip.png`.
```

(Replace `<FBA_MS_HERE>`, `<MED>`, `<P95>` with the actual measured values.)

### Step C.3: Append dated entry to `docs/superpowers/fba-dev-progress.md`

Prepend the following section before existing entries (it's a chronological log; latest first):

```markdown
## 2026-05-XX — Phase 2 (SqueezingBall) complete

**Acceptance criteria** from `docs/superpowers/specs/2026-05-16-cudatests-full-reproduction-spec.md` §5 Phase 2:
- ✓ Visual: ball squeezed by 2 cylinder pairs, friction-driven lateral drift, settles on plane
- ✓ Tests: 86/86 FBA unit tests pass (83 prior + 3 new for kinematic cylinder path)
- ✓ Perf: `mean_ms ≤ RealSim baseline` (FBA <X> vs RealSim 53.99 = <RATIO>×)

**Commits this phase** (chronological, all on `ziqiu/fba-solver-design`):
- `<sha1>` Add per-shape kinematic angular velocity for SolverFBA friction (Task A)
- `<sha2>` Add CudaTests SqueezingBall demo for SolverFBA (Task B)
- `<sha3>` Document SqueezingBall demo and Phase 2 completion (Task C)

**New infrastructure**:
- `SolverFBA(shape_angular_velocity={shape_idx: ω_rad_s})` — per-shape angular
  velocity about local +Z (Newton's cylinder primitive axis).  Surface anchor
  velocity is computed at update_contacts and injected into Stage B tangent
  residual as `dt·dot(t, v_anchor)`.  Normal contact handling unchanged.

**Next phase**: P3 CrossingGingerbreadman — TET NH gingerbreadman_10k pulled through a 13-cylinder corridor (gravity=0, PULLING pin, all cylinders mu=0).  Reuses P1's PULLING infrastructure and P2's many-cylinder broad-phase pattern.
```

(Substitute `<sha1>`, `<sha2>`, `<sha3>`, `<X>`, `<RATIO>` after the previous commits land.)

### Step C.4: Pre-commit + commit

```bash
cd /home/ziqiu/work/newton
uvx pre-commit run -a 2>&1 | tail -10
git add scripts/contact_demos_out/README.md docs/superpowers/fba-dev-progress.md
git commit -m "Document SqueezingBall demo and Phase 2 completion"
```

If pre-commit modifies unrelated files (ruff format on stale demos), leave those changes unstaged (out of scope for this commit).

---

## Acceptance Checklist

- [ ] All 86 FBA unit tests pass: `uv run --extra dev -m newton.tests -k test_solver_fba`
- [ ] `scripts/fba_cudatest_rows.json` has a `SqueezingBall` row with `stable=True` and `mean_ms ≤ 53.99` (or documented gap if > 1.0×)
- [ ] `scripts/contact_demos_out/demo5/squeezing_ball_strip.png` exists and shows the squeeze/drag/settle behaviour
- [ ] `docs/superpowers/fba-dev-progress.md` has the 2026-05-XX Phase 2 entry
- [ ] `git log --oneline -3` shows the three task commits on `ziqiu/fba-solver-design`

---

## Out of Scope (Deferred to Phase 3+)

- Multi-axis spin (a single shape rotating about an axis other than its primitive +Z). RealSim's CableGrabRaptor gripper ROLLING pins around arbitrary axes — handled in Phase 7 via the `set_pin_targets` route, not via this kinematic shape API.
- Time-varying ω (acceleration / stop conditions). Phase 2 cylinders spin at constant rate.
- Per-shape mass-spring damping. Neither RealSim nor FBA implements damping (Audit §N).

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `cross(axis, r_local)` sign error → ball pushed wrong direction | Med | Step B.3 visual check; if wrong, flip sign in `v_anchor = -ω · cross(...)` |
| `shape_transform.q` is in body-local frame for shapes attached to dynamic bodies | Low | Phase 2 all cylinders are body=-1; if Phase 3+ needs dynamic-body cylinders, apply `body_q` first |
| Schur cache invalidation misses the kinematic-offset change | Med | `update_contacts` calls `_invalidate_schur_cache()` first — same path as Phase 1, no extra plumbing needed |
| Friction overshoots and ball ejects from cyl pair | Low | `mu=0.5` + closed-form Coulomb cone projection is well-tested in Phase 1's Demo 3 |
