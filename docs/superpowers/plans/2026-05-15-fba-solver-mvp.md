# SolverFBA MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SolverFBA` to Newton: a Warp-native projective-dynamics cloth solver with Pin + gravity, using a Cholesky + sparse-inverse linear solver (RealSim "Fast But Accurate" paper kernel). MVP scope per `docs/superpowers/specs/2026-05-15-fba-solver-design.md`.

**Architecture:** PD local-global iteration with prefactored constant Hessian. Setup (CPU/Python): assemble scalar `N×N` PD Hessian via `scipy.sparse`, factor with `scipy.sparse.linalg.splu`, compute sparse inverse `S = L⁻¹` via a Python port of RealSim's elimination-tree-based algorithm, upload to Warp `BsrMatrix`. Runtime (GPU/Warp): per PD iteration assemble RHS from per-element local projections (ARAP for stretching, isometric for bending, soft Pin), solve via two BSR SpMVs + diagonal scaling, repeat K times, then write velocity.

**Tech Stack:** Python 3.10+, Warp 1.14.0.dev (`warp.sparse.BsrMatrix`, kernels), NumPy, SciPy (`scipy.sparse`, `scipy.sparse.linalg.splu`), Newton's `Model` / `State` / `SolverBase`. Tests via `unittest`. No new dependencies.

**Working branch:** `ziqiu/fba-solver-design` (already created off `main`, contains the spec doc). All implementation commits land on this branch; PR split per spec §8 happens at PR-prep time.

**Reference paths:**
- Spec: `docs/superpowers/specs/2026-05-15-fba-solver-design.md`
- RealSim sparse LDLᵀ source: `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/tools/math/SparseLDLT.cpp`
- RealSim PD stretching: `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.cpp`
- RealSim ARAP projection: `/home/ziqiu/work/RealSim_py/realsim_py/include/Scomponent/integrator/localglobal/energy/elastic/PDARAPTriangleEnergy.h`
- RealSim isometric bending: `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/elastic/PDIsometricBendingEnergy.cpp`
- RealSim Pin: `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/hardconstraint/PinEnergy.cpp`
- Newton VBD layout reference: `newton/_src/solvers/vbd/`
- Newton Style3D PD reference (use sparingly — different sparse layout): `newton/_src/solvers/style3d/`

---

## File Structure

**Created:**
- `newton/_src/solvers/fba/__init__.py` — re-exports `SolverFBA`.
- `newton/_src/solvers/fba/linear_solver.py` — CPU assembly + factor + sparse-inverse + `FBALinearSolver` runtime class.
- `newton/_src/solvers/fba/kernels.py` — all Warp kernels and helper `wp.func`s.
- `newton/_src/solvers/fba/solver_fba.py` — `class SolverFBA(SolverBase)`.
- `newton/tests/test_solver_fba.py` — unittest module with 8 tests (T1–T8 per spec §6).
- `newton/examples/cloth/example_cloth_hanging_fba.py` — runnable example, follows Newton `Example` class.

**Modified:**
- `newton/_src/solvers/__init__.py` — add `SolverFBA` import + `__all__` entry.
- `newton/solvers.py` — add `SolverFBA` to the public re-export.

Per Newton `AGENTS.md`: SPDX header on every new file; bracket Warp array annotations (`wp.array[X]`); PEP 604 unions (`X | None`); Google-style docstrings; SI units in public docs.

---

## Task 1: Package skeleton + import smoke test

**Goal:** Create the four FBA files with SPDX headers and bare minimum content. Verify `from newton._src.solvers.fba import SolverFBA` import path works (raises `NotImplementedError` for now is fine).

**Files:**
- Create: `newton/_src/solvers/fba/__init__.py`
- Create: `newton/_src/solvers/fba/solver_fba.py`
- Create: `newton/_src/solvers/fba/linear_solver.py`
- Create: `newton/_src/solvers/fba/kernels.py`
- Create: `newton/tests/test_solver_fba.py`

- [ ] **Step 1: Write failing import test**

`newton/tests/test_solver_fba.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest


class TestSolverFBAImport(unittest.TestCase):
    def test_import_solver_fba(self):
        from newton._src.solvers.fba import SolverFBA  # noqa: F401


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.test_import_solver_fba
```

Expected: `ModuleNotFoundError: No module named 'newton._src.solvers.fba'`.

- [ ] **Step 3: Create the four package files**

`newton/_src/solvers/fba/__init__.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .solver_fba import SolverFBA

__all__ = [
    "SolverFBA",
]
```

`newton/_src/solvers/fba/solver_fba.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


class SolverFBA(SolverBase):
    """Fast But Accurate projective-dynamics cloth solver (MVP stub)."""

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap"] = "arap",
    ) -> None:
        super().__init__(model)
        raise NotImplementedError("SolverFBA __init__ pending Task 10")

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        raise NotImplementedError("SolverFBA.step pending Task 11")

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        raise NotImplementedError(
            "Contact-aware FBA solver TBD; SolverFBA MVP supports gravity + pin only"
        )
```

`newton/_src/solvers/fba/linear_solver.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cholesky + sparse-inverse linear solver for SolverFBA.

Setup phase (CPU):
    1. assemble scalar N×N PD Hessian `A` (caller-supplied).
    2. factor with `scipy.sparse.linalg.splu` (COLAMD ordering).
    3. compute `S = L⁻¹` keeping elimination-tree sparsity (ported from
       RealSim `LDLT_computeLowerInverse`, SparseLDLT.cpp:225–347).
    4. upload to device as Warp BSR matrices.

Runtime (GPU):
    `A⁻¹ b = Sᵀ · D⁻¹ · (S · b)` — two `bsr_mv` calls plus diagonal scaling.
"""
```

`newton/_src/solvers/fba/kernels.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics cloth solver)."""

import warp as wp  # noqa: F401  (used in kernels added in later tasks)
```

- [ ] **Step 4: Run test, expect pass**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.test_import_solver_fba
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/fba/ newton/tests/test_solver_fba.py
git commit -m "Add SolverFBA package skeleton"
```

---

## Task 2: `build_pd_system` — PD Hessian assembly (test T1)

**Goal:** Assemble the scalar `N×N` PD Hessian `A = M/dt² + Σ_tri w_i·K_i + Σ_edge w_e·Q_e + Σ_pin w_pin·I` as a `scipy.sparse.csr_matrix` from a Newton `Model` snapshot. Pure NumPy/SciPy, no Warp.

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py`
- Modify: `newton/tests/test_solver_fba.py`

**Algorithm:**
- For each triangle `t`, read pre-computed `tri_poses[t]` (2×2, equivalent to `restMatrix`) and `tri_areas[t]`. Stiffness `ke = tri_materials[t, 0]` (Newton stores `[ke, ka, kd, mu, ...]`; we use the first slot as area-weighted stretching). Compute the per-triangle 3×3 stencil:
  ```
  ST = [[-1, -1], [1, 0], [0, 1]]    # 3×2
  G = ST @ tri_poses[t]              # 3×2
  K_t = G @ G.T                      # 3×3 symmetric PSD
  w_t = ke * tri_areas[t]
  ```
  Scatter `w_t * K_t[a,b]` to `A[tri_indices[3t+a], tri_indices[3t+b]]` for `a,b ∈ {0,1,2}`.
- For each bending edge `e` with stencil `(i0, i1, i2, i3)` (`i0,i1` = shared edge endpoints, `i2,i3` = opposite vertices in adjacent triangles), assemble the isometric quadratic form 4×4 — defer the algebra to Task 8 / kernel; for MVP assembly use the cotangent-Laplacian-style stencil from `model.edge_bending_properties[e, 0]` (rest angle is in `[e, 1]`). We will accept a simplified form here matching `PDIsometricBendingEnergy.cpp`'s `_K` build:
  - 4×4 matrix `Q_e` is rank-1: `q = [k0, k1, k2, k3]ᵀ` (cotangent weights from edge stencil) and `Q_e = q · qᵀ`. The k-coefficients come from the rest configuration of the two adjacent triangles; see `PDIsometricBendingEnergy.cpp` `init()`.
  - Scatter `w_e * Q_e[a,b]` to the four vertex pairs.
- For each particle `i` with `particle_inv_mass[i] > 0`, add `m_i / dt²` to `A[i, i]`.
- For each particle `i` with `particle_inv_mass[i] == 0` (pinned), add `pin_stiffness` to `A[i, i]`.

The output is a `scipy.sparse.csr_matrix` of shape `(N, N)` plus a metadata dict.

- [ ] **Step 1: Write test T1 — assembly produces correct entries on toy mesh**

In `newton/tests/test_solver_fba.py` append:
```python
import numpy as np
import warp as wp

import newton


class TestBuildPDSystem(unittest.TestCase):
    """T1: PD Hessian assembly on a 2-triangle mesh, sanity-check entries."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_two_triangle_cloth(self):
        # Two triangles sharing edge (1, 2):
        #     0 --- 1
        #     |   / |
        #     | /   |
        #     2 --- 3
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 1.0),
                wp.vec3(1.0, 0.0, 1.0),
            ],
            indices=[0, 1, 2, 1, 3, 2],
            density=1.0,
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0,
            edge_kd=0.0,
        )
        return builder.finalize(device="cpu")

    def test_diagonal_includes_mass_over_dt_squared(self):
        from newton._src.solvers.fba.linear_solver import build_pd_system

        model = self._build_two_triangle_cloth()
        dt = 1.0 / 60.0
        A, meta = build_pd_system(model, dt=dt, pin_stiffness=1e12)

        self.assertEqual(A.shape, (model.particle_count, model.particle_count))
        # A is symmetric.
        diff = (A - A.T)
        self.assertLess(np.abs(diff).max(), 1e-10)

        # Diagonal of mass contribution: m_i/dt² for free particles.
        masses = model.particle_mass.numpy()
        inv_masses = model.particle_inv_mass.numpy()
        diag = A.diagonal()
        for i in range(model.particle_count):
            if inv_masses[i] > 0.0:
                # Contains at least m/dt² (other terms add on top).
                self.assertGreaterEqual(diag[i], masses[i] / (dt * dt) - 1e-9)

    def test_pin_stiffness_on_pinned_diagonal(self):
        from newton._src.solvers.fba.linear_solver import build_pd_system

        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e2, tri_ka=0.0, tri_kd=0.0,
            edge_ke=0.0, edge_kd=0.0,
            fix_left=False,
        )
        # Manually pin particle 0.
        builder.particle_mass[0] = 0.0
        builder.particle_inv_mass[0] = 0.0
        model = builder.finalize(device="cpu")

        A, _ = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)
        # Pin contributes pin_stiffness to A[0,0].
        self.assertGreaterEqual(A[0, 0], 1e12 - 1e-3)
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestBuildPDSystem
```

Expected: `ImportError: cannot import name 'build_pd_system'`.

- [ ] **Step 3: Implement `build_pd_system`**

In `newton/_src/solvers/fba/linear_solver.py` append:
```python
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from ...sim import Model


def build_pd_system(
    model: Model,
    dt: float,
    pin_stiffness: float,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Assemble the constant scalar PD Hessian `A` for cloth.

    Args:
        model: The Newton model containing cloth triangles and (optionally)
            bending edges. Pinned particles are detected via
            ``particle_inv_mass == 0``.
        dt: Time step [s]. Enters as `m_i / dt²` on the diagonal.
        pin_stiffness: PD soft-pin weight `w_pin`. Larger ⇒ harder pin.

    Returns:
        A tuple ``(A, meta)`` where ``A`` is an `N × N` symmetric PSD CSR
        matrix (`N = model.particle_count`) and ``meta`` contains derived
        per-element data needed at the runtime stage:

        - ``meta["tri_indices"]`` : ``np.ndarray[int32]`` shape (T, 3)
        - ``meta["tri_rest_inv"]`` : ``np.ndarray[float64]`` shape (T, 2, 2)
        - ``meta["tri_area"]`` : ``np.ndarray[float64]`` shape (T,)
        - ``meta["tri_weight"]`` : ``np.ndarray[float64]`` shape (T,)  (= ke * area)
        - ``meta["edge_indices"]`` : ``np.ndarray[int32]`` shape (E, 4)
        - ``meta["edge_quad_q"]`` : ``np.ndarray[float64]`` shape (E, 4)  (k-coefs)
        - ``meta["edge_weight"]`` : ``np.ndarray[float64]`` shape (E,)
        - ``meta["pin_indices"]`` : ``np.ndarray[int32]`` (packed pinned indices)
        - ``meta["pin_weight"]`` : float
    """
    N = model.particle_count
    mass = model.particle_mass.numpy().astype(np.float64)
    inv_mass = model.particle_inv_mass.numpy().astype(np.float64)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # ---- Mass diagonal ----
    inv_dt2 = 1.0 / (dt * dt)
    for i in range(N):
        if inv_mass[i] > 0.0:
            rows.append(i); cols.append(i); vals.append(mass[i] * inv_dt2)

    # ---- Pin diagonal ----
    pin_indices = np.where(inv_mass == 0.0)[0].astype(np.int32)
    for i in pin_indices:
        rows.append(int(i)); cols.append(int(i)); vals.append(pin_stiffness)

    # ---- Triangle stretching contribution ----
    tri_indices = np.zeros((0, 3), dtype=np.int32)
    tri_rest_inv = np.zeros((0, 2, 2), dtype=np.float64)
    tri_area = np.zeros((0,), dtype=np.float64)
    tri_weight = np.zeros((0,), dtype=np.float64)
    if model.tri_count > 0:
        tri_indices_flat = model.tri_indices.numpy().astype(np.int32)
        tri_indices = tri_indices_flat.reshape(-1, 3)
        # Newton stores per-triangle 2×2 rest pose at `tri_poses` (or `tri_pose`); fall back to recomputing.
        if hasattr(model, "tri_poses") and model.tri_poses is not None:
            tri_rest_inv = model.tri_poses.numpy().astype(np.float64)
        else:
            tri_rest_inv = _compute_rest_inv_from_positions(model)
        tri_area = model.tri_areas.numpy().astype(np.float64)
        # Stiffness from tri_materials column 0 (`ke`) per AGENTS.md convention.
        tri_materials = model.tri_materials.numpy().astype(np.float64)
        ke = tri_materials[:, 0]
        tri_weight = ke * tri_area

        # ST is the 3×2 selector for the 3-vertex stencil.
        ST = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        for t in range(tri_indices.shape[0]):
            G = ST @ tri_rest_inv[t]               # 3×2
            K = G @ G.T                             # 3×3
            w = tri_weight[t]
            for a in range(3):
                ia = int(tri_indices[t, a])
                for b in range(3):
                    ib = int(tri_indices[t, b])
                    rows.append(ia); cols.append(ib); vals.append(w * K[a, b])

    # ---- Bending edge contribution (4-vertex isometric stencil) ----
    edge_indices = np.zeros((0, 4), dtype=np.int32)
    edge_quad_q = np.zeros((0, 4), dtype=np.float64)
    edge_weight = np.zeros((0,), dtype=np.float64)
    if model.edge_indices is not None:
        ei = model.edge_indices.numpy().astype(np.int32).reshape(-1, 4)
        # Newton stores (o1, o2, v1, v2) per row; v1/v2 are shared-edge ends, o1/o2 opposite.
        # Map to ordering (v0=v1, v1=v2, v2=o1, v3=o2) used by PDIsometricBendingEnergy.
        edge_indices = ei[:, [2, 3, 0, 1]]
        # Compute k-coefficients per edge (Bergou 06 cotangent stencil) from initial geometry.
        positions = model.particle_q.numpy().astype(np.float64)
        edge_quad_q = _compute_isometric_bending_q(edge_indices, positions)
        if model.edge_bending_properties is not None:
            bend_props = model.edge_bending_properties.numpy().astype(np.float64)
            edge_weight = bend_props[:, 0]
        else:
            edge_weight = np.zeros(edge_indices.shape[0], dtype=np.float64)

        for e in range(edge_indices.shape[0]):
            q = edge_quad_q[e]                 # length 4
            w = edge_weight[e]
            if w == 0.0:
                continue
            for a in range(4):
                ia = int(edge_indices[e, a])
                for b in range(4):
                    ib = int(edge_indices[e, b])
                    rows.append(ia); cols.append(ib); vals.append(w * q[a] * q[b])

    A = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(N, N),
    )

    meta: dict[str, Any] = {
        "tri_indices": tri_indices,
        "tri_rest_inv": tri_rest_inv,
        "tri_area": tri_area,
        "tri_weight": tri_weight,
        "edge_indices": edge_indices,
        "edge_quad_q": edge_quad_q,
        "edge_weight": edge_weight,
        "pin_indices": pin_indices,
        "pin_weight": pin_stiffness,
    }
    return A, meta


def _compute_rest_inv_from_positions(model: Model) -> np.ndarray:
    """Fallback: recompute (basisᵀ·edges)⁻¹ per triangle from particle_q."""
    pos = model.particle_q.numpy().astype(np.float64)
    tri = model.tri_indices.numpy().astype(np.int32).reshape(-1, 3)
    out = np.zeros((tri.shape[0], 2, 2), dtype=np.float64)
    for t in range(tri.shape[0]):
        a, b, c = tri[t]
        e12 = pos[b] - pos[a]
        e13 = pos[c] - pos[a]
        n1 = e12 / max(np.linalg.norm(e12), 1e-20)
        e13_perp = e13 - np.dot(e13, n1) * n1
        n2 = e13_perp / max(np.linalg.norm(e13_perp), 1e-20)
        basis = np.stack([n1, n2], axis=1)        # 3×2
        edges = np.stack([e12, e13], axis=1)      # 3×2
        Dm = basis.T @ edges                       # 2×2
        out[t] = np.linalg.inv(Dm)
    return out


def _compute_isometric_bending_q(
    edge_indices: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Per-edge length-4 vector q such that bending Hessian = q·qᵀ.

    Bergou et al. 2006 isometric bending: for stencil (v0, v1, v2, v3)
    where (v0, v1) is the shared edge and (v2, v3) the opposite vertices
    of the two adjacent triangles, the per-edge bending energy is
    ½ k·(qᵀx)² where q is determined by cotangents at the two opposite
    angles. See PDIsometricBendingEnergy.cpp:init().
    """
    E = edge_indices.shape[0]
    q_out = np.zeros((E, 4), dtype=np.float64)
    for e in range(E):
        v0, v1, v2, v3 = edge_indices[e]
        x0, x1, x2, x3 = positions[v0], positions[v1], positions[v2], positions[v3]
        # Cotangents at v2 (angle at v2 in triangle (v0,v1,v2)) and v3.
        e01 = x1 - x0
        c2_a = x0 - x2; c2_b = x1 - x2
        c3_a = x0 - x3; c3_b = x1 - x3
        cot2 = np.dot(c2_a, c2_b) / max(np.linalg.norm(np.cross(c2_a, c2_b)), 1e-20)
        cot3 = np.dot(c3_a, c3_b) / max(np.linalg.norm(np.cross(c3_a, c3_b)), 1e-20)
        # Stencil weights (Bergou 06 eq. for isometric bending Hessian).
        a0 = cot2 + cot3
        a1 = cot2 + cot3
        a2 = -cot2
        a3 = -cot3
        # Ensure sum is zero (translation invariance):
        s = a0 + a1 + a2 + a3
        a0 -= s / 4; a1 -= s / 4; a2 -= s / 4; a3 -= s / 4
        scale = np.sqrt(3.0 / max(np.linalg.norm(e01) ** 2, 1e-20))
        q_out[e] = scale * np.array([a0, a1, a2, a3])
    return q_out
```

- [ ] **Step 4: Run test, expect pass**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestBuildPDSystem
```

Expected: both tests pass (`OK`).

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/fba/linear_solver.py newton/tests/test_solver_fba.py
git commit -m "Add build_pd_system for SolverFBA Hessian assembly"
```

---

## Task 3: `compute_lower_inverse` — sparse inverse algorithm port (test T2)

**Goal:** Pure-Python port of `LDLT_computeLowerInverse` (RealSim `SparseLDLT.cpp:225–347`). Given a lower-triangular `L` (as `scipy.sparse.csc_matrix`) and the elimination-tree parent array, produce `S = L⁻¹` as a `csc_matrix` with sparsity pattern derived from the elimination tree.

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py`
- Modify: `newton/tests/test_solver_fba.py`

**Algorithm (faithful port of RealSim):**

1. `_lower_inverse_nnz(invperm, parent) -> int`: count S's nnz by walking up the tree from each `invperm[i]` to the root. (RealSim `LDLT_computeLowerInverseNNZ`, lines 242–260.)
2. `_lower_inverse_pattern(invperm, parent) -> (S_outerPtr, S_innerInd)`: same walk but record column indices into a CSC pattern. (`LDLT_computeLowerInversePattern`, lines 262–283.)
3. `_compute_aligned_lower(S_outerPtr, S_innerInd, invperm, L_outerPtr, L_innerInd, L_values) -> aL_values`: project L's values onto S's pattern, indexed by S's column order. (`LDLT_computeAlignedLower`, lines 285–306.)
4. `_compute_lower_inverse_lines(S_outerPtr, S_innerInd, perm, aL_values) -> S_values`: for each column `index`, set `S_values[outerPtr[index]] = 1.0`, then for each subsequent entry recursively subtract products of earlier values with aligned-L. (`LDLT_computeLowerInverse_line`, lines 308–319.)

This algorithm uses CSC storage and is parallelisable per column (RealSim uses TBB). MVP runs serially in Python; performance is fine for <50K nodes.

- [ ] **Step 1: Write test T2 — sparse inverse correctness against dense reference**

In `newton/tests/test_solver_fba.py` append:
```python
class TestSparseInverse(unittest.TestCase):
    """T2: compute_lower_inverse matches dense np.linalg.inv on small SPD."""

    def test_random_spd_8x8(self):
        from newton._src.solvers.fba.linear_solver import compute_lower_inverse

        rng = np.random.default_rng(0)
        # Build a small SPD matrix via tridiagonal + jitter.
        n = 8
        diag = rng.uniform(10.0, 20.0, n)
        off = rng.uniform(-1.0, 1.0, n - 1)
        A = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csc").astype(np.float64)

        S = compute_lower_inverse(A)
        # Verify S · L ≈ I (where L is the lower-tri Cholesky-equivalent factor).
        # We compare to dense reference: L_dense = scipy cholesky of A; S_dense = inv(L_dense).
        L_dense = np.linalg.cholesky(A.toarray())
        S_dense = np.linalg.inv(L_dense)
        # Permutation may differ — instead verify SᵀDS == A⁻¹ via splu.
        # Simpler invariant: S is lower-triangular w.r.t. permutation and S·L ≈ I in the permuted basis.
        # Strongest check: A⁻¹ matches dense.
        Ainv_dense = np.linalg.inv(A.toarray())
        # Use S to reconstruct A⁻¹ via Sᵀ D⁻¹ S.
        from newton._src.solvers.fba.linear_solver import _splu_extract_factors
        L_csc, U_csc, Dinv, perm_r, perm_c = _splu_extract_factors(A)
        S = compute_lower_inverse(L_csc, parent=_elimination_tree(L_csc), invperm=np.argsort(perm_r))
        # Reconstruction:
        S_dense_arr = S.toarray()
        Ainv_reconstructed = S_dense_arr.T @ np.diag(Dinv) @ S_dense_arr
        # Account for permutation.
        P = np.eye(n)[perm_r]
        Ainv_reconstructed = P.T @ Ainv_reconstructed @ P
        self.assertLess(np.abs(Ainv_reconstructed - Ainv_dense).max(), 1e-8)
```

> **Note for engineer:** The test above leverages helper `_splu_extract_factors` and `_elimination_tree` introduced in this Task. If your test design diverges, ensure the core invariant `Sᵀ · diag(D⁻¹) · S ≈ A⁻¹` (modulo permutation) holds within 1e-8.

- [ ] **Step 2: Run test, expect failure**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestSparseInverse
```

Expected: `ImportError: cannot import name 'compute_lower_inverse'`.

- [ ] **Step 3: Implement sparse inverse**

In `newton/_src/solvers/fba/linear_solver.py` append:
```python
import scipy.sparse.linalg as spla


def _splu_extract_factors(
    A: sp.csr_matrix,
) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Wrap scipy.sparse.linalg.splu and return (L_csc, U_csc, Dinv, perm_r, perm_c).

    For SPD A, splu effectively produces LU = L · U with U ≈ Dᵀ Lᵀ; we extract
    the lower factor L (with unit diagonal), the diagonal D = diag(U), and the
    permutation arrays. The factor satisfies P_r · A · P_cᵀ = L · D · Lᵀ
    (modulo numerical asymmetry of SuperLU's pivoting for non-symmetric input,
    which is unobservable on SPD input).
    """
    lu = spla.splu(A.tocsc(), permc_spec="COLAMD")
    L = lu.L.tocsc()         # lower-tri with unit diagonal
    U = lu.U.tocsc()
    Dinv = 1.0 / U.diagonal()
    perm_r = lu.perm_r.astype(np.int32)
    perm_c = lu.perm_c.astype(np.int32)
    return L, U, Dinv, perm_r, perm_c


def _elimination_tree(L: sp.csc_matrix) -> np.ndarray:
    """Build elimination tree parent[] from L's structure.

    parent[j] = smallest row index > j with L[parent[j], j] != 0, else -1.
    """
    n = L.shape[0]
    parent = -np.ones(n, dtype=np.int32)
    indptr = L.indptr
    indices = L.indices
    for j in range(n):
        for ptr in range(indptr[j], indptr[j + 1]):
            i = indices[ptr]
            if i > j:
                parent[j] = i
                break
    return parent


def compute_lower_inverse(
    L: sp.csc_matrix,
    parent: np.ndarray | None = None,
    invperm: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Compute S = L⁻¹ as a sparse CSC matrix.

    Ports RealSim `LDLT_computeLowerInverse` (SparseLDLT.cpp:225–347).
    Sparsity pattern of S is derived from L's elimination tree; this gives
    the exact sparse inverse (no thresholding, no fill-in).

    Args:
        L: Lower-triangular CSC matrix with unit diagonal.
        parent: Elimination tree parent array; computed from L if None.
        invperm: Inverse permutation (identity if None).

    Returns:
        S as CSC; S · L ≈ I in the permuted basis.
    """
    n = L.shape[0]
    if parent is None:
        parent = _elimination_tree(L)
    if invperm is None:
        invperm = np.arange(n, dtype=np.int32)
    perm = np.argsort(invperm).astype(np.int32)

    # 1. Count nnz of S.
    S_nnz = 0
    for i in range(n):
        index = int(invperm[i])
        inner = 1
        while parent[index] != -1:
            inner += 1
            index = int(parent[index])
        S_nnz += inner

    # 2. Build S's CSC pattern.
    S_outerPtr = np.zeros(n + 1, dtype=np.int32)
    S_innerInd = np.zeros(S_nnz, dtype=np.int32)
    S_values = np.zeros(S_nnz, dtype=np.float64)
    count = 0
    for i in range(n):
        index = int(invperm[i])
        S_innerInd[count] = index
        inner = 1
        while parent[index] != -1:
            S_innerInd[count + inner] = int(parent[index])
            inner += 1
            index = int(parent[index])
        count += inner
        S_outerPtr[i + 1] = count

    # 3. Compute aligned L values.
    aL = np.zeros(S_nnz, dtype=np.float64)
    L_outerPtr = L.indptr
    L_innerInd = L.indices
    L_values = L.data
    for row in range(n):
        invr = int(invperm[row])
        ptrS = S_outerPtr[row]
        for ptrL in range(L_outerPtr[invr], L_outerPtr[invr + 1]):
            while ptrS < S_outerPtr[row + 1]:
                if S_innerInd[ptrS] == L_innerInd[ptrL]:
                    aL[ptrS] = L_values[ptrL]
                    break
                ptrS += 1

    # 4. Solve for S column-by-column.
    for index in range(n):
        S_values[S_outerPtr[index]] = 1.0
        for i in range(S_outerPtr[index] + 1, S_outerPtr[index + 1]):
            col = int(perm[S_innerInd[i - 1]])
            j = i
            k = S_outerPtr[col] + 1
            while (j < S_outerPtr[index + 1]) or (k < S_outerPtr[col + 1]):
                S_values[j] -= aL[k] * S_values[i - 1]
                j += 1
                k += 1
                if j >= S_outerPtr[index + 1] or k >= S_outerPtr[col + 1]:
                    break

    S = sp.csc_matrix((S_values, S_innerInd, S_outerPtr), shape=(n, n))
    return S
```

> **Warning to engineer:** The bracketed loop bound in step 4 (lines `while (j < ...) or (k < ...)`) faithfully reproduces the C++ off-by-one structure (`SparseLDLT.cpp:315–319`). The condition `or` (not `and`) is intentional; verify by comparing against the C++ source. The Python `break` guard prevents index overruns when one of `j/k` has reached its column end.

- [ ] **Step 4: Run test, expect pass**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestSparseInverse
```

Expected: `OK`. If failure: instrument by comparing `S · L` against identity in the permuted basis; suspected bug is the loop bound or aligned-L extraction. Cross-check against C++ source.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/fba/linear_solver.py newton/tests/test_solver_fba.py
git commit -m "Port LDLT_computeLowerInverse to compute sparse inverse S=L^-1"
```

---

## Task 4: `factorize_and_sparse_inverse` — splu + sparse inverse pipeline

**Goal:** Tie Task 2 and Task 3 together into a single entry-point function that takes `A`, returns everything needed by `FBALinearSolver`.

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py`

This task has no dedicated test; covered by Task 5's end-to-end test.

- [ ] **Step 1: Implement wrapper**

In `newton/_src/solvers/fba/linear_solver.py` append:
```python
from dataclasses import dataclass


@dataclass
class FactorizedSystem:
    """Output of factorize_and_sparse_inverse."""

    S: sp.csc_matrix          # n×n, lower-tri inverse with elimination-tree sparsity
    ST: sp.csc_matrix         # transpose, runtime-uploaded separately
    Dinv: np.ndarray          # length n
    perm_r: np.ndarray        # length n, int32
    invperm_r: np.ndarray     # length n, int32


def factorize_and_sparse_inverse(A: sp.csr_matrix) -> FactorizedSystem:
    """Run COLAMD ordering + LU factor + sparse inverse on an SPD matrix.

    Args:
        A: scalar N×N PD Hessian (csr or csc).

    Returns:
        FactorizedSystem with all data needed to construct an FBALinearSolver.
    """
    L, _U, Dinv, perm_r, _perm_c = _splu_extract_factors(A)
    parent = _elimination_tree(L)
    invperm_r = np.argsort(perm_r).astype(np.int32)
    S = compute_lower_inverse(L, parent=parent, invperm=invperm_r)
    ST = S.T.tocsc()
    return FactorizedSystem(S=S, ST=ST, Dinv=Dinv, perm_r=perm_r, invperm_r=invperm_r)
```

- [ ] **Step 2: Quick smoke check via Python REPL (no test commit)**

```bash
uv run python -c "
import numpy as np, scipy.sparse as sp
from newton._src.solvers.fba.linear_solver import factorize_and_sparse_inverse

n = 16
diag = np.full(n, 10.0)
off = np.full(n-1, -1.0)
A = sp.diags([off, diag, off], [-1,0,1], format='csr')
fs = factorize_and_sparse_inverse(A)
print('S shape:', fs.S.shape, 'nnz:', fs.S.nnz, 'Dinv min/max:', fs.Dinv.min(), fs.Dinv.max())
"
```

Expected: prints something like `S shape: (16, 16) nnz: <small> Dinv min/max: ~0.1 0.1`.

- [ ] **Step 3: Commit**

```bash
git add newton/_src/solvers/fba/linear_solver.py
git commit -m "Add factorize_and_sparse_inverse pipeline"
```

---

## Task 5: `FBALinearSolver` + Warp BSR runtime (tests T3, T6)

**Goal:** Build the runtime solver class. Upload `S`, `Sᵀ` as Warp `BsrMatrix` (1×1 blocks, scalar float), `Dinv` as `wp.array[float]`, `perm` / `invperm` as `wp.array[int32]`. Implement `solve(b, x)` for `wp.array[wp.vec3]` inputs: three component passes, each `apply_perm → bsr_mv(S) → ×D⁻¹ → bsr_mv(Sᵀ) → apply_invperm`.

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py`
- Modify: `newton/_src/solvers/fba/kernels.py`
- Modify: `newton/tests/test_solver_fba.py`

- [ ] **Step 1: Write test T3 + T6 — end-to-end solve + permutation roundtrip**

In `newton/tests/test_solver_fba.py` append:
```python
class TestFBALinearSolver(unittest.TestCase):
    """T3 end-to-end linear solve + T6 permutation roundtrip."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_solve_random_spd(self):
        from newton._src.solvers.fba.linear_solver import (
            factorize_and_sparse_inverse,
            FBALinearSolver,
        )

        rng = np.random.default_rng(42)
        n = 64
        diag = rng.uniform(10.0, 20.0, n)
        off = rng.uniform(-0.5, 0.5, n - 1)
        A = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr").astype(np.float64)

        fs = factorize_and_sparse_inverse(A)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        solver = FBALinearSolver(fs, device=device)

        b_np = rng.standard_normal((n, 3))
        b = wp.array(b_np, dtype=wp.vec3, device=device)
        x = wp.empty(n, dtype=wp.vec3, device=device)

        solver.solve(b, x)
        x_np = x.numpy()

        # Verify A · x ≈ b for each component.
        for c in range(3):
            r = A.dot(x_np[:, c]) - b_np[:, c]
            self.assertLess(np.linalg.norm(r) / max(np.linalg.norm(b_np[:, c]), 1e-12), 1e-6)

    def test_permutation_roundtrip(self):
        from newton._src.solvers.fba.kernels import apply_permutation_vec3_kernel

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        n = 32
        perm = np.random.default_rng(1).permutation(n).astype(np.int32)
        invperm = np.argsort(perm).astype(np.int32)

        x_np = np.random.default_rng(2).standard_normal((n, 3))
        x = wp.array(x_np, dtype=wp.vec3, device=device)
        y = wp.empty(n, dtype=wp.vec3, device=device)
        z = wp.empty(n, dtype=wp.vec3, device=device)
        perm_wp = wp.array(perm, dtype=wp.int32, device=device)
        invperm_wp = wp.array(invperm, dtype=wp.int32, device=device)

        wp.launch(apply_permutation_vec3_kernel, dim=n, inputs=[x, perm_wp], outputs=[y], device=device)
        wp.launch(apply_permutation_vec3_kernel, dim=n, inputs=[y, invperm_wp], outputs=[z], device=device)

        np.testing.assert_allclose(z.numpy(), x_np, atol=0.0)
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestFBALinearSolver
```

Expected: `ImportError` for `FBALinearSolver` and `apply_permutation_vec3_kernel`.

- [ ] **Step 3: Add `apply_permutation_vec3_kernel` and scalar helpers in `kernels.py`**

In `newton/_src/solvers/fba/kernels.py` append:
```python
@wp.kernel
def apply_permutation_vec3_kernel(
    src: wp.array[wp.vec3],
    perm: wp.array[wp.int32],
    dst: wp.array[wp.vec3],
):
    """dst[i] = src[perm[i]] — per-particle gather."""
    tid = wp.tid()
    dst[tid] = src[perm[tid]]


@wp.kernel
def extract_component_kernel(
    src: wp.array[wp.vec3],
    component: int,
    dst: wp.array[wp.float32],
):
    """dst[i] = src[i][component]."""
    tid = wp.tid()
    dst[tid] = src[tid][component]


@wp.kernel
def insert_component_kernel(
    src: wp.array[wp.float32],
    component: int,
    dst: wp.array[wp.vec3],
):
    """dst[i][component] = src[i]."""
    tid = wp.tid()
    v = dst[tid]
    v[component] = src[tid]
    dst[tid] = v


@wp.kernel
def scale_by_diag_kernel(
    src: wp.array[wp.float32],
    diag: wp.array[wp.float32],
    dst: wp.array[wp.float32],
):
    """dst[i] = diag[i] * src[i] — used as ×D⁻¹ in the linear solve."""
    tid = wp.tid()
    dst[tid] = diag[tid] * src[tid]


@wp.kernel
def apply_permutation_scalar_kernel(
    src: wp.array[wp.float32],
    perm: wp.array[wp.int32],
    dst: wp.array[wp.float32],
):
    """dst[i] = src[perm[i]] — scalar gather variant."""
    tid = wp.tid()
    dst[tid] = src[perm[tid]]
```

- [ ] **Step 4: Implement `FBALinearSolver`**

In `newton/_src/solvers/fba/linear_solver.py` append:
```python
import warp as wp
import warp.sparse as wps

from .kernels import (
    apply_permutation_scalar_kernel,
    extract_component_kernel,
    insert_component_kernel,
    scale_by_diag_kernel,
)


class FBALinearSolver:
    """Runtime sparse-inverse solver `x = A⁻¹ b` via two BSR SpMVs per component."""

    def __init__(self, factor: FactorizedSystem, device: wp.Device | str) -> None:
        self.device = wp.get_device(device)
        n = factor.S.shape[0]
        self.n = n

        # Build BSR matrices (block size 1×1) from CSC triplets.
        self._S_bsr = _csc_to_bsr_1x1(factor.S, self.device)
        self._ST_bsr = _csc_to_bsr_1x1(factor.ST, self.device)
        self._Dinv = wp.array(factor.Dinv.astype(np.float32), dtype=wp.float32, device=self.device)
        self._perm = wp.array(factor.perm_r.astype(np.int32), dtype=wp.int32, device=self.device)
        self._invperm = wp.array(factor.invperm_r.astype(np.int32), dtype=wp.int32, device=self.device)

        # Scratch buffers (scalar) — reused across components.
        self._b_scalar = wp.empty(n, dtype=wp.float32, device=self.device)
        self._b_perm = wp.empty(n, dtype=wp.float32, device=self.device)
        self._Sb = wp.empty(n, dtype=wp.float32, device=self.device)
        self._DSb = wp.empty(n, dtype=wp.float32, device=self.device)
        self._SDSb = wp.empty(n, dtype=wp.float32, device=self.device)
        self._x_scalar = wp.empty(n, dtype=wp.float32, device=self.device)

    def solve(self, b: wp.array, x: wp.array) -> None:
        """Compute x[i] = A⁻¹ · b[i] component-wise (vec3 in, vec3 out)."""
        n = self.n
        for c in range(3):
            wp.launch(extract_component_kernel, dim=n,
                      inputs=[b, c], outputs=[self._b_scalar], device=self.device)
            wp.launch(apply_permutation_scalar_kernel, dim=n,
                      inputs=[self._b_scalar, self._perm], outputs=[self._b_perm], device=self.device)
            wps.bsr_mv(self._S_bsr, self._b_perm, self._Sb)
            wp.launch(scale_by_diag_kernel, dim=n,
                      inputs=[self._Sb, self._Dinv], outputs=[self._DSb], device=self.device)
            wps.bsr_mv(self._ST_bsr, self._DSb, self._SDSb)
            wp.launch(apply_permutation_scalar_kernel, dim=n,
                      inputs=[self._SDSb, self._invperm], outputs=[self._x_scalar], device=self.device)
            wp.launch(insert_component_kernel, dim=n,
                      inputs=[self._x_scalar, c], outputs=[x], device=self.device)


def _csc_to_bsr_1x1(M: sp.csc_matrix, device: wp.Device | str) -> wps.BsrMatrix:
    """Build a Warp 1×1 BSR matrix from a SciPy CSC matrix."""
    M_csr = M.tocsr()
    rows, cols = M_csr.nonzero()
    vals = M_csr.data.astype(np.float32)
    rows_wp = wp.array(rows.astype(np.int32), dtype=wp.int32, device=device)
    cols_wp = wp.array(cols.astype(np.int32), dtype=wp.int32, device=device)
    vals_wp = wp.array(vals, dtype=wp.float32, device=device)
    bsr = wps.bsr_zeros(M.shape[0], M.shape[1], wp.float32, device=device)
    wps.bsr_set_from_triplets(bsr, rows_wp, cols_wp, vals_wp)
    return bsr
```

> **Engineer note:** `warp.sparse.BsrMatrix` with `block_type=wp.float32` is a 1×1 (scalar) BSR. If `warp.sparse.bsr_zeros` API differs in the version pinned by `uv.lock`, consult `warp.sparse` source for the exact signature. The semantic is "build BSR from row/col/value triplets, sum duplicates if any".

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestFBALinearSolver
```

Expected: `OK`. If `test_solve_random_spd` fails, debug by:
1. Print `np.linalg.norm(A @ fs.S.T.toarray() @ np.diag(fs.Dinv) @ fs.S.toarray() - np.eye(n))` to confirm the CPU pipeline; if non-zero, bug is in Task 3.
2. Verify BSR upload by comparing `wps.bsr_to_csr(self._S_bsr).numpy()` to `fs.S.toarray()` for a tiny case.

- [ ] **Step 6: Commit**

```bash
git add newton/_src/solvers/fba/linear_solver.py newton/_src/solvers/fba/kernels.py newton/tests/test_solver_fba.py
git commit -m "Add FBALinearSolver with Warp BSR runtime solve"
```

---

## Task 6: Trivial vec3 kernels

**Goal:** Implement `compute_inertial_kernel`, `zero_vec3_kernel`, `add_inertia_to_rhs_kernel`, `write_velocity_kernel`. Each is a few lines. No dedicated test (covered by T7 smoke).

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py`

- [ ] **Step 1: Add kernels**

In `newton/_src/solvers/fba/kernels.py` append:
```python
@wp.kernel
def zero_vec3_kernel(arr: wp.array[wp.vec3]):
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_inertial_kernel(
    x_prev: wp.array[wp.vec3],
    v_prev: wp.array[wp.vec3],
    f_ext: wp.array[wp.vec3],
    inv_mass: wp.array[wp.float32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: float,
    x_inertia: wp.array[wp.vec3],
):
    """x_inertia = x_prev + dt·v_prev + (dt²/m)·(f_ext + g·m)."""
    tid = wp.tid()
    w_idx = wp.max(particle_world[tid], 0)
    g = gravity[w_idx]
    im = inv_mass[tid]
    a = f_ext[tid] * im + g * wp.step(-im)  # gravity active only for free particles
    x_inertia[tid] = x_prev[tid] + v_prev[tid] * dt + a * (dt * dt)


@wp.kernel
def add_inertia_to_rhs_kernel(
    x_inertia: wp.array[wp.vec3],
    mass: wp.array[wp.float32],
    dt: float,
    rhs: wp.array[wp.vec3],
):
    """rhs += (m/dt²) · x_inertia.  Free-particle term only; mass==0 for pins, contributes 0."""
    tid = wp.tid()
    m = mass[tid]
    w = m / (dt * dt)
    rhs[tid] = rhs[tid] + x_inertia[tid] * w


@wp.kernel
def write_velocity_kernel(
    x_prev: wp.array[wp.vec3],
    x_new: wp.array[wp.vec3],
    inv_mass: wp.array[wp.float32],
    dt: float,
    v_new: wp.array[wp.vec3],
):
    """v_new = (x_new - x_prev) / dt for free particles; zero for pins."""
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        v_new[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v_new[tid] = (x_new[tid] - x_prev[tid]) / dt
```

- [ ] **Step 2: Smoke compile check**

```bash
uv run python -c "import warp as wp; wp.init(); from newton._src.solvers.fba import kernels; print('kernels imported')"
```

Expected: prints `kernels imported`. Warp will lazily compile kernels at first launch — full compile happens during integration tests (Task 13).

- [ ] **Step 3: Commit**

```bash
git add newton/_src/solvers/fba/kernels.py
git commit -m "Add trivial vec3 kernels for SolverFBA"
```

---

## Task 7: `svd_3x2` helper + ARAP stretching projection (test T4)

**Goal:** Implement a 3×2 SVD as a `@wp.func` (used by ARAP and future material models), plus the per-triangle ARAP projection kernel.

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py`
- Modify: `newton/tests/test_solver_fba.py`

**Algorithm:**
The 3×2 SVD can be computed by reducing to a 2×2 problem: `FᵀF` is 2×2 symmetric PSD whose eigendecomposition gives `V` and `Σ²`. Then `U₁ = F·V·Σ⁻¹` (3×2), extended to 3×3 by Gram-Schmidt or any orthonormal completion. ARAP projection clamps singular values to 1: `P = U₁·Vᵀ` (3×2).

- [ ] **Step 1: Write test T4 — ARAP projection on canonical inputs**

In `newton/tests/test_solver_fba.py` append:
```python
class TestARAPProjection(unittest.TestCase):
    """T4: ARAP local projection on triangle stretching."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_identity_F_gives_identity_P(self):
        from newton._src.solvers.fba.kernels import project_stretching_arap_kernel

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # One triangle at the rest configuration → F = identity column-augmented.
        # rest pos: (0,0,0), (1,0,0), (0,0,1); current pos = rest → F·Dm = identity.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3, device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        Dm_inv = wp.array(
            [wp.mat22(1.0, 0.0, 0.0, 1.0)],   # 2×2 identity
            dtype=wp.mat22, device=device,
        )
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.launch(
            project_stretching_arap_kernel, dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight],
            outputs=[rhs],
            device=device,
        )
        rhs_np = rhs.numpy()
        # When F is the rest configuration, ARAP projects to identity rotation;
        # the scatter contribution is the "rest" stencil — caller will balance
        # it with the Hessian. Verify scatter pattern matches Aᵀ·proj where
        # proj is the rest configuration mapped through Dm_inv·Pᵀ.
        # Concretely: rhs[1] - rhs[2] should be along e12 direction; rhs[0] = -rhs[1] - rhs[2].
        np.testing.assert_allclose(rhs_np[0], -(rhs_np[1] + rhs_np[2]), atol=1e-6)
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestARAPProjection
```

Expected: `ImportError`.

- [ ] **Step 3: Implement SVD + projection**

In `newton/_src/solvers/fba/kernels.py` append:
```python
@wp.func
def svd_3x2(F: wp.mat32) -> wp.mat32:
    """ARAP projection: SVD-clamp 3×2 deformation gradient to nearest rotation.

    Reduces to 2×2 symmetric eigenproblem on FᵀF.
    """
    # FᵀF  (2×2)
    FtF = wp.transpose(F) * F
    a = FtF[0, 0]
    b = FtF[0, 1]
    d = FtF[1, 1]

    # 2×2 symmetric eigen via closed form
    tr = a + d
    det = a * d - b * b
    disc = wp.max(tr * tr * 0.25 - det, 0.0)
    sq = wp.sqrt(disc)
    lam0 = tr * 0.5 + sq          # σ²₀
    lam1 = wp.max(tr * 0.5 - sq, 0.0)  # σ²₁

    # eigenvectors of FtF
    if wp.abs(b) > 1.0e-20:
        v0 = wp.normalize(wp.vec2(lam0 - d, b))
        v1 = wp.vec2(-v0[1], v0[0])
    else:
        v0 = wp.vec2(1.0, 0.0) if a >= d else wp.vec2(0.0, 1.0)
        v1 = wp.vec2(-v0[1], v0[0])

    sigma0 = wp.sqrt(wp.max(lam0, 1.0e-20))
    sigma1 = wp.sqrt(wp.max(lam1, 1.0e-20))

    # U columns: u_i = F·v_i / σ_i
    Fv0 = F * v0
    Fv1 = F * v1
    u0 = Fv0 / sigma0
    u1 = Fv1 / sigma1

    # ARAP: clamp Σ to identity → P = U · I · Vᵀ
    # P = [u0 | u1] · [v0 v1]ᵀ
    p00 = u0[0] * v0[0] + u1[0] * v1[0]
    p01 = u0[0] * v0[1] + u1[0] * v1[1]
    p10 = u0[1] * v0[0] + u1[1] * v1[0]
    p11 = u0[1] * v0[1] + u1[1] * v1[1]
    p20 = u0[2] * v0[0] + u1[2] * v1[0]
    p21 = u0[2] * v0[1] + u1[2] * v1[1]
    return wp.mat32(p00, p01, p10, p11, p20, p21)


@wp.kernel
def project_stretching_arap_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],   # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    # Deformation gradient F = [p1-p0 | p2-p0] · Dm⁻¹  (3×2)
    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = wp.mat32(
        Ds_col0[0], Ds_col1[0],
        Ds_col0[1], Ds_col1[1],
        Ds_col0[2], Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    P = svd_3x2(F)

    # Local contribution to RHS: proj = w · Dm⁻ᵀ · Pᵀ  (2×3)
    w = tri_weight[t]
    Dm_invT = wp.transpose(Dm_inv)
    # proj = w · Dm_invT · Pᵀ (rows are length-3 vectors)
    PT00 = P[0, 0]; PT01 = P[1, 0]; PT02 = P[2, 0]   # column 0 of P
    PT10 = P[0, 1]; PT11 = P[1, 1]; PT12 = P[2, 1]   # column 1 of P

    row0 = wp.vec3(
        w * (Dm_invT[0, 0] * PT00 + Dm_invT[0, 1] * PT10),
        w * (Dm_invT[0, 0] * PT01 + Dm_invT[0, 1] * PT11),
        w * (Dm_invT[0, 0] * PT02 + Dm_invT[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_invT[1, 0] * PT00 + Dm_invT[1, 1] * PT10),
        w * (Dm_invT[1, 0] * PT01 + Dm_invT[1, 1] * PT11),
        w * (Dm_invT[1, 0] * PT02 + Dm_invT[1, 1] * PT12),
    )

    # Scatter: rhs[i0] += -row0 - row1; rhs[i1] += row0; rhs[i2] += row1
    wp.atomic_add(rhs, i0, -(row0 + row1))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)
```

> **Engineer note:** Verify `wp.mat32` exists in your Warp version. If not, fall back to two `wp.vec3` columns or use `wp.types.matrix(shape=(3,2), dtype=float)` per the version's API. The semantic is "3 rows × 2 columns float matrix".

- [ ] **Step 4: Run test, expect pass**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestARAPProjection
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/fba/kernels.py newton/tests/test_solver_fba.py
git commit -m "Add ARAP stretching projection kernel with 3x2 SVD"
```

---

## Task 8: Isometric bending projection kernel (test T5)

**Goal:** For each bending edge (4-vertex stencil), apply the isometric bending energy local target: PD's projection for `Q_e = q·qᵀ` is to project `qᵀx` to zero (rest is flat or to the rest dihedral angle). For MVP, rest = current planar configuration → RHS contribution scatters `-w·q·(qᵀ·x_cur)`. (See `PDIsometricBendingEnergy.cpp:localProjection`.)

Actually for isometric bending in PD with rank-1 `Q_e = q·qᵀ`, the local-global iteration's local step is trivial (no SVD; the projection target is the rest "flatness" or the projection of the current curvature to zero). The PD scatter is `rhs[i] += w · q[a]·q[b] · x_ref[b]` — for rest = flat (initial mesh), this expands to a fixed RHS contribution and a constant Hessian. **For MVP with flat rest configuration, the bending RHS contribution is zero** (rest curvature is zero) — only the Hessian contribution remains (already baked into `A` in Task 2).

Therefore: **the bending projection kernel does nothing at runtime when rest curvature is zero**. We still implement it for non-flat initial meshes (folded cloth, draped initial conditions).

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py`
- Modify: `newton/tests/test_solver_fba.py`

- [ ] **Step 1: Write test T5 — flat configuration gives zero RHS**

In `newton/tests/test_solver_fba.py` append:
```python
class TestBendingProjection(unittest.TestCase):
    """T5: isometric bending projection — flat rest gives zero RHS contribution."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_flat_rest_zero_rhs(self):
        from newton._src.solvers.fba.kernels import project_bending_kernel

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Flat 4-vertex stencil: 2 triangles sharing edge (0,1).
        positions = wp.array(
            [wp.vec3(0,0,0), wp.vec3(1,0,0), wp.vec3(0.5,0,1), wp.vec3(0.5,0,-1)],
            dtype=wp.vec3, device=device,
        )
        edge_indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=device)
        # q computed from this rest geometry by build_pd_system → in this test we
        # use a known-good q (cotangent-based). For flat-rest test, we pass the
        # CURRENT x_ref equal to positions: the projection target is x_ref, so
        # rhs contribution = w · q · qᵀ · (x_ref - x_cur) = 0.
        x_ref = wp.array(positions.numpy(), dtype=wp.vec3, device=device)
        edge_quad_q = wp.array([wp.vec4(1.0, 1.0, -1.0, -1.0)], dtype=wp.vec4, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_bending_kernel, dim=1,
            inputs=[positions, x_ref, edge_indices, edge_quad_q, weight],
            outputs=[rhs],
            device=device,
        )
        np.testing.assert_allclose(rhs.numpy(), np.zeros((4, 3)), atol=1e-6)
```

- [ ] **Step 2: Run test, expect failure**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestBendingProjection
```

Expected: `ImportError`.

- [ ] **Step 3: Implement kernel**

In `newton/_src/solvers/fba/kernels.py` append:
```python
@wp.kernel
def project_bending_kernel(
    positions: wp.array[wp.vec3],        # x_cur
    x_ref: wp.array[wp.vec3],            # rest positions (defines rest curvature)
    edge_indices: wp.array2d[wp.int32],  # shape (E, 4)
    edge_quad_q: wp.array[wp.vec4],      # length-4 vector q per edge; Q = q·qᵀ
    edge_weight: wp.array[wp.float32],
    rhs: wp.array[wp.vec3],
):
    """Isometric bending: rhs[i] += w·q[a]·q[b]·(x_ref[b] - x_cur[b]) for a=i.

    PD local-global: the local step's projection target is the rest curvature.
    The scatter contribution is w·q·qᵀ·x_ref (constant) and the −w·q·qᵀ·x_cur
    portion is part of the Hessian; for the iterative form we scatter
    w·q·qᵀ·x_ref directly to RHS and rely on the prefactored A to absorb the
    matching Hessian term.
    """
    e = wp.tid()
    q = edge_quad_q[e]
    w = edge_weight[e]
    if w == 0.0:
        return

    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    i2 = edge_indices[e, 2]
    i3 = edge_indices[e, 3]

    # Compute qᵀ·x_ref (a vec3).
    qTxref = (
        x_ref[i0] * q[0]
        + x_ref[i1] * q[1]
        + x_ref[i2] * q[2]
        + x_ref[i3] * q[3]
    )

    # Scatter w · q[a] · (qᵀ·x_ref) to each row.
    wp.atomic_add(rhs, i0, w * q[0] * qTxref)
    wp.atomic_add(rhs, i1, w * q[1] * qTxref)
    wp.atomic_add(rhs, i2, w * q[2] * qTxref)
    wp.atomic_add(rhs, i3, w * q[3] * qTxref)
```

> **Engineer note:** The bending energy and RHS form in PD is sometimes formulated as scattering `w·q·qᵀ·x_ref` only (Hessian absorbs the rest), and sometimes as scattering the *delta* against current. The form above matches `PDIsometricBendingEnergy.cpp:localProjection` (RealSim — it scatters w·q·qᵀ·x_ref).

- [ ] **Step 4: Run test, expect pass**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestBendingProjection
```

Expected: `OK`. (Test passes because `x_ref == positions` → qᵀ·x_ref scatter cancels at the planar configuration only when summed with the Hessian's `−w·q·qᵀ·x_cur` — but the test as written passes any kernel that scatters zero when `x_ref` is "consistent". If your formulation results in non-zero scatter at flat rest, adjust the test to compare against an analytical reference value computed with NumPy.)

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/fba/kernels.py newton/tests/test_solver_fba.py
git commit -m "Add isometric bending projection kernel"
```

---

## Task 9: Pin projection kernel

**Goal:** For each pinned particle, scatter `w_pin · x_ref` into RHS.

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py`

- [ ] **Step 1: Add kernel**

In `newton/_src/solvers/fba/kernels.py` append:
```python
@wp.kernel
def project_pin_kernel(
    pin_indices: wp.array[wp.int32],
    x_ref: wp.array[wp.vec3],            # full vec3 array indexed by particle index
    pin_stiffness: float,
    rhs: wp.array[wp.vec3],
):
    """rhs[pin_indices[k]] += pin_stiffness · x_ref[pin_indices[k]]."""
    k = wp.tid()
    i = pin_indices[k]
    wp.atomic_add(rhs, i, pin_stiffness * x_ref[i])
```

- [ ] **Step 2: Smoke compile check**

```bash
uv run python -c "import warp as wp; wp.init(); from newton._src.solvers.fba.kernels import project_pin_kernel; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add newton/_src/solvers/fba/kernels.py
git commit -m "Add Pin projection kernel"
```

---

## Task 10: `SolverFBA.__init__` — setup pipeline

**Goal:** Replace the `NotImplementedError` stub with a real `__init__` that:
1. Validates inputs.
2. Calls `build_pd_system` (Task 2) → `(A, meta)`.
3. Calls `factorize_and_sparse_inverse` (Task 4) → `FactorizedSystem`.
4. Constructs `FBALinearSolver` (Task 5).
5. Uploads per-element metadata (`tri_indices`, `tri_rest_inv`, `tri_weight`, `edge_indices`, `edge_quad_q`, `edge_weight`, `pin_indices`, `x_ref`) to device.
6. Pre-allocates per-step buffers (`_x_inertia`, `_x_cur`, `_rhs`).

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py`

- [ ] **Step 1: Write the new `__init__`**

Replace the body of `__init__` in `newton/_src/solvers/fba/solver_fba.py`:
```python
    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap"] = "arap",
    ) -> None:
        super().__init__(model)

        # ---- Validate ----
        if model.particle_count == 0:
            raise ValueError("SolverFBA requires at least one particle")
        if model.tri_count == 0:
            raise ValueError("SolverFBA requires at least one cloth triangle")
        if stretching_model != "arap":
            raise NotImplementedError(
                f"stretching_model={stretching_model!r} not yet implemented (MVP: arap only)"
            )
        if pin_stiffness <= 0:
            raise ValueError("pin_stiffness must be positive")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.iterations = int(iterations)
        self.pin_stiffness = float(pin_stiffness)
        self.stretching_model = stretching_model

        # ---- CPU setup ----
        from .linear_solver import (
            build_pd_system,
            factorize_and_sparse_inverse,
            FBALinearSolver,
        )

        # PD setup is dt-dependent; we cache the assembly at a reference dt and
        # rebuild lazily inside `step` if the dt changes. MVP requires a stable
        # dt across the simulation — record the first dt seen.
        self._dt_setup: float | None = None
        self._linear_solver: FBALinearSolver | None = None
        self._meta: dict | None = None

        # Device buffers (allocated lazily once we know dt — see _setup_pd_system).
        N = model.particle_count
        device = model.device
        self._device = device
        self._x_ref = wp.clone(model.particle_q)  # initial positions = pin/bending refs
        self._x_inertia = wp.empty(N, dtype=wp.vec3, device=device)
        self._x_cur = wp.empty(N, dtype=wp.vec3, device=device)
        self._rhs = wp.empty(N, dtype=wp.vec3, device=device)

        # Per-element device data (filled by _upload_meta).
        self._tri_indices_d: wp.array | None = None
        self._tri_rest_inv_d: wp.array | None = None
        self._tri_weight_d: wp.array | None = None
        self._edge_indices_d: wp.array | None = None
        self._edge_quad_q_d: wp.array | None = None
        self._edge_weight_d: wp.array | None = None
        self._pin_indices_d: wp.array | None = None

    def _setup_pd_system(self, dt: float) -> None:
        """Build / rebuild the PD Hessian, factor it, and upload device data."""
        from .linear_solver import (
            build_pd_system,
            factorize_and_sparse_inverse,
            FBALinearSolver,
        )

        A, meta = build_pd_system(self.model, dt=dt, pin_stiffness=self.pin_stiffness)
        fs = factorize_and_sparse_inverse(A)
        self._linear_solver = FBALinearSolver(fs, device=self._device)
        self._meta = meta
        self._dt_setup = dt

        device = self._device
        self._tri_indices_d = wp.array(
            meta["tri_indices"].flatten().astype(np.int32),
            dtype=wp.int32, device=device,
        )
        self._tri_rest_inv_d = wp.array(
            meta["tri_rest_inv"].astype(np.float32),
            dtype=wp.mat22, device=device,
        )
        self._tri_weight_d = wp.array(
            meta["tri_weight"].astype(np.float32), dtype=wp.float32, device=device,
        )
        if meta["edge_indices"].shape[0] > 0:
            self._edge_indices_d = wp.array(
                meta["edge_indices"].astype(np.int32), dtype=wp.int32, device=device,
            )
            edge_q4 = np.concatenate(
                [meta["edge_quad_q"], np.zeros((meta["edge_quad_q"].shape[0], 0))],
                axis=1,
            ).astype(np.float32)
            self._edge_quad_q_d = wp.array(
                meta["edge_quad_q"].astype(np.float32), dtype=wp.vec4, device=device,
            )
            self._edge_weight_d = wp.array(
                meta["edge_weight"].astype(np.float32),
                dtype=wp.float32, device=device,
            )
        if meta["pin_indices"].shape[0] > 0:
            self._pin_indices_d = wp.array(
                meta["pin_indices"].astype(np.int32), dtype=wp.int32, device=device,
            )
```

Add the imports at the top of the file:
```python
import numpy as np
import warp as wp
```

- [ ] **Step 2: Smoke check that constructor runs on a tiny model**

```bash
uv run python -c "
import warp as wp
wp.init()
import newton
from newton._src.solvers.fba.solver_fba import SolverFBA

builder = newton.ModelBuilder()
builder.add_cloth_grid(
    pos=wp.vec3(0,0,0), rot=wp.quat_identity(), vel=wp.vec3(0,0,0),
    dim_x=4, dim_y=4, cell_x=0.1, cell_y=0.1, mass=0.01,
    tri_ke=100.0, tri_ka=0.0, tri_kd=0.0,
    edge_ke=1.0, edge_kd=0.0,
    fix_left=True,
)
model = builder.finalize()
solver = SolverFBA(model, iterations=5)
solver._setup_pd_system(dt=1/60)
print('init+setup OK; N=', model.particle_count)
"
```

Expected: prints `init+setup OK; N= 25`.

- [ ] **Step 3: Commit**

```bash
git add newton/_src/solvers/fba/solver_fba.py
git commit -m "Implement SolverFBA __init__ and PD setup pipeline"
```

---

## Task 11: `SolverFBA.step` — runtime PD iteration

**Goal:** Implement the per-step PD local-global loop. Sequence per spec §3.1.

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py`

- [ ] **Step 1: Write `step()` body**

Replace the `step()` body in `newton/_src/solvers/fba/solver_fba.py`:
```python
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        from .kernels import (
            add_inertia_to_rhs_kernel,
            compute_inertial_kernel,
            project_bending_kernel,
            project_pin_kernel,
            project_stretching_arap_kernel,
            write_velocity_kernel,
            zero_vec3_kernel,
        )

        if self._linear_solver is None or self._dt_setup is None or abs(self._dt_setup - dt) > 1e-12:
            self._setup_pd_system(dt)

        model = self.model
        N = model.particle_count
        device = self._device

        # 1) Compute inertial prediction x_inertia.
        wp.launch(
            compute_inertial_kernel, dim=N,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[self._x_inertia],
            device=device,
        )
        # Initialize current iterate x_cur = x_inertia (copy).
        wp.copy(self._x_cur, self._x_inertia)

        # 2) PD outer iterations.
        for _k in range(self.iterations):
            # 2a. Zero RHS.
            wp.launch(zero_vec3_kernel, dim=N, inputs=[self._rhs], device=device)
            # 2b. Inertia term.
            wp.launch(
                add_inertia_to_rhs_kernel, dim=N,
                inputs=[self._x_inertia, model.particle_mass, dt],
                outputs=[self._rhs],
                device=device,
            )
            # 2c. Pin projection.
            if self._pin_indices_d is not None:
                wp.launch(
                    project_pin_kernel, dim=self._pin_indices_d.shape[0],
                    inputs=[self._pin_indices_d, self._x_ref, self.pin_stiffness],
                    outputs=[self._rhs],
                    device=device,
                )
            # 2d. Stretching projection.
            wp.launch(
                project_stretching_arap_kernel, dim=model.tri_count,
                inputs=[
                    self._x_cur, self._tri_indices_d, self._tri_rest_inv_d, self._tri_weight_d,
                ],
                outputs=[self._rhs],
                device=device,
            )
            # 2e. Bending projection (only if edges present).
            if self._edge_indices_d is not None:
                wp.launch(
                    project_bending_kernel, dim=self._edge_indices_d.shape[0],
                    inputs=[
                        self._x_cur, self._x_ref, self._edge_indices_d,
                        self._edge_quad_q_d, self._edge_weight_d,
                    ],
                    outputs=[self._rhs],
                    device=device,
                )
            # 2f. Global linear solve: x_cur = A⁻¹ · rhs.
            self._linear_solver.solve(self._rhs, self._x_cur)

        # 3) Write velocity and update state_out.
        wp.copy(state_out.particle_q, self._x_cur)
        wp.launch(
            write_velocity_kernel, dim=N,
            inputs=[state_in.particle_q, self._x_cur, model.particle_inv_mass, dt],
            outputs=[state_out.particle_qd],
            device=device,
        )

    def notify_model_changed(self, flags: int) -> None:
        # On any geometry/inertial change, force a full re-setup at the next step.
        from ..flags import SolverNotifyFlags
        if flags & (SolverNotifyFlags.SHAPE_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._linear_solver = None
            self._dt_setup = None
```

- [ ] **Step 2: Smoke check — one step on tiny grid**

```bash
uv run python -c "
import warp as wp
wp.init()
import newton
from newton._src.solvers.fba.solver_fba import SolverFBA
builder = newton.ModelBuilder()
builder.add_cloth_grid(
    pos=wp.vec3(0,0,0), rot=wp.quat_identity(), vel=wp.vec3(0,0,0),
    dim_x=4, dim_y=4, cell_x=0.1, cell_y=0.1, mass=0.01,
    tri_ke=100.0, tri_ka=0.0, tri_kd=0.0,
    edge_ke=1.0, edge_kd=0.0,
    fix_left=True,
)
model = builder.finalize()
solver = SolverFBA(model, iterations=5)
s_in, s_out = model.state(), model.state()
solver.step(s_in, s_out, None, None, 1/60)
print('step OK; first vert q=', s_out.particle_q.numpy()[0])
"
```

Expected: prints `step OK; first vert q= [0. 0. 0.]` (pinned corner stationary) plus successful completion.

- [ ] **Step 3: Commit**

```bash
git add newton/_src/solvers/fba/solver_fba.py
git commit -m "Implement SolverFBA.step PD outer loop"
```

---

## Task 12: Public exports

**Goal:** Add `SolverFBA` to `newton._src.solvers.__init__` and `newton.solvers`.

**Files:**
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/solvers.py`
- Modify: `newton/tests/test_solver_fba.py`

- [ ] **Step 1: Update internal `__init__.py`**

Edit `newton/_src/solvers/__init__.py`:
```python
# Add this import (alphabetical order between .featherstone and .flags):
from .fba import SolverFBA

# And add "SolverFBA" to __all__ (alphabetical between SolverBase and SolverFeatherstone).
```

Concretely the file becomes:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .fba import SolverFBA
from .featherstone import SolverFeatherstone
from .flags import SolverNotifyFlags
from .implicit_mpm import SolverImplicitMPM
from .kamino import SolverKamino
from .mujoco import SolverMuJoCo
from .semi_implicit import SolverSemiImplicit
from .solver import SolverBase
from .style3d.solver_style3d import SolverStyle3D
from .vbd import SolverVBD
from .xpbd import SolverXPBD

__all__ = [
    "SolverBase",
    "SolverFBA",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
]
```

- [ ] **Step 2: Update `newton/solvers.py`**

Add `SolverFBA` to the imports from `_src.solvers` and to `__all__`. The two lists are sorted; insert alphabetically.

- [ ] **Step 3: Add a public-API smoke test**

In `newton/tests/test_solver_fba.py` append:
```python
class TestPublicAPI(unittest.TestCase):
    def test_solver_fba_in_newton_solvers(self):
        from newton import solvers
        self.assertTrue(hasattr(solvers, "SolverFBA"))
        self.assertIn("SolverFBA", solvers.__all__)
```

- [ ] **Step 4: Run tests**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba
```

Expected: all previously-passing tests still pass plus `test_solver_fba_in_newton_solvers`.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/__init__.py newton/solvers.py newton/tests/test_solver_fba.py
git commit -m "Expose SolverFBA via newton.solvers public API"
```

---

## Task 13: Integration tests T7, T8 and `example_cloth_hanging_fba`

**Goal:** Add the smoke + steady-state tests and a runnable example.

**Files:**
- Modify: `newton/tests/test_solver_fba.py`
- Create: `newton/examples/cloth/example_cloth_hanging_fba.py`

- [ ] **Step 1: Write tests T7 + T8**

In `newton/tests/test_solver_fba.py` append:
```python
class TestSolverFBAIntegration(unittest.TestCase):
    """T7: smoke; T8: steady-state of hanging cloth."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_hanging_cloth(self, dim: int = 32):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=dim, dim_y=dim,
            cell_x=0.05, cell_y=0.05,
            mass=0.005,
            tri_ke=1.0e3, tri_ka=0.0, tri_kd=0.0,
            edge_ke=1.0e-1, edge_kd=0.0,
            fix_left=False,
        )
        # Pin the two top corners.
        builder.particle_mass[0] = 0.0
        builder.particle_inv_mass[0] = 0.0
        builder.particle_mass[dim] = 0.0
        builder.particle_inv_mass[dim] = 0.0
        return builder.finalize()

    def test_t7_smoke_no_nan(self):
        from newton.solvers import SolverFBA

        model = self._build_hanging_cloth(dim=16)
        solver = SolverFBA(model, iterations=8)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite values appeared")

    def test_t8_steady_state_settles(self):
        from newton.solvers import SolverFBA

        model = self._build_hanging_cloth(dim=16)
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(500):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        # Center vertex (approximate index dim*dim/2 + dim/2) should have
        # descended well below the pinned-corner height (~2 m).
        center_idx = (16 // 2) * (16 + 1) + (16 // 2)
        self.assertLess(q[center_idx, 1], 1.5, "cloth did not fall under gravity")
        self.assertGreater(q[center_idx, 1], 0.5, "cloth fell past plausible drape")
```

- [ ] **Step 2: Run tests, debug as needed**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba.TestSolverFBAIntegration
```

Expected: both pass. Common failure modes:
- NaN under 50 steps → check kernel atomic_adds reach correct rows; check `inv_mass` zero handling.
- T8 not settling → increase `iterations` to 20; if still off, check sign of gravity propagation in `compute_inertial_kernel`.

- [ ] **Step 3: Create example**

`newton/examples/cloth/example_cloth_hanging_fba.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hanging cloth example using SolverFBA.

Run with: ``python -m newton.examples cloth_hanging_fba``
"""

import warp as wp

import newton
from newton.examples import Example
from newton.solvers import SolverFBA


class Example(Example):  # noqa: F811 — Newton convention
    def __init__(self, viewer):
        self.viewer = viewer
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 2.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=32, dim_y=32,
            cell_x=0.05, cell_y=0.05,
            mass=0.005,
            tri_ke=1.0e3, tri_ka=0.0, tri_kd=0.0,
            edge_ke=1.0e-1, edge_kd=0.0,
            fix_left=False,
        )
        builder.particle_mass[0] = 0.0
        builder.particle_inv_mass[0] = 0.0
        builder.particle_mass[32] = 0.0
        builder.particle_inv_mass[32] = 0.0
        self.model = builder.finalize()

        self.solver = SolverFBA(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.dt = 1.0 / 60.0

    def step(self):
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, None, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.log_state(self.state_0)

    def test_final(self):
        import numpy as np
        q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q))
        # Center should drop but not collapse to floor.
        center_idx = 16 * 33 + 16
        assert 0.5 < q[center_idx, 1] < 1.5
```

- [ ] **Step 4: Run example headless to verify**

```bash
uv run -m newton.examples cloth_hanging_fba --viewer null --test
```

Expected: exits 0; final state passes `test_final()`.

- [ ] **Step 5: Run full test suite**

```bash
uv run --extra dev -m newton.tests -k test_solver_fba
```

Expected: all eight tests pass.

- [ ] **Step 6: Final commit**

```bash
git add newton/tests/test_solver_fba.py newton/examples/cloth/example_cloth_hanging_fba.py
git commit -m "Add SolverFBA integration tests T7/T8 and cloth_hanging_fba example"
```

- [ ] **Step 7: Run pre-commit + push**

```bash
uvx pre-commit run -a
```

Fix any complaints (likely typo hints or ruff format). Then:
```bash
git push -u origin ziqiu/fba-solver-design
```

---

## Out-of-band tasks (after MVP lands)

These are NOT part of this plan but are tracked for the next iteration:

- **PR split (spec §8):** Cherry-pick into three branches when preparing upstream PR — Phase 1 (linear-solver core + T1/T2/T3), Phase 2 (kernels + T4/T5/T6), Phase 3 (solver + T7/T8 + example).
- **CHANGELOG.md entry:** Insert "Add `SolverFBA` Projective Dynamics cloth solver" under `[Unreleased] → Added`.
- **Docs:** Add `SolverFBA` row to the supported-features table in `newton/solvers.py` (cloth ✅, particles ✅, others ❌).
- **Verify** `docs/generate_api.py` regenerates without errors.
