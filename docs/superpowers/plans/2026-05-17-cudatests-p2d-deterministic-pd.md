# Task P2-D: Deterministic Particle-Centered PD Reductions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate GPU nondeterminism in SolverFBA's PD pipeline so that 3 consecutive runs of the SqueezingBall demo produce bit-identical `min_y` / `x_drift` / `z_drift` (or differ only at FP-epsilon level ~1e-6).

**Architecture:** Replace `wp.atomic_add` scatter from per-element threads (one thread per tri/tet/edge, scatters into 3-4 vertex slots) with a two-kernel reduction: (1) a *compute* kernel runs per-element and writes its full contribution to a per-element scratch buffer (`element_count × vertices_per_element × vec3`), then (2) a *gather* kernel runs per-particle and reads the contributions from incident elements via a precomputed CSR adjacency. The CSR adjacency `(offsets[P+1], entries[total_incidence])` is built once in `_setup_pd_system` and reused across steps. No atomics → bit-deterministic by construction.

**Tech Stack:** Python 3, `uv`, `warp`, `numpy`, Newton (`/home/ziqiu/work/newton`), `unittest`.

---

## Pre-flight Checks

- [ ] **Confirm branch.**
```bash
cd /home/ziqiu/work/newton
git status
git log --oneline -3
```
Expected: `On branch ziqiu/fba-solver-design`. HEAD = `09e4f855` (contact sort).

- [ ] **Confirm baseline FBA tests pass.**
```bash
uv run --extra dev -m newton.tests -k test_solver_fba
```
Expected: 87 tests OK.

- [ ] **Confirm the reproducer captures the variance.**
```bash
for i in 1 2 3; do
  echo "=== Run $i ==="
  uv run python scripts/fba_demo5_squeezing_ball.py 2>&1 | grep -E "min_y|x_drift"
done
```
Expected: `min_y` values differ across runs by > 0.1 (variance up to ~5 units). After P2-D, this command should produce 3 identical lines (or differ only at the 4th decimal).

---

## File Map

- **Modify** `newton/_src/solvers/fba/solver_fba.py`
  - `_setup_pd_system`: build particle→element CSR adjacency for tri, tet, and bending edges; allocate per-element contribution scratch buffers
  - Per-step: launch new (compute, gather) kernel pairs instead of the old single-kernel atomic scatter

- **Modify** `newton/_src/solvers/fba/kernels.py`
  - For each of 7 energies (tri ARAP/Corot/NH, tet ARAP/Corot/NH, bending), split the existing kernel into:
    - `*_compute_kernel`: one thread per element, writes its full contribution (e.g. `(3 × vec3)` for tri, `(4 × vec3)` for tet) to a scratch buffer
    - `*_gather_kernel`: one thread per particle, reads its incident contributions via CSR adjacency
  - For the isodof Jᵀλ scatter (`build_jt_lambda_vec3_kernel`), apply the same compute/gather split: per-contact compute, per-particle gather

- **Test** `newton/tests/test_solver_fba.py`
  - Add `SolverFBADeterminismTests` that asserts bit-identical output across two SqueezingBall steps with identical seed
  - For each refactored kernel, add a regression test that captures `rhs` after one step, compares atomic-path vs reduction-path within `1e-6` tolerance

---

## CSR Adjacency Design

For tet bodies, the adjacency must map each particle `p` to a list of (tet_index, local_vertex_index_in_tet) where `local_vertex_index_in_tet ∈ {0, 1, 2, 3}`.

Build at init time:
```python
def build_particle_tet_csr(tet_indices: np.ndarray, n_particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR adjacency from tets to particles.

    Args:
        tet_indices: ``(num_tets, 4)`` particle indices per tet.
        n_particles: Total particle count.

    Returns:
        ``(offsets, tet_index, local_vertex_index)`` where:
        - ``offsets`` has shape ``(n_particles + 1,)``: ``offsets[p+1] - offsets[p]``
          is the number of tets incident to particle ``p``.
        - ``tet_index`` has shape ``(total_incidence,)``: the tet index for
          each entry.
        - ``local_vertex_index`` has shape ``(total_incidence,)``: which
          of the tet's 4 local vertices is particle ``p`` (0, 1, 2, or 3).
    """
    # Build entries as list[(particle, tet_idx, local_vert)].
    entries = []
    for t_idx, tet in enumerate(tet_indices):
        for local_v, p in enumerate(tet):
            entries.append((int(p), t_idx, local_v))
    entries.sort()  # group by particle
    offsets = np.zeros(n_particles + 1, dtype=np.int32)
    tet_idx_arr = np.zeros(len(entries), dtype=np.int32)
    local_v_arr = np.zeros(len(entries), dtype=np.int32)
    for i, (p, t, v) in enumerate(entries):
        tet_idx_arr[i] = t
        local_v_arr[i] = v
        offsets[p + 1] += 1
    np.cumsum(offsets, out=offsets)
    return offsets, tet_idx_arr, local_v_arr
```

Analogous functions for tri (3 verts/element) and edge (4 verts/bending stencil).

---

## Task A: Adjacency infrastructure

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py:_setup_pd_system` (or wherever `_tet_indices_d`, `_tri_indices_d`, `_edge_indices_d` are uploaded)
- Modify: `newton/_src/solvers/fba/linear_solver.py` (host-side helpers — adjacency builders fit naturally here next to existing mesh utilities)
- Test: `newton/tests/test_solver_fba.py`

- [ ] **Step A.1: Add adjacency builders to `linear_solver.py`.**

Append to `linear_solver.py`:
```python
def build_particle_element_csr(
    element_indices: np.ndarray,
    n_particles: int,
    n_verts_per_element: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR adjacency from elements (tri/tet/edge) to particles.

    Args:
        element_indices: ``(num_elements, n_verts_per_element)`` particle indices.
        n_particles: Total particle count.
        n_verts_per_element: 3 (triangle), 4 (tet or bending edge stencil).

    Returns:
        ``(offsets, element_idx, local_vertex_idx)`` where ``offsets`` has shape
        ``(n_particles + 1,)``, and the two index arrays have shape
        ``(total_incidence,)`` with ``total_incidence == num_elements · n_verts_per_element``.
    """
    num_elements = element_indices.shape[0]
    total_incidence = num_elements * n_verts_per_element
    # Build (particle, element, local) entries, sort by particle.
    particles = element_indices.reshape(-1).astype(np.int64)
    elements = np.repeat(np.arange(num_elements, dtype=np.int64), n_verts_per_element)
    locals_ = np.tile(np.arange(n_verts_per_element, dtype=np.int64), num_elements)
    order = np.argsort(particles, kind="stable")
    particles = particles[order]
    elements = elements[order]
    locals_ = locals_[order]
    # Build offsets.
    offsets = np.zeros(n_particles + 1, dtype=np.int32)
    counts = np.bincount(particles, minlength=n_particles).astype(np.int32)
    offsets[1:] = np.cumsum(counts)
    return offsets, elements.astype(np.int32), locals_.astype(np.int32)
```

- [ ] **Step A.2: Add a small unit test for the builder.**

Append to `test_solver_fba.py`:
```python
class ParticleElementCsrTests(unittest.TestCase):
    """Adjacency CSR builder produces correct (offsets, element_idx, local_vertex_idx)."""

    def test_basic_tet_adjacency(self) -> None:
        import numpy as np
        from newton._src.solvers.fba.linear_solver import build_particle_element_csr

        # 2 tets sharing vertices 1 and 2 (a simple bipyramid-ish):
        # tet 0: verts (0, 1, 2, 3)
        # tet 1: verts (1, 2, 3, 4)
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(tets, n_particles=5, n_verts_per_element=4)

        # Vertex 0: only in tet 0 at local index 0.
        # Vertex 1: in tet 0 (local=1) and tet 1 (local=0).
        # Vertex 2: in tet 0 (local=2) and tet 1 (local=1).
        # Vertex 3: in tet 0 (local=3) and tet 1 (local=2).
        # Vertex 4: only in tet 1 at local index 3.
        self.assertEqual(offsets.tolist(), [0, 1, 3, 5, 7, 8])
        # Vertex 1's entries: (tet=0, local=1), (tet=1, local=0).
        v1_slice = slice(offsets[1], offsets[2])
        elem_v1 = sorted(zip(elem_idx[v1_slice].tolist(), local_v[v1_slice].tolist()))
        self.assertEqual(elem_v1, [(0, 1), (1, 0)])

    def test_tri_adjacency(self) -> None:
        import numpy as np
        from newton._src.solvers.fba.linear_solver import build_particle_element_csr

        tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(tris, n_particles=4, n_verts_per_element=3)
        self.assertEqual(offsets.tolist(), [0, 1, 3, 5, 6])
        v1_slice = slice(offsets[1], offsets[2])
        elem_v1 = sorted(zip(elem_idx[v1_slice].tolist(), local_v[v1_slice].tolist()))
        self.assertEqual(elem_v1, [(0, 1), (1, 0)])
```

- [ ] **Step A.3: Run the unit test (expect FAIL — function not yet implemented).**
```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k ParticleElementCsr
```

- [ ] **Step A.4: Implement the function in `linear_solver.py` (per Step A.1).**

- [ ] **Step A.5: Re-run, expect PASS.**

- [ ] **Step A.6: Upload adjacencies to device in `_setup_pd_system`.**

In `_setup_pd_system`, after existing tet / tri / edge indices upload, add:
```python
        # Particle-centered CSR adjacency for deterministic PD reductions.
        n_p = self.model.particle_count
        if tet_indices_np is not None and tet_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(tet_indices_np, n_p, 4)
            self._particle_tet_offsets_d = wp.array(offs, dtype=int, device=dev)
            self._particle_tet_element_d = wp.array(elem_idx, dtype=int, device=dev)
            self._particle_tet_local_d = wp.array(local_v, dtype=int, device=dev)
        if tri_indices_np is not None and tri_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(tri_indices_np, n_p, 3)
            self._particle_tri_offsets_d = wp.array(offs, dtype=int, device=dev)
            self._particle_tri_element_d = wp.array(elem_idx, dtype=int, device=dev)
            self._particle_tri_local_d = wp.array(local_v, dtype=int, device=dev)
        if edge_indices_np is not None and edge_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(edge_indices_np, n_p, 4)
            self._particle_edge_offsets_d = wp.array(offs, dtype=int, device=dev)
            self._particle_edge_element_d = wp.array(elem_idx, dtype=int, device=dev)
            self._particle_edge_local_d = wp.array(local_v, dtype=int, device=dev)
```

- [ ] **Step A.7: Commit.**
```bash
git add newton/_src/solvers/fba/linear_solver.py newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Add particle-element CSR adjacency builder for deterministic PD reductions"
```

---

## Task B: Refactor `project_stretching_arap_tet_kernel` (the first kernel, lowest risk — used by PullingWooper)

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py:project_stretching_arap_tet_kernel`
- Modify: `newton/_src/solvers/fba/solver_fba.py` (step() — the tet ARAP launch site)
- Test: `newton/tests/test_solver_fba.py` (new regression test)

- [ ] **Step B.1: Read the existing kernel.**

`kernels.py:project_stretching_arap_tet_kernel` (find by name). Note:
- Inputs: positions, tet_indices (`(T, 4)`), tet weights, Dm_inv (rest-shape inverse), rhs (output, atomic-add target)
- Per-tet: computes deformation gradient `F = D · Dm_inv`, projects via `project_arap_3x3` to get `R = U·Vᵀ`, computes `H · w · Rᵀ · D` and atomic-scatters to the 4 vertex slots of rhs.

The contribution to a particle from one of its incident tets is a deterministic vec3 — write it to a scratch buffer per tet (per-tet `(4, vec3)`), then a gather kernel sums per-particle from the adjacency.

- [ ] **Step B.2: Write the failing regression test.**

Append to `test_solver_fba.py`:
```python
class SolverFBATetArapDeterminismTests(unittest.TestCase):
    """tet ARAP rhs scatter is bit-deterministic across reruns."""

    def test_two_runs_identical_rhs(self) -> None:
        import newton
        import numpy as np
        from newton.solvers import SolverFBA

        def run_once_capture_rhs() -> np.ndarray:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-10.0)
            # 3x3x3 soft cube — has ~80 tets, plenty of scatter conflicts per vertex.
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=3, dim_y=3, dim_z=3,
                cell_x=0.1, cell_y=0.1, cell_z=0.1,
                density=500.0, k_mu=5.0e3, k_lambda=5.0e3, k_damp=0.0,
            )
            model = builder.finalize()
            solver = SolverFBA(model, iterations=1, friction=False, stretching_model="arap")
            s_in = model.state(); s_out = model.state()
            for _ in range(3):
                s_in.clear_forces()
                solver.step(s_in, s_out, None, None, 1.0 / 60.0)
                s_in, s_out = s_out, s_in
            return s_in.particle_q.numpy().copy()

        q1 = run_once_capture_rhs()
        q2 = run_once_capture_rhs()
        # Float32 positions; tolerance ~1e-6 reflects FP-order independence,
        # which is the property the refactor guarantees.
        np.testing.assert_allclose(q1, q2, atol=1e-6,
                                   err_msg="tet ARAP scatter produced non-deterministic output")
```

Run it — it will FAIL on HEAD because of atomic scatter (or pass with low cube count and low contact conflict — depending on luck).

- [ ] **Step B.3: Add the compute kernel.**

In `kernels.py`, add `project_stretching_arap_tet_compute_kernel` that:
- Takes the same inputs as the old kernel, PLUS a `contributions: wp.array2d[wp.vec3]` of shape `(num_tets, 4)`
- For each tet (one thread per tet), computes the same contribution as before, but instead of atomic-scattering to `rhs[v_idx]`, writes to `contributions[tet_idx, local_v]`

The contribution per local vertex `i`:
```
contributions[tet_idx, i] = w · (Dm_inv · Rᵀ)[i, :]   (or equivalent — match the existing formula exactly)
```

(Exact formula: extract from the existing kernel's `wp.atomic_add(rhs, v_idx_i, contribution_i)` lines. Read the existing kernel carefully so the new kernel produces the SAME vec3 per (tet, local_v) pair.)

- [ ] **Step B.4: Add the gather kernel.**

In `kernels.py`, add `gather_per_particle_kernel`:
```python
@wp.kernel
def gather_per_particle_kernel(
    contributions: wp.array2d[wp.vec3],   # (num_elements, n_verts_per_element)
    offsets: wp.array[int],               # (n_particles + 1,)
    element_idx: wp.array[int],
    local_idx: wp.array[int],
    rhs: wp.array[wp.vec3],               # (n_particles,)
):
    p = wp.tid()
    start = offsets[p]
    end = offsets[p + 1]
    sum_p = wp.vec3(0.0, 0.0, 0.0)
    for k in range(start, end):
        sum_p = sum_p + contributions[element_idx[k], local_idx[k]]
    rhs[p] = rhs[p] + sum_p
```

This is the universal gather kernel — used by ALL the refactored energies, since they all share the same CSR adjacency pattern just with different `n_verts_per_element`.

(Note: `rhs[p] = rhs[p] + sum_p` — *not* `=` alone, because the inertial prediction kernel writes `rhs[p]` first. The gather adds the energy term on top. Verify by reading the existing scatter sequence in `step()`.)

- [ ] **Step B.5: Update the launch site in `solver_fba.py`.**

Find the existing `wp.launch(project_stretching_arap_tet_kernel, ...)` call inside `step()`. Replace with the two-kernel sequence:
```python
            if self._has_tets:
                # Allocate per-tet scratch on the fly (cheap — (T, 4) vec3).
                # If perf needs amortization, hoist to __init__.
                num_tets = self._num_tets
                tet_contrib = wp.zeros(shape=(num_tets, 4), dtype=wp.vec3, device=device)
                wp.launch(
                    project_stretching_arap_tet_compute_kernel,
                    dim=num_tets,
                    inputs=[self._x_cur, self._tet_indices_d, self._tet_weights_d, self._Dm_inv_d],
                    outputs=[tet_contrib],
                    device=device,
                )
                wp.launch(
                    gather_per_particle_kernel,
                    dim=N,
                    inputs=[tet_contrib, self._particle_tet_offsets_d,
                            self._particle_tet_element_d, self._particle_tet_local_d],
                    outputs=[self._rhs],
                    device=device,
                )
```

(Adapt to the existing variable names — `self._x_cur`, `self._rhs`, etc. — by reading the existing kernel launch site.)

- [ ] **Step B.6: Run the new regression test, expect PASS.**

- [ ] **Step B.7: Run the full FBA suite.**
```bash
uv run --extra dev -m newton.tests -k test_solver_fba
```
Expected: 89 PASS (87 prior + 1 ParticleElementCsr + 1 TetArapDeterminism). If any prior test fails, the refactored kernel produces different numerical output than the original — debug the contribution formula match. Allow `atol=1e-5` for any contact / NSN tests since FP-order changes can show through.

- [ ] **Step B.8: Re-run PullingWooper to confirm the refactor doesn't regress perf.**
```bash
uv run python scripts/fba_demo4_pulling_wooper.py 2>&1 | tail -5
```
Expected: `mean_ms` close to 10.5 ms (was 10.54 on the isodof path). If perf regresses > 50%, profile — the per-tet scratch alloc inside `step()` may need hoisting to `__init__`.

- [ ] **Step B.9: Commit.**
```bash
git add newton/_src/solvers/fba/kernels.py newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Refactor tet ARAP projection to particle-centered reduction"
```

---

## Task C: Refactor `project_stretching_corotational_tet_kernel`

Same pattern as Task B. The compute kernel writes per-tet `(4, vec3)` contributions; gather kernel is the same `gather_per_particle_kernel` from B.4. The only new code is the energy-specific compute kernel.

Steps:
- [ ] Read existing `project_stretching_corotational_tet_kernel` formula
- [ ] Add `project_stretching_corotational_tet_compute_kernel` mirroring tet ARAP's pattern
- [ ] Replace launch in `step()`
- [ ] Add `SolverFBATetCorotDeterminismTests` (mirror of the ARAP test class, just with `stretching_model="corotational"` + `mu=1e3, lam=1e3`)
- [ ] Run tests, expect 90 PASS
- [ ] Commit

---

## Task D: Refactor `project_stretching_neohookean_tet_kernel`

Same pattern. Add `*_compute_kernel`, replace launch, add determinism test (`stretching_model="neohookean"`), commit.

After Task D: 91 tests pass; tet path (PullingWooper, SqueezingBall ball) is fully deterministic.

---

## Task E: Refactor the three tri kernels (`project_stretching_*_tri_kernel`)

Each tri kernel has `n_verts_per_element=3`. The gather kernel is the SAME universal `gather_per_particle_kernel` (it indexes whatever element CSR you pass).

- [ ] Task E.1: tri ARAP (used by Demo 1, 3, StretchingCloth, SharpCorner, ParallelEnvTest)
- [ ] Task E.2: tri Corot
- [ ] Task E.3: tri NH (used by ClothOnKnives)
- [ ] Add `SolverFBATriDeterminismTests` (cloth grid scene)

After Task E: 94 tests pass; tri path is deterministic.

---

## Task F: Refactor `project_bending_kernel`

Bending stencil is 4 vertices (the 2 triangles sharing an edge). `n_verts_per_element=4` (matches tet, but uses a different adjacency CSR).

- [ ] Refactor `project_bending_kernel` into compute + gather (compute writes `(num_edges, 4)` vec3 contributions, gather uses `_particle_edge_*` CSR)
- [ ] Add `SolverFBABendingDeterminismTests` (small curved-rest cloth scene from Task H' — reuses an existing test fixture)
- [ ] 95 tests pass

---

## Task G: Refactor `build_jt_lambda_vec3_kernel` (isodof Jᵀλ scatter)

This is the per-contact-row → per-particle accumulation in the isodof Schur path. It's not a per-element energy term but follows the same pattern:
- Compute kernel: one thread per contact-row, writes contribution `λ[r] · Jᵀ[r, :]` to per-row buffer of shape `(total_rows, max_particles_per_row)` (for particle contacts, `max_particles_per_row = 1`)
- Gather kernel: per-particle, reads from contact-row CSR adjacency

In fact, for particle contacts where each row touches exactly 1 particle, this collapses to: for each particle, sum `λ[r]` of the (small) set of rows that touch it. The adjacency is just `contact_row → particle`, inverted.

Steps:
- [ ] Read `build_jt_lambda_vec3_kernel`
- [ ] Refactor compute + gather (or eliminate the kernel entirely by computing the per-particle sum directly from the contact arrays already on device)
- [ ] Test: `SolverFBAIsodofJtLambdaDeterminismTests` (3 runs of the cloth-on-plane Stage A scene, identical output)
- [ ] 96 tests pass
- [ ] Commit

---

## Task H: SqueezingBall full determinism verification

- [ ] **Step H.1: Run the reproducer 3 times.**
```bash
cd /home/ziqiu/work/newton
for i in 1 2 3; do
  echo "=== Run $i ==="
  uv run python scripts/fba_demo5_squeezing_ball.py 2>&1 | grep -E "min_y|x_drift|z_drift"
done
```
**Acceptance**: 3 runs produce identical `min_y` (or differ by < 1e-3). If still varies > 0.01, an atomic op was missed — grep `atomic_add` again in `kernels.py` and trace which one fires in this scene.

- [ ] **Step H.2: Run the full FBA suite + the wooper demo for regression check.**
```bash
uv run --extra dev -m newton.tests -k test_solver_fba
uv run python scripts/fba_demo4_pulling_wooper.py 2>&1 | tail -5
```
Expected: 96 tests OK. Wooper `mean_ms` within 20% of pre-refactor 10.5 ms.

- [ ] **Step H.3: Capture the deterministic baseline values in the demos README.**

Append to `scripts/contact_demos_out/README.md`:
```markdown
## Demo 5 — SqueezingBall (kinematic friction + deterministic PD)

(... existing content ...)

After Task P2-D (particle-centered PD reductions), 3 consecutive runs of this demo produce bit-identical particle positions. The previous variance (~5 unit spread in `min_y`) is gone.
```

---

## Acceptance Checklist

- [ ] All 96 FBA unit tests pass (87 prior + 9 new across Tasks A–G)
- [ ] SqueezingBall demo: 3 consecutive runs produce identical `min_y` (within FP epsilon)
- [ ] PullingWooper demo: `mean_ms` within 20% of pre-refactor baseline (10.5 ms)
- [ ] No `wp.atomic_add` calls remain in `newton/_src/solvers/fba/kernels.py` — `grep "atomic_add" newton/_src/solvers/fba/kernels.py` returns nothing

---

## Out of Scope

- **Newton's collision pipeline atomics** (`geometry/kernels.py:1127`). The contact-array order from `pipeline.collide()` remains non-deterministic at the Newton level; we handle it via the lexicographic sort in `update_contacts` (commit `09e4f855`). Fixing it at Newton's pipeline level is a separate effort upstream.
- **Warp's BSR SpMV** (transpose=False is documented deterministic; transpose=True is not used in FBA).
- **Performance optimization** of the per-element scratch buffers — current design allocates fresh each step. Hoisting to `__init__` is a follow-up if perf shows it matters (Task B.8 perf check informs this).

---

## Risk Register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Compute kernel produces different per-(element, local_v) vec3 than the original atomic scatter (formula transcription error) | Med | High | Each task's regression test compares against the prior atomic version with `atol=1e-5` — catches transcription bugs before commit |
| Per-step scratch buffer allocation dominates perf at small mesh sizes | Med | Med | Task B.8 checks PullingWooper perf; if regression > 50%, hoist allocation to `__init__` |
| 96-test suite reveals a prior bug masked by atomic-noise | Low | Med | Investigate per-failure; do NOT just relax tolerance |
| isodof Jᵀλ refactor (Task G) breaks Stage A friction-free contact (PullingWooper) | Low | High | Task G's regression test specifically covers Stage A; full FBA suite + wooper run gate the commit |