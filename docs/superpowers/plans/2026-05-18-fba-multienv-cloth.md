# Plan: SolverFBA multi-env support via LiteNSN — contact-rich cloth

**Date:** 2026-05-18
**Status:** Drafted, not started.
**Owner:** TBD
**Estimated effort:** ~2 weeks (Phase 0 multi-env smoke 0.5d, Phase 1 single-env baseline 1-2d, Phase 2 LiteNSN 1w, Phase 3 multi-env scaling 3-5d).
**Target scenario:** `RealSim_py/simulation/config/CudaTests/ParallelEnvTest` — 25 cloth-on-sphere environments (5×1×5 grid, 10 m spacing).

---

## Why

We have not validated SolverFBA on **parallel environments**. RealSim ships `parallelEnv` natively (see `RealSim_py/simulation/simulation.cpp:75-136`) and pairs it with a **LiteNSN** constraint solver that swaps the dense Schur `W = H · A⁻¹ · Hᵀ` for a sparse `W = H · M⁻¹ · Hᵀ`. The combination is what makes contact-rich cloth scenes at scale tractable.

**Memory + compute math for ParallelEnvTest** (25 envs, 1013-particle cloth-on-sphere each, ~500 contacts/env after settling):

| Quantity | Full NSN (dense W) | LiteNSN (sparse W) |
|---|---|---|
| W shape | 37500 × 37500 (one block per env merged into flat dense) | 37500 × 37500 BSR, ~3 nnz per row |
| W memory (fp64) | **11.25 GB** | **~1 MB** |
| Schur build (extrapolated from Demo 5's 0.87 ms/iter @ 500 row) | 0.87 × 625 = **544 ms/iter** | ≤ **0.5 ms/iter** |
| Cross-env wasted compute | (NK)² = 25² block-diag fill-in | 0 (sparse W block-diagonal naturally) |

Dense W at this scale is **literally infeasible** on a 32 GB RTX 5090 — let alone larger fleets. Sparse W stays in tens of MB even at 100+ envs. **LiteNSN is a prerequisite, not an optimization.**

The single-env cloth benefit is smaller: ~500 contact rows → 18 MB dense W, fits and is fast. LiteNSN single-env will likely be **slightly slower** than full NSN due to sparse iter overhead. So the design needs **both modes** behind a `nsn_schur_mode` flag, with full NSN as default and LiteNSN explicitly opt-in for cloth-parallel-env workloads. Non-cloth (softbody / tet) scenes do NOT get the LiteNSN path — see the locked architectural decision below.

## What changes

Two orthogonal pieces of work, joined by a third validation phase:

1. **New SolverFBA constraint mode: `nsn_schur_mode = "lite"`** — adds a sparse Schur build (`build_schur_lite`) and a sparse-matvec PCR path. Switchable per-instance, default `"full"`. Code lives entirely under `newton/_src/solvers/fba/`.

2. **New example: cloth-on-sphere FBA** — `newton/examples/cloth/example_cloth_on_sphere_fba.py`, mirroring RealSim's ParallelEnvTest single-env geometry. Standalone usefulness for SolverFBA contact testing.

3. **Multi-env validation harness** — a script (NOT an example, since user-facing examples shouldn't `replicate()` by default) that builds N copies of the cloth-on-sphere model and measures step time + memory vs N, exercising both `"full"` and `"lite"` modes.

The PD outer loop, A_FBA matrix assembly, and pin handling are **unchanged**. The linear pre-solve already factorizes `A_FBA` once and uses `bsr_mv` for application; that path is reused.

## Pre-conditions

- `viewer_gl.log_mesh` texture fix is in (commit `24c85b8a`). Required for cloth checker textures in any visualization.
- `wp.sparse.bsr_mm` and `bsr_mv` exist in our pinned Warp build. **Verified** (`grep` finds both in `warp/sparse.py`).
- `builder.replicate(builder, world_count, spacing)` works for cloth. **Untested for cloth** — Phase 0 is the first stress (contact-free, no sphere). Sphere replication is exercised later in Phase 3.

## Architectural decisions

### LOCKED: build sparse W via warp.sparse `bsr_mm`, not cuSPARSE directly

RealSim's LiteNSN calls `cusparseSpGEMM` (`CUDASparseInverseSolver.cpp:368-477`). Newton already wraps cuSPARSE through `wp.sparse.bsr_mm`, so we don't need a second wrapping. If `bsr_mm` perf turns out poor on these contact-graph shapes we can drop to a custom Warp kernel (per-particle scan over row pairs sharing that particle); we'll measure before going there.

### LOCKED: sparse Schur output as `wp.sparse.BsrMatrix[wp.float64]`

Match the existing `A_FBA` precision and storage (`linear_solver.py:754` says `block_type=wp.float64`).

### LOCKED: PCR sparse path uses `bsr_mv`; preconditioner = `W.diag()`

Newton's current dense PCR (`nsn_pcr_solver.py`) uses `wp.launch_tiled(matmul_kernel, ...)` for the matvec. The sparse path replaces it with `wps.bsr_mv(W_bsr, p, q)`. Jacobi preconditioner from `W.diag` (same primitive RealSim uses, `LiteNonSmoothNewton.cpp:211-243`).

### LOCKED: LiteNSN scoped to cloth scenes only; full NSN stays default

LiteNSN trades elastic coupling for sparsity. For softbody / tet-based scenes (Demos 2/4/5, TwistingBar variants), contact resolution depends on elastic restoring forces propagating through the full A matrix — dropping that into M⁻¹ degrades convergence catastrophically. Cloth is different: contacts are predominantly *normal* against a rigid obstacle (sphere, ground), so M⁻¹ captures enough of the local dynamics for FB-Newton to converge.

**Decision:**
- LiteNSN is validated **only on cloth-on-sphere** (Phase 2.4 single-env, Phase 3 multi-env).
- Demos 2/3/4/5 keep `nsn_schur_mode="full"` and are out of scope for LiteNSN testing.
- Default stays `"full"`; users explicitly opt into `"lite"` for cloth parallel-env workloads.

### LOCKED: validate physics with full NSN first

Phase 1 builds cloth-on-sphere with the **current solver (`"full"`)** before touching LiteNSN. This establishes a known-good Tier 1 reference. Without it, any post-LiteNSN bug ("did the math break or did the implementation break?") is unbisectable.

### OPEN: outer iteration count for LiteNSN

RealSim's ParallelEnvTest uses **10 NSN iters with FB function and 1e-9 tol** at the linear solver layer. Whether SolverFBA's current outer loop (`nsn_iterations=1` for FB-Newton) needs an increase under LiteNSN coupling loss is empirical. Phase 2.4 will measure.

## Phase 0 — contact-free 2-env hanging-cloth sanity check (~0.5 day)

**Goal:** Validate that `builder.replicate()` + SolverFBA's PD path (no NSN, no Schur, no LiteNSN) handles two independent cloths correctly. This isolates the **replicate mechanism + PD + bending + gravity + pin** layer from all contact-solver complexity. If it fails here, every later phase is unbisectable.

The hanging-cloth example (`newton/examples/cloth/example_hanging_cloth_fba.py`, commit `503eb30f`) is the right fixture: 10225-particle cloth, top-edge pin via `particle_mass = 0`, bending=1e-3, no contacts. Two copies of this should evolve identically modulo a fixed translation.

### Step 0.1 — extract `build_single_env_builder()` from hanging cloth

Factor the model-construction half of `example_hanging_cloth_fba.Example.__init__` into a free function that returns a `ModelBuilder` (not a finalized Model). The existing example continues to work — it just calls this helper, then finalizes.

```python
def build_hanging_cloth_builder() -> newton.ModelBuilder:
    """Construct a single-env hanging-cloth ModelBuilder (no .finalize())."""
    ...
    return builder
```

This pattern is the same one Phase 3.1 will need for cloth-on-sphere, so we're paying the refactor cost once.

### Step 0.2 — `scripts/fba_multienv_hanging_cloth_smoke.py`

A standalone smoke-test script (NOT an example, since `replicate()` doesn't belong in the user-facing example surface yet):

```python
sub = build_hanging_cloth_builder()
main = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-10.0)
main.replicate(sub, world_count=2, spacing=(5.0, 0.0, 0.0))
model = main.finalize()
solver = SolverFBA(model, iterations=10)  # full default config
...run 100 frames...
```

### Step 0.3 — correctness acceptance gates

Run 100 frames at dt=0.05. Verify:

1. **Tier 1 binary (must pass):** Both envs stay finite. No NaN, no >100 m drift.
2. **Tier 2 (must pass within fp32 round-off):** The two envs evolve identically modulo their static spacing offset:
   ```python
   q = state.particle_q.numpy().reshape(2, n_per_env, 3)
   offset = np.array([5.0, 0.0, 0.0])
   diff = q[1] - (q[0] + offset)
   assert np.linalg.norm(diff) / np.linalg.norm(q[0]) < 1e-5
   ```
   A failure here means **some kernel is incorrectly aggregating across worlds** — probably a global reduction, or a non-`world_idx`-aware gravity / dt path. Stop and find it.
3. **Tier 3 diagnostic:** Step time is roughly 2× single-env (allow 1.5×-3×). Sub-linear scaling here is a free win; super-linear (> 3×) suggests AMD reordering didn't find the block-diagonal structure in the PD prefactor (see Risks).

### Step 0.4 — if Tier 2 fails, bisect

The failure modes are well-defined:
- Gravity wrong: `model.gravity` is per-world `wp.array[wp.vec3]`; verify `solver.step` reads `gravity[particle_world[tid]]`, not `gravity[0]`. Newton's base `integrate_particles` (`solver.py:35`) already does this — but SolverFBA may have its own external-force injection path that bypasses it.
- PD prefactor cross-env coupling: check `wp.sparse.bsr_get_diag(A_FBA)` per particle; off-diagonal entries should only connect particles in the same world.
- Pin kernel cross-env: check `particle_inv_mass = 0` is respected per-world.

Document what was found in a follow-up spec.

---

## Phase 1 — single-env cloth-on-sphere baseline (1-2 days)

**Goal:** Newton example matching ParallelEnvTest geometry runs to completion with **full NSN**, and the rendered trajectory visually matches RealSim ground truth.

### Step 1.1 — example_cloth_on_sphere_fba.py

Build the single-env scene mirroring `CudaTests/ParallelEnvTest/cloth0.json`:
- `square_1013P.obj` (path `/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_1013P.obj`)
- Rx(90°) then scale 3.5 then trans (0, 3.5, 0)
- ARAP (`stretching_model="arap"`), tri_ke = 2·μ from Young=1e5, Poisson=0.4
- `edge_ke` for `bending=20.0` — note: the scale mismatch we hit on the hanging cloth (RealSim 0.1 ≠ Newton edge_ke 0.1) needs revisiting here. Start with the literal value, observe behavior, tune if needed.
- `add_shape_sphere(radius=3.0)` at origin, `is_visible=True` (use visualradius 2.95 as the visual size only if we feel like splitting visual/collision; otherwise use 3.0 directly)
- `friction=True`, `mu=0.3`
- `gravity=-10` Y-up, `dt=0.01`, PD iter=5
- `nsn_iterations=1` (Newton's FB-Newton default)
- `pin_stiffness` — no pins, but solver still needs a value; carry forward Demo 5's setting

**Test:** runs for 300 frames at `--viewer null` with no NaN, particle bounding box stays reasonable (cloth descends from y=3.5 to ~y=0.5 or so, mass-conserving stays roughly flat afterward as it drapes on the sphere).

### Step 1.2 — capture RealSim ground-truth trajectories (BOTH solver modes)

Algorithm-fair acceptance requires comparing **same-algorithm vs same-algorithm**. RealSim ParallelEnvTest defaults to LiteNSN; for Newton-full validation we need a RealSim-full reference too. Capture both, overriding the constraint solver:

```bash
# Baseline A — RealSim FULL NSN, single env (for Phase 1 Newton-full validation)
realsim --config CudaTests/ParallelEnvTest \
        --override 'parallelEnv.matrix=[1,1,1]' \
        --override 'constraintsolver.type=NonSmoothNewton_CUDA'
# → scripts/realsim_baseline/cloth_on_sphere_full_ref.npz

# Baseline B — RealSim LITE NSN, single env (for Phase 2.4 Newton-lite validation)
realsim --config CudaTests/ParallelEnvTest \
        --override 'parallelEnv.matrix=[1,1,1]'
# → scripts/realsim_baseline/cloth_on_sphere_lite_ref.npz

# Baseline C — RealSim LITE NSN, full N=25 (for Phase 3.5 multi-env validation)
realsim --config CudaTests/ParallelEnvTest
# → scripts/realsim_baseline/parallel_env_cloth_lite_ref.npz
```

Each `.npz` contains:
- `positions` — particle positions per frame (single-env: `(300, 1013, 3)`; multi-env: `(300, 25, 1013, 3)`)
- `n_contacts` — saturation curve per frame
- `step_ms` — RealSim per-step time

**Newton-mode → RealSim-baseline pairing:**

| Newton run | Compared against | Why |
|---|---|---|
| Phase 1.3 (Newton `"full"`) | Baseline A (RealSim full NSN) | apples-to-apples |
| Phase 2.4 (Newton `"lite"`) | Baseline B (RealSim LiteNSN) | apples-to-apples |
| Phase 3.5 (Newton `"lite"` × 25) | Baseline C (RealSim LiteNSN × 25) | apples-to-apples |
| Newton `"full"` vs Newton `"lite"` | *not compared* | different algorithms; drift expected and not gated |

If we can't reproduce any of A/B/C, the relevant phase is blocked. Capture all three upfront in Step 1.2.

### Step 1.3 — visual + numerical validation

Render Newton preview frames at 0 / 50 / 150 / 299; alongside, dump Newton's per-frame particle positions to `/tmp/fba_cloth_on_sphere.npz`.

**Acceptance gate (all must pass):**
1. **Tier 1 binary** — cloth makes contact with sphere by frame ~50 (visible drape, particles below y < 3.0 cloth surface).
2. **Tier 2 trajectory match — Newton `"full"` vs RealSim full NSN baseline (A)** — per-particle position error on the *settled* frame (frame 200, after initial drape transient):
   ```python
   rs = np.load("scripts/realsim_baseline/cloth_on_sphere_full_ref.npz")["positions"][200]
   fba = np.load("/tmp/fba_cloth_on_sphere_full.npz")["positions"][200]
   per_particle_err = np.linalg.norm(rs - fba, axis=1)
   assert np.percentile(per_particle_err, 95) < 0.05, "Tier 2: p95 per-particle drift > 5cm"
   assert per_particle_err.max() < 0.15, "Tier 2: max per-particle drift > 15cm"
   ```
   Apples-to-apples (both sides use full NSN). 5 cm @ p95 / 15 cm max is calibrated to the **RealSim FBA verification tiers** memory (Demo 4-style drape, looser than Demo 5 contact). Tightens to 1-2 cm if achievable, but 5/15 is the gate.
3. **Tier 3 diagnostics** — contact count saturates around frame 100 (cloth settled), doesn't grow indefinitely. Compare `n_contacts[150:300]` mean against RealSim's — should be within ±20% (cloth seating is sensitive to friction state, exact contact count not strict).

### Step 1.4 — perf baseline

`SolverFBA.get_timing_summary` with `enable_perf_timing=True`. Record step_mean_ms, Schur build ms, NSN inner ms. **This becomes the regression baseline** for LiteNSN's Phase 2.4 comparison.

## Phase 2 — LiteNSN implementation (~1 week)

### Step 2.1 — sparse W build (`build_schur_lite`)

In `linear_solver.py`, add a sibling to `_build_schur_isodof`:

```python
def _build_schur_lite(
    self,
    row_particle, row_particle_d, row_dir_d, row_alpha_d,
    total_rows,
) -> wps.BsrMatrix:
    # 1. Build H as BsrMatrix (1 block per contact row, 3 DOFs per block)
    H = self._build_jacobian_bsr(row_particle_d, row_dir_d, total_rows)
    # 2. Scale columns of H by sqrt(1/m_i) (or equivalently scale Hᵀ rows)
    M_inv_HT = self._build_jacobian_scaled_T(row_particle_d, row_dir_d, row_alpha_d, total_rows)
    # 3. W = H · (M⁻¹ Hᵀ) via sparse SpGEMM
    W = wps.bsr_mm(H, M_inv_HT)
    return W
```

The `row_alpha_d` scaling absorbs friction compliance (existing FB-Newton convention).

**Subtask 2.1.a — H as BSR:** Newton already has `row_particle` (which particle each row touches) and `row_dir` (the 3-vec direction). Convert to BSR with block size 3, one nonzero block per row, in particle-index column order. ~50 lines.

**Subtask 2.1.b — scaled Hᵀ:** Either pre-scale and use `bsr_transposed()` from warp.sparse, or directly assemble `M⁻¹ Hᵀ` row-by-row. Pick whichever is cleaner.

**Subtask 2.1.c — `bsr_mm`:** One call. Result is a BsrMatrix.

**Test:** numerical regression — for a small fixed scene (e.g. 10-row toy), compare `W_lite = H · diag(1/m) · Hᵀ` against `W_full = H · A⁻¹ · Hᵀ`. Verify the **sparsity patterns** are identical (same contact pairs share particles) and the **values are correctly scaled** by `1/m_p` vs full A⁻¹ on the diagonal.

### Step 2.2 — sparse PCR inner (`nsn_pcr_solver.py`)

Add a sparse variant of `pcr_solve`. Existing dense path uses `wp.launch_tiled(matmul_kernel, dim=(n,1), ...)`. Replace with:

```python
wps.bsr_mv(W_bsr, p, q)  # q = W · p
```

Preconditioner: extract `W.diag()` once at build time, store inverse, apply componentwise.

**Test:** on the same 10-row toy, `pcr_solve_dense(W_full, rhs)` and `pcr_solve_sparse(W_full_as_bsr, rhs)` give identical results (within fp64 round-off, ~1e-12).

### Step 2.3 — solver_fba.py integration

```python
class SolverFBA:
    def __init__(self, ..., nsn_schur_mode: Literal["full", "lite"] = "full"):
        ...
```

Dispatch in NSN step:
```python
if self._nsn_schur_mode == "lite":
    W = self._linear.build_schur_lite(...)
    dlam = pcr_solve_sparse(W, rhs, max_iter, tol)
else:
    W = self._linear.build_schur_complement(...)  # existing dense path
    dlam = pcr_solve_dense(W, rhs, max_iter, tol)
```

**Test:** run `example_cloth_on_sphere_fba` (Phase 1 scene) with `nsn_schur_mode="lite"` for 50 smoke frames. Confirm: stays finite, no kernel-launch errors, contact count looks sensible (~RealSim n_contacts at frame 50 within ±50%, looser than the final-state gate below since this is just a "does it launch" check).

LiteNSN is **scoped to cloth scenes only**; we do NOT test or claim correctness on Demo 5 (SqueezingBall), Demo 4 (PullingWooper), or any tet-softbody/twist scenarios. The "lite" path bypasses elastic coupling at the Schur level, which is a poor fit for softbody NSN where elastic forces are tightly coupled with contact resolution. Softbody demos keep `nsn_schur_mode="full"`.

### Step 2.4 — single-env cloth-on-sphere LiteNSN parity validation

Same scene as Phase 1, same dt, same 300 frames. Compared against **RealSim LiteNSN baseline (B)** from Step 1.2 — apples-to-apples (both sides use LiteNSN).

**Note on Newton self-consistency:** we explicitly do **not** require `"lite"` to match Newton's `"full"` output. LiteNSN's Schur (W = H · M⁻¹ · Hᵀ) is a mathematically distinct algorithm from full NSN's (W = H · A⁻¹ · Hᵀ), so drift between modes is expected and not a defect. Acceptance is keyed only on each Newton mode matching its same-algorithm RealSim counterpart.

**Acceptance gate (all must pass):**
1. **Tier 1 binary** — cloth contacts sphere by frame ~50 (same as Phase 1.3).
2. **Tier 2 — Newton `"lite"` vs RealSim LiteNSN baseline (B):** per-particle drift on the settled frame (200) — **p95 < 5 cm, max < 15 cm** vs `cloth_on_sphere_lite_ref.npz`. Same budget as Phase 1.3 because both gates are same-algorithm comparisons.
3. **Tier 3 diagnostics** — log per-frame NSN outer iter count and PCR iter count. Document the iter-count delta vs Phase 1.3's full-NSN run (informational; not gated).

**Perf measurement:** vs Phase 1.4 baseline, document:
- Schur build delta (expect much faster — sparse build)
- NSN inner delta (expect slower — more PCR iters, sparser matrix less cache-friendly)
- End-to-end step time delta (expect roughly even single-env; cloth is the wrong scene to "win" on)

If LiteNSN is more than 3× slower than full NSN at single-env cloth-on-sphere, profile and tune (preconditioner choice, sparsity-pattern caching across PCR iters, etc.). Do NOT block Phase 3 on closing this gap — multi-env is where the win lives, and the single-env regression is expected on cloth (low contact count, dense W cheaply fits in cache).

## Phase 3 — multi-env scaling (3-5 days)

### Step 3.1 — replicate harness script

`scripts/fba_multienv_cloth_bench.py`:

```python
import newton
from newton.examples.cloth.example_cloth_on_sphere_fba import build_single_env_builder

main = newton.ModelBuilder()
sub = build_single_env_builder()  # factor out the builder construction from the Example
main.replicate(sub, world_count=args.n_envs, spacing=(10.0, 10.0, 10.0))
model = main.finalize()

solver = SolverFBA(model, ..., nsn_schur_mode=args.mode)
# run 300 frames, time each step
```

This needs `example_cloth_on_sphere_fba.py` to expose a `build_single_env_builder()` helper that the script can import. Refactor in Phase 1 (same pattern Phase 0 already applied to hanging cloth).

### Step 3.2 — scaling test

Run with N ∈ {1, 4, 9, 16, 25}, both modes. Plot step_time vs N.

**Acceptance gate:**
1. `"lite"` mode scales **linearly** in N up to N=25 (step_time(25) ≤ 30 × step_time(1) — accommodates some constant overhead).
2. `"full"` mode either OOMs above some N or scales O(N²) (expected, validates the motivation for the work).
3. At N=25, `"lite"` step time matches RealSim ParallelEnvTest within 3×. (RealSim reference perf needs to be captured — see Step 3.3.)

### Step 3.3 — RealSim ground-truth multi-env reference (Baseline C, captured in Step 1.2)

This baseline is already captured in Step 1.2 (Baseline C: RealSim LiteNSN × 25, the default `ParallelEnvTest` config). Loaded here for the multi-env perf and trajectory acceptance gates below.

`scripts/realsim_baseline/parallel_env_cloth_lite_ref.npz`:
- `positions: (300, 25, 1013, 3) float32`
- `n_contacts_per_env: (300, 25) int32` — if RealSim emits per-env counts; otherwise total
- `step_ms: (300,) float32`

### Step 3.4 — cross-env consistency (Newton-internal)

All 25 envs start identical and have identical sphere geometry (no per-env randomization). Their final states should be **physically identical** (modulo fp non-associativity). Validate:
```python
q = state.particle_q.numpy().reshape(25, 1013, 3)
offsets = compute_world_offsets(25, (10,10,10), Axis.Y)  # newton's helper
for i in range(1, 25):
    diff = (q[i] - offsets[i]) - (q[0] - offsets[0])
    err = np.linalg.norm(diff) / np.linalg.norm(q[0] - offsets[0])
    assert err < 1e-3, f"env {i} diverged from env 0"
```

If this fails, it diagnoses a **world-coupling bug** — some kernel is incorrectly aggregating across worlds. Stop and find it.

### Step 3.5 — RealSim trajectory parity per env (apples-to-apples: lite vs lite)

For each of the 25 envs, the Newton-LiteNSN trajectory should match the RealSim-LiteNSN reference (Baseline C) at frame 200 (settled state). Same algorithm both sides:

```python
ref = np.load("scripts/realsim_baseline/parallel_env_cloth_lite_ref.npz")
rs = ref["positions"][200]   # (25, 1013, 3)
fba = state.particle_q.numpy().reshape(25, 1013, 3) - offsets[:, None, :]  # un-offset to local

per_env_p95 = np.array([np.percentile(np.linalg.norm(rs[i] - fba[i], axis=1), 95)
                        for i in range(25)])
per_env_max = np.array([np.linalg.norm(rs[i] - fba[i], axis=1).max() for i in range(25)])

assert per_env_p95.max() < 0.05, f"env-worst p95 drift {per_env_p95.max():.3f} > 5 cm"
assert per_env_max.max() < 0.15, f"env-worst max drift {per_env_max.max():.3f} > 15 cm"
```

**Acceptance budget reasoning:** identical to Phase 2.4 (same algorithm both sides, 5/15 cm budget). If a coupling-related drift were going to leak into LiteNSN at scale, Step 3.4 catches it as cross-env *inconsistency*; Step 3.5 catches it as RealSim divergence. Together they pin down "Newton multi-env LiteNSN reproduces RealSim multi-env LiteNSN."

If Step 2.4 (single-env lite) passed but Step 3.5 fails *for env 0 only*: regression in the multi-env path. If it fails *uniformly across all envs*: numeric divergence at scale; profile and bisect.

## Out of scope (deferred to follow-up plans)

- **`nsn_schur_mode="lite"` for any non-cloth scene**: softbody / tet / twist / stretch demos all keep `"full"`. The M⁻¹ approximation discards elastic coupling that those scenes depend on. Validating `"lite"` outside cloth-on-sphere is explicitly NOT part of this plan; if a future plan wants to extend LiteNSN coverage, it owns its own correctness validation.
- **Per-env randomization**: ParallelEnvTest replicates one cloth identically. Per-env perturbation (initial pose, mass, material) is RL-training territory and adds infrastructure (per-world parameter arrays) we don't need yet.
- **Per-env gravity / dt / timestep**: similarly deferred.
- **CPU LiteNSN path**: GPU only. CPU NSN was deleted in `2026-05-17-fba-nsn-gpu-port.md` Step 13.
- **Newton's `wp.sparse.bsr_mm` perf tuning**: if `bsr_mm` is slow on our contact-graph shapes, document and either fall back to a custom kernel (Phase 2 contingency) or work around at the matrix-build layer (don't try to optimize warp.sparse itself).
- **Newton viewer rendering 25 envs**: not part of this plan. The viewer probably needs `viewer.set_world_offsets()` to visually separate them; that integration is a one-off if/when we want a multi-env video.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `wp.sparse.bsr_mm` doesn't accept the contact-Jacobian sparsity pattern (block-size mismatch, asymmetric structure) | Phase 2.1 blocks for days while we write custom SpGEMM | Subtask 2.1.c smoke-tests `bsr_mm` on the toy scene **before** wiring into solver. If it fails, drop to per-particle custom Warp kernel (well-defined fallback, ~2 day cost). |
| LiteNSN convergence is so much worse than full NSN that doubling outer iter count erases the per-iter speedup | LiteNSN ends up no faster at large N | Step 2.4 measures iter-count delta. If outer iter cost grows >3× relative to dense, switch from Jacobi preconditioner to **block-Jacobi at per-particle granularity** (`LiteNonSmoothNewton.cpp:233-238` does exactly this for friction triples). |
| Multi-env block-diagonal A is **not detected by warp.sparse's AMD reordering**, so the PD pre-factor balloons in fill-in | step time scales O(N²) instead of O(N), defeating the purpose | Phase 0 Tier 3 (step-time vs single-env) is the first signal; Step 3.2 then quantifies at scale. If AMD reordering doesn't find the block structure, manually pre-partition particles by world (each contiguous chunk = one env) so the symbolic factorization can't help itself. |
| Cloth + sphere contact + ARAP at dt=0.01 with bending=20 explodes (parallel to our `hanging_cloth` 0.1→1e-3 backoff) | Phase 1 stalls on physics, blocks the whole plan | Bisect bending strength: try 20, then 1, then 0.01. Document Newton's edge_ke normalization vs RealSim's. If a Newton-side fix to edge_ke is needed, that's a separate plan (out of scope here — just empirically tune until stable). |
| `builder.replicate` doesn't handle the cloth+sphere case (e.g. sphere shape not properly replicated) | Phase 3 blocks at construction | Step 3.1 first verifies the model finalizes correctly with N=4 before going to 25. Inspect `model.particle_world`, `model.shape_world` arrays for correctness. The Phase 0 smoke already de-risks cloth-only replication. |

## Done definition

This plan is done when:
1. ✅ Phase 0 contact-free 2-env hanging-cloth smoke test passes Tier 1 + Tier 2 (env-to-env identity modulo spacing).
2. ✅ `example_cloth_on_sphere_fba.py` passes Phase 1 acceptance gates with `nsn_schur_mode="full"`:
   - Step 1.3 Tier 1 (contact by frame 50)
   - Step 1.3 **Tier 2: per-particle drift vs RealSim ref at frame 200 — p95 < 5 cm, max < 15 cm**
   - Step 1.3 Tier 3 (contact saturation within ±20% of RealSim)
3. ✅ `SolverFBA(nsn_schur_mode="lite")` is callable and passes Phase 2.4 cloth-on-sphere single-env Tier 2 (apples-to-apples: Newton-lite vs RealSim-LiteNSN baseline B, p95 < 5 cm / max < 15 cm). (No Newton-lite vs Newton-full self-consistency gate — different algorithms, drift expected. No softbody / non-cloth correctness claims.)
4. ✅ Phase 3.2 scaling test: `"lite"` scales linearly in N up to 25, step time within 3× RealSim's ParallelEnvTest reference.
5. ✅ Phase 3.4 cross-env consistency check passes (env-to-env divergence < 1e-3 relative).
6. ✅ **Phase 3.5 RealSim trajectory parity at scale passes (per-env p95 < 5 cm, max < 15 cm vs RealSim ParallelEnvTest reference).**
7. ✅ Plan-level commit on `ziqiu/fba-solver-design` (or a follow-up branch) with the example, solver mode, harness script, captured RealSim baselines under `scripts/realsim_baseline/`, and a follow-up spec documenting any RealSim-vs-FBA drift observed at scale.
