# Task P2-E: Full RealSim NSN Port Implementation Plan (v2 — corrected)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port RealSim's complete `NonSmoothNewton::solve` semantics into SolverFBA. After this lands, SqueezingBall should reach the y=-10 plane with `|x_drift|, |z_drift| < 1.5`.

**v2 changes** from the earlier broken simplification: this plan reflects the full RealSim NSN structure including `omega` weighting, position-LCP coupling, `dλ` (delta) output, and the asymmetric h formula between normal and friction rows. Source spec from a verbatim read of `NonSmoothNewton.cpp:102-170, 332-378`.

---

## Pre-flight Checks

- [ ] Branch on `ziqiu/fba-solver-design`, HEAD = `08b60e5d` (P2E-A landed).
- [ ] `git status` working tree clean.
- [ ] FBA suite 103/103 passes.
- [ ] **The P2E-A helpers (`fb_unilateral_row`, `fb_frictional_row`) MUST be REWRITTEN** because they used the wrong h formula for the normal row and didn't return `omega`. This plan supersedes them. See Task A.

---

## Reference: corrected algorithm (per RealSim source spec)

**Per call to `_solve_nsn_*`** (one NSN solve = one outer Local-Global iter on RealSim side; FBA can call multiple iters by stacking):

```
Given inputs:
  W       : (3M, 3M) Schur (or (M, M) for unilateral)
  r       : residual = pene0 - α·J·x_unc
  pene0   : per-row anchor projection (constant within step)
  μ       : per-contact friction coefficient
  λ_in    : current λ iterate (warm-start; zero on first iter)
  dt      : timestep

Per outer NSN iter:
  # Compute current penetration at λ_in (since J·x_corrected = J·x_unc + W·λ_in
  # and pene0 = J·x_unc + r, we get J·x_corrected - pene0 = W·λ_in - r).
  penetration = W · λ_in - r        # vector of length 3M (or M)

  # Per-row FB evaluation: returns (omega, compliance, h).
  for each row in each contact:
    compute omega[r], compliance[r], h[r] via FB helpers (see Task A formulas).

  # Schur LHS: A_schur = diag(omega) · W · diag(omega) + diag(compliance)
  A_schur = (omega outer omega) * W + diag(compliance)

  # Schur RHS: rhs = (1/dt²) · (h - omega · J·x_corrected)
  #         where J·x_corrected = J·x_unc + W·λ_in = (pene0 - r) + W·λ_in
  J_x = (pene0 - r) + W · λ_in
  rhs = (1 / dt²) · (h - omega * J_x)

  # Solve for delta lambda.
  dλ = A_schur⁻¹ · rhs

  # Update lambda.
  λ_out = λ_in + dλ

  # Box clamp (boundConstraintForces, NonSmoothNewton.cpp:380-408):
  #   normal: λ_n ← max(0, min(λ_n, maxforce))
  #   friction: |λ_t| ≤ μ·λ_n PER AXIS (rectangular box, NOT circular cone)
  for each contact:
    λ_out[3c]    = max(0, λ_out[3c])
    cone = μ[c] · λ_out[3c]
    λ_out[3c+1] = clip(λ_out[3c+1], -cone, +cone)
    λ_out[3c+2] = clip(λ_out[3c+2], -cone, +cone)

  λ_in = λ_out   # for next NSN iter

return λ_out
```

**Critical math notes:**

1. **`dλ` is a DELTA**, not the new lambda. λ accumulates across iters.
2. **Normal row's h formula DIFFERS from friction row's**:
   - Normal row (`NonSmoothNewton.cpp:340`):
     ```
     pene = penetration[n]
     plam = precond[n] · λ_n
     root = √(pene² + plam²)
     omega[n] = 1 - pene / root      # CAN exceed 1 in deep penetration
     compliance[n] = (1 - plam / root) · (precond[n] / dt²)
     h[n] = -(pene + plam - root) + omega[n] · (penetration[n] + pene0[n])
     ```
   - Friction row, **active branch** when `λ_n > 0` (line 357-377):
     ```
     abspenevel = |penetration[t] / dt|
     tmp = precond[t] · (μ · λ_n - |λ_t|)
     root = √(abspenevel² + tmp²)
     omega[t] = 1
     compliance[t] = ((root - tmp) / (abspenevel + μ·precond[t]·λ_n - root)) · (precond[t] / dt)
     h[t] = -dt² · compliance[t] · λ_t + pene0[t]
     ```
   - Friction row, **inactive branch** when `λ_n ≤ 0` (line 348-356):
     ```
     omega[t] = 0
     compliance[t] = 1 / dt
     h[t] = -dt · λ_t
     ```
3. **`pene0[t]` semantically** is `t·anchor_at_current_frame`, equivalent to FBA's `_contact_tangent1_offset_h` (which is `dot(t1, world_anchor) + dt·dot(t1, v_anchor)`). Plumbing already exists.
4. **`precond[k] = 1 / W_kk`** (the diagonal of W, inverted). RealSim sets it in `prepare`; we compute from `np.diag(W)`.
5. **`omega` can exceed 1** in deep penetration — that's intentional. Don't clamp.

---

## Task A (REWRITE): FB helpers returning `(omega, compliance, h)`

The existing P2E-A helpers must be REPLACED. Their h formula was wrong for the normal row, and they didn't return omega.

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py` (replace existing fb_* helpers)
- Modify: `newton/tests/test_solver_fba.py` (update FBNonsmoothFunctionTests)

### Step A.1: Update tests

Replace the existing `FBNonsmoothFunctionTests` class in `test_solver_fba.py` with:

```python
class FBNonsmoothFunctionTests(unittest.TestCase):
    """FB unilateral and frictional row helpers — return (omega, compliance, h)."""

    def test_unilateral_signature(self) -> None:
        from newton._src.solvers.fba.solver_fba import fb_unilateral_row
        omega, compliance, h = fb_unilateral_row(
            penetration=0.0, lam=0.0, precond=1.0, dt=0.01, pene0=0.0
        )
        self.assertTrue(np.isfinite(omega))
        self.assertTrue(np.isfinite(compliance))
        self.assertTrue(np.isfinite(h))

    def test_unilateral_stick(self) -> None:
        """Open contact (penetration > 0) with lam = 0 → omega near 0 (inactive)."""
        from newton._src.solvers.fba.solver_fba import fb_unilateral_row
        omega, compliance, h = fb_unilateral_row(
            penetration=0.1, lam=0.0, precond=1.0, dt=0.01, pene0=0.0
        )
        # ω = 1 - pene/root = 1 - 0.1/0.1 = 0.
        self.assertAlmostEqual(omega, 0.0, places=5)

    def test_unilateral_penetration(self) -> None:
        """Penetrating contact (penetration < 0) → omega > 1 (active, deep)."""
        from newton._src.solvers.fba.solver_fba import fb_unilateral_row
        omega, _, _ = fb_unilateral_row(
            penetration=-0.1, lam=0.0, precond=1.0, dt=0.01, pene0=0.0
        )
        # ω = 1 - (-0.1)/0.1 = 2.0. Can exceed 1 per source spec.
        self.assertAlmostEqual(omega, 2.0, places=3)

    def test_frictional_inactive(self) -> None:
        """λ_n ≤ 0 → omega = 0, compliance = 1/dt, h = -dt·λ_t."""
        from newton._src.solvers.fba.solver_fba import fb_frictional_row
        omega, compliance, h = fb_frictional_row(
            penetration=0.05, lam_t=10.0, lam_n=0.0, mu=0.5,
            precond=1.0, dt=0.01, pene0=0.0,
        )
        self.assertAlmostEqual(omega, 0.0, places=5)
        self.assertAlmostEqual(compliance, 100.0, places=3)  # 1/dt
        self.assertAlmostEqual(h, -0.01 * 10.0, places=6)

    def test_frictional_active_stick(self) -> None:
        """Active contact + cone slack > 0 → omega = 1, compliance small."""
        from newton._src.solvers.fba.solver_fba import fb_frictional_row
        omega, compliance, h = fb_frictional_row(
            penetration=1e-6, lam_t=10.0, lam_n=100.0, mu=0.5,
            precond=1.0, dt=0.01, pene0=0.0,
        )
        self.assertAlmostEqual(omega, 1.0, places=5)
        self.assertLess(compliance, 5.0)  # stick: small compliance

    def test_frictional_active_slip(self) -> None:
        """Active contact + on cone → omega = 1, compliance large (~1/dt)."""
        from newton._src.solvers.fba.solver_fba import fb_frictional_row
        omega, compliance, h = fb_frictional_row(
            penetration=0.1, lam_t=50.0, lam_n=100.0, mu=0.5,
            precond=1.0, dt=0.01, pene0=0.0,
        )
        self.assertAlmostEqual(omega, 1.0, places=5)
        self.assertGreater(compliance, 10.0)
```

### Step A.2: Replace fb_unilateral_row

```python
def fb_unilateral_row(
    penetration: float,
    lam: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Fischer-Burmeister evaluation for a unilateral (normal) contact row.

    Per RealSim NonSmoothNewton.cpp:332-341. Returns ``(omega, compliance, h)``.

    For the **normal** row the h formula includes ``omega · (penetration + pene0)``;
    do NOT use the simpler friction-row formula here.

    Args:
        penetration: ``J·q − pene0`` at current iterate (m).
        lam: Current normal lambda (N).
        precond: ``1 / W_ii`` (1/N).
        dt: Timestep (s).
        pene0: ``n·anchor`` (m).

    Returns:
        ``(omega, compliance, h)``:
        - ``omega``: per-row weighting ``1 − pene/√(pene² + (precond·lam)²)``.
          May exceed 1 in deep penetration (penetration < 0); do not clamp.
        - ``compliance``: diagonal regularization ``(1 − precond·lam/root)
          · (precond / dt²)``.
        - ``h``: RHS contribution ``-(pene + precond·lam − root)
          + omega · (penetration + pene0)``.
    """
    pene = penetration
    plam = precond * lam
    root = math.sqrt(pene * pene + plam * plam)
    if root < 1e-30:
        # Degenerate; both args zero.
        return 0.0, precond / (dt * dt), pene0
    omega = 1.0 - pene / root
    compliance = (1.0 - plam / root) * (precond / (dt * dt))
    h = -(pene + plam - root) + omega * (penetration + pene0)
    return omega, compliance, h


def fb_frictional_row(
    penetration: float,
    lam_t: float,
    lam_n: float,
    mu: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Fischer-Burmeister evaluation for a frictional (tangent) contact row.

    Per RealSim NonSmoothNewton.cpp:343-378. Returns ``(omega, compliance, h)``.

    Inactive contact (``lam_n ≤ 0``): ``omega = 0``, ``compliance = 1/dt``,
    ``h = -dt · lam_t``. The friction row is decoupled and dissipates.

    Active contact (``lam_n > 0``): smooth complementarity between slip speed
    ``|penetration|/dt`` and cone slack ``μ·lam_n − |lam_t|``. ``omega = 1``
    (binary), compliance varies between near-zero (stick) and ``1/dt`` (slip).
    """
    if lam_n <= 0.0:
        return 0.0, 1.0 / dt, -dt * lam_t

    abspenevel = abs(penetration / dt)
    tmp = precond * (mu * lam_n - abs(lam_t))
    root = math.sqrt(abspenevel * abspenevel + tmp * tmp)
    denom = abspenevel + mu * precond * lam_n - root
    if abs(denom) < 1e-30:
        compliance = 1.0 / dt
    else:
        compliance = ((root - tmp) / denom) * (precond / dt)
    h = -(dt * dt) * compliance * lam_t + pene0
    return 1.0, compliance, h
```

### Step A.3: Run tests

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k FBNonsmoothFunction
```
Expected: 6 PASS.

### Step A.4: Full suite

```bash
uv run --extra dev -m newton.tests -k test_solver_fba
```
Expected: 103 + 1 = 104 PASS (one extra test from A.1's additional `test_unilateral_penetration`).

### Step A.5: Commit

```bash
git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Rewrite FB helpers per RealSim spec (return omega; correct normal h formula)"
```

---

## Task B: Implement `_solve_nsn_coulomb` and `_solve_nsn_unilateral` per RealSim spec

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py`

### Step B.1: Replace `_solve_nsn_coulomb` body

```python
    def _solve_nsn_coulomb(
        self,
        W: np.ndarray,
        r: np.ndarray,
        mu: np.ndarray,
        pene0: np.ndarray,
        max_iters: int = 1,
        lam_init: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        """RealSim NonSmoothNewton port for frictional LCP.

        Implements the dλ-Newton update from NonSmoothNewton.cpp:102-170 + 343-378,
        with omega weighting and position-LCP coupling. λ accumulates across
        max_iters iterations.

        Args:
            W: ``(3M, 3M)`` Schur complement.
            r: ``(3M,)`` residual ``pene0 − α·J·x_unc`` from ``_compute_contact_residual_friction``.
            mu: ``(M,)`` per-contact friction coefficient.
            pene0: ``(3M,)`` per-row anchor projection.
            max_iters: NSN iterations (default 1 — RealSim's Local-Global outer iter
                count, here exposed as the inner Newton count since FBA invokes
                this solver inside its own PD outer loop).
            lam_init: Optional warm-start.
            dt: Timestep (s).

        Returns:
            λ of shape ``(3M,)``.
        """
        M = len(mu)
        if M == 0:
            return np.zeros(0, dtype=np.float64)
        size_total = 3 * M

        if lam_init is not None and lam_init.shape == (size_total,):
            lam = lam_init.astype(np.float64, copy=True)
        else:
            lam = np.zeros(size_total, dtype=np.float64)

        diag_W = np.diag(W)
        precond = np.where(np.abs(diag_W) > 1e-12, 1.0 / np.maximum(diag_W, 1e-12), 1.0)

        for _ in range(max_iters):
            # Current penetration at lambda.
            penetration = W @ lam - r

            # Per-row FB evaluation.
            omega = np.zeros(size_total, dtype=np.float64)
            compliance = np.zeros(size_total, dtype=np.float64)
            h = np.zeros(size_total, dtype=np.float64)
            for c in range(M):
                idx_n = 3 * c
                idx_t1 = 3 * c + 1
                idx_t2 = 3 * c + 2
                on, cn, hn = fb_unilateral_row(
                    penetration=penetration[idx_n],
                    lam=lam[idx_n],
                    precond=precond[idx_n],
                    dt=dt,
                    pene0=pene0[idx_n],
                )
                ot1, ct1, ht1 = fb_frictional_row(
                    penetration=penetration[idx_t1],
                    lam_t=lam[idx_t1],
                    lam_n=lam[idx_n],
                    mu=mu[c],
                    precond=precond[idx_t1],
                    dt=dt,
                    pene0=pene0[idx_t1],
                )
                ot2, ct2, ht2 = fb_frictional_row(
                    penetration=penetration[idx_t2],
                    lam_t=lam[idx_t2],
                    lam_n=lam[idx_n],
                    mu=mu[c],
                    precond=precond[idx_t2],
                    dt=dt,
                    pene0=pene0[idx_t2],
                )
                omega[idx_n] = on
                omega[idx_t1] = ot1
                omega[idx_t2] = ot2
                compliance[idx_n] = cn
                compliance[idx_t1] = ct1
                compliance[idx_t2] = ct2
                h[idx_n] = hn
                h[idx_t1] = ht1
                h[idx_t2] = ht2

            # Schur LHS: A = diag(omega) · W · diag(omega) + diag(compliance).
            A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)

            # Schur RHS: rhs = (1/dt²) · (h - omega · J·x_corrected)
            #            J·x_corrected = J·x_unc + W·λ = (pene0 - r) + W·λ.
            J_x = (pene0 - r) + W @ lam
            rhs = (1.0 / (dt * dt)) * (h - omega * J_x)

            # Solve for delta lambda.
            try:
                dlam = np.linalg.solve(A_schur, rhs)
            except np.linalg.LinAlgError:
                break
            lam = lam + dlam

            # boundConstraintForces: clamp box per contact.
            for c in range(M):
                lam_n_c = max(0.0, lam[3 * c])
                lam[3 * c] = lam_n_c
                cone = mu[c] * lam_n_c
                lam[3 * c + 1] = float(np.clip(lam[3 * c + 1], -cone, cone))
                lam[3 * c + 2] = float(np.clip(lam[3 * c + 2], -cone, cone))

        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
        return lam
```

### Step B.2: Replace `_solve_nsn_unilateral` body

```python
    def _solve_nsn_unilateral(
        self,
        W: np.ndarray,
        r: np.ndarray,
        pene0: np.ndarray,
        max_iters: int = 1,
        lam_init: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        """RealSim NonSmoothNewton port for unilateral LCP (Stage A)."""
        M = len(r)
        if M == 0:
            return np.zeros(0, dtype=np.float64)
        if lam_init is not None and lam_init.shape == (M,):
            lam = lam_init.astype(np.float64, copy=True)
        else:
            lam = np.zeros(M, dtype=np.float64)

        diag_W = np.diag(W)
        precond = np.where(np.abs(diag_W) > 1e-12, 1.0 / np.maximum(diag_W, 1e-12), 1.0)

        for _ in range(max_iters):
            penetration = W @ lam - r
            omega = np.zeros(M, dtype=np.float64)
            compliance = np.zeros(M, dtype=np.float64)
            h = np.zeros(M, dtype=np.float64)
            for c in range(M):
                on, cn, hn = fb_unilateral_row(
                    penetration=penetration[c],
                    lam=lam[c],
                    precond=precond[c],
                    dt=dt,
                    pene0=pene0[c],
                )
                omega[c] = on
                compliance[c] = cn
                h[c] = hn

            A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)
            J_x = (pene0 - r) + W @ lam
            rhs = (1.0 / (dt * dt)) * (h - omega * J_x)

            try:
                dlam = np.linalg.solve(A_schur, rhs)
            except np.linalg.LinAlgError:
                break
            lam = lam + dlam
            np.maximum(lam, 0.0, out=lam)

        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
        return lam
```

### Step B.3: Update call sites in `step()`

Same as before — pass `pene0_b` (Coulomb) or `pene0_a` (unilateral) and `dt`. See plan v1 Step B.3 for the exact code (unchanged).

### Step B.4: Run FBA suite

```bash
uv run --extra dev -m newton.tests -k test_solver_fba
```
Expected: most pass. Some test tolerances may need relaxing (1e-3 → 1e-2) if FB-Newton produces slightly different per-iter behavior than PGS. If a test fails with magnitude > 1e-2, debug — likely a sign or formula error.

**Critical test**: `single_plane_contact_pushes_particle_away` MUST pass (this was the failing test in v1 due to `h=0` bug). With the corrected normal-row h formula, this should now produce a non-zero force.

### Step B.5: PullingWooper regression

```bash
uv run python scripts/fba_demo4_pulling_wooper.py 2>&1 | tail -5
```
Expected: stable, `min_y ≈ -8.4`, `pulled = 10`. Slower (dense LU per iter); `mean_ms` may reach 40-80 ms.

### Step B.6: SqueezingBall test

```bash
for i in 1 2 3; do
  echo "=== Run $i ==="
  uv run python scripts/fba_demo5_squeezing_ball.py 2>&1 | grep -E "min_y|x_drift|z_drift"
done
```

Acceptance:
- 3 runs bit-identical (numpy linalg is deterministic)
- `min_y ≤ -8`
- `|x_drift|, |z_drift| < 1.5`

If `min_y > -6`, BLOCKED — formula or sign error remains.

### Step B.7: Commit

```bash
git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Port RealSim NonSmoothNewton solver (omega + position-LCP coupling)"
```

---

## Task C: Documentation

Same as v1 Task C — update README Demo 5 section + progress doc with trajectory parity results.

---

## Acceptance Checklist

- [ ] All FBA tests pass (104+ expected)
- [ ] SqueezingBall 3-run bit-identical
- [ ] SqueezingBall `min_y ≤ -8` and `|drift| < 1.5`
- [ ] PullingWooper still stable, `min_y ≈ -8.4`
- [ ] `mean_ms` for both demos within 5× of pre-P2-E baseline

---

## Risk Register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| `omega · W · omega` outer product unstable when omega ≈ 0 → singular A_schur | High | Med | Compliance diagonal regularization prevents singular; ε epsilon-denom in fb helpers catches degenerate |
| Normal row h formula transcription error | Med | High | Step A.1 tests verify per-regime values; Step B.4 catches integration failures |
| `J·x_unc = pene0 - r` derivation only valid for α=1 (which we have, but worth checking) | Low | Med | All FBA contacts use α=1 since they're particle contacts |
| Convergence requires multiple iters (max_iters=1 not enough) | Med | Low | RealSim's Local-Global runs 5 iters total per step. FBA's NSN here is invoked per PD outer iter, so 5 PD × 1 NSN ≈ 5 NSN iters total, comparable. |
| `boundConstraintForces` box vs cone shape (RealSim's box is non-physical but lossy in slip) | Low | Low | Documented as known divergence; either keep box (RealSim parity) or use cone (physical correctness). |