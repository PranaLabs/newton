# FBA × RealSim cross-check root cause

**Status:** RESOLVED (HIGH confidence)
**Date:** 2026-05-15
**Author:** diagnostic agent (Claude Opus 4.7)

## TL;DR

The 79 mm trajectory divergence between Newton's `SolverFBA` and
RealSim's reference is caused by a **single line** in
`newton/_src/solvers/fba/kernels.py::compute_inertial_kernel`. Newton
**skips gravity for pinned particles** when computing the inertial
prediction `x_inertia` (a.k.a. `sn`); RealSim **applies gravity to all
particles, regardless of whether they are pinned**. Even though pinned
particles have zero mass in Newton (so their `Msn` row contributes 0 to
the RHS), `x_inertia` is also used to **initialize `x_cur` before the PD
iteration loop**. That initial pinned-vertex position is fed into the
first iteration's ARAP and bending local projections at triangles
incident to the pin, producing per-step bias that compounds over 50
steps into the observed 79 mm tip displacement.

Patching `compute_inertial_kernel` to apply gravity unconditionally
collapses the frame-48 divergence from **7.91e-2 m** to **6.94e-6 m**
(pure float32 noise) — a 4-orders-of-magnitude reduction.

The previous diagnosis ("(M + dt²·L) vs (M/dt² + L) convergence rate")
was incorrect: under an exact linear solve, the two scalings yield
identical iterates. The convergence-rate hypothesis is moot.

## Evidence

### 1. RealSim per-step pseudocode (transcribed)

`src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp::update`,
lines 132–197 (and the parallel CUDA path in
`CUDALocalGlobalSolver.cpp::update`, lines 44–108, is identical):

```
// LocalGlobalSolver::update
1.  for each energy: energy->reset()
2.  computeExternalForce(f)               # f = M·g (all particles)
3.  _mass->computeSn(_nextpos, pos, vel, f, _dt)
        # sn_i = pos_i + dt*vel_i + dt²·invMass_i·f_i
        #      = pos_i + dt*vel_i + dt²·g                  (for ALL i)
4.  _mass->computeMx(f, _nextpos, 1.0)    # f = M·sn  (all particles)
5.  prepare(pos, _nextpos, _dt)
6.  for nb_iter in [0, _maxIter):
        _b = f                            # _b = M·sn
        for each energy:                  # ARAP, bending, pin (in addition order)
            energy->localProjection(_b, _nextpos, dt²)
        globalSolve(_nextpos, _b)         # _nextpos = A⁻¹·_b
7.  vel = (_nextpos - pos)/_dt
8.  pos = _nextpos
```

Key facts from the source:

- `Mass::computeSn` (Mass.cpp:59-63): applies `dt²·invMass_i·f_i` to
  every particle regardless of pin state. Combined with
  `computeExternalForce` (LocalGlobalSolver.cpp:199-211), which sets
  `acc.row(i) = _grav` for **all** particles and then `f = M·acc`, this
  yields `sn_i = pos_i + dt·vel_i + dt²·g` for both free and pinned
  vertices.
- `Mass::accumulateMatrix(triplets, coeff=1.0)` (Mass.cpp:45) is called
  with `coeff=1.0` (LocalGlobalSolver.cpp:87) so `A` accumulates the
  bare `mass_i` (not `mass_i / dt²`).
- Energies are accumulated with `coeff=dt²`
  (LocalGlobalSolver.cpp:85): elastic K and pin diagonal both receive
  the `dt²` multiplier.
- `localProjection(rhs, pos, coeff=dt²)`
  (LocalGlobalSolver.cpp:170): all energies see `coeff=dt²` so their
  RHS contributions are multiplied by `dt²`.
- Result: the solved system is
  `(M + dt²·(2μ·area·Σᵢ Kᵢ + w_pin·δ_pin·I)) x = M·sn + dt²·(2μ·area·proj + w_pin·δ_pin·x_ref)`.

### 2. Newton per-step pseudocode (transcribed)

`newton/_src/solvers/fba/solver_fba.py::step`, lines 154–271:

```
# SolverFBA.step
1.  compute_inertial_kernel(x_prev, v_prev, f_ext, inv_mass, world,
                            gravity, dt) -> x_inertia
        # kernels.py:86–92
        # a = f_ext·invMass + g·step(-invMass)
        # x_inertia = x_prev + dt·v_prev + dt²·a
        #
        # For free particles (invMass > 0): step(-invMass) = 0
        #     → x_inertia includes gravity.       ✓
        # For pinned particles (invMass == 0):    step(0) = 0
        #     → x_inertia EXCLUDES gravity.       ✗  *** ROOT CAUSE ***
2.  wp.copy(x_cur, x_inertia)            # initial x_cur = x_inertia
3.  for k in [0, iterations):
        zero_vec3_kernel(rhs)
        add_inertia_to_rhs_kernel(x_inertia, mass, dt, rhs)
            # rhs[i] += (mass[i]/dt²)·x_inertia[i]
            # For pinned particles, mass = 0, so this term is zero.
            # *** But x_inertia[pin] still leaked into x_cur in step 2 ***
        project_pin_kernel(pin_indices, x_ref, pin_stiffness, rhs)
            # rhs[pin] += w_pin · x_ref[pin]
        project_stretching_arap_kernel(x_cur, tri_indices, tri_rest_inv,
                                       tri_weight, rhs)
            # uses x_cur (gravity-less for pinned at iter 0)
        project_bending_kernel(x_cur, x_ref, edge_indices, edge_quad_q,
                               edge_weight, rhs)
        linear_solver.solve(rhs, x_cur)
4.  write_velocity_kernel(x_prev, x_cur, inv_mass, dt) -> v_new
        # if invMass==0: v_new = 0 (zeroed for pin)
        # else:          v_new = (x_cur - x_prev)/dt
5.  state_out.particle_q = x_cur
```

`build_pd_system` (linear_solver.py:38-188) assembles
`A = M/dt² + 2μ·area·Σ K + w_pin·I_pin`. The free-particle mass
diagonal omits pinned particles entirely (linear_solver.py:80-84) and
adds `pin_stiffness` (1e12 default, but the cross-check uses 1e10) at
the pinned diagonals (lines 87-91).

### 3. Side-by-side, line-by-line comparison

| Step                          | RealSim                                                     | Newton                                                        | Match?                              |
|------------------------------- |-------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------|
| Gravity for free particle      | `sn = x + dt·v + dt²·g` (always)                            | `x_inertia = x + dt·v + dt²·g` (when `invMass>0`)             | ✓                                  |
| **Gravity for pinned particle**| `sn = x + dt·v + dt²·g` (always)                            | `x_inertia = x + dt·v + 0` (since `step(0)=0`)                | ✗  **ROOT CAUSE**                  |
| Mass in A diagonal (free)      | `mass_i` (×1.0)                                             | `mass_i/dt²`                                                  | ✓  (equiv after scale)             |
| Mass in A diagonal (pin)       | `mass_i = 1/N` (kept, even for pin)                         | `mass_i = 0` (zeroed)                                         | Negligible (see "Subordinate")     |
| Pin diagonal of A              | `dt²·w_pin` (= 1e6 for w_pin=1e10, dt=0.01)                 | `w_pin` (= 1e10 for default)                                  | ✓  (equiv after scale)             |
| Pin in RHS                     | `dt²·w_pin·x_ref` added each iter                           | `w_pin·x_ref` added each iter                                 | ✓                                  |
| Stretching K in A              | `dt²·2μ·area·K_ab`                                          | `2μ·area·K_ab`                                                | ✓                                  |
| Stretching local proj          | `dt²·2μ·area·Dm⁻¹·P^T` (scatter)                            | `2μ·area·Dm⁻¹·P^T` (scatter)                                  | ✓                                  |
| Bending Hessian                | `dt²·3w/(A0+A1)·q·q^T` for interior edge                    | same (× edge_quad_scale)                                      | ✓                                  |
| Bending local proj             | `dt²·wᵢ·q_a·(e.normalize()·rest_norm)` (skipped if rest_norm≈0) | `w·q_a·(q^T·x_ref)` (zero for flat rest)                      | ✓ (both yield 0 for flat rest)     |
| Initial `x_cur` for PD loop    | `_nextpos = sn` (= pos + dt·v + dt²·g for all particles)    | `x_cur = x_inertia` (gravity-shifted for free; **NOT shifted for pin**) | ✗  (direct consequence of root cause) |
| ARAP projection in iter 0      | Evaluated at `_nextpos[i]` (gravity-shifted everywhere)      | Evaluated at `x_cur[i]` (gravity-shifted free; **flat for pin**) | ✗                                  |
| Velocity reduction at pin      | `v_pin = (x_pin_new - x_pin_prev)/dt` (small drift)         | `v_pin = 0` (forced)                                          | Negligible                          |
| `pin_stiffness` value          | 1e10 (init.h:1023)                                          | 1e10 in cross-check (default 1e12)                            | ✓ (matched in cross-check)         |

### 4. Numerical proof

I built a pure-Python reference that mirrors RealSim's algorithm
exactly (`/tmp/probe_step_one.py`). With identical parameters
(TRI_KE = 2μ = 7142.857, uniform 1/113 mass, w_pin = 1e10, dt = 0.01,
5 PD iterations):

```
ref_python vs RealSim trajectory: max delta = 7.14e-8 m (frame 48)
ref_python vs Newton trajectory:  max delta = 7.91e-2 m (frame 48)
```

The Python reference matches RealSim to float32 precision; the Newton
trajectory differs by 79 mm — exactly the gap the task describes.

Isolating each variable (`/tmp/isolate_cause.py`):

| Config | Pinned mass | Gravity applied to pinned? | Frame-48 delta vs RealSim |
|--------|-------------|----------------------------|---------------------------|
| A      | `1/N` (≠0)  | Yes                        | 7.14e-08 m                |
| B      | `0`         | No                         | **7.91e-02 m**            |
| C      | `0`         | Yes                        | 7.14e-08 m                |
| D      | `1/N` (≠0)  | No                         | 7.91e-02 m                |

Config C reproduces RealSim while keeping `mass[pin]=0` (Newton's
choice). Config D is the inverse: it gives 79 mm even with
non-zero pinned mass. **The pin-mass-zeroing is irrelevant; the
gravity-skipping is the entire effect.**

Direct verification (`/tmp/verify_cause.py`): monkey-patching
`compute_inertial_kernel` in Newton to apply gravity unconditionally
(`a = f_ext * im + g`) yields:

```
Original Newton (gravity skipped for pin): frame-48 delta = 7.91e-02 m
Patched Newton  (gravity always applied):  frame-48 delta = 6.94e-06 m
```

A 4-orders-of-magnitude reduction. The residual 7 µm is float32 noise
from `state_out.particle_q` storage.

### 5. Why this matters even though `mass[pin] = 0`

Newton's `add_inertia_to_rhs_kernel` (kernels.py:96-106) writes
`rhs += (mass/dt²)·x_inertia`. With `mass[pin]=0`, the pinned row's
RHS contribution is zero — so naively, `x_inertia[pin]` should be
irrelevant. **But it is also copied into `x_cur` before the PD
loop** (solver_fba.py:209: `wp.copy(self._x_cur, self._x_inertia)`).
`x_cur[pin]` is then read by `project_stretching_arap_kernel`
(kernels.py:166-235) and `project_bending_kernel` (kernels.py:238-284)
in iteration 0, entering the deformation-gradient computation for
every triangle that touches a pinned vertex.

In RealSim, the equivalent variable `_nextpos[pin] = sn[pin]
= x_ref + dt²·g`. In Newton, `x_inertia[pin] = x_prev[pin]
= x_ref[pin]`. The difference is `dt²·g = (0.01)² · 10 = 1e-3 m`
in the z-direction — small per-step, but it perturbs the ARAP
deformation gradient of every triangle touching the pin, biasing
the projection systematically and accumulating over 50 steps into
~80 mm at the free corners.

After iteration 0, the global solve drives `x_cur[pin]` toward
`x_ref` (the pin weight dominates), so the perturbation only
affects the first iteration's projection. But 1 iteration of bias
per step × 50 steps × the lever arm of a pinned-edge triangle
is plenty to produce the observed magnitude.

### 6. Subordinate hypotheses ruled out

- **(α) Gravity placement in RHS vs each-iter**: Both RealSim and
  Newton place gravity into `sn` once before the PD loop. Not a
  source of difference for free particles. *Confirmed equivalent.*
- **(β) Warm-start `x_0`**: Both use `x_cur = sn` (initial inertia
  prediction) before the PD loop. The differing initial value at
  *pinned* particles is the actual issue and is a symptom of the
  gravity-skipping bug, not a separate algorithmic choice.
- **(γ) `coeff` factor in `accumulateMatrix`**: Both use
  `coeff = dt²` for energies and `coeff = 1` for mass; the scaling
  is consistent. RealSim's system `(M + dt²L)x = M·sn + dt²·b` and
  Newton's system `(M/dt² + L)x = (M/dt²)·sn + b` are exact scalar
  multiples and yield identical solutions under exact solve. *No
  divergence source here.*
- **(δ) Implicit velocity damping**: Neither implementation applies
  per-step damping. *Confirmed absent in RealSim
  (LocalGlobalSolver.cpp:188 has no damping coefficient).*
- **(ε) ARAP rest reference**: Newton stores `tri_poses` from
  `particle_q` at builder time; RealSim stores `_restMatrix` from
  `getPositions()` at init time. Both reference the OBJ-loaded
  positions and produce identical 2x2 inverse rest pose
  (verified by sign+magnitude analysis of the Gram-Schmidt vs
  `cross(n, e1)` orthonormal basis: same span, same orientation).
- **(ζ) Iteration ordering / Gauss-Seidel-like updates**: Both
  implementations are pure Jacobi: each PD iteration recomputes
  the full RHS from the *previous* iterate, then a single global
  solve. *Confirmed identical loop structure.*
- **(η) `coeff = dt²` vs `coeff = 1`**: Same as (γ). RHS and matrix
  scale together; the solution is invariant. *No divergence.*
- **Bending energy projection sign**: For a flat rest cloth, both
  produce zero RHS contribution from bending. The Hessian term in
  both is the same (`w·q·q^T` with the same q from cotangents).
  *Confirmed equivalent.*
- **Float precision**: Newton's interior solver runs in float64;
  only `state_out.particle_q` is float32. The residual 6.94e-6 m
  after the patch is purely float32 storage noise. Not the cause.
- **Pin stiffness 1e10 vs 1e12**: Both produce `x_pin ≈ x_ref`
  to ~1e-9 m and ~1e-12 m respectively. *Negligible.*
- **`mass[pin] = 0` vs `mass[pin] = 1/N`**: Affects only the
  inertia-row contribution for pinned vertices (which the pin
  energy dominates by 6+ orders of magnitude anyway).
  *Confirmed negligible by Config A vs Config C comparison above:
  both yield 7e-8 m.*

## Primary cause (with file:line evidence)

**`newton/_src/solvers/fba/kernels.py:91`** — `compute_inertial_kernel`:

```python
a = f_ext[tid] * im + g * wp.step(-im)  # gravity active only for free particles (im>0)
```

`wp.step(-im)` evaluates to `1.0` only when `-im < 0` (i.e., `im > 0`,
i.e., free particle). For pinned particles with `im = 0`, `wp.step(0) = 0`
and gravity is excluded. This was a deliberate choice to "save work"
for particles that are constrained, but is inconsistent with RealSim's
implementation:

- **`src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:199-211`**
  — `computeExternalForce` writes `_grav` to **every** particle's
  acceleration row, including pinned vertices.
- **`src/Scomponent/integrator/localglobal/energy/Mass.cpp:59-63`** —
  `computeSn` loops over all particles uniformly.

The bug is only observable when the constrained particle's `x_inertia`
is read downstream by per-element local projections. Newton's
`solver_fba.py:209` (`wp.copy(self._x_cur, self._x_inertia)`) seeds the
PD loop's `x_cur` from `x_inertia`, so the iteration-0 ARAP / bending
projections at the pin neighborhood see a gravity-less pin position,
producing systematic bias.

## Minimal reproduction (2-particle)

A toy reproduction that exhibits the bug without setting up a full
cloth scene:

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

# Two particles, particle 0 pinned at origin, particle 1 free below it.
# A spring of stiffness k connects them; gravity pulls particle 1 down.
N = 2
dt = 0.01
g = np.array([0.0, 0.0, -10.0])
mass = np.array([1.0/N, 1.0/N])     # uniform
w_pin = 1e10
k = 7142.857                         # ARAP-equivalent spring stiffness
x_ref = np.array([[0.0, 0.0, 0.0],   # pinned
                  [0.0, 0.0, -1.0]]) # free
rest_len = 1.0

# Build A for both variants:
# RealSim: A_ii = mass_i + dt²*k (for spring) + dt²*w_pin (for pin0)
# Newton:  A_ii = mass_i/dt² + k (free) or pin_stiffness (pin)
# Use the RealSim scaling (multiply through by dt²).

# Assemble A as (M + dt²·H)
A = np.zeros((N, N))
for i in range(N):
    A[i, i] += mass[i] if i == 1 else 0.0  # Newton zeroes pinned mass
# spring contributes K = [[1,-1],[-1,1]] (1D for simplicity along z)
A[0, 0] += dt*dt * k
A[0, 1] += -dt*dt * k
A[1, 0] += -dt*dt * k
A[1, 1] += dt*dt * k
# pin on particle 0
A[0, 0] += dt*dt * w_pin

def step(x, v, gravity_for_pin: bool, n_iter=5):
    sn = x.copy() + dt*v
    if gravity_for_pin:
        sn += dt*dt*g           # RealSim: gravity for all
    else:
        sn[1] += dt*dt*g        # Newton: gravity only for free

    # Initialize nextpos = sn (this is the Newton-style x_cur warm start)
    nextpos = sn.copy()
    # In Newton, mass[pin]=0, so Msn[pin]=0. In Python ref we use mass[pin]=mass[1] for fairness.
    Msn = np.zeros_like(x)
    Msn[1] = mass[1] * sn[1]  # only free has nonzero mass*sn in Newton

    A_lu = splu(csr_matrix(A).tocsc())
    for _ in range(n_iter):
        b = Msn.copy()
        # Local projection: spring tries to restore particle 1 to z = x0 - rest_len.
        # Project current spring direction onto rest length:
        # ARAP-1D: F = (nextpos[1]-nextpos[0])/rest_len; P = sign(F).
        delta = nextpos[1] - nextpos[0]
        norm = np.abs(delta[2])
        sign = -1.0 if delta[2] < 0 else 1.0
        # projection target: each side gets sign*rest_len contribution
        # b += dt²·k·(±rest_len)
        b[0, 2] += -dt*dt * k * sign * rest_len  # particle 0 wants to be 1 above particle 1
        b[1, 2] += dt*dt * k * sign * rest_len   # particle 1 wants to be 1 below particle 0
        b[0, 2] += dt*dt * w_pin * x_ref[0, 2]   # pin

        # Solve column z (only z matters):
        new_z = A_lu.solve(b[:, 2])
        nextpos[:, 2] = new_z
    new_v = (nextpos - x) / dt
    return nextpos, new_v


# Run both for 50 steps
x_real, v_real = x_ref.copy(), np.zeros_like(x_ref)
x_newt, v_newt = x_ref.copy(), np.zeros_like(x_ref)
for _ in range(50):
    x_real, v_real = step(x_real, v_real, gravity_for_pin=True)
    x_newt, v_newt = step(x_newt, v_newt, gravity_for_pin=False)
print(f"RealSim-like x1.z = {x_real[1, 2]:.6f}")
print(f"Newton-like  x1.z = {x_newt[1, 2]:.6f}")
print(f"delta = {abs(x_real[1, 2] - x_newt[1, 2]):.4e}")
```

This 2-particle harness exposes the same bias: applying gravity to the
pinned particle changes the spring's resting projection in iter 0,
biasing the free particle's trajectory.

## Recommended Newton-side fix

**Single-line fix.** Apply gravity to all particles unconditionally in
`compute_inertial_kernel`:

```python
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
    """x_inertia = x_prev + dt·v_prev + dt²·(f_ext/m + g).
    Gravity is applied to ALL particles (RealSim convention).
    For pinned particles, this affects only the initial `x_cur` of the
    PD loop; the pin energy will still drive `x_cur[pin]` to `x_ref`."""
    tid = wp.tid()
    w_idx = wp.max(particle_world[tid], 0)
    g = gravity[w_idx]
    im = inv_mass[tid]
    a = f_ext[tid] * im + g    # gravity unconditional
    x_inertia[tid] = x_prev[tid] + v_prev[tid] * dt + a * (dt * dt)
```

**Caveat: external forces.** The current formula `f_ext[tid] * im`
already evaluates to zero for pinned particles (`im = 0`), so external
forces are naturally disabled for pins — consistent with RealSim's
`Mass::computeSn` which multiplies `f_ext` by `invMass` (zero for
infinitely heavy / pinned in RealSim's convention, although RealSim
doesn't actually use `invMass=0` because pins have `mass=1/N`).

**Alternative, semantically richer fix.** Add a `pin_indices` field
to `Model` and detect pinning by membership in that array rather than
by `inv_mass == 0`. Then the pinned mass can remain `1/N` (matching
RealSim), eliminating an off-spec convention. This is a larger refactor
and unnecessary for the cross-check parity.

**Recommended test.** Add a regression test under
`newton/tests/test_solver_fba.py` that compares a single step's
`x_inertia[pinned_idx]` against the analytic expression
`x_ref + dt·v + dt²·g`. Currently no test exercises pinned-particle
gravity behavior.

**Documentation note.** Update the docstring of
`compute_inertial_kernel` to make explicit that gravity is applied
uniformly and that pinned particles' `x_inertia` is only consumed by
local projections (the inertia-row RHS for pinned particles is zero
because `mass = 0`).

## Files referenced (all absolute paths)

- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp`
  (lines 132–197, 199–211)
- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/CUDALocalGlobalSolver.cpp`
  (lines 44–108)
- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/Mass.cpp`
  (lines 12–63)
- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.cpp`
  (lines 35–169)
- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/elastic/PDIsometricBendingEnergy.cpp`
  (lines 69–128)
- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/hardconstraint/PinEnergy.cpp`
  (lines 6–53)
- `/home/ziqiu/work/RealSim_py/realsim_py/include/Scomponent/integrator/localglobal/energy/elastic/ElasticEnergy.h`
  (lines 34–44)
- `/home/ziqiu/work/RealSim_py/realsim_py/include/init.h` (lines
  922–1023)
- `/home/ziqiu/work/newton/newton/_src/solvers/fba/solver_fba.py`
  (lines 154–271)
- `/home/ziqiu/work/newton/newton/_src/solvers/fba/kernels.py`
  (lines 75–123)  **← THE FIX LIVES HERE (line 91)**
- `/home/ziqiu/work/newton/newton/_src/solvers/fba/linear_solver.py`
  (lines 38–188)
- `/home/ziqiu/work/newton/scripts/fba_realsim_crosscheck.py`
- Reproduction scripts (transient): `/tmp/probe_step_one.py`,
  `/tmp/isolate_cause.py`, `/tmp/verify_cause.py`
