# Phase 1.3.a — NH LBFGS Port Spec

**Date:** 2026-05-17
**Goal:** Port RealSim's LBFGS-based local projection for Tri/Tet NH to Warp, replacing FBA's
5-iter fixed Newton. Implementation-blocker for SqueezingBall (NH ball under squeezing must
pass through gap intact — binary success).

**Scope:** Design only. No code changes. All design questions in §4 require explicit user sign-off
before implementation begins.

---

## 1. RealSim's LBFGS implementation

### 1.1 LBFGS template signature

`deps/mcloptlib/include/MCL/LBFGS.hpp:36-41`

```cpp
template<typename Scalar, int DIM, int M=8>
class LBFGS : public Minimizer<Scalar,DIM> {
private:
    typedef Eigen::Matrix<Scalar,DIM,1> VecX;
    typedef Eigen::Matrix<Scalar,DIM,M> MatM;
    typedef Eigen::Matrix<Scalar,M,1> VecM;
```

Instantiations used in NH:

- `PDNeohookeanTriangleEnergyParallel.h:19, 149` — `mcl::optlib::LBFGS<double, 2>` (2D singular-value problem)
- `PDNeohookeanTetrahedronEnergyParallel.h:19, 162` — `mcl::optlib::LBFGS<double, 3>` (3D singular-value problem)

**Both NH variants take the default history depth `M = 8`.** No override seen anywhere in
`include/Scomponent/integrator/`. The history slot count `M=8` is therefore the canonical port
target (not 2 as the controller speculated).

**Default settings inherited from `Minimizer::Settings` (`Minimizer.hpp:66-70`):**

| Setting          | Default                                  |
|------------------|------------------------------------------|
| `verbose`        | `0`                                      |
| `max_iters`      | `50` (overridden in LBFGS ctor at LBFGS.hpp:47) |
| `ls_max_iters`   | `100000`                                 |
| `ls_decrease`    | `1e-4` (Armijo sufficient-decrease constant `c_1`) |
| `ls_method`      | `LSMethod::BacktrackingCubic`            |

No NH-elastic call site overrides these (`init()` only calls `problem.set_lame(mu, lam)`;
the solver vector is default-constructed → keeps defaults).

### 1.2 LBFGS step algorithm

`LBFGS.hpp:52-151` — full `minimize(problem, x, x0)` body. `x` is the iterate; `x0` is the
*previous-step PD anchor* `σ_0` (snapshot of singular values at PD outer-iter start), passed
through to `problem.value/gradient` for the quadratic penalty `(k/2)·‖x-x0‖²`. **Do not confuse
`x0` with the previous LBFGS iterate.**

**Local state declared inside `minimize` (LBFGS.hpp:53-67):**
```cpp
MatM s, y;                          // shape (DIM, M), columns = history
VecM alpha, rho;                    // shape (M,), two-loop scalars
VecX grad, q, grad_old, x_old, x_last;
```
This state is **stack-local**, so **LBFGS history is freshly zeroed on every `minimize` call**.
Per PD outer iter, per element, the solver restarts from an empty history. (See §3.2.)

**Initial gradient (LBFGS.hpp:70-73):**
```cpp
problem.gradient(x, x0, grad);
Scalar gamma_k = 1.0;
Scalar alpha_init = 1.0;
```

**Main loop (LBFGS.hpp:79-147):**

```cpp
for (int k=0; k<max_iters; ++k) {
    x_old = x;
    grad_old = grad;
    q = grad;
    global_iter++;

    // L-BFGS first-loop recursion (LBFGS.hpp:86-92)
    int iter = std::min(M, k);
    for (int i = iter-1; i >= 0; --i) {
        rho(i) = 1.0 / (s.col(i).dot(y.col(i)));
        alpha(i) = rho(i) * s.col(i).dot(q);
        q = q - alpha(i)*y.col(i);
    }

    // Scale by gamma_k, then second-loop recursion (LBFGS.hpp:94-99)
    q = gamma_k * q;
    for (int i = 0; i < iter; ++i) {
        Scalar beta = rho(i) * q.dot(y.col(i));
        q = q + (alpha(i) - beta) * s.col(i);
    }

    // Descent check + restart on non-descent (LBFGS.hpp:101-108)
    Scalar dir = q.dot(grad);
    if (dir <= 0) {
        q = grad;
        max_iters -= k;
        k = 0;
        alpha_init = std::min(1.0, 1.0 / grad.template lpNorm<Eigen::Infinity>());
    }

    // Line search on direction p = -q (LBFGS.hpp:110-115)
    Scalar rate = this->linesearch(x, x0, -q, problem, alpha_init);
    if (rate <= 0) return FAILURE;

    // Step + convergence (LBFGS.hpp:117-119)
    x_last = x;
    x -= rate * q;
    if (problem.converged(x_last, x, grad)) break;

    // Recompute gradient and history pair (LBFGS.hpp:121-135)
    problem.gradient(x, x0, grad);
    VecX s_temp = x - x_old;
    VecX y_temp = grad - grad_old;

    if (k < M) {
        s.col(k) = s_temp;  y.col(k) = y_temp;
    } else {
        s.leftCols(M-1) = s.rightCols(M-1).eval();
        s.rightCols(1) = s_temp;
        y.leftCols(M-1) = y.rightCols(M-1).eval();
        y.rightCols(1) = y_temp;
    }

    // gamma update (LBFGS.hpp:137-145)
    Scalar denom = y_temp.dot(y_temp);
    if (std::abs(denom) <= 0) break;
    gamma_k = s_temp.dot(y_temp) / denom;
    alpha_init = 1.0;
}
```

Key observations:

- **Two-loop recursion is the textbook Nocedal-Wright form.** `q` holds `-r_k = -H_k·g_k`. The
  line search direction is `p = -q` (LBFGS.hpp:110), so the actual step is `x -= rate * q`
  (LBFGS.hpp:118), equivalent to `x += rate*p`.
- **History order:** `col(0) = oldest, col(min(M,k)-1) = newest`. First-loop runs newest→oldest;
  second-loop runs oldest→newest. After M iters, a left-shift evicts col(0) and writes the
  newest pair into `col(M-1)`. **Note: the eviction is a copy `s.leftCols(M-1) = s.rightCols(M-1).eval()` —
  a non-circular buffer.** Port faithfully or use a circular index (semantically equivalent).
- **Initial Hessian scaling `γ_k`:** Nocedal-Wright "method 1" — `γ_k = ⟨s_k, y_k⟩ / ⟨y_k, y_k⟩`
  computed *after* each accepted step (LBFGS.hpp:144), `γ_0 = 1.0`. Applied at LBFGS.hpp:95 between
  the two loops as `q ← γ_k · q`. This is `H_0 = γ_k · I` per-iteration.
- **Non-descent restart:** if `q·grad ≤ 0`, the search direction is non-descent. Code falls back
  to steepest descent (`q = grad`, so `-q = -grad`), resets `k` to 0 (next iter discards history,
  inner loops see `iter = min(M, 0) = 0` and skip), decrements remaining budget. The
  `alpha_init` after restart uses `min(1, 1/‖grad‖∞)` to avoid overshooting in a bad basis.

### 1.3 Wolfe / Backtracking line search

**Method actually used:** `BacktrackingCubic` (Minimizer.hpp default, no override). **Armijo only —
no curvature condition (`c_2`).** Source `Backtracking.hpp:74-145`.

```cpp
// Backtracking.hpp:79-113 (BacktrackingCubic::search)
static inline Scalar search(int verbose, int max_iters, Scalar decrease,
                            const VecX &x, const VecX &x0, const VecX &p,
                            const Problem<Scalar,DIM> &problem, Scalar alpha0) {
    const Scalar t_eps = std::numeric_limits<Scalar>::epsilon();
    if (p.norm() <= t_eps) return decrease;

    Scalar alpha = alpha0;
    VecX grad;
    Scalar fx0 = problem.gradient(x, x0, grad);  // gradient at x (along with f value)
    Scalar gtp = grad.dot(p);
    Scalar fxp = fx0;
    Scalar alphap = alpha;

    int iter = 0;
    for (; iter < max_iters; ++iter) {
        Scalar fxa = problem.value(x + alpha*p, x0);
        Scalar fx0_fxa = fx0 + alpha*decrease*gtp;     // Armijo: f(x+αp) ≤ f(x) + c_1·α·∇f·p
        if (fxa <= fx0_fxa) break;

        // Cubic safeguarded interpolation
        Scalar alpha_tmp = iter == 0
            ? ( gtp / (2.0 * (fx0 + gtp - fxa)) )       // quadratic on first failure
            : cubic( fx0, gtp, fxa, alpha, fxp, alphap );
        fxp = fxa;
        alphap = alpha;
        alpha = range(alpha_tmp, 0.1*alpha, 0.5*alpha); // clamp into [0.1α, 0.5α]
    }
    if (iter >= max_iters) return -1;                   // failure → LBFGS returns FAILURE
    return alpha;
}
```

`cubic(...)` (Backtracking.hpp:129-143): 2x2 linear system on `[fxa, fxp]` vs `[α, αp]`
endpoints, returns the larger root of the cubic-derivative quadratic. Falls back to
`-gtp / (2·r[1])` if the cubic degenerates to a quadratic.

`range(...)` (Backtracking.hpp:116-120): scalar `clamp(alpha, low, high)`. For per-iter behavior,
`alpha` shrinks by no more than `0.5×` per failed Armijo check and by no less than `10×` (i.e.
`alpha` lives in `[0.1·alpha, 0.5·alpha]`).

**Critical Armijo constants:**

| Symbol         | Value                          | RealSim source                |
|----------------|--------------------------------|-------------------------------|
| `c_1` (decrease) | `1e-4`                        | `Minimizer.hpp:67`            |
| `c_2`          | *not used* (Armijo only)       | —                             |
| `α_init`       | `1.0` after each accepted step | `LBFGS.hpp:73, 145`           |
| `α_init` (restart) | `min(1, 1/‖g‖∞)`           | `LBFGS.hpp:107`               |
| backtrack factor | `[0.1, 0.5]` of current `α` via cubic safeguard | `Backtracking.hpp:104` |
| `ls_max_iters` | `100000`                       | `Minimizer.hpp:67`            |

**Warp porting implication:** the cubic interpolation is cheap (one 2x2 solve) and self-contained
per element. A pure-`0.5×` backtrack (no cubic) is simpler and a candidate simplification — surfaced
in §4 as a design question.

### 1.4 Convergence test

`HyperelasticProblemS.h:9-12` (3D) and `:67-70` (2D), both inheriting from
`mcl::optlib::Problem<double, DIM>`:

```cpp
bool converged(const VecX &x0, const VecX &x1, const VecX &grad) const override {
    return ( grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6 );
}
```

- Tolerance: **`1e-6`** (hard-coded, double precision).
- Both gradient-norm OR step-norm trigger exit. Disjunction means "either small enough".
- Convergence checked **after** the new `x` (line 119 in LBFGS.hpp) but **before** the new gradient
  is recomputed. So the `grad` passed into `converged()` is the gradient at the *previous* iterate
  `x_old`, not the new `x`. The step-norm check `(x_last - x).norm()` uses the actually-taken step
  `rate·q`.
- Max outer LBFGS iters: **`50`** (`LBFGS.hpp:47`).

### 1.5 NH energy / gradient (Tri 2D)

`HyperelasticProblemS.h:73-120` (`NHProjectionProblem2D`):

Variable: `x = (σ_0, σ_1) ∈ R²` — the two singular values of `F` (a 3x2 matrix's two non-trivial
singular values). `x0 = σ_init` snapshot at LBFGS start. Constants: `μ` (shear), `λ`, `k = 2μ`
(bulk).

**Energy density** (HyperelasticProblemS.h:81-91):
```cpp
J      = σ_0 · σ_1
I_1    = σ_0² + σ_1²
I_3    = J²
log_I3 = log(I_3) = 2·log(J)
Ψ(σ)   = 0.5·μ·(I_1 − log_I3 − 3)  +  0.125·λ·log_I3²
       = (μ/2)·(σ_0² + σ_1² − 2·log(J) − 3)  +  (λ/2)·log²(J)
```

**Objective** (HyperelasticProblemS.h:93-102):
```cpp
value(x, x0) = Ψ(x) + (k/2)·‖x − x0‖²   if x_0 ≥ 0 and x_1 ≥ 0
             = +∞                       if any x_i < 0
```
The `+∞` barrier on `x_i < 0` is the line-search's only guard against the J ≤ 0 NaN region.
Comment in source: `// No Mr. Linesearch, you have gone too far!` (HyperelasticProblemS.h:97).

**Gradient** (HyperelasticProblemS.h:104-116):
```cpp
∇Ψ(σ)_i  =  μ·(σ_i − 1/σ_i) + λ·log(J)·(1/σ_i)
∇(x,x0)_i = ∇Ψ(σ)_i + k·(σ_i − σ_init_i)
```
Raises `std::runtime_error` if `J ≤ 0`. In the LBFGS loop this would crash; the `value()` barrier
is designed to prevent the line search from ever reaching there. **For Warp this must be a no-throw
path: gracefully clamp σ_i to a small ε and continue, or signal failure.** See §4.

**No Hessian used.** LBFGS does not call `problem.hessian()`. The base class supplies a
finite-difference Hessian (`Problem.hpp:111-130`) but it's only invoked by Newton-method
subclasses. **Confirmed: LBFGS port needs gradient only.**

### 1.6 NH energy / gradient (Tet 3D)

`HyperelasticProblemS.h:15-63` (`NHProjectionProblem3D`):

Same shape as 2D but `x ∈ R³`:
```cpp
J      = σ_0 · σ_1 · σ_2
I_1    = σ_0² + σ_1² + σ_2²
I_3    = J²
log_I3 = log(I_3) = 2·log(J)
Ψ(σ)   = 0.5·μ·(I_1 − log_I3 − 3)  +  0.125·λ·log_I3²
```

**Objective** (HyperelasticProblemS.h:35-44): identical `+∞` barrier on any `x_i < 0`.

**Gradient** (HyperelasticProblemS.h:46-59):
```cpp
∇(x, x0)_i = μ·(σ_i − 1/σ_i) + λ·log(J)·(1/σ_i) + k·(σ_i − σ_init_i)
```

**Pre-LBFGS sigma fix-up** (PDNeohookeanTetrahedronEnergyParallel.h:69-78):
```cpp
const real eps = 1e-6;
if (|σ_0|<eps && |σ_1|<eps && |σ_2|<eps) { σ_0 = σ_1 = σ_2 = eps; }
if (σ_2 < 0.0) { σ_2 = -σ_2; }
```
The third singular value is unflipped to make all σ_i ≥ 0 before entering LBFGS. The 2D
equivalent (PDNeohookeanTriangleEnergyParallel.h:68-72) only does the all-three-tiny check, no
sign flip — because in 2D `signedEigenSVD` is not used; `JacobiSVD<Mat3x2>` returns non-negative
singular values natively.

### 1.7 Scattering after projection

After `solver[i].minimize(problem[i], sigma, sigma0)` returns, RealSim reconstructs P and
scatters into a per-element buffer `_proj[i]`, then later accumulates per-particle.

**Tri** (PDNeohookeanTriangleEnergyParallel.h:76-79, 115-122):
```cpp
// Reconstruct P (3x2) from updated sigma
Mat3x2R P = svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();

real wi = coeff * weight * area[i];
res[i] = wi * (restMatrix[i] * P.transpose());   // 2x3 = D^T · P^T scaled by w_i

// Per-particle accumulation
rhs.row(tri[0]) += -res[i].row(0) - res[i].row(1);
rhs.row(tri[1]) +=  res[i].row(0);
rhs.row(tri[2]) +=  res[i].row(1);
```

**Tet** (PDNeohookeanTetrahedronEnergyParallel.h:82-89, 125-133):
```cpp
S = diag(sigma_0, sigma_1, sigma_2);
Mat3R P = U * S * V.transpose();
res[i] = wi * (DmInv[i] * P.transpose());        // 3x3
rhs.row(tetra[0]) += -res[i].row(0) - res[i].row(1) - res[i].row(2);
rhs.row(tetra[1]) +=  res[i].row(0);
rhs.row(tetra[2]) +=  res[i].row(1);
rhs.row(tetra[3]) +=  res[i].row(2);
```

Both match FBA's existing ARAP/Corot scatter pattern (kernels.py:973-976, kernels.py:812-815)
exactly. **The post-projection scatter requires zero changes — only `sigma_proj` differs.**

`coeff` is a PD outer-iter scaling factor; `weight` is the material weight (RealSim's `_weight =
ke`). The product `coeff · weight · area` corresponds to FBA's `tri_weight_d` (which already
folds `ke · area`).

---

## 2. FBA's current Newton implementation

### 2.1 Tri NH projection in `kernels.py`

`newton/_src/solvers/fba/kernels.py:1303-1355` — `project_neohookean_sigma(sigma_sq, mu, lam) → wp.vec2`.

Input: **squared** singular values `sigma_sq` from `wp.svd2(F^T F)`. Inside:
```python
eps = 1.0e-6
sigma0_0 = sqrt(max(sigma_sq[0], eps²))
sigma0_1 = sqrt(max(sigma_sq[1], eps²))
k = 2.0 * mu
sigma_0 = sigma0_0
sigma_1 = sigma0_1

for _i in range(5):                                  # FIXED 5 ITERS, NO CONVERGENCE CHECK
    sigma_0 = max(sigma_0, eps)
    sigma_1 = max(sigma_1, eps)
    J = sigma_0 * sigma_1
    log_J = log(J)
    inv_0 = 1.0 / sigma_0
    inv_1 = 1.0 / sigma_1

    # Gradient (identical to RealSim's NHProjectionProblem2D::gradient)
    grad_0 = mu*(sigma_0 - inv_0) + lam*log_J*inv_0 + k*(sigma_0 - sigma0_0)
    grad_1 = mu*(sigma_1 - inv_1) + lam*log_J*inv_1 + k*(sigma_1 - sigma0_1)

    # Hessian (NEWTON-SPECIFIC — RealSim's LBFGS does not use this)
    h_diag_factor = mu + lam - lam*log_J
    H_00 = mu + h_diag_factor*inv_0² + k
    H_11 = mu + h_diag_factor*inv_1² + k
    H_01 = lam / J

    det = H_00*H_11 - H_01²
    dx_0 = (H_11*grad_0 - H_01*grad_1) / det
    dx_1 = (H_00*grad_1 - H_01*grad_0) / det

    sigma_0 -= dx_0                                  # FULL NEWTON STEP, NO LINE SEARCH
    sigma_1 -= dx_1

sigma_0 = max(sigma_0, eps)                          # clamp at end
sigma_1 = max(sigma_1, eps)
return wp.vec2(sigma_0, sigma_1)
```

Used by `project_stretching_neohookean_kernel` (line 1359) and
`project_stretching_neohookean_compute_kernel` (line 1467). Both reconstruct P from
`U * diag(sigma_proj) * V^T` using `wp.svd2(F^T F)` outputs.

### 2.2 Tet NH projection in `kernels.py`

`newton/_src/solvers/fba/kernels.py:629-718` — `project_neohookean_sigma3d(sigma, mu, lam) → wp.vec3`.

Input: `sigma` directly from `wp.svd3(F)` (already non-negative). Same algorithmic skeleton:
- Same gradient formula (kernels.py:684-686).
- 3x3 Hessian via closed-form cofactor inverse (kernels.py:697-709).
- **Fixed 5 Newton iters, no line search, no convergence check.**
- Clamp σ_i ≥ ε before each iter and after exit.

Used by `project_stretching_neohookean_tet_kernel` (line 721) and
`project_stretching_neohookean_tet_compute_kernel` (line 818).

### 2.3 "5 fixed Newton iters" — audit claim verified

- **5 fixed iterations:** confirmed (`for _i in range(5):` at kernels.py:671 and 1326).
- **No convergence check:** confirmed.
- **No line search:** confirmed (raw `sigma -= dx`).
- **No Hessian damping (Levenberg-Marquardt, trust region, etc.):** confirmed.
- **Per-iter clamp `σ ≥ ε`:** present, but does not protect against the `J ≤ 0` region that
  RealSim's `value()` guards via `+∞`. In practice the closed-form Hessian inverse can drive
  σ_i below ε mid-iter; the clamp at the *start* of the next iter saves the next log/inv, but
  the gradient used for `dx` came from a clamped state and may not be physically meaningful.
- **No symmetric Hessian projection / PSD safeguard:** the Hessian is symmetric by construction
  (`H_01 = lam/(σ_0·σ_1)`, etc.) but indefinite for `log_J · lam > μ + lam` (i.e. large J). No
  modification when indefinite — Newton can take an ascent step.

This last point matters: under compression, σ_i < 1 → `log_J < 0` → `h_diag_factor = μ + λ - λ·log_J >
μ + λ` (more positive), Hessian stays PD. Under expansion, σ_i > 1 → `log_J > 0` → `h_diag_factor`
decreases, can flip sign of the diagonal entries when `λ·log_J > μ + λ`, at which point the
Newton step diverges. RealSim's LBFGS sidesteps this entirely because it never builds the Hessian.

The SqueezingBall failure mode (ball explodes under compression) is at the opposite end of the
J range, where the analytic Hessian *is* PD — so the failure cannot be naive indefinite-Hessian
divergence. More likely: under high compression, `1/σ_i` blows up, the closed-form `dx` is huge,
and the 5-iter cap leaves σ far from optimum at a non-physical configuration. The LBFGS line
search ensures monotone descent, which is the discriminating property.

---

## 3. Warp-implementation constraints

### 3.1 LBFGS history storage

Per element (one per tri or tet), the persistent inside-kernel state is:
- `s : (DIM, M) = (DIM·8) doubles` — DIM=2 for tri (16 doubles = 128 bytes), DIM=3 for tet (24 doubles = 192 bytes)
- `y : (DIM, M)` — same size
- `α : (M,) = 8 doubles = 64 bytes`
- `ρ : (M,) = 8 doubles = 64 bytes`
- `grad, q, grad_old, x_old, x_last : 5×DIM doubles` — 10–15 doubles
- Scalars: `γ_k, α_init`, plus loop indices

**Key point:** in RealSim this state lives in the call-stack frame of `minimize()`. In a Warp
`@wp.func`, local stack-allocated arrays of fixed size are supported (Warp scalars/vec/mat or
`wp.tile`-style locals are fine for small DIM·M). The 2D tri case needs 16+16+8+8+2·5 = 58
doubles ≈ 464 bytes per thread; the 3D tet case needs 24+24+8+8+15 = 79 doubles ≈ 632 bytes per
thread.

**Storage options:**

(a) **All-stack-local** (preferred if Warp supports it): declare `s`, `y`, etc. as local `mat`
or fixed-size arrays inside the `@wp.func`. Zero allocator overhead, hits L1/registers. RealSim
relies on this — porting faithfully means stack-local. Warp's `@wp.func` permits local
`wp.mat`/`wp.vec` and small fixed-size local arrays via type templates.

(b) **Pre-allocated global scratch `wp.array2d` keyed by element index:** shape `(T, 2·M·DIM)`.
For 10K elements, tri = 10K · 32 = 320 KB; tet = 10K · 48 = 480 KB of GMEM. Cheap. Used if (a)
turns out unsupported in Warp's `@wp.func` for the required size/shape.

(c) **Shared memory:** Warp does not expose CUDA shared memory at the Python API level (tile API
is the closest); skip.

**Recommendation:** Start with (a). If Warp rejects `wp.mat[float, 3, 8]`-style locals (or the
register pressure tanks throughput), fall back to (b). Either way RealSim's algorithm is
identical.

### 3.2 LBFGS persistence across PD outer iters

**RealSim resets LBFGS history at every `localProjection` call.** Verified:

- `_solver` is a `std::vector<LBFGS<...>>` resized once in `init()`
  (PDNeohookeanTetrahedronEnergyParallel.h:111). After resize, no `_solver[i]` member is ever
  written — only `_solver[i].minimize(...)` is invoked.
- `LBFGS::minimize` (LBFGS.hpp:52) declares `s, y, alpha, rho, grad, q, grad_old, x_old, x_last`
  as **function-local variables** (LBFGS.hpp:53-56). They have no class storage; they're stack-allocated
  per call, default-initialized to zero by Eigen.

**Conclusion: history is reset per PD outer iter, per element.** The Warp port must do the same.
Caching across outer iters is **not** faithful (and the user may want it as a perf experiment
later — surfaced in §4).

### 3.3 Convergence early-exit in Warp kernel

RealSim's `for (int k=0; k<max_iters; ++k)` includes an in-loop `break`. Warp's `@wp.func` with
a Python `for` over `range(max_iters)` can be `break`-exited; the GPU thread that breaks early
sits idle waiting for the warp's other threads to finish. Per-element divergence cost is
proportional to the slowest-converging element in the warp.

The FBA precedent for similar "iter-with-early-exit" Warp kernels: `_solve_nsn_*` is CPU-side
(NumPy), not a Warp kernel, so it's not a direct precedent. The relevant precedent inside `@wp.func`
is the SVD/closed-form per-element solves, all of which are fixed-iter. **NH LBFGS would be the
first Warp `@wp.func` in FBA with a variable-iter break.**

**Two patterns:**

(a) **Run all `max_iters` always, ignore early-exit.** Simpler, deterministic, wastes work (up to
50/avg_iters factor — typically 5×–10× given LBFGS converges in 5–10 iters near optimum). May be
acceptable if NH cost is a small slice of frame time.

(b) **`break` on convergence inside the loop.** GPU threads idle on warp-divergence basis.
Realistic given Warp supports it. Implement with `if converged: break`.

**Recommendation:** (b) — break inside the loop. The convergence test is cheap (3 doubles + 1
norm). Threads in a converged-fast warp do waste some cycles waiting for the slow thread, but
this is small compared to running 50 iters for everyone. Default `max_iters = 50` per RealSim.

### 3.4 Line search in Warp

**Line search needs:**

- Compute `f(x + α·p)` per trial step (one energy eval, includes a `log(J)` + 4–6 mul/add).
- Hold scalars `α, α_prev, f_prev, gtp, fx0` per element — trivial.
- Maintain a backtracking loop with cubic safeguard.

Each LBFGS iter triggers ≥1 line-search iter (the first try at `α=1.0` usually accepts because
the LBFGS Hessian approximation is good). When it doesn't, the cubic interpolation chooses a new
α in `[0.1·α_prev, 0.5·α_prev]` and tries again. The `+∞` barrier on σ_i < 0 makes early rejections
fast in the LBFGS code (one comparison, no full eval).

**Critical Warp concern:** RealSim's `Backtracking::search` does `problem.gradient(x, x0, grad)`
at the start (Backtracking.hpp:88) to get `fx0` and `gtp`. But the caller (LBFGS::minimize)
already has `grad` from the previous iter. **The port should pass `fx0` and `grad` into the line
search, not recompute** — this is a perf improvement RealSim leaves on the table because the
finite-diff fallback in `Problem::gradient` is generic. For NH the closed-form gradient is cheap
enough that recomputing is a non-issue but it's still avoidable work.

**Cubic safeguard simplification candidates (design Q in §4):**

- **Full BacktrackingCubic** (faithful port) — cubic + range clamp.
- **Plain Backtracking** (`Backtracking::search` at Backtracking.hpp:34-68, also in RealSim) —
  `α *= 0.7` per failure, no interpolation. Strictly less aggressive; expected to need ~1.4×
  more line-search iters on average.
- **Pure halving** (`α *= 0.5`) — even simpler.

Faithfulness argument: the user (RealSim author) chose `BacktrackingCubic` as the LBFGS default.
Diverging here is the kind of "I thought this was equivalent" pitfall flagged in
`realsim-port-discipline`. Recommend faithful BacktrackingCubic for v1.

### 3.5 Per-element parallelism

FBA's existing NH kernels already have `t = wp.tid()` per tri/tet (kernels.py:752, 1386, 845, 1493).
**LBFGS preserves this exactly — every element runs its own inner LBFGS, no cross-element
synchronization or atomic ops inside the LBFGS loop.** Only the final scatter (which already
exists in FBA's `compute_kernel`s) is unchanged.

The only meaningful difference vs the current Newton kernel is per-thread state size (§3.1) and
warp divergence cost (§3.3).

---

## 4. Design questions for the controller (USER) to resolve

Each requires an explicit Y/N or numeric answer before implementation. Default = "faithful to
RealSim" unless RealSim itself doesn't pin the choice.

### Q1. LBFGS history depth `M`

- **RealSim:** `M = 8` (default template param, no override; see §1.1).
- **Recommendation:** **`M = 8`**, faithful.
- **Cost:** tri = 16 extra doubles per element vs `M=2`; tet = 24. Negligible.

### Q2. Persist LBFGS history across PD outer iters?

- **RealSim:** **No, reset every call** (§3.2 confirmed by reading `minimize` body).
- **Recommendation:** **No, reset per outer iter**, faithful. Add a `persist_lbfgs_history`
  constructor flag (default False) so a future perf experiment can flip it.
- **Risk if persist=True:** LBFGS history built around the previous-iter `σ_0` is mostly
  irrelevant when `σ_0` shifts; can produce non-descent directions more often.

### Q3. Line search method

- **RealSim:** `BacktrackingCubic` (§1.3, Minimizer.hpp default, no override).
- **Options:** (a) BacktrackingCubic faithful, (b) plain Backtracking `τ=0.7`, (c) pure halving.
- **Recommendation:** **(a) BacktrackingCubic, faithful.** Cubic is 5–10 ops per trial; cheap.
  `realsim-port-discipline` rules against (b)/(c).

### Q4. Armijo decrease constant `c_1`

- **RealSim:** `1e-4` (Minimizer.hpp:67).
- **Recommendation:** **`1e-4`**, faithful.

### Q5. Convergence tolerance and disjunction

- **RealSim:** `‖grad‖ < 1e-6  OR  ‖x_prev − x‖ < 1e-6` (HyperelasticProblemS.h:11, 69).
- **Recommendation:** **Faithful both predicates, both `1e-6`.** Use float64 (RealSim uses double).
- **Note:** FBA currently uses float32 in some kernel paths; this is the first NH-internal computation
  that needs `wp.float64` per the tolerance. Verify Warp supports `wp.float64` in `@wp.func`
  scope (it does, but kernel signatures may need explicit double `mu, lam` casts).

### Q6. Max outer LBFGS iters `max_iters`

- **RealSim:** `50` (LBFGS.hpp:47).
- **Recommendation:** **`50`** for tri and tet. Allow tuning via constructor kwarg (default 50)
  in case warp divergence cost becomes a hotspot.

### Q7. Initial Hessian scaling `γ_0`

- **RealSim:** `γ_0 = 1.0` on first iter; `γ_k = ⟨s, y⟩ / ⟨y, y⟩` thereafter (Nocedal-Wright
  "method 1"). See §1.2.
- **Recommendation:** **Faithful.**

### Q8. Non-descent restart behavior

- **RealSim:** if `q·grad ≤ 0`: discard history, fall back to scaled steepest descent with
  `α_init = min(1, 1/‖g‖∞)`, and effectively decrement the LBFGS budget by `k` (LBFGS.hpp:101-108).
- **Recommendation:** **Faithful.** The budget-decrement is mildly tricky to express in a fixed
  `range(max_iters)` Warp loop; suggested implementation is a `restart_count` counter and a `total_iters`
  that decrements on restart.

### Q9. `J ≤ 0` and `σ_i < 0` handling

- **RealSim:** `value()` returns `+∞` for σ_i < 0, blocking the line search; `gradient()` throws
  `std::runtime_error` for J ≤ 0. The `+∞` barrier is supposed to prevent any gradient call ever
  seeing J ≤ 0, but the precondition isn't enforced by the type system.
- **Warp constraint:** Cannot throw exceptions inside a `@wp.func`.
- **Options:** (a) Clamp σ_i to `eps = 1e-6` inside `gradient()` (and document the divergence
  from RealSim), (b) Return a sentinel ("invalid") from `gradient()` and have the LBFGS loop
  detect and reject the line-search trial. (c) Trust the `+∞` barrier in `value()` and let
  `gradient()` produce NaN/Inf in the corner case where the line search never accepts.
- **Recommendation:** **(a) clamp σ_i ≥ 1e-6 inside the gradient evaluator** AND ensure `value()`
  returns a huge finite number (e.g. `1e30`) instead of true `+∞` for σ_i < 0 — `+∞` interacts
  poorly with NaN-propagating GPU ops and `1e30` is "infinity" for any practical Armijo test.
  Document the small divergence from RealSim's `+∞` and `runtime_error`. (Surfaced explicitly per
  port-discipline rule "I chose X instead of RealSim's Y because Z".)

### Q10. Sigma initial fix-up

- **RealSim tri:** all-three-tiny → clamp σ_i to `eps = 1e-6`.
- **RealSim tet:** same all-three-tiny clamp + `σ_2 = |σ_2|` (flip sign if negative).
- **Recommendation:** **Faithful tri and tet pre-LBFGS fix-up.** Note that FBA's tri path uses
  `wp.svd2(F^T F)` which already returns non-negative σ²; the input `sigma_sq` is non-negative
  but tiny values still need the all-three-tiny clamp. FBA's tet path uses `wp.svd3(F)` which
  returns non-negative SVs already (no signed SVD), so the `σ_2 < 0` flip is a **no-op in Warp**.
  Keep the all-three-tiny clamp; document that the σ_2 flip is unnecessary because of `wp.svd3`'s
  unsigned output.

### Q11. Per-element storage layout

- **Options:** (a) stack-local fixed-size arrays inside `@wp.func`, (b) global `wp.array2d`
  scratch of shape `(T, 2·M·DIM)`, (c) hybrid (history on stack, large temporaries on scratch).
- **Recommendation:** **(a) first; benchmark and fall back to (b) if register pressure or
  spilling is the bottleneck.** No `@wp.func` in FBA today exceeds ~30 doubles of stack-local
  state, so 60–80 doubles is uncharted; verify with `wp.config.verbose_warnings`.

### Q12. Share kernel between Tri and Tet?

- **Options:** (a) two separate `@wp.func`s (`project_neohookean_sigma2_lbfgs` and `…_sigma3_lbfgs`),
  parameterized by DIM at compile time. (b) one templated function via `wp.func` overloads on
  `wp.vec2` vs `wp.vec3`.
- **Recommendation:** **(a) two separate functions.** Warp's typing is duck-typed at JIT time;
  one templated function risks accidental wrong-DIM dispatch and obscures debugging. Tri and tet
  diverge only in the loop bodies (`J = σ_0·σ_1` vs `σ_0·σ_1·σ_2`); duplication is small (~100
  lines each) and clarifies the maintenance contract.

### Q13. Float32 vs Float64

- **RealSim:** `double` throughout.
- **Tolerance:** `1e-6` matches both fp32 and fp64; convergence won't differ much.
- **Recommendation:** **Use `wp.float64` inside the LBFGS `@wp.func` (locals + arithmetic),** cast
  back to `wp.float32` only when emitting `sigma_proj` for the scatter. Rationale: NH energy
  `log(J²)·log(J²)·λ/8` is moderately ill-conditioned near J=1 (catastrophic cancellation in
  `I_1 − log_I3 − 3`); fp64 preserves the gradient's small-magnitude regime. FBA already uses
  `wp.float64` in `linear_solver.py`. Cost: ~2× ALU vs fp32 — but NH is not the frame's hotspot.

### Q14. PSD safeguard / damping?

- **RealSim LBFGS doesn't need this** — the Hessian is implicit. Skip.
- **Note:** FBA's *Newton* kernel would benefit from a safeguard, but we're replacing it. N/A.

### Q15. Test mode coverage

- Should the LBFGS path be the only Tri/Tet NH path going forward, or should the 5-iter Newton
  be retained as a `nh_solver: Literal["newton5", "lbfgs"]` constructor flag for regression?
- **Recommendation:** **Add `nh_solver` flag, default `"lbfgs"`,** keep `"newton5"` available for
  one Newton release. After SqueezingBall is verified binary-passing, deprecate `"newton5"`.

---

## 5. Test plan (do NOT implement; describe)

### 5.1 Unit test — single-element NH minimization

For a single tri (DIM=2) and a single tet (DIM=3):
- Pick `μ = 1e4, λ = 1e4` (cloth-like); pick `σ_0 = (1.5, 0.7)` (mild stretch + compression);
- Run the Warp LBFGS `@wp.func` from `sigma = sigma0` (anchor).
- Reference solution: `scipy.optimize.minimize(method='L-BFGS-B', maxiter=50, tol=1e-6,
  options={'maxcor': 8})` on the same `Ψ + (k/2)‖σ−σ_0‖²` objective.
- Assert `‖σ_warp − σ_scipy‖ < 1e-5`. Tighter than the 1e-6 grad tolerance because both stop at
  near-stationary points.

Repeat for 3–5 anchors covering the parameter space: extreme compression (σ_0 = 0.1, 0.1),
near-rest (σ_0 = (1.0, 1.0)), extreme stretch (σ_0 = (3.0, 1.0)). For tet: add (1.0, 1.0, 0.05).

### 5.2 Cross-validate gradient symbolically

For one anchor, compute the gradient three ways:
- The hand-derived analytic gradient in the kernel.
- `scipy.optimize.approx_fprime` finite difference (`eps = 1e-7`).
- Symbolic differentiation via `sympy` of `Ψ + (k/2)‖σ−σ_0‖²`.
All three within `1e-5`. Catches transcription errors.

### 5.3 Integration test — SqueezingBall

- Run `fba_squeezing_ball_cudatests.py` in offline mode (per `realsim-test-mode` memory) with new LBFGS NH.
- **Binary success: ball passes through the gap intact** (per `realsim-port-success-binary`).
  Concretely, no particle has `|particle_q| > 50` (explosion sentinel) at any frame, and
  `ball_center.y` reaches RealSim's `min_y ≈ -10.02` by frame N (N ≈ 200; verify against
  RealSim's `output_obj_0.abc` reference trajectory).
- Compare frame-by-frame trajectory drift against RealSim reference; target < 1e-3 m at each
  recorded frame. Drift > 1e-3 m + binary success = parity issue, queue for follow-up; drift
  < 1e-3 m = full parity.

### 5.4 Regression test — PullingWooper

- Run `fba_pulling_wooper_cudatests.py` with LBFGS NH.
- **Acceptance:** `min_y ≈ -8.41`, `pulled = 10` (matches Phase 2 baseline). Equivalent stability
  to current 5-iter Newton.

### 5.5 Regression test — all existing NH demos pass without change

- TwistingBar, BarSlam, any other NH softbody in CudaTests reproduction list.
- **Acceptance:** binary success on each (per `realsim-port-success-binary`).

### 5.6 Convergence-iteration histogram

- Instrument the LBFGS Warp kernel to write `total_iters` per element into a debug array.
- Run one frame of SqueezingBall; report median + max iter count across elements.
- Sanity check: median ≤ ~10 (LBFGS converges near optimum from a good anchor), max ≤ 50 (cap).
- If max often hits 50, increase the cap or investigate non-descent behavior.

---

## 6. Estimated implementation effort

- **Read time:** done.
- **LBFGS `@wp.func` skeleton (one for tri, one for tet):** 1 day. Direct port; ~250 lines each.
- **BacktrackingCubic line search `@wp.func`:** 0.5 day. ~50 lines; needs careful cubic-coeff
  transcription against `Backtracking.hpp:129-143`.
- **Energy/gradient `@wp.func` (`value_nh`, `gradient_nh`):** 0.25 day each (tri/tet). ~30 lines.
- **Integration into `project_stretching_neohookean_*_compute_kernel`:** 0.5 day (replace one
  line, plumb extra kernel args for `max_iters`, `tol`, `c_1`, `M`).
- **Unit tests (§5.1, §5.2):** 1 day. SciPy comparison + symbolic gradient validation.
- **SqueezingBall integration verification:** 0.5 day (run, render frames, compare to RealSim
  `.abc`).
- **Bug-fix budget (port-discipline insurance):** 1 day. Past port pitfalls (NSN simplification,
  bending H' rederivation, sign flips) suggest this is non-trivial.

**Total: ~5 working days. Bucket: Medium (M).**

---

## 7. Open issues that may block implementation

### O1. Warp `@wp.func` local fixed-size arrays of size `2·DIM·M`

Need to verify Warp supports declaring `s : wp.array_local[wp.float64, 2*8]` or equivalent. If
not, fall back to global scratch (§3.1.b). Trivial day-1 spike.

### O2. `wp.float64` inside `@wp.func` with `wp.svd2/svd3` outputs

`wp.svd2` and `wp.svd3` likely return `float` (32-bit). Need to cast inside the func.
Verify no precision loss between `wp.svd3` (fp32) → fp64 LBFGS → fp32 scatter.

### O3. `J ≤ 0` corner case

If line search occasionally misses the `value() = +∞` barrier (e.g. due to fp32 SVD rounding of
already-near-zero σ_i), `gradient()` will see `log(J) = log(small)` → large negative number →
unbounded step direction. Mitigation: clamp σ_i ≥ 1e-6 inside `gradient()` and emit a sentinel
that the LBFGS loop treats as line-search failure. This is the recommendation in Q9.

### O4. Determinism across runs

Warp `@wp.func` arithmetic is deterministic within a thread; LBFGS history accumulates floating
errors deterministically per thread. **No atomic ops are added by this change** — the existing
deterministic `compute_kernel + gather_per_particle_kernel` pattern is preserved (FBA's `solver_fba.py:835`
already routes NH through this). No new nondeterminism.

### O5. Energy bowl shape at extreme compression

The NH energy `Ψ → +∞` as `σ_i → 0⁺` (because of `log(J²)`). The local minimum the LBFGS targets
is finite and well-defined for σ_0 > 0 anchor, but the basin can be narrow. The `+∞` barrier
keeps LBFGS in the feasible region; the line-search backtracking handles the narrow basin. The
5-iter Newton solver doesn't enforce monotone descent, so it can leave the basin → SqueezingBall
explosion. **This is the leading hypothesis for why the Newton substitution fails and LBFGS
will fix it.** Implementation must verify by running SqueezingBall after the port.

### O6. Float32 weight × float64 sigma_proj round-trip

When `Ψ(F) + (k/2)·‖σ−σ_0‖²` reaches a near-zero minimum, fp32 cannot resolve the small
gradient magnitude. The final `sigma_proj` cast to fp32 may differ from RealSim's double-precision
output at the 6th decimal place. Drift at the 1e-7 m level is expected and acceptable per
existing FBA parity targets (1e-3 m).

### O7. PD outer-iter coefficient `coeff`

RealSim passes `coeff` into `NeohookeanTriangleProjectionInRange` (PDNeohookeanTriangleEnergyParallel.h:24-25,
113) and multiplies the per-element weight `wi = coeff * weight * area[i]`. FBA's current Tri NH
kernel uses `tri_weight_d` (precomputed `ke·area`) and **does not** apply a runtime `coeff` —
this is hard-coded `coeff = 1` semantics. Trace back into FBA's `localProjection` call: in
RealSim, `coeff` is `dt² · ω_pd` or similar; need to verify FBA's `tri_weight_d` already absorbs
this. If yes (likely — Phase 0 audit didn't flag this), no change. **If `coeff` is missing
in FBA, this is an unrelated bug separate from LBFGS** and is out-of-scope for 1.3.a. Surface to
controller; do not fix in this task.

### O8. Restart behavior in a fixed `range(max_iters)` Warp loop

RealSim's LBFGS restart (`k = 0; max_iters -= k`) modifies the loop counter mid-iter. Warp's
Python-compatible `range` doesn't permit this. Equivalent implementation: separate counter
`iters_done` that increments each pass, `iters_remaining = max_iters - iters_done`, plus a
`history_count` that resets to 0 on restart. The inner `iter = min(M, history_count)` and the
shift logic depend on `history_count`, not `k`. **This is a faithful re-implementation, not a
semantic change** — surface it for review.

---

## 8. Summary of cited file:line pairs

**RealSim:**
- `deps/mcloptlib/include/MCL/LBFGS.hpp:36-152` — LBFGS template + minimize body.
- `deps/mcloptlib/include/MCL/Minimizer.hpp:52-108` — base class, settings defaults, linesearch dispatcher.
- `deps/mcloptlib/include/MCL/Backtracking.hpp:34-68, 74-145` — Backtracking and BacktrackingCubic.
- `deps/mcloptlib/include/MCL/Problem.hpp:34-131` — Problem base + finite gradient/Hessian.
- `include/Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h:5-120` — NH 2D/3D energy/value/gradient.
- `include/Scomponent/integrator/localglobal/energy/elastic/PDNeohookeanTriangleEnergyParallel.h:1-154` — Tri NH wrapper, SVD, LBFGS call, scatter.
- `include/Scomponent/integrator/localglobal/energy/elastic/PDNeohookeanTetrahedronEnergyParallel.h:1-167` — Tet NH wrapper, SVD, LBFGS call, scatter.

**Newton FBA:**
- `newton/_src/solvers/fba/kernels.py:629-718` — `project_neohookean_sigma3d` (3D Newton).
- `newton/_src/solvers/fba/kernels.py:1303-1355` — `project_neohookean_sigma` (2D Newton).
- `newton/_src/solvers/fba/kernels.py:721-815, 818-903` — Tet NH scatter kernels.
- `newton/_src/solvers/fba/kernels.py:1359-1463, 1467-1561` — Tri NH scatter kernels.
- `newton/_src/solvers/fba/solver_fba.py:181-184, 725-751, 835-903` — NH dispatch in PD outer loop.
- `docs/superpowers/specs/2026-05-17-fba-realsim-audit-v2.md:95-119` — audit components E and H.
