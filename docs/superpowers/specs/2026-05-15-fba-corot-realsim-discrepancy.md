# SolverFBA Corotational vs RealSim TRI_COROTATION: Known Trajectory Discrepancy

**Date:** 2026-05-15  
**Status:** Documented; no Newton code change required.

## TL;DR

Newton's `SolverFBA` Corotational stretching uses the standard symmetric corotational energy from the elasticity literature (Sifakis 2012, Bouaziz et al. 2014). RealSim's `CorotProjectionProblem2D` implements an **anisotropic variant** due to an inadvertent use of `Eigen::Vector2d::trace()`, which returns only the first component for a column vector. Trajectory-level cross-check between the two Corotational implementations therefore disagrees at the mm level on stretched cloth, while ARAP and Neo-Hookean cross-checks agree to micron level.

## Symptom

Cross-check setup (matches `TestFBARealSimAgreement`):
- 113-vertex cloth, 50 frames, dt=0.01, gravity=(0,0,-10), 5 PD iterations
- Pin top two corners; soft pin stiffness 1e10
- Material: E=1e4 Pa, ν=0.4 → μ = 3571.43, λ = 14285.71

| Stretching model | Max L2 delta (newton[k] vs realsim[k+1]) | Status |
|------------------|------------------------------------------|--------|
| ARAP             | 5.99 µm at frame 48                      | ✅ pass |
| Neo-Hookean      | 9.46 µm at frame 48                      | ✅ pass |
| **Corotational** | **14.1 mm at frame 33**                  | ⚠️  expected discrepancy |

The extension demo (10K vertices, ν=0.45) showed the same pattern: Newton Corot has necking ratio 0.804; RealSim Corot has 0.693. Both clearly show necking, but the magnitudes differ.

## Root Cause

RealSim's `include/Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h:CorotProjectionProblem2D::energy_density`:

```cpp
Eigen::Vector2d I(2);
I[0] = 1.0;
I[1] = 1.0;

double t1 = _mu * (x - I).squaredNorm();
double trace = (x - I).trace();           // ← returns x[0] only, not x[0] + x[1]
double t2 = 0.5 * _lambda * trace * trace;
```

`Eigen::Vector2d` is `Eigen::Matrix<double, 2, 1>` (2x1 column vector). Eigen's `.trace()` is defined as `sum of A(i,i) for i in [0, min(rows, cols))`. For a 2x1 matrix `min(2, 1) = 1`, so `.trace()` returns `A(0, 0)` — the first component only.

Verified empirically with `/tmp/test_eigen_trace.cpp` using RealSim's own Eigen distribution:

```
Vector2d(3, 5).trace() = 3        ← only x[0]
Vector2d(3, 5).sum()   = 8        ← what the comment intended
Vector3d(2, 4, 7).trace() = 2     ← same in 3D
Vector3d(2, 4, 7).sum()   = 13
```

RealSim's intended trace operation (per the source comment `tr(Sig - I)`) was almost certainly `(σ_0 - 1) + (σ_1 - 1)` in 2D, i.e. `.sum()`. The `.trace()` call silently returns `σ_0 - 1`, making the energy:

$$E_{\text{RealSim Corot 2D}} = \mu\,((\sigma_0 - 1)^2 + (\sigma_1 - 1)^2) + \tfrac{\lambda}{2}(\sigma_0 - 1)^2$$

$$= (\mu + \tfrac{\lambda}{2})(\sigma_0 - 1)^2 + \mu(\sigma_1 - 1)^2$$

— an **anisotropic** energy that only constrains the first singular value via λ.

Newton's `project_corotational_sigma` (in `newton/_src/solvers/fba/kernels.py`) uses the standard symmetric formulation:

$$E_{\text{Newton Corot}} = \mu\,||\sigma - I||^2 + \tfrac{\lambda}{2}(\operatorname{tr}(\sigma - I))^2$$

with `tr(σ - I) = (σ_0 - 1) + (σ_1 - 1)`. This matches Sifakis (2012), Bouaziz et al. (2014), and RealSim's own ARAP and NH formulations.

## Why ARAP and NH Don't Show This

- **ARAP**: projects to nearest rotation; no trace term in the energy.
- **NH**: energy uses `J = σ_0 · σ_1` (always a scalar product, no Eigen `.trace()` call) and `log(J)`. RealSim's NH is therefore symmetric in σ.
- **Corot only**: uses `(x - I).trace()` on a Vector2d explicitly.

## Quantitative Comparison at σ₀ = (1.5, 1.5), μ=3571, λ=14286

| Model | σ_proj | Notes |
|-------|--------|-------|
| Newton Corot (symmetric) | (1.083, 1.083) | both equal by symmetry |
| RealSim Corot (anisotropic via `.trace()`) | (1.125, 1.250) | σ_1 less constrained → larger drift |
| Newton NH (Newton iteration) | (~1.09, ~1.09) | symmetric, similar to Newton Corot |

## Decision

**Keep Newton's symmetric implementation.** Rationale:

1. It matches the standard corotational elasticity literature.
2. It is consistent with RealSim's own ARAP and NH (both symmetric in σ).
3. Visually Newton Corot still produces clear necking (ratio 0.804) — qualitatively correct.
4. ARAP and NH cross-checks pass at micron level, validating the surrounding kernel + linear-solver infrastructure.

Documentation: `project_corotational_sigma` docstring notes the discrepancy and points to this spec. Newton remains the canonical-physics implementation; RealSim's Corot will diverge at trajectory level until upstream RealSim is patched (`.trace()` → `.sum()`).

## Future Action (out of scope here)

Consider opening an issue/PR against `RealSim_py` to replace `.trace()` with `.sum()` in:
- `include/Scomponent/integrator/localglobal/energy/elastic/HyperelasticProblemS.h`
  - `CorotProjectionProblem2D::energy_density`
  - `CorotProjectionProblem2D::gradient` (same bug pattern)
  - `CorotProjectionProblem3D::energy_density`
  - `CorotProjectionProblem3D::gradient`

After such a fix, re-running our cross-check should bring Newton Corot vs RealSim Corot residual down to <10 µm, matching ARAP and NH.
