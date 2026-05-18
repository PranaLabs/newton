# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Literal

import numpy as np
import warp as wp

from ...geometry.types import GeoType
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


def fb_unilateral_row(
    penetration: float,
    lam: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Fischer-Burmeister evaluation for a unilateral (normal) contact row.

    Port of RealSim ``NonSmoothNewton.cpp:332-341``. Returns
    ``(omega, compliance, h)``.

    Args:
        penetration: ``J·q − pene0`` at current iterate (m).
        lam: Current normal lambda (N).
        precond: ``1 / W_ii`` (1/N).
        dt: Timestep (s).
        pene0: ``n·anchor`` (m).

    Returns:
        ``(omega, compliance, h)``:
        - ``omega``: per-row weighting ``1 − pene/√(pene² + (precond·lam)²)``.
          May exceed 1 in deep penetration (penetration < 0); intentional.
        - ``compliance``: diagonal regularization for the Schur LHS.
        - ``h``: row contribution to the Schur RHS,
          ``−(pene + precond·lam − root) + omega · (penetration + pene0)``.
    """
    pene = penetration
    plam = precond * lam
    root = math.sqrt(pene * pene + plam * plam)
    if root < 1e-30:
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

    Port of RealSim ``NonSmoothNewton.cpp:343-378``. Returns
    ``(omega, compliance, h)``.

    Inactive contact (``lam_n ≤ 0``): ``omega = 0``, ``compliance = 1/dt``,
    ``h = -dt · lam_t``.

    Active contact (``lam_n > 0``): smooth complementarity between slip speed
    ``|penetration|/dt`` and cone slack ``μ·lam_n − |lam_t|``. ``omega = 1``,
    compliance varies between near-zero (stick) and ``~1/dt`` (slip).
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


def compute_tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute two orthonormal tangent vectors ``(t1, t2)`` from a unit normal.

    Convention mirrors RealSim's ``BaseCollision::generateTangentDirections``:
    pick a reference direction (world X, or world Y if normal is nearly X),
    cross with normal to get t1, then cross normal with t1 to get t2.

    Args:
        n: Unit normal vector, shape ``(3,)``.

    Returns:
        Tuple ``(t1, t2)`` each of shape ``(3,)``, float64 unit vectors
        orthogonal to ``n`` and to each other.
    """
    n = np.asarray(n, dtype=np.float64)
    if abs(n[0]) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1) + 1e-30
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-30
    return t1, t2


def _quat_rotate_z_axis(q: np.ndarray) -> np.ndarray:
    """Apply quaternion ``q = (qx, qy, qz, qw)`` to the local +Z unit vector.

    Returns the world-frame direction of a shape's local +Z, used as the spin
    axis for Newton's cylinder primitive (which extends along local +Z).
    """
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dtype=np.float64,
    )


def _transform_point(pos: np.ndarray, quat: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a rigid transform ``T = (pos, quat)`` to point ``p``.

    Warp stores transforms as ``[px, py, pz, qx, qy, qz, qw]``.

    Args:
        pos: Translation [m], shape ``(3,)``.
        quat: Quaternion ``[qx, qy, qz, qw]``.
        p: Point to transform, shape ``(3,)``.

    Returns:
        Transformed point, shape ``(3,)``.
    """
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    # Rotate p by quaternion then add translation.
    # Formula: q * p * q_conj where q = (qx, qy, qz, qw).
    # Using Rodrigues formula: v' = v + 2*qw*(q_vec x v) + 2*(q_vec x (q_vec x v))
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    rx = px + qw * tx + (qy * tz - qz * ty)
    ry = py + qw * ty + (qz * tx - qx * tz)
    rz = pz + qw * tz + (qx * ty - qy * tx)
    return np.array([rx + float(pos[0]), ry + float(pos[1]), rz + float(pos[2])], dtype=np.float64)


class SolverFBA(SolverBase):
    """Fast But Accurate projective-dynamics cloth solver.

    Implements a Projective Dynamics (PD) local-global iteration for cloth
    with isotropic stretching (ARAP or corotational), isometric bending,
    and soft Pin constraints. The PD Hessian is prefactored once via scipy
    SuperLU (COLAMD ordering); the sparse inverse `S = L⁻¹` is computed
    exactly using the elimination tree (ported from RealSim
    `LDLT_computeLowerInverse`) and uploaded to Warp BSR matrices. The
    runtime linear solve evaluates ``A⁻¹·b = Sᵀ · D⁻¹ · S · b`` via two
    GPU SpMVs per coordinate component, avoiding the inherently
    sequential GPU triangular solve.

    Stretching models:

    - ``"arap"`` — As-Rigid-As-Possible: projects deformation gradient to
      the nearest rotation (singular values clamped to 1).  No Lamé
      parameters needed; stiffness is fully encoded in ``tri_ke``.
    - ``"corotational"`` — Corotational linear elasticity: closed-form solve
      on singular values of F (2x2 for cloth, 3x3 Sherman-Morrison for tets).
      Requires ``mu`` and ``lam`` (first and second Lame parameters [Pa]).
    - ``"neohookean"`` — Neo-Hookean hyperelasticity: 5-iteration Newton solve
      on singular values of F (2x2 for cloth, 3x3 cofactor inverse for tets).
      Requires ``mu`` and ``lam`` (first and second Lame parameters [Pa]).

    Notes:
        - Float32 ``particle_q`` output: the interior linear solver
          operates in float64, but ``state_out.particle_q`` (vec3) is
          float32. This caps achievable position precision at ~1e-7 m.

        - Stability tracks `dt^2 * tri_ke / mass` — the standard CFL-like
          factor for implicit elastic dynamics. Realistic cloth parameters
          (tri_ke ~ 1e4 for ~10 kPa Young's modulus) keep arbitrarily
          sized grids stable with 5-10 PD iterations. Pathologically
          soft inputs (tri_ke < 200 for the default test setup) drive
          unphysical large oscillations that eventually overflow. The
          implementation faithfully reproduces RealSim's PD formulas
          (verified by hand against PDTriangleStretchingEnergy.cpp and
          PDIsometricBendingEnergy.cpp); the previous "dim>=24 unstable"
          claim was an artifact of using tri_ke=100 (~1000x softer than
          real cloth) in the test scenarios.

    See also:
        :class:`~newton.solvers.SolverStyle3D` — Newton's other PD cloth
        solver with different sparse-matrix layout (ELL + PCG).
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap", "corotational", "neohookean"] = "arap",
        mu: float | None = None,
        lam: float | None = None,
        friction: bool = True,
        mu_per_pair_override: np.ndarray | None = None,
        nsn_iterations: int = 1,
        lambda_cap: float | None = None,
        use_isodof: bool = True,
        shape_angular_velocity: dict[int, float] | None = None,
        nh_solver: Literal["newton5", "lbfgs"] = "lbfgs",
        enable_perf_timing: bool = False,
    ) -> None:
        """
        Args:
            model: The :class:`~newton.Model` containing cloth/tet attributes to integrate.
            iterations: Number of PD local-global iterations per step.
            pin_stiffness: Soft-pin stiffness [N/m].
            stretching_model: Stretching model — ``"arap"``, ``"corotational"``, or ``"neohookean"``.
            mu: First Lamé parameter [Pa]; required for ``"corotational"``/``"neohookean"``.
            lam: Second Lamé parameter [Pa]; required for ``"corotational"``/``"neohookean"``.
            friction: If ``True`` (default) use Stage B Coulomb friction NSN solver;
                otherwise use Stage A unilateral NSN.
            mu_per_pair_override: Optional ``(num_pairs,)`` array of friction
                coefficients overriding ``model.shape_material_mu`` lookups.
            nsn_iterations: Number of FB-Newton linearization steps per
                NSN call. Defaults to ``1``, matching RealSim's single
                FB-Newton step per ``build`` -> ``solve`` ->
                ``applyConstraintCorrection`` sequence in
                ``NonSmoothNewton.cpp:102-171``. RealSim's scene.json
                ``constraintsolver.iterations: 10`` is the **PCR Schur
                LCP solver's** iterative-convergence cap, not a FB-Newton
                outer loop count -- FBA uses a direct ``np.linalg.solve``
                on the Schur block which is equivalent to PCR-to-
                convergence on dense SPD systems. Values >1 add extra
                FB-Newton linearizations beyond what RealSim does; useful
                for stress-testing the FB residual but breaks 1:1 parity.
            lambda_cap: Optional per-step clamp on contact impulse magnitude in
                **physical force units [N]**, matching RealSim's
                ``constraintsolver.maxforce`` scene.json field. The internal
                Lagrange multiplier ``λ_FBA = λ_R/dt²`` is scaled, so the clamp
                is internally divided by ``dt²`` at each NSN solve. ``None``
                disables clamping.
            use_isodof: If ``True`` (default, Task P), build the Schur
                complement via the isodof-restricted path that computes only
                the ``A^{-1}[i, j]`` entries for the unique contacted
                particles, skipping the multi-RHS Cholesky solve over ``M``
                columns of ``J^T``. Mirrors RealSim's
                ``CUDASparseInverseSolver::addHAinvHT_gpu``. Set to ``False``
                only for regression testing against the legacy path.
            shape_angular_velocity: Optional ``{shape_index: ω_rad_s}`` mapping.
                Each listed shape rotates about its local +Z axis (Newton's
                cylinder primitive axis) at the given angular speed.  Surfaces
                with nonzero ω contribute a tangential anchor velocity
                ``v_anchor = −ω · axis_world × (world_anchor − shape_center)``
                into the Stage B friction residual; the shape geometry itself
                stays static.  Sign matches RealSim CudaTests'
                ``cylindercollisions[i].rollingvel`` (tangent direction
                ``normal × axis``).  Defaults to no kinematic motion.
            nh_solver: Local-projection inner solver for the Neo-Hookean
                stretching model.  ``"lbfgs"`` (default) uses an LBFGS+Armijo
                cubic-backtracking line search ported from RealSim
                ``mcl::optlib::LBFGS<double, 2/3>`` (M=8, c_1=1e-4, tol=1e-6,
                max_iters=50; LBFGS.hpp:36-152, Backtracking.hpp:74-145,
                HyperelasticProblemS.h:5-120). ``"newton5"`` keeps FBA's
                legacy 5-iter fixed Newton (no line search, no convergence
                check) for regression. Ignored unless
                ``stretching_model="neohookean"``.
            enable_perf_timing: If ``True``, instrument :meth:`step` with
                per-PD-iter ``wp.synchronize_device()``-bracketed timers for
                the local energy projection, Schur ``W`` build, linear solve,
                and NSN inner phase. Each PD outer iter appends one entry to
                the corresponding ``_timing_*_ms_per_iter`` list. Off by
                default — synchronization adds ~50us overhead per call, so
                this is opt-in for matched-granularity profiling against
                RealSim's ``LocalGlobalSolver::printTimer`` output.
        """
        super().__init__(model)

        # ---- Validate ----
        if model.particle_count == 0:
            raise ValueError("SolverFBA requires at least one particle")
        if model.tri_count == 0 and (not hasattr(model, "tet_count") or model.tet_count == 0):
            raise ValueError("SolverFBA requires at least one cloth triangle or tetrahedral element")
        if stretching_model not in ("arap", "corotational", "neohookean"):
            raise NotImplementedError(
                f"stretching_model={stretching_model!r} not yet implemented; "
                "supported: 'arap', 'corotational', 'neohookean'"
            )
        if nh_solver not in ("newton5", "lbfgs"):
            raise ValueError(f"nh_solver={nh_solver!r} not supported; choose 'newton5' or 'lbfgs'")
        if stretching_model in ("corotational", "neohookean"):
            if mu is None or lam is None:
                raise ValueError(
                    f"stretching_model={stretching_model!r} requires both mu and lam (first and second Lamé parameters)"
                )
            if mu <= 0:
                raise ValueError("mu must be positive")
            if lam <= 0:
                raise ValueError("lam must be positive")
        if pin_stiffness <= 0:
            raise ValueError("pin_stiffness must be positive")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.iterations = int(iterations)
        self.pin_stiffness = float(pin_stiffness)
        self.stretching_model = stretching_model
        self.nh_solver = nh_solver
        self._mu = float(mu) if mu is not None else None
        self._lam = float(lam) if lam is not None else None
        self.friction = bool(friction)
        self._mu_per_pair_override = (
            np.asarray(mu_per_pair_override, dtype=np.float64) if mu_per_pair_override is not None else None
        )
        self.nsn_iterations = int(nsn_iterations)
        self.lambda_cap = lambda_cap
        self.use_isodof = bool(use_isodof)

        # Per-shape kinematic angular velocity (rad/s about local +Z).  RealSim
        # parity for ``cylindercollisions.rollingvel``.
        n_shapes = int(model.shape_count) if hasattr(model, "shape_count") and model.shape_count else 0
        self._shape_omega_h = np.zeros(max(n_shapes, 1), dtype=np.float64)
        if shape_angular_velocity is not None:
            for s_idx, omega in shape_angular_velocity.items():
                if 0 <= int(s_idx) < n_shapes:
                    self._shape_omega_h[int(s_idx)] = float(omega)
        # Device mirror consumed by the Step 5.5 GPU contact kernels; the
        # host array stays the source of truth (small, rarely modified).
        self._shape_omega_d = wp.array(self._shape_omega_h, dtype=wp.float64, device=model.device)

        # PD setup is dt-dependent; we cache the assembly at a reference dt and
        # rebuild lazily inside `step` if the dt changes.
        self._dt_setup: float | None = None
        self._linear_solver = None
        self._meta: dict | None = None

        # Device buffers (allocated lazily once we know dt - see _setup_pd_system).
        N = model.particle_count
        device = model.device
        self._device = device
        self._x_ref = wp.clone(model.particle_q)  # initial positions = pin/bending refs
        self._x_inertia = wp.empty(N, dtype=wp.vec3, device=device)
        self._x_cur = wp.empty(N, dtype=wp.vec3, device=device)
        self._rhs = wp.empty(N, dtype=wp.vec3, device=device)

        # Phase 4 Stage A — contact state (0 = no contacts registered yet).
        self._contact_count: int = 0

        # Task L: cache ``W`` and the device buffer of ``A⁻¹ · Jᵀ`` across
        # PD outer iterations within a single ``step()`` call.  Both depend
        # only on (J, A) which are constant within a step (J is set once
        # by :meth:`update_contacts`, A is fixed unless dt or the model
        # changes).  The cache is invalidated whenever either input could
        # change: by :meth:`update_contacts` (J change) and by
        # :meth:`notify_model_changed` / :meth:`_setup_pd_system` (A change).
        self._cached_W: np.ndarray | None = None
        # ``_A_inv_Jt_d`` itself lives on the :class:`FBALinearSolver` and
        # is reused by :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.apply_lambda_correction_combined`;
        # we only need to track whether that buffer is fresh for the current step.
        self._cached_A_inv_Jt_valid: bool = False

        # λ persistent across PD outer iters within a step. RealSim parity:
        # cuda_lambda.setZero once per frame in prepare_gpu, then λ += dλ
        # accumulates over PD outer iters.
        self._lam_unilateral_persistent: np.ndarray | None = None
        self._lam_coulomb_persistent: np.ndarray | None = None
        # ω persistent across PD outer iters: needed by NSN iter-0 to compute
        # penetration at the previously-corrected position.
        self._omega_unilateral_persistent: np.ndarray | None = None
        self._omega_coulomb_persistent: np.ndarray | None = None

        # Optional single-frame diagnostic dump of per-PD-outer-iter intermediate
        # variables, used for line-by-line parity with RealSim. Enabled via
        # :meth:`configure_diagnostic_dump`. ``_step_count`` is incremented at the
        # start of each :meth:`step` call; when it equals ``_diag_frame``, the
        # solver records intermediate state and writes the buffers as ``.npz``
        # to ``_diag_out_path`` at end of step.
        self._diag_frame: int | None = None
        self._diag_out_path: Path | None = None
        self._diag_buffers: dict[str, np.ndarray] = {}
        self._step_count: int = 0

        # Per-element device data (filled by _setup_pd_system).
        self._tri_indices_d = None
        self._tri_rest_inv_d = None
        self._tri_weight_d = None
        self._edge_indices_d = None
        self._edge_quad_q_d = None
        self._edge_weight_d = None
        self._edge_norm_d = None
        self._pin_indices_d = None

        # Tet device data (filled by _setup_pd_system).
        self._tet_indices_d = None
        self._tet_rest_inv_d = None
        self._tet_weight_d = None
        # Per-(tet, local vertex) scratch for deterministic (compute → gather)
        # PD scatter; allocated in _setup_pd_system, reused per step.
        self._tet_contrib_d: wp.array | None = None
        # Per-(tri, local vertex) scratch for deterministic (compute → gather)
        # PD cloth-stretching scatter; allocated in _setup_pd_system, reused per step.
        self._tri_contrib_d: wp.array | None = None
        # Per-(edge, local vertex) scratch for deterministic (compute → gather)
        # PD isometric bending scatter; allocated in _setup_pd_system, reused per step.
        self._edge_contrib_d: wp.array | None = None

        # Particle-centered CSR adjacency for deterministic PD reductions (Task P2-D-A).
        # Lets the PD scatter use (compute → gather) instead of atomic_add.
        self._particle_tet_offsets_d = None
        self._particle_tet_element_d = None
        self._particle_tet_local_d = None
        self._particle_tri_offsets_d = None
        self._particle_tri_element_d = None
        self._particle_tri_local_d = None
        self._particle_edge_offsets_d = None
        self._particle_edge_element_d = None
        self._particle_edge_local_d = None

        # Opt-in per-PD-iter perf instrumentation. Matches the granularity of
        # RealSim's ``LocalGlobalSolver::printTimer`` (Schur, Local, Linear
        # solve, NSN inner).  Each list grows by one entry per PD outer iter
        # when :attr:`enable_perf_timing` is ``True``. Synchronization adds
        # measurable overhead, so this is opt-in.
        self.enable_perf_timing = bool(enable_perf_timing)
        self._timing_local_ms_per_iter: list[float] = []
        self._timing_schur_ms_per_iter: list[float] = []
        self._timing_linear_solve_ms_per_iter: list[float] = []
        self._timing_nsn_inner_ms_per_iter: list[float] = []

    def reset_timing(self) -> None:
        """Clear all per-PD-iter timing buffers.

        Useful between warm-up steps and the measured window.
        """
        self._timing_local_ms_per_iter.clear()
        self._timing_schur_ms_per_iter.clear()
        self._timing_linear_solve_ms_per_iter.clear()
        self._timing_nsn_inner_ms_per_iter.clear()

    def get_timing_summary(self, last_n_steps: int | None = 50) -> dict:
        """Return mean per-PD-iter timings over the last ``last_n_steps`` steps.

        Args:
            last_n_steps: Number of trailing PD outer iterations to include.
                The buffer grows by :attr:`iterations` entries per ``step()``
                call, so passing ``50`` means "last 50 steps" only if all
                steps ran the same number of PD iters (default behaviour).
                Pass ``None`` to average over the full history.

        Returns:
            Dict with keys ``local_ms``, ``linear_solve_ms``, ``schur_ms``,
            ``nsn_inner_ms``, each a mean (``None`` if no samples were
            recorded for that phase — e.g. ``schur_ms`` for contact-free
            demos). Also includes ``n_samples`` (count of PD iters used).
        """
        if last_n_steps is None:
            window = None
        else:
            window = int(last_n_steps) * int(self.iterations)

        def _mean(buf: list[float]) -> float | None:
            if not buf:
                return None
            slice_ = buf[-window:] if window is not None else buf
            return sum(slice_) / len(slice_) if slice_ else None

        return {
            "local_ms": _mean(self._timing_local_ms_per_iter),
            "linear_solve_ms": _mean(self._timing_linear_solve_ms_per_iter),
            "schur_ms": _mean(self._timing_schur_ms_per_iter),
            "nsn_inner_ms": _mean(self._timing_nsn_inner_ms_per_iter),
            "n_samples": len(self._timing_local_ms_per_iter)
            if window is None
            else min(window, len(self._timing_local_ms_per_iter)),
        }

    def configure_diagnostic_dump(self, frame: int, out_path: str) -> None:
        """Configure a single-frame diagnostic dump of PD intermediate state.

        On the ``frame``-th call to :meth:`step` (1-indexed), the solver
        captures per-PD-outer-iter intermediate variables (``x_pre``,
        ``x_inertia``, ``rhs_k{k}``, ``x_post_solve_k{k}``) into an in-memory
        buffer and writes them to ``out_path`` as a ``.npz`` archive once
        the step completes. Used to compare against RealSim's binary dumps
        for line-by-line parity verification.

        Args:
            frame: 1-indexed step number at which to capture the dump.
            out_path: Destination ``.npz`` file path.
        """
        self._diag_frame = int(frame)
        self._diag_out_path = Path(out_path)
        self._diag_buffers = {}

    def _setup_pd_system(self, dt: float) -> None:
        """Build / rebuild the PD Hessian, factor it, and upload device data."""
        from .linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            build_pd_system,
            factorize_and_sparse_inverse,
        )

        A, meta = build_pd_system(self.model, dt=dt, pin_stiffness=self.pin_stiffness)
        fs = factorize_and_sparse_inverse(A)
        self._linear_solver = FBALinearSolver(fs, device=self._device)
        self._meta = meta
        self._dt_setup = dt
        # A changed → any cached W / A^{-1} J^T is stale.
        self._invalidate_schur_cache()

        device = self._device
        self._tri_indices_d = wp.array(
            meta["tri_indices"].flatten().astype(np.int32),
            dtype=wp.int32,
            device=device,
        )
        self._tri_rest_inv_d = wp.array(
            meta["tri_rest_inv"].astype(np.float32),
            dtype=wp.mat22,
            device=device,
        )
        self._tri_weight_d = wp.array(
            meta["tri_weight"].astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
        if meta["edge_indices"].shape[0] > 0:
            self._edge_indices_d = wp.array(
                meta["edge_indices"].astype(np.int32),
                dtype=wp.int32,
                device=device,
            )
            self._edge_quad_q_d = wp.array(
                meta["edge_quad_q"].astype(np.float32),
                dtype=wp.vec4,
                device=device,
            )
            # Combine user weight with isometric scale 3/(A0+A1) into a single float per edge.
            edge_w = meta["edge_weight"] * meta.get("edge_quad_scale", np.ones_like(meta["edge_weight"]))
            self._edge_weight_d = wp.array(
                edge_w.astype(np.float32),
                dtype=wp.float32,
                device=device,
            )
            # Rest curvature magnitude per edge: ``‖q·x_rest‖``. The local
            # projection pulls ``q·x_cur`` back to this magnitude (not all the
            # way to a flat configuration), matching RealSim
            # ``PDIsometricBendingEnergy::computeRestBendingEnergy`` which
            # caches ``_norm[i]`` at init. Computed here from the cotangent
            # stencil and rest positions so the meta dict stays geometry-only.
            x_rest = self.model.particle_q.numpy().astype(np.float64)
            q = meta["edge_quad_q"]
            stencil_rest = x_rest[meta["edge_indices"]]  # (E, 4, 3)
            qTx_rest = (q[:, :, None] * stencil_rest).sum(axis=1)  # (E, 3)
            edge_norm = np.linalg.norm(qTx_rest, axis=1)  # (E,)
            self._edge_norm_d = wp.array(
                edge_norm.astype(np.float32),
                dtype=wp.float32,
                device=device,
            )
        if meta["pin_indices"].shape[0] > 0:
            self._pin_indices_d = wp.array(
                meta["pin_indices"].astype(np.int32),
                dtype=wp.int32,
                device=device,
            )
        if meta["tet_indices"].shape[0] > 0:
            self._tet_indices_d = wp.array(
                meta["tet_indices"].flatten().astype(np.int32),
                dtype=wp.int32,
                device=device,
            )
            self._tet_rest_inv_d = wp.array(
                meta["tet_rest_inv"].astype(np.float32),
                dtype=wp.mat33,
                device=device,
            )
            self._tet_weight_d = wp.array(
                meta["tet_weight"].astype(np.float32),
                dtype=wp.float32,
                device=device,
            )

        # Particle-centered CSR adjacency for deterministic PD reductions (Task P2-D-A).
        # Lets the PD scatter use (compute → gather) instead of atomic_add.
        from .linear_solver import build_particle_element_csr  # noqa: PLC0415

        n_p = int(self.model.particle_count)
        tet_indices_np = meta["tet_indices"]
        tri_indices_np = meta["tri_indices"]
        edge_indices_np = meta["edge_indices"]

        if tet_indices_np is not None and tet_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(tet_indices_np, n_p, 4)
            self._particle_tet_offsets_d = wp.array(offs, dtype=wp.int32, device=device)
            self._particle_tet_element_d = wp.array(elem_idx, dtype=wp.int32, device=device)
            self._particle_tet_local_d = wp.array(local_v, dtype=wp.int32, device=device)
            # Per-(tet, local vertex) scratch buffer for deterministic
            # (compute → gather) PD reduction. Reused across energies
            # (only one stretching_model is active per step).
            self._tet_contrib_d = wp.zeros(
                shape=(int(self.model.tet_count), 4),
                dtype=wp.vec3,
                device=device,
            )
        else:
            self._particle_tet_offsets_d = None
            self._particle_tet_element_d = None
            self._particle_tet_local_d = None
            self._tet_contrib_d = None

        if tri_indices_np is not None and tri_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(tri_indices_np, n_p, 3)
            self._particle_tri_offsets_d = wp.array(offs, dtype=wp.int32, device=device)
            self._particle_tri_element_d = wp.array(elem_idx, dtype=wp.int32, device=device)
            self._particle_tri_local_d = wp.array(local_v, dtype=wp.int32, device=device)
            # Per-(tri, local vertex) scratch buffer for deterministic
            # (compute → gather) PD reduction. Reused across energies
            # (only one stretching_model is active per step).
            self._tri_contrib_d = wp.zeros(
                shape=(int(tri_indices_np.shape[0]), 3),
                dtype=wp.vec3,
                device=device,
            )
        else:
            self._particle_tri_offsets_d = None
            self._particle_tri_element_d = None
            self._particle_tri_local_d = None
            self._tri_contrib_d = None

        if edge_indices_np is not None and edge_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(edge_indices_np, n_p, 4)
            self._particle_edge_offsets_d = wp.array(offs, dtype=wp.int32, device=device)
            self._particle_edge_element_d = wp.array(elem_idx, dtype=wp.int32, device=device)
            self._particle_edge_local_d = wp.array(local_v, dtype=wp.int32, device=device)
            # Per-(edge, local vertex) scratch buffer for deterministic
            # (compute → gather) PD bending reduction.
            self._edge_contrib_d = wp.zeros(
                shape=(int(edge_indices_np.shape[0]), 4),
                dtype=wp.vec3,
                device=device,
            )
        else:
            self._particle_edge_offsets_d = None
            self._particle_edge_element_d = None
            self._particle_edge_local_d = None
            self._edge_contrib_d = None

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the cloth state by one implicit Euler step using PD.

        Args:
            state_in: Input simulation state. ``particle_q`` and
                ``particle_qd`` are read; ``particle_f`` provides external
                per-particle forces [N].
            state_out: Output simulation state. ``particle_q`` and
                ``particle_qd`` are written; other fields are untouched.
            control: Unused in MVP (cloth has no actuated joints).
            contacts: Unused in MVP (contact-aware FBA is future work).
            dt: Time step [s]. Triggers `_setup_pd_system` rebuild on first
                step or when changed (PD Hessian depends on dt).
        """
        from .kernels import (  # noqa: PLC0415
            add_inertia_to_rhs_kernel,
            compute_inertial_kernel,
            gather_per_particle_kernel,
            project_bending_compute_kernel,
            project_pin_kernel,
            project_stretching_arap_compute_kernel,
            project_stretching_arap_tet_compute_kernel,
            project_stretching_corotational_compute_kernel,
            project_stretching_corotational_tet_compute_kernel,
            project_stretching_neohookean_compute_kernel,
            project_stretching_neohookean_compute_kernel_lbfgs,
            project_stretching_neohookean_tet_compute_kernel,
            project_stretching_neohookean_tet_compute_kernel_lbfgs,
            write_velocity_kernel,
            zero_vec3_kernel,
        )

        # Diagnostic dump: increment step counter at the START so frame N
        # corresponds to the N-th step() call (1-indexed). _diag_active becomes
        # True only on the configured frame; if not configured, it is always
        # False and all dump code below is a no-op.
        self._step_count += 1
        _diag_active = self._diag_frame is not None and self._step_count == self._diag_frame
        if _diag_active:
            self._diag_buffers = {
                "x_pre": state_in.particle_q.numpy().astype(np.float64).copy(),
            }

        if self._linear_solver is None or self._dt_setup is None or abs(self._dt_setup - dt) > 1e-12:
            self._setup_pd_system(dt)

        # Ingest contact data once per step (if provided).
        if contacts is not None:
            self.update_contacts(contacts, state_in)
        # Note: if contacts is None, self._contact_count retains its previous value (0 by default).
        # Call with contacts=None to keep existing behaviour; explicitly pass contacts to activate.
        # Reset contact count when contacts=None so the path stays disabled.
        if contacts is None:
            self._contact_count = 0
            self._invalidate_schur_cache()

        has_contacts = self._contact_count > 0

        # Apply kinematic anchor advancement: tangent_offset = base + dt · dot(t, v_anchor).
        # update_contacts cached base offsets + v_anchor; the dt-dependent
        # shift is applied here on device (kernel) so the per-step
        # ``tangent1_d.numpy() / tangent2_d.numpy() / v_anchor_d.numpy()``
        # downloads no longer fire. When ω = 0 everywhere, v_anchor is
        # zero and this is a no-op (offsets equal base).
        if self.friction and self._contact_count > 0 and hasattr(self, "_contact_v_anchor_d"):
            M_kin = self._contact_count
            from . import kernels as K_kin  # noqa: PLC0415

            wp.launch(
                K_kin.apply_tangent_kinematic_shift_kernel,
                dim=M_kin,
                inputs=[
                    self._contact_tangent1_d,
                    self._contact_tangent2_d,
                    self._contact_v_anchor_d,
                    self._contact_tangent1_offset_d,  # base (set in update_contacts)
                    self._contact_tangent2_offset_d,  # base
                    wp.float64(dt),
                ],
                outputs=[
                    self._contact_tangent1_offset_shifted_d,
                    self._contact_tangent2_offset_shifted_d,
                ],
                device=self._device,
            )

        # RealSim parity: λ = 0 once per step (per frame), then accumulates
        # across the PD outer iters below. ω carries the previous step's
        # weighting so iter-0 of the next call can compute penetration at the
        # already-corrected position.
        M = self._contact_count
        if M > 0:
            if self._lam_unilateral_persistent is None or self._lam_unilateral_persistent.shape != (M,):
                self._lam_unilateral_persistent = np.zeros(M, dtype=np.float64)
            else:
                self._lam_unilateral_persistent.fill(0.0)
            if self._lam_coulomb_persistent is None or self._lam_coulomb_persistent.shape != (3 * M,):
                self._lam_coulomb_persistent = np.zeros(3 * M, dtype=np.float64)
            else:
                self._lam_coulomb_persistent.fill(0.0)
            if self._omega_unilateral_persistent is None or self._omega_unilateral_persistent.shape != (M,):
                self._omega_unilateral_persistent = np.zeros(M, dtype=np.float64)
            else:
                self._omega_unilateral_persistent.fill(0.0)
            if self._omega_coulomb_persistent is None or self._omega_coulomb_persistent.shape != (3 * M,):
                self._omega_coulomb_persistent = np.zeros(3 * M, dtype=np.float64)
            else:
                self._omega_coulomb_persistent.fill(0.0)

            # Device-resident persistent warm-start buffers. Sized to the
            # current row count and zeroed at step entry, matching the host
            # arrays above. The hot solver path keeps the device versions
            # authoritative across PD outer iters; the host arrays are
            # refreshed once at end-of-step for tests / diagnostics.
            n_rows_max = 3 * M
            need_alloc = (
                not hasattr(self, "_lam_unilateral_persistent_d")
                or self._lam_unilateral_persistent_d is None
                or self._lam_unilateral_persistent_d.size < n_rows_max
            )
            if need_alloc:
                self._lam_unilateral_persistent_d = wp.zeros(n_rows_max, dtype=wp.float64, device=self._device)
                self._lam_coulomb_persistent_d = wp.zeros(n_rows_max, dtype=wp.float64, device=self._device)
                self._omega_unilateral_persistent_d = wp.zeros(n_rows_max, dtype=wp.float64, device=self._device)
                self._omega_coulomb_persistent_d = wp.zeros(n_rows_max, dtype=wp.float64, device=self._device)
            else:
                self._lam_unilateral_persistent_d.zero_()
                self._lam_coulomb_persistent_d.zero_()
                self._omega_unilateral_persistent_d.zero_()
                self._omega_coulomb_persistent_d.zero_()

        model = self.model
        N = model.particle_count
        device = self._device

        # 1) Compute inertial prediction x_inertia.
        wp.launch(
            compute_inertial_kernel,
            dim=N,
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

        if _diag_active:
            self._diag_buffers["x_inertia"] = self._x_inertia.numpy().astype(np.float64).copy()

        # 2) PD outer iterations.
        _perf_on = self.enable_perf_timing
        for _k in range(self.iterations):
            # ---- Local phase (energy projection + RHS assembly) ----
            if _perf_on:
                wp.synchronize_device()
                _t_local_0 = time.perf_counter()
            # Zero RHS.
            wp.launch(zero_vec3_kernel, dim=N, inputs=[self._rhs], device=device)
            # Inertia term.
            wp.launch(
                add_inertia_to_rhs_kernel,
                dim=N,
                inputs=[self._x_inertia, model.particle_mass, dt],
                outputs=[self._rhs],
                device=device,
            )
            # Pin projection.
            if self._pin_indices_d is not None:
                wp.launch(
                    project_pin_kernel,
                    dim=self._pin_indices_d.shape[0],
                    inputs=[self._pin_indices_d, self._x_ref, self.pin_stiffness],
                    outputs=[self._rhs],
                    device=device,
                )
            # Tri (cloth) stretching projection — deterministic compute+gather.
            if self._tri_indices_d is not None and model.tri_count > 0:
                if self.stretching_model == "arap":
                    wp.launch(
                        project_stretching_arap_compute_kernel,
                        dim=model.tri_count,
                        inputs=[
                            self._x_cur,
                            self._tri_indices_d,
                            self._tri_rest_inv_d,
                            self._tri_weight_d,
                        ],
                        outputs=[self._tri_contrib_d],
                        device=device,
                    )
                    wp.launch(
                        gather_per_particle_kernel,
                        dim=N,
                        inputs=[
                            self._tri_contrib_d,
                            self._particle_tri_offsets_d,
                            self._particle_tri_element_d,
                            self._particle_tri_local_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "corotational":
                    wp.launch(
                        project_stretching_corotational_compute_kernel,
                        dim=model.tri_count,
                        inputs=[
                            self._x_cur,
                            self._tri_indices_d,
                            self._tri_rest_inv_d,
                            self._tri_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._tri_contrib_d],
                        device=device,
                    )
                    wp.launch(
                        gather_per_particle_kernel,
                        dim=N,
                        inputs=[
                            self._tri_contrib_d,
                            self._particle_tri_offsets_d,
                            self._particle_tri_element_d,
                            self._particle_tri_local_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "neohookean":
                    nh_tri_kernel = (
                        project_stretching_neohookean_compute_kernel_lbfgs
                        if self.nh_solver == "lbfgs"
                        else project_stretching_neohookean_compute_kernel
                    )
                    wp.launch(
                        nh_tri_kernel,
                        dim=model.tri_count,
                        inputs=[
                            self._x_cur,
                            self._tri_indices_d,
                            self._tri_rest_inv_d,
                            self._tri_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._tri_contrib_d],
                        device=device,
                    )
                    wp.launch(
                        gather_per_particle_kernel,
                        dim=N,
                        inputs=[
                            self._tri_contrib_d,
                            self._particle_tri_offsets_d,
                            self._particle_tri_element_d,
                            self._particle_tri_local_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
            # Bending projection — deterministic (compute → gather) scatter.
            if self._edge_indices_d is not None:
                wp.launch(
                    project_bending_compute_kernel,
                    dim=self._edge_indices_d.shape[0],
                    inputs=[
                        self._x_cur,
                        self._edge_indices_d,
                        self._edge_quad_q_d,
                        self._edge_weight_d,
                        self._edge_norm_d,
                    ],
                    outputs=[self._edge_contrib_d],
                    device=device,
                )
                wp.launch(
                    gather_per_particle_kernel,
                    dim=N,
                    inputs=[
                        self._edge_contrib_d,
                        self._particle_edge_offsets_d,
                        self._particle_edge_element_d,
                        self._particle_edge_local_d,
                    ],
                    outputs=[self._rhs],
                    device=device,
                )
            # Tet ARAP projection.
            if self._tet_indices_d is not None and model.tet_count > 0:
                if self.stretching_model == "arap":
                    # Deterministic (compute → gather) tet ARAP scatter.
                    wp.launch(
                        project_stretching_arap_tet_compute_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                        ],
                        outputs=[self._tet_contrib_d],
                        device=device,
                    )
                    wp.launch(
                        gather_per_particle_kernel,
                        dim=N,
                        inputs=[
                            self._tet_contrib_d,
                            self._particle_tet_offsets_d,
                            self._particle_tet_element_d,
                            self._particle_tet_local_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "corotational":
                    # Deterministic (compute → gather) tet Corot scatter.
                    wp.launch(
                        project_stretching_corotational_tet_compute_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._tet_contrib_d],
                        device=device,
                    )
                    wp.launch(
                        gather_per_particle_kernel,
                        dim=N,
                        inputs=[
                            self._tet_contrib_d,
                            self._particle_tet_offsets_d,
                            self._particle_tet_element_d,
                            self._particle_tet_local_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "neohookean":
                    # Deterministic (compute → gather) tet NH scatter.
                    nh_tet_kernel = (
                        project_stretching_neohookean_tet_compute_kernel_lbfgs
                        if self.nh_solver == "lbfgs"
                        else project_stretching_neohookean_tet_compute_kernel
                    )
                    wp.launch(
                        nh_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._tet_contrib_d],
                        device=device,
                    )
                    wp.launch(
                        gather_per_particle_kernel,
                        dim=N,
                        inputs=[
                            self._tet_contrib_d,
                            self._particle_tet_offsets_d,
                            self._particle_tet_element_d,
                            self._particle_tet_local_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
            if _diag_active:
                self._diag_buffers[f"rhs_k{_k}"] = self._rhs.numpy().astype(np.float64).copy()

            if _perf_on:
                wp.synchronize_device()
                self._timing_local_ms_per_iter.append(1000.0 * (time.perf_counter() - _t_local_0))
                _t_lin_0 = time.perf_counter()

            # Global linear solve: x_unc = A^-1 . rhs  (unconstrained).
            self._linear_solver.solve(self._rhs, self._x_cur)

            if _perf_on:
                wp.synchronize_device()
                self._timing_linear_solve_ms_per_iter.append(1000.0 * (time.perf_counter() - _t_lin_0))

            if _diag_active:
                self._diag_buffers[f"x_post_solve_k{_k}"] = self._x_cur.numpy().astype(np.float64).copy()

            if has_contacts:
                # --- Schur-complement NSN contact correction ---
                M = self._contact_count
                ls = self._linear_solver

                if self.friction and hasattr(self, "_contact_tangent1_d"):
                    # Stage B: 3M Schur complement with Coulomb cone projection.
                    # Task L: reuse cached W and ls._A_inv_Jt_d across PD outer
                    # iters — both depend only on (J, A), which are fixed
                    # within a step.
                    if _perf_on:
                        wp.synchronize_device()
                        _t_schur_0 = time.perf_counter()
                    if self._cached_W is None or not self._cached_A_inv_Jt_valid:
                        self._cached_W = ls.build_schur_complement(
                            M,
                            self._contact_particle_d,
                            self._contact_normal_d,
                            self._contact_alpha_d,
                            self._contact_tangent1_d,
                            self._contact_tangent2_d,
                            use_isodof=self.use_isodof,
                        )
                        self._cached_A_inv_Jt_valid = True
                    if _perf_on:
                        wp.synchronize_device()
                        self._timing_schur_ms_per_iter.append(1000.0 * (time.perf_counter() - _t_schur_0))
                        _t_nsn_0 = time.perf_counter()
                    # Bottleneck #2 fix: residual + pene0 + warm-start all
                    # device-resident. The PD outer iter now does ZERO
                    # host<->device transfers around the NSN call.
                    self._ensure_nsn_inner_buffers(3 * M)
                    from . import kernels as K_step  # noqa: PLC0415

                    wp.launch(
                        K_step.compute_contact_residual_coulomb_kernel,
                        dim=M,
                        inputs=[
                            self._contact_offset_d,
                            self._contact_tangent1_offset_shifted_d,
                            self._contact_tangent2_offset_shifted_d,
                            self._contact_alpha_d,
                            self._contact_normal_d,
                            self._contact_tangent1_d,
                            self._contact_tangent2_d,
                            self._contact_particle_d,
                            self._x_cur,
                        ],
                        outputs=[self._nsn_r_d],
                        device=device,
                    )
                    wp.launch(
                        K_step.build_coulomb_pene0_kernel,
                        dim=M,
                        inputs=[
                            self._contact_offset_d,
                            self._contact_tangent1_offset_shifted_d,
                            self._contact_tangent2_offset_shifted_d,
                        ],
                        outputs=[self._nsn_pene0_d],
                        device=device,
                    )
                    self._solve_nsn_coulomb_gpu(
                        None,
                        None,
                        None,
                        None,
                        max_iters=self.nsn_iterations,
                        dt=dt,
                        W_device=ls.W_device_view(),
                        r_device=self._nsn_r_d,
                        pene0_device=self._nsn_pene0_d,
                        mu_device=self._contact_mu_d,
                        lam_init_device=self._lam_coulomb_persistent_d,
                        omega_init_device=self._omega_coulomb_persistent_d,
                        skip_download=True,
                        m_contacts=M,
                    )
                    # Persist device-resident lam / omega for next PD iter.
                    wp.copy(self._lam_coulomb_persistent_d, self._nsn_lam_d, count=3 * M)
                    wp.copy(self._omega_coulomb_persistent_d, self._nsn_omega_d, count=3 * M)
                    # RealSim in-iter combined-form correction (Task 1.3.g,
                    # A-tier alignment): mirrors NonSmoothNewton.cpp:151-167's
                    # ``applyConstraintCorrection``. The magnitude guard was
                    # dropped along with the host download — the cost of a
                    # no-op Cholesky solve is small relative to the saved
                    # ``_nsn_lam_apply_d.numpy()`` round-trip.
                    ls.apply_lambda_correction_combined(self._nsn_lam_apply_d, self._rhs, self._x_cur)
                    if _perf_on:
                        wp.synchronize_device()
                        self._timing_nsn_inner_ms_per_iter.append(1000.0 * (time.perf_counter() - _t_nsn_0))
                else:
                    # Stage A: M Schur complement, unilateral (λ ≥ 0) only.
                    if _perf_on:
                        wp.synchronize_device()
                        _t_schur_0 = time.perf_counter()
                    if self._cached_W is None or not self._cached_A_inv_Jt_valid:
                        self._cached_W = ls.build_schur_complement(
                            M,
                            self._contact_particle_d,
                            self._contact_normal_d,
                            self._contact_alpha_d,
                            use_isodof=self.use_isodof,
                        )
                        self._cached_A_inv_Jt_valid = True
                    if _perf_on:
                        wp.synchronize_device()
                        self._timing_schur_ms_per_iter.append(1000.0 * (time.perf_counter() - _t_schur_0))
                        _t_nsn_0 = time.perf_counter()
                    # Bottleneck #2 fix: residual + pene0 + warm-start all
                    # device-resident.
                    self._ensure_nsn_inner_buffers(M)
                    from . import kernels as K_step  # noqa: PLC0415

                    wp.launch(
                        K_step.compute_contact_residual_unilateral_kernel,
                        dim=M,
                        inputs=[
                            self._contact_offset_d,
                            self._contact_alpha_d,
                            self._contact_normal_d,
                            self._contact_particle_d,
                            self._x_cur,
                        ],
                        outputs=[self._nsn_r_d],
                        device=device,
                    )
                    # Stage A pene0 == contact_offset (no kinematic shift).
                    # Reuse the device buffer directly.
                    self._solve_nsn_unilateral_gpu(
                        None,
                        None,
                        None,
                        max_iters=self.nsn_iterations,
                        dt=dt,
                        W_device=ls.W_device_view(),
                        r_device=self._nsn_r_d,
                        pene0_device=self._contact_offset_d,
                        lam_init_device=self._lam_unilateral_persistent_d,
                        omega_init_device=self._omega_unilateral_persistent_d,
                        skip_download=True,
                        m_rows=M,
                    )
                    wp.copy(self._lam_unilateral_persistent_d, self._nsn_lam_d, count=M)
                    wp.copy(self._omega_unilateral_persistent_d, self._nsn_omega_d, count=M)
                    # RealSim in-iter combined-form correction (Task 1.3.g);
                    # see Stage B comment above re: dropped magnitude guard.
                    ls.apply_lambda_correction_combined(self._nsn_lam_apply_d, self._rhs, self._x_cur)
                    if _perf_on:
                        wp.synchronize_device()
                        self._timing_nsn_inner_ms_per_iter.append(1000.0 * (time.perf_counter() - _t_nsn_0))

        # 3) Write velocity and update state_out.
        wp.copy(state_out.particle_q, self._x_cur)
        wp.launch(
            write_velocity_kernel,
            dim=N,
            inputs=[state_in.particle_q, self._x_cur, model.particle_inv_mass, dt],
            outputs=[state_out.particle_qd],
            device=device,
        )

        # End-of-step refresh of host-mirror persistent buffers. Done once
        # per step (not per PD outer iter) so tests / diagnostics that read
        # ``solver._lam_*_persistent`` continue to see populated arrays
        # while the hot path remains free of per-PD-iter device->host
        # downloads. When M==0 the host arrays were left at None / unchanged.
        if has_contacts:
            if self.friction and hasattr(self, "_lam_coulomb_persistent_d"):
                self._lam_coulomb_persistent = self._lam_coulomb_persistent_d.numpy()[: 3 * M].astype(
                    np.float64, copy=True
                )
                self._omega_coulomb_persistent = self._omega_coulomb_persistent_d.numpy()[: 3 * M].astype(
                    np.float64, copy=True
                )
            elif hasattr(self, "_lam_unilateral_persistent_d"):
                self._lam_unilateral_persistent = self._lam_unilateral_persistent_d.numpy()[:M].astype(
                    np.float64, copy=True
                )
                self._omega_unilateral_persistent = self._omega_unilateral_persistent_d.numpy()[:M].astype(
                    np.float64, copy=True
                )

        # Diagnostic dump: flush captured buffers to .npz at end of step.
        if _diag_active and self._diag_out_path is not None:
            self._diag_out_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "N": int(N),
                "PD_iter": int(self.iterations),
                "frame": int(self._diag_frame),
                "dt": float(dt),
            }
            np.savez(
                str(self._diag_out_path),
                _meta_N=np.int64(meta["N"]),
                _meta_PD_iter=np.int64(meta["PD_iter"]),
                _meta_frame=np.int64(meta["frame"]),
                _meta_dt=np.float64(meta["dt"]),
                **self._diag_buffers,
            )
            print(
                f"[SolverFBA] wrote diagnostic dump for frame {self._diag_frame} "
                f"to {self._diag_out_path} ({len(self._diag_buffers)} arrays)"
            )

    def set_pin_targets(self, target_positions) -> None:
        """Update reference positions for pinned particles (dynamic pin).

        For pinned particles (``inv_mass == 0``), the soft Pin energy pulls them
        toward ``target_positions[i]``.  For free particles the value is unused
        but the array must cover all particles.

        Args:
            target_positions: Per-particle target positions [particle_count, 3].
                Accepts either a :class:`warp.array` (``dtype=wp.vec3``) or a
                NumPy array of shape ``(particle_count, 3)`` in float32.
        """
        if isinstance(target_positions, wp.array):
            self._x_ref.assign(target_positions)
        else:
            self._x_ref.assign(np.asarray(target_positions, dtype=np.float32))

    def notify_model_changed(self, flags: int) -> None:
        # On any geometry/inertial change, force a full re-setup at the next step.
        from ..flags import SolverNotifyFlags  # noqa: PLC0415

        if flags & (SolverNotifyFlags.SHAPE_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._linear_solver = None
            self._dt_setup = None
            self._invalidate_schur_cache()

    def _invalidate_schur_cache(self) -> None:
        """Drop the per-step ``W`` / ``A⁻¹·Jᵀ`` cache.

        Called whenever either input (``J`` from
        :meth:`update_contacts` or the Cholesky factor of ``A`` from
        :meth:`_setup_pd_system` / :meth:`notify_model_changed`) might
        change so the next :meth:`step` rebuilds the Schur complement.

        The persistent λ/ω warm-start buffers
        (``_lam_unilateral_persistent``, ``_lam_coulomb_persistent``,
        ``_omega_unilateral_persistent``, ``_omega_coulomb_persistent``)
        are intentionally left untouched here. Per-step entry in
        :meth:`step` reallocates them when the contact count ``M``
        changes and otherwise zero-fills them in-place, matching
        RealSim's ``cuda_lambda.setZero(_num_constraint)`` pattern in
        ``CUDANonSmoothNewton::prepare_gpu`` — a per-step concern that
        is decoupled from Schur factor invalidation.
        """
        self._cached_W = None
        self._cached_A_inv_Jt_valid = False

    # ------------------------------------------------------------------
    # Phase 4 Stage A — contact state (allocated lazily on first call).
    # ------------------------------------------------------------------

    def _ensure_contact_buffers(self, max_contacts: int) -> None:
        """Lazily allocate per-step contact buffers sized to ``max_contacts``."""
        if hasattr(self, "_contact_buf_max") and self._contact_buf_max >= max_contacts:
            return
        device = self._device
        cap = max(max_contacts, 16)
        self._contact_buf_max = cap
        self._contact_particle_d = wp.empty(cap, dtype=wp.int32, device=device)
        self._contact_normal_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_alpha_d = wp.empty(cap, dtype=wp.float32, device=device)
        self._contact_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
        # Stage B: tangent directions and per-contact friction mu.
        self._contact_tangent1_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_tangent2_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_mu_h = np.zeros(cap, dtype=np.float64)  # host-side mu array
        # Step 5.5: device-resident scratch buffers consumed by the
        # ``update_contacts`` GPU kernels (world anchor, tangent offsets,
        # spinning flag, v_anchor). Sized to ``cap`` once here so kernel
        # launches in :meth:`update_contacts` never re-allocate.
        self._contact_shape_d = wp.empty(cap, dtype=wp.int32, device=device)
        self._contact_body_pos_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_world_anchor_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_is_spinning_d = wp.empty(cap, dtype=wp.int32, device=device)
        self._contact_tangent1_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
        self._contact_tangent2_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
        self._contact_v_anchor_d = wp.empty(cap, dtype=wp.vec3d, device=device)
        # Per-step kinematic-shifted tangent offsets (base + dt·t·v_anchor).
        # Stage B residual + interleaved pene0 read these directly so the
        # ``dt`` shift no longer round-trips through host arrays.
        self._contact_tangent1_offset_shifted_d = wp.empty(cap, dtype=wp.float64, device=device)
        self._contact_tangent2_offset_shifted_d = wp.empty(cap, dtype=wp.float64, device=device)
        # Per-contact friction mu, device-resident fp64. Populated in
        # :meth:`update_contacts` to avoid the per-PD-iter host->device
        # upload that previously fired from the Stage B NSN driver.
        self._contact_mu_d = wp.empty(cap, dtype=wp.float64, device=device)

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Ingest the active particle-vs-shape contact set for the next step.

        Reads the soft contact fields from ``contacts`` and builds device-side
        Jacobian arrays (particle index, normal, alpha, signed-distance offset)
        that the Schur-complement path inside :meth:`step` will consume.

        For Stage A all contacts are particle-vs-static-shape so ``alpha = 1.0``.
        The signed-distance offset is ``dot(normal, world_anchor)`` where
        ``world_anchor`` is:

        - For dynamic bodies (``body_index >= 0``): ``wp.transform_point(body_q, body_pos)``,
          since ``soft_contact_body_pos`` is in body-local frame.
        - For static shapes (``body_index < 0``): ``body_pos`` directly, since
          ``soft_contact_body_pos`` is already in world frame (``X_wb = identity`` so
          ``X_ws = X_bs``, and the kernel stores ``wp.transform_point(X_bs, x_local)``).

        Args:
            contacts: :class:`~newton.Contacts` populated by a prior
                :meth:`~newton.CollisionPipeline.collide` call.
            state: Optional current :class:`~newton.State`; unused in Stage A
                (penetration offset derived purely from the contact anchor).
        """
        # Contact data is about to change → drop the Schur cache so the
        # next ``step()`` rebuilds ``W`` / ``A⁻¹·Jᵀ`` with fresh J.
        self._invalidate_schur_cache()

        M_raw = int(contacts.soft_contact_count.numpy()[0])
        if M_raw == 0:
            self._contact_count = 0
            return

        # Pull contact data to host for the lexsort (intentional S.1
        # for deterministic contact ordering; M is small so this is fast).
        # The collision pipeline writes contacts to slots assigned by
        # ``wp.atomic_add(soft_contact_count, 0, 1)`` so per-frame order
        # depends on GPU thread scheduling — downstream Gauss-Seidel and
        # atomic ``J^T λ`` accumulation are order-sensitive, hence we sort.
        particle_h = contacts.soft_contact_particle.numpy()[:M_raw]
        shape_h = contacts.soft_contact_shape.numpy()[:M_raw]
        # soft_contact_body_pos convention (from create_soft_contacts kernel):
        #   - body_index >= 0 (dynamic body): contact point in body-local frame.
        #   - body_index  < 0 (static shape): contact point already in world frame
        #     (because X_wb = identity so X_ws = X_bs, and body_pos = X_bs * x_local).
        body_pos_h = contacts.soft_contact_body_pos.numpy()[:M_raw]
        normal_h = contacts.soft_contact_normal.numpy()[:M_raw]  # world frame

        # Filter out sentinel entries (particle == -1).
        valid_mask = particle_h >= 0
        particle_h = particle_h[valid_mask]
        shape_h = shape_h[valid_mask]
        body_pos_h = body_pos_h[valid_mask]
        normal_h = normal_h[valid_mask]
        M = int(particle_h.shape[0])

        if M == 0:
            self._contact_count = 0
            return

        # Deterministic contact ordering (host-side lexsort — see comment above).
        sort_keys = (
            normal_h[:, 2].astype(np.float64),
            normal_h[:, 1].astype(np.float64),
            normal_h[:, 0].astype(np.float64),
            shape_h.astype(np.int64),
            particle_h.astype(np.int64),
        )
        order = np.lexsort(sort_keys)
        particle_h = particle_h[order]
        shape_h = shape_h[order]
        body_pos_h = body_pos_h[order]
        normal_h = normal_h[order]

        self._ensure_contact_buffers(M)

        # Step 5.5: per-contact bookkeeping (world_anchor, n·anchor offset,
        # tangent basis, t·anchor tangent offsets, v_anchor for rolling
        # cylinders) runs on GPU.  Only the lexsort above and the
        # singleton ``soft_contact_count`` read still hit the host;
        # everything else stays device-resident.
        from .kernels import (  # noqa: PLC0415
            compute_normal_offset_kernel,
            compute_tangent_basis_kernel,
            compute_tangent_offsets_kernel,
            compute_v_anchor_kernel,
            compute_world_anchor_kernel,
        )

        model = self.model
        device = self._device
        alpha_h = np.ones(M, dtype=np.float32)

        # Upload (sorted) compact contact arrays to device once.
        self._contact_particle_d.assign(particle_h.astype(np.int32))
        self._contact_normal_d.assign(normal_h.astype(np.float32))
        self._contact_alpha_d.assign(alpha_h)
        self._contact_shape_d.assign(shape_h.astype(np.int32))
        self._contact_body_pos_d.assign(body_pos_h.astype(np.float32))

        # Resolve model fields used by the GPU kernels.  When a field is
        # absent or empty we still pass a non-null dummy buffer through
        # the kernel and gate the lookup with a ``has_*`` flag, matching
        # the CPU branches that previously skipped these lookups.
        body_q_arr = model.body_q if hasattr(model, "body_q") and model.body_q is not None else None
        has_body_q = wp.int32(1 if body_q_arr is not None and body_q_arr.size > 0 else 0)
        if body_q_arr is None or body_q_arr.size == 0:
            body_q_arr = wp.empty(1, dtype=wp.transform, device=device)

        shape_body_arr = model.shape_body if hasattr(model, "shape_body") and model.shape_body is not None else None
        has_shape_body = wp.int32(1 if shape_body_arr is not None and shape_body_arr.size > 0 else 0)
        if shape_body_arr is None or shape_body_arr.size == 0:
            shape_body_arr = wp.empty(1, dtype=wp.int32, device=device)

        shape_type_arr = model.shape_type if hasattr(model, "shape_type") and model.shape_type is not None else None
        has_shape_type = wp.int32(1 if shape_type_arr is not None and shape_type_arr.size > 0 else 0)
        if shape_type_arr is None or shape_type_arr.size == 0:
            shape_type_arr = wp.empty(1, dtype=wp.int32, device=device)

        shape_transform_arr = (
            model.shape_transform if hasattr(model, "shape_transform") and model.shape_transform is not None else None
        )
        has_shape_transform = wp.int32(1 if shape_transform_arr is not None and shape_transform_arr.size > 0 else 0)
        if shape_transform_arr is None or shape_transform_arr.size == 0:
            shape_transform_arr = wp.empty(1, dtype=wp.transform, device=device)

        has_shape_omega = wp.int32(1 if self._shape_omega_d.size > 0 else 0)

        # Kernel 1: world-frame anchor.  Static-shape contacts pass body_pos
        # through; dynamic contacts apply body_q.
        wp.launch(
            compute_world_anchor_kernel,
            dim=M,
            inputs=[
                self._contact_body_pos_d,
                self._contact_shape_d,
                shape_body_arr,
                body_q_arr,
                has_body_q,
                has_shape_body,
            ],
            outputs=[self._contact_world_anchor_d],
            device=device,
        )

        # Kernel 2: normal offset with sphere/cylinder cushion.
        wp.launch(
            compute_normal_offset_kernel,
            dim=M,
            inputs=[
                self._contact_normal_d,
                self._contact_world_anchor_d,
                self._contact_shape_d,
                shape_type_arr,
                has_shape_type,
                wp.int32(int(GeoType.SPHERE)),
                wp.int32(int(GeoType.CYLINDER)),
            ],
            outputs=[self._contact_offset_d],
            device=device,
        )

        # Host mirrors of the basic contact fields are still consumed by
        # the CPU residual (``_compute_contact_residual*``); a follow-on
        # step in the plan moves the residual to GPU and removes these.
        self._contact_particle_h = particle_h.astype(np.int32)
        self._contact_normal_h = normal_h.astype(np.float32)
        self._contact_alpha_h = alpha_h
        self._contact_offset_h = self._contact_offset_d.numpy()[:M].copy()
        self._contact_count = M

        # Stage B: tangent basis, tangent offsets, kinematic anchor velocity.
        if self.friction:
            wp.launch(
                compute_tangent_basis_kernel,
                dim=M,
                inputs=[
                    self._contact_normal_d,
                    self._contact_shape_d,
                    self._shape_omega_d,
                    shape_transform_arr,
                    has_shape_omega,
                    has_shape_transform,
                ],
                outputs=[
                    self._contact_tangent1_d,
                    self._contact_tangent2_d,
                    self._contact_is_spinning_d,
                ],
                device=device,
            )
            wp.launch(
                compute_tangent_offsets_kernel,
                dim=M,
                inputs=[
                    self._contact_tangent1_d,
                    self._contact_tangent2_d,
                    self._contact_world_anchor_d,
                ],
                outputs=[
                    self._contact_tangent1_offset_d,
                    self._contact_tangent2_offset_d,
                ],
                device=device,
            )
            wp.launch(
                compute_v_anchor_kernel,
                dim=M,
                inputs=[
                    self._contact_is_spinning_d,
                    self._contact_shape_d,
                    self._shape_omega_d,
                    shape_transform_arr,
                    self._contact_world_anchor_d,
                ],
                outputs=[self._contact_v_anchor_d],
                device=device,
            )

            # Host mirrors (transitional — Step 6 moves the friction
            # residual to GPU and we can drop most of these).
            self._contact_tangent1_h = self._contact_tangent1_d.numpy()[:M].copy()
            self._contact_tangent2_h = self._contact_tangent2_d.numpy()[:M].copy()
            self._contact_v_anchor_h = self._contact_v_anchor_d.numpy()[:M].copy()
            t1_offset_h = self._contact_tangent1_offset_d.numpy()[:M].copy()
            t2_offset_h = self._contact_tangent2_offset_d.numpy()[:M].copy()
            self._contact_tangent1_offset_h_base = t1_offset_h.copy()
            self._contact_tangent2_offset_h_base = t2_offset_h.copy()
            self._contact_tangent1_offset_h = t1_offset_h
            self._contact_tangent2_offset_h = t2_offset_h

            # Per-contact friction μ. RealSim uses the shape's material μ directly
            # (no particle-side μ, no mixing) — see RealSim collision shape mu
            # propagation. FBA previously did `sqrt(particle_mu * shape_mu)` (VBD-style
            # mixing), which diverged from RealSim whenever a demo didn't override
            # via mu_per_pair_override. Per Q-G2 user decision 2026-05-17, align to
            # RealSim: shape μ directly. particle_mu kept as a fallback only when
            # the shape has no material μ (purely a robustness path; no in-scope
            # demo hits it).
            particle_mu = float(getattr(model, "particle_mu", 0.5))
            mu_h = np.zeros(M, dtype=np.float64)
            if self._mu_per_pair_override is not None:
                for c in range(M):
                    mu_h[c] = float(self._mu_per_pair_override[c])
            else:
                shape_mat_mu = model.shape_material_mu.numpy() if hasattr(model, "shape_material_mu") else None
                for c in range(M):
                    s_idx = int(shape_h[c])
                    if shape_mat_mu is not None and s_idx >= 0 and s_idx < len(shape_mat_mu):
                        mu_h[c] = float(shape_mat_mu[s_idx])
                    else:
                        mu_h[c] = particle_mu
            self._contact_mu_h = mu_h[:M]
            # Push mu to the device-resident buffer once per contact-set
            # update; the Stage B NSN inner driver reads from this fp64
            # array directly, eliminating the per-PD-iter ``M*8`` byte
            # host -> device upload.
            mu_padded = np.zeros(self._contact_mu_d.size, dtype=np.float64)
            mu_padded[:M] = self._contact_mu_h
            self._contact_mu_d.assign(mu_padded)

    # ------------------------------------------------------------------
    # Phase 4 Stage A — Schur-complement NSN helpers
    # ------------------------------------------------------------------

    def _compute_contact_residual_friction(self, x_np: np.ndarray) -> np.ndarray:
        """Compute the 3M-vector residual ``r = [r_n, r_t1, r_t2]`` for each contact.

        For each contact c:

        - ``r_n[c]  = offset_n[c]  - alpha[c] * dot(n[c],  x_np[p[c]])``
        - ``r_t1[c] = offset_t1[c] - alpha[c] * dot(t1[c], x_np[p[c]])``
        - ``r_t2[c] = offset_t2[c] - alpha[c] * dot(t2[c], x_np[p[c]])``

        where ``offset_n = dot(n, world_anchor)`` and
        ``offset_t1 = dot(t1, world_anchor)``,
        ``offset_t2 = dot(t2, world_anchor)`` (set in :meth:`update_contacts`).

        The tangent residuals measure displacement from the contact anchor
        along each tangent direction, so they are zero when the particle
        sits exactly at the anchor.  Using the global origin (offset = 0)
        was incorrect and inflated residuals by ~10x for off-origin
        contacts, driving unphysically large friction impulses.

        Args:
            x_np: Unconstrained solution, shape ``(N, 3)``, float32.

        Returns:
            Residual vector of shape ``(3M,)``, float64, ordered
            ``[r_n_0, r_t1_0, r_t2_0, r_n_1, ...]``.
        """
        M = self._contact_count
        r = np.zeros(3 * M, dtype=np.float64)
        for c in range(M):
            ip = int(self._contact_particle_h[c])
            xp = x_np[ip].astype(np.float64)
            alpha = float(self._contact_alpha_h[c])
            n = self._contact_normal_h[c].astype(np.float64)
            t1 = self._contact_tangent1_h[c].astype(np.float64)
            t2 = self._contact_tangent2_h[c].astype(np.float64)
            r[3 * c + 0] = float(self._contact_offset_h[c]) - alpha * float(np.dot(n, xp))
            r[3 * c + 1] = float(self._contact_tangent1_offset_h[c]) - alpha * float(np.dot(t1, xp))
            r[3 * c + 2] = float(self._contact_tangent2_offset_h[c]) - alpha * float(np.dot(t2, xp))
        return r

    def _compute_contact_residual(self, x_np: np.ndarray) -> np.ndarray:
        """Compute ``r[c] = alpha[c] * dot(n[c], x_np[p[c]]) - offset[c]``.

        Args:
            x_np: Current unconstrained solution, shape ``(N, 3)``, float32/64.

        Returns:
            Residual vector of shape ``(M,)``, float64.
        """
        M = self._contact_count
        r = np.zeros(M, dtype=np.float64)
        for c in range(M):
            ip = int(self._contact_particle_h[c])
            # r[c] = c_offset - J·x_unc  (positive when particle penetrates)
            r[c] = self._contact_offset_h[c] - float(self._contact_alpha_h[c]) * float(
                np.dot(self._contact_normal_h[c], x_np[ip])
            )
        return r

    def _solve_nsn_unilateral(
        self,
        W: np.ndarray,
        r: np.ndarray,
        pene0: np.ndarray,
        max_iters: int = 1,
        lam_init: np.ndarray | None = None,
        omega_init: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy CPU/numpy NSN driver for unilateral LCP (Stage A).

        .. deprecated:: Step 13 (NSN GPU port)
            No longer dispatched by :meth:`step`; the GPU driver
            :meth:`_solve_nsn_unilateral_gpu` is unconditional.  Retained
            only as a parity oracle for
            ``newton.tests.test_fba_nsn_gpu_driver`` and
            ``newton.tests.test_solver_fba``.  Do not call from new code.

        Returns ``(lam, omega, lam_apply)``:
          - ``lam`` (force units, for warm-start): RealSim's accumulated λ.
          - ``omega``: per-row weighting from the last NSN iter.
          - ``lam_apply`` = ``dt² · ω · lam`` (position-LCP units): the value to
            pass into :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.apply_lambda_correction_combined`
            (which adds ``A⁻¹·Jᵀ·lam_apply`` to ``x_unc``).

        The two scales differ because RealSim's correction is
        ``Δq = dt² · A⁻¹ · Jᵀ · (ω · λ_force)``, whereas FBA's
        ``apply_lambda_correction_combined`` adds ``A⁻¹·Jᵀ·λ`` directly. ``omega``
        from the previous step is needed so that the next call can compute the
        correct iter-0 ``penetration = -r + dt²·W·(ω·λ)`` accounting for the
        previously-applied correction.
        """
        M = len(r)
        if M == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        if lam_init is not None and lam_init.shape == (M,):
            lam = lam_init.astype(np.float64, copy=True)
        else:
            lam = np.zeros(M, dtype=np.float64)
        if omega_init is not None and omega_init.shape == (M,):
            omega = omega_init.astype(np.float64, copy=True)
        else:
            omega = np.zeros(M, dtype=np.float64)

        diag_W = np.diag(W)
        # RealSim convention: precond[i] = dt² · W_ii for unilateral rows.
        precond = (dt * dt) * np.where(np.abs(diag_W) > 1e-12, np.maximum(diag_W, 1e-12), 1.0)

        for _ in range(max_iters):
            # penetration = -r + dt²·W·(ω·λ). With warm-start ω·λ this reflects
            # the gap at the position currently corrected by the previous iter.
            penetration = -r + (dt * dt) * (W @ (omega * lam))
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
            # J·x_corrected = J·x_unc + dt²·W·(ω·λ) = (pene0 − r) + dt²·W·(ω·λ).
            J_x = (pene0 - r) + (dt * dt) * (W @ (omega * lam))
            rhs = (1.0 / (dt * dt)) * (h - omega * J_x)

            try:
                dlam = np.linalg.solve(A_schur, rhs)
            except np.linalg.LinAlgError:
                break
            lam = lam + dlam
            # No lam_n >= 0 clamp here. RealSim's boundConstraintForces
            # (NonSmoothNewton.cpp:391-394, unilateral branch) explicitly
            # comments out `if(_lambda[cid] < 0.0) _lambda[cid] = 0.0;` —
            # FB-Newton allows transient negative lambda inside the inner
            # solve and lets the FB residual converge. The previous
            # np.maximum(lam, 0.0) clamp converted Stage A toward
            # Signorini-PGS (see realsim-port-discipline P2-E v1 failure).

        if self.lambda_cap is not None:
            cap_internal = self.lambda_cap / (dt * dt)
            np.clip(lam, -cap_internal, cap_internal, out=lam)
        lam_apply = (dt * dt) * omega * lam
        return lam, omega, lam_apply

    def _solve_nsn_coulomb(
        self,
        W: np.ndarray,
        r: np.ndarray,
        mu: np.ndarray,
        pene0: np.ndarray,
        max_iters: int = 1,
        lam_init: np.ndarray | None = None,
        omega_init: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy CPU/numpy NSN driver for the frictional LCP (Stage B).

        .. deprecated:: Step 13 (NSN GPU port)
            No longer dispatched by :meth:`step`; the GPU driver
            :meth:`_solve_nsn_coulomb_gpu` is unconditional.  Retained
            only as a parity oracle for
            ``newton.tests.test_fba_nsn_gpu_driver`` and
            ``newton.tests.test_solver_fba``.  Do not call from new code.

        RealSim NonSmoothNewton port for the frictional LCP.

        Implements the dλ-Newton update from ``NonSmoothNewton.cpp:102-170``
        + ``343-378``, with omega weighting and position-LCP coupling. λ
        accumulates across ``max_iters`` iterations.

        Args:
            W: ``(3M, 3M)`` Schur complement.
            r: ``(3M,)`` residual ``pene0 − α·J·x_unc`` from
                ``_compute_contact_residual_friction``.
            mu: ``(M,)`` per-contact friction coefficient.
            pene0: ``(3M,)`` per-row anchor projection. Constant within step.
            max_iters: NSN iterations (default 1).
            lam_init: Optional warm-start ``(3M,)`` lambda.
            omega_init: Optional warm-start ``(3M,)`` omega from previous call.
            dt: Timestep (s).

        Returns:
            ``(lam, omega, lam_apply)``:
              - ``lam`` (force units, ``(3M,)``): RealSim-style accumulated λ
                with layout ``[λ_n_0, λ_t1_0, λ_t2_0, ...]``. Use as warm-start.
              - ``omega``: per-row weighting from the last NSN iter.
              - ``lam_apply`` = ``dt² · ω · lam``: pass into
                :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.apply_lambda_correction_combined`.
        """
        M = len(mu)
        if M == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        size_total = 3 * M

        if lam_init is not None and lam_init.shape == (size_total,):
            lam = lam_init.astype(np.float64, copy=True)
        else:
            lam = np.zeros(size_total, dtype=np.float64)
        if omega_init is not None and omega_init.shape == (size_total,):
            omega = omega_init.astype(np.float64, copy=True)
        else:
            omega = np.zeros(size_total, dtype=np.float64)

        diag_W = np.diag(W)
        # RealSim convention: precond[3c]   = dt² · W_ii  (unilateral row)
        #                     precond[3c+1] = dt   · W_ii  (tangent rows)
        #                     precond[3c+2] = dt   · W_ii
        precond = np.zeros(size_total, dtype=np.float64)
        for c in range(M):
            w_n = max(float(abs(diag_W[3 * c])), 1e-12)
            w_t1 = max(float(abs(diag_W[3 * c + 1])), 1e-12)
            w_t2 = max(float(abs(diag_W[3 * c + 2])), 1e-12)
            precond[3 * c] = (dt * dt) * w_n
            precond[3 * c + 1] = dt * w_t1
            precond[3 * c + 2] = dt * w_t2

        for _ in range(max_iters):
            # penetration = -r + dt²·W·(ω·λ); on iter 0 with ω=0 this is -r.
            penetration = -r + (dt * dt) * (W @ (omega * lam))

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

            # Schur LHS.
            A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)
            # J·x_corrected = J·x_unc + dt²·W·(ω·λ) = (pene0 − r) + dt²·W·(ω·λ).
            J_x = (pene0 - r) + (dt * dt) * (W @ (omega * lam))
            rhs = (1.0 / (dt * dt)) * (h - omega * J_x)

            try:
                dlam = np.linalg.solve(A_schur, rhs)
            except np.linalg.LinAlgError:
                break
            lam = lam + dlam

            # boundConstraintForces (NonSmoothNewton.cpp:395-403, friction
            # branch): signed cone clamp. lam_n is NOT clamped to >= 0 inside
            # the inner loop (the `if(_lambda[cid] < 0.0) = 0` line is
            # explicitly commented out in RealSim). Tangent clamp is two
            # independent if-statements applied in order; when lam_n < 0 this
            # saturates lam_t to -mu*lam_n (positive) rather than producing a
            # degenerate np.clip with low > high. Mirroring RealSim's exact
            # mutation order:
            for c in range(M):
                lam_n_c = lam[3 * c]  # SIGNED, per RealSim
                upper = mu[c] * lam_n_c
                lower = -mu[c] * lam_n_c
                if lam[3 * c + 1] > upper:
                    lam[3 * c + 1] = upper
                if lam[3 * c + 1] < lower:
                    lam[3 * c + 1] = lower
                if lam[3 * c + 2] > upper:
                    lam[3 * c + 2] = upper
                if lam[3 * c + 2] < lower:
                    lam[3 * c + 2] = lower

        if self.lambda_cap is not None:
            cap_internal = self.lambda_cap / (dt * dt)
            np.clip(lam, -cap_internal, cap_internal, out=lam)
        lam_apply = (dt * dt) * omega * lam
        return lam, omega, lam_apply

    # ------------------------------------------------------------------
    # NSN inner-loop GPU drivers (Steps 6-10 of the NSN GPU port)
    # ------------------------------------------------------------------
    #
    # The drivers below mirror :meth:`_solve_nsn_unilateral` and
    # :meth:`_solve_nsn_coulomb` but execute the per-iteration sequence
    # (residual → penetration → FB rows → Schur build → linear solve →
    # lambda update → box clamp → cap clip) on the device. They are
    # transitional: the host-side ``W`` / ``r`` / ``lam_init`` / ``omega_init``
    # arguments are accepted for direct CPU-parity testing and to keep the
    # ``step()`` plumbing decisions in Step 11. The new methods exit with
    # host-side ``(lam, omega, lam_apply)`` for the same reason.
    #
    # See ``docs/superpowers/plans/2026-05-17-fba-nsn-gpu-port.md`` for the
    # roll-out plan; RealSim reference is ``NonSmoothNewton.cpp:102-171``.

    def _ensure_pcr_solver(self, max_n: int) -> None:
        """Lazily create / resize the NSN-Schur PCR solver."""
        from .nsn_pcr_solver import NSNPCRSolver  # noqa: PLC0415

        existing = getattr(self, "_pcr_solver", None)
        if existing is None or existing.max_n < max_n:
            self._pcr_solver = NSNPCRSolver(
                max_n=max_n,
                device=self._device,
                # Tight tolerance so PCR ≈ ``np.linalg.solve`` to fp64 noise.
                tol=1.0e-10,
                max_iter=max(1000, 2 * max_n),
            )

    def _ensure_nsn_inner_buffers(self, n_rows: int) -> None:
        """Allocate or resize the device-resident NSN inner-loop scratch.

        Buffers grow with a 1.5x headroom factor so a slowly climbing contact
        count does not realloc the dense ``(cap, cap)`` matrices every frame.
        Demo 5 contact counts can ramp by tens of contacts between frames; a
        per-frame realloc otherwise pegs the device allocator and stalls.
        """
        existing_n = int(getattr(self, "_nsn_inner_n", 0))
        if existing_n >= n_rows:
            return
        # 1.5x growth factor on top of the strict minimum.  ``max`` against the
        # current capacity ensures monotonic growth and avoids shrinking after
        # a transient contact-count spike.
        cap = max(n_rows, int(existing_n * 1.5) + 1, 16)
        device = self._device
        self._nsn_inner_n = cap
        self._nsn_lam_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_omega_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_compliance_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_h_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_penetration_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_precond_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_r_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_pene0_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_rhs_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_dlam_d = wp.zeros(cap, dtype=wp.float64, device=device)
        self._nsn_lam_apply_d = wp.zeros(cap, dtype=wp.float64, device=device)
        # 2D buffers are square in the NSN setting.  ``_nsn_W_d`` is used only
        # as a fallback when the caller passes W as a host array (tests / CPU
        # parity path).  The hot solver path passes the device-resident
        # ``FBALinearSolver._W_device_d`` directly via ``W_device=`` and never
        # touches this buffer, so its memory cost is only paid in test mode.
        self._nsn_W_d = wp.zeros((cap, cap), dtype=wp.float64, device=device)
        self._nsn_a_schur_d = wp.zeros((cap, cap), dtype=wp.float64, device=device)

    def _solve_nsn_unilateral_gpu(
        self,
        W: np.ndarray | None,
        r: np.ndarray | None,
        pene0: np.ndarray | None,
        max_iters: int = 1,
        lam_init: np.ndarray | None = None,
        omega_init: np.ndarray | None = None,
        dt: float = 0.01,
        W_device: wp.array2d[wp.float64] | None = None,
        r_device: wp.array[wp.float64] | None = None,
        pene0_device: wp.array[wp.float64] | None = None,
        lam_init_device: wp.array[wp.float64] | None = None,
        omega_init_device: wp.array[wp.float64] | None = None,
        skip_download: bool = False,
        m_rows: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GPU port of :meth:`_solve_nsn_unilateral` (Stage A NSN inner).

        Sequence per FB-Newton iter (mirrors the CPU loop body 1:1):

            1. ``penetration = -r + dt²·W·(ω·λ)``
            2. ``(ω, c, h) = fb_unilateral_row(...)`` per row
            3. ``A_schur = ωωᵀ ⊙ W + diag(c)``
            4. ``rhs = (1/dt²)·(h - ω·J·x_corrected)``
            5. ``dλ = A_schur⁻¹·rhs`` via :class:`NSNPCRSolver`
            6. ``λ += dλ``

        After the loop, applies the optional ``lambda_cap / dt²``
        symmetric clip and returns ``lam_apply = dt²·ω·λ``.

        Args:
            W: ``(M, M)`` Schur complement (host fp64). May be ``None`` when
                ``W_device`` is provided.
            r: ``(M,)`` host residual. May be ``None`` when ``r_device`` is
                provided.
            pene0: ``(M,)`` per-row anchor projection. May be ``None`` when
                ``pene0_device`` is provided.
            max_iters: FB-Newton iters; matches ``nsn_iterations``.
            lam_init: Optional ``(M,)`` warm-start lambda (host).
            omega_init: Optional ``(M,)`` warm-start omega (host).
            dt: Timestep (s).
            W_device: Optional pre-populated ``(M, M)`` device-resident view of
                the Schur W (e.g. ``FBALinearSolver.W_device_view()``).  When
                provided, ``W`` is ignored and the per-PD-iter ``cap*cap*8``
                host -> device upload is skipped.  The view must remain alive
                for the duration of the call.
            r_device: Optional device-resident ``(>=M,)`` fp64 residual.
                Eliminates the per-PD-iter host->device upload of ``r``.
            pene0_device: Optional device-resident ``(>=M,)`` fp64 anchor
                projection. Eliminates the per-PD-iter host->device upload
                of ``pene0``.
            lam_init_device: Optional device-resident ``(>=M,)`` fp64
                warm-start lambda (takes precedence over ``lam_init``).
            omega_init_device: Optional device-resident ``(>=M,)`` fp64
                warm-start omega (takes precedence over ``omega_init``).
            skip_download: When ``True``, returns ``(empty, empty, empty)``
                arrays — the device-resident ``_nsn_lam_d`` / ``_nsn_omega_d``
                / ``_nsn_lam_apply_d`` buffers hold the authoritative result
                and the caller is expected to consume them on device.
            m_rows: Override row count when ``r_device`` is supplied (needed
                because device arrays carry capacity, not active size).

        References:
            RealSim ``NonSmoothNewton.cpp:102-171`` (Newton step assembly)
            and ``:332-341`` (unilateral FB row).
        """
        from . import kernels as K  # noqa: PLC0415

        if m_rows is not None:
            M = int(m_rows)
        elif r is not None:
            M = int(len(r))
        elif r_device is not None:
            M = int(r_device.shape[0])
        else:
            M = 0
        if M == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )

        device = self._device
        self._ensure_nsn_inner_buffers(M)
        self._ensure_pcr_solver(M)

        cap = self._nsn_inner_n
        # Bottleneck #1 fix: when the caller supplies a device-resident W view
        # (the hot solver path), use it directly — the kernels read only the
        # ``[:M, :M]`` block and the FBALinearSolver._W_device_d buffer already
        # holds the freshly built ``J A^{-1} J^T``.  Otherwise fall back to the
        # zero-padded upload path used by tests that pass a host numpy W.
        if W_device is not None and W_device.shape[0] >= M and W_device.shape[1] >= M:
            W_d = W_device
        else:
            W_pad = np.zeros((cap, cap), dtype=np.float64)
            W_pad[:M, :M] = W.astype(np.float64)
            self._nsn_W_d.assign(W_pad)
            W_d = self._nsn_W_d
        # Residual / pene0: prefer device buffers when supplied — eliminates
        # the per-PD-iter ``cap*8`` host upload that previously fired every
        # outer iter (Demo 5: ~6000 transfers/run).
        if r_device is not None and r_device.shape[0] >= M:
            r_d = r_device
        else:
            r_pad = np.zeros(cap, dtype=np.float64)
            r_pad[:M] = r.astype(np.float64)
            self._nsn_r_d.assign(r_pad)
            r_d = self._nsn_r_d
        if pene0_device is not None and pene0_device.shape[0] >= M:
            pene0_d = pene0_device
        else:
            pene0_pad = np.zeros(cap, dtype=np.float64)
            pene0_pad[:M] = pene0.astype(np.float64)
            self._nsn_pene0_d.assign(pene0_pad)
            pene0_d = self._nsn_pene0_d
        # Warm-start: device buffers take precedence; otherwise upload from
        # host arrays (legacy CPU path used by tests).
        if lam_init_device is not None and lam_init_device.shape[0] >= M:
            # Copy first M entries into the working buffer (preserves
            # warm-start while letting the kernel write into the scratch).
            wp.copy(self._nsn_lam_d, lam_init_device, count=M)
            # Zero the tail above M to keep the cap-sized buffer clean.
            if cap > M:
                self._nsn_lam_d[M:cap].zero_()
        else:
            lam_h = np.zeros(cap, dtype=np.float64)
            if lam_init is not None and lam_init.shape == (M,):
                lam_h[:M] = lam_init.astype(np.float64)
            self._nsn_lam_d.assign(lam_h)
        if omega_init_device is not None and omega_init_device.shape[0] >= M:
            wp.copy(self._nsn_omega_d, omega_init_device, count=M)
            if cap > M:
                self._nsn_omega_d[M:cap].zero_()
        else:
            omega_h = np.zeros(cap, dtype=np.float64)
            if omega_init is not None and omega_init.shape == (M,):
                omega_h[:M] = omega_init.astype(np.float64)
            self._nsn_omega_d.assign(omega_h)

        dt_w = wp.float64(dt)
        n_rows_w = wp.int32(M)

        # Step 7-prep: precond = dt² · max(W_ii, 1e-12) (Stage A).
        wp.launch(
            K.compute_precond_unilateral_kernel,
            dim=M,
            inputs=[W_d, dt_w],
            outputs=[self._nsn_precond_d],
            device=device,
        )

        for _ in range(int(max_iters)):
            # Step 7: penetration = -r + dt²·W·(ω·λ).
            wp.launch(
                K.compute_penetration_kernel,
                dim=M,
                inputs=[
                    r_d,
                    W_d,
                    self._nsn_omega_d,
                    self._nsn_lam_d,
                    n_rows_w,
                    dt_w,
                ],
                outputs=[self._nsn_penetration_d],
                device=device,
            )

            # Step 9 — FB rows (unilateral) overwrite omega/compliance/h.
            wp.launch(
                K.compute_unilateral_fb_kernel,
                dim=M,
                inputs=[
                    self._nsn_penetration_d,
                    self._nsn_lam_d,
                    self._nsn_precond_d,
                    pene0_d,
                    dt_w,
                ],
                outputs=[
                    self._nsn_omega_d,
                    self._nsn_compliance_d,
                    self._nsn_h_d,
                ],
                device=device,
            )

            # build_a_schur_kernel: A_schur = ωωᵀ ⊙ W + diag(c). Launched
            # over the active MxM block.
            wp.launch(
                K.build_a_schur_kernel,
                dim=(M, M),
                inputs=[
                    W_d,
                    self._nsn_omega_d,
                    self._nsn_compliance_d,
                ],
                outputs=[self._nsn_a_schur_d],
                device=device,
            )

            # Step 8: rhs = (1/dt²) · (h - ω · J·x_corrected).
            wp.launch(
                K.compute_nsn_rhs_kernel,
                dim=M,
                inputs=[
                    self._nsn_h_d,
                    self._nsn_omega_d,
                    pene0_d,
                    r_d,
                    W_d,
                    self._nsn_lam_d,
                    n_rows_w,
                    dt_w,
                ],
                outputs=[self._nsn_rhs_d],
                device=device,
            )

            # PCR Schur solve: ``A_schur · dlam = rhs`` on the active
            # M-sized slice. Warp 1.14 supports basic strided slicing on
            # device arrays; the resulting views share storage with the
            # pre-allocated buffers (no copy) so the PCR solver writes
            # ``dlam`` straight into ``_nsn_dlam_d``.
            a_view = self._nsn_a_schur_d[:M, :M]
            rhs_view = self._nsn_rhs_d[:M]
            dlam_view = self._nsn_dlam_d[:M]
            try:
                self._pcr_solver.solve(a_view, rhs_view, dlam_view)
            except Exception:
                # Match CPU LinAlgError fall-through: stop iterating, keep
                # current lambda.
                break

            # λ += dλ.
            wp.launch(
                K.axpy_lambda_kernel,
                dim=M,
                inputs=[self._nsn_dlam_d],
                outputs=[self._nsn_lam_d],
                device=device,
            )

        # Optional lambda_cap (in internal units = physical / dt²).
        if self.lambda_cap is not None:
            cap_internal = float(self.lambda_cap) / (dt * dt)
            wp.launch(
                K.lambda_cap_clip_kernel,
                dim=M,
                inputs=[wp.float64(cap_internal)],
                outputs=[self._nsn_lam_d],
                device=device,
            )

        # lam_apply = dt² · ω · lam.
        wp.launch(
            K.compute_lam_apply_kernel,
            dim=M,
            inputs=[self._nsn_lam_d, self._nsn_omega_d, dt_w],
            outputs=[self._nsn_lam_apply_d],
            device=device,
        )

        if skip_download:
            # Hot solver path: caller consumes the device buffers directly
            # (``_nsn_lam_d`` / ``_nsn_omega_d`` / ``_nsn_lam_apply_d``).
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty

        lam = self._nsn_lam_d.numpy()[:M].astype(np.float64, copy=True)
        omega = self._nsn_omega_d.numpy()[:M].astype(np.float64, copy=True)
        lam_apply = self._nsn_lam_apply_d.numpy()[:M].astype(np.float64, copy=True)
        return lam, omega, lam_apply

    def _solve_nsn_coulomb_gpu(
        self,
        W: np.ndarray | None,
        r: np.ndarray | None,
        mu: np.ndarray | None,
        pene0: np.ndarray | None,
        max_iters: int = 1,
        lam_init: np.ndarray | None = None,
        omega_init: np.ndarray | None = None,
        dt: float = 0.01,
        W_device: wp.array2d[wp.float64] | None = None,
        r_device: wp.array[wp.float64] | None = None,
        pene0_device: wp.array[wp.float64] | None = None,
        mu_device: wp.array[wp.float64] | None = None,
        lam_init_device: wp.array[wp.float64] | None = None,
        omega_init_device: wp.array[wp.float64] | None = None,
        skip_download: bool = False,
        m_contacts: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GPU port of :meth:`_solve_nsn_coulomb` (Stage B NSN inner).

        Identical structure to :meth:`_solve_nsn_unilateral_gpu` with row
        count ``3M`` and the frictional FB block replacing the unilateral
        per-row evaluation. After the Newton update, the per-contact
        signed-cone tangent clamp from RealSim
        ``NonSmoothNewton.cpp:395-403`` runs on device via
        :func:`coulomb_box_clamp_kernel`.

        Args:
            W: ``(3M, 3M)`` Schur complement (host fp64). May be ``None``
                when ``W_device`` is provided.
            r: ``(3M,)`` host residual. May be ``None`` when ``r_device``
                is provided.
            mu: ``(M,)`` per-contact friction coefficient (host). May be
                ``None`` when ``mu_device`` is provided.
            pene0: ``(3M,)`` per-row anchor projection. May be ``None``
                when ``pene0_device`` is provided.
            max_iters: FB-Newton iters.
            lam_init: Optional ``(3M,)`` warm-start lambda (host).
            omega_init: Optional ``(3M,)`` warm-start omega (host).
            dt: Timestep (s).
            W_device: Optional pre-populated ``(3M, 3M)`` device-resident view
                of the Schur W.  When provided, ``W`` is ignored and the
                per-PD-iter ``cap*cap*8`` host -> device upload is skipped.
            r_device: Optional device-resident ``(>=3M,)`` fp64 residual.
            pene0_device: Optional device-resident ``(>=3M,)`` fp64 anchor
                projection (interleaved ``[n, t1, t2]`` per contact).
            mu_device: Optional device-resident ``(>=M,)`` fp64 friction
                coefficient (avoids the per-PD-iter mu upload).
            lam_init_device: Optional device-resident ``(>=3M,)`` fp64
                warm-start lambda.
            omega_init_device: Optional device-resident ``(>=3M,)`` fp64
                warm-start omega.
            skip_download: When ``True``, suppresses the device->host copy
                of ``(lam, omega, lam_apply)`` — the caller is expected to
                read the device buffers directly.
            m_contacts: Override active contact count when only device
                arrays are provided.

        Returns:
            ``(lam, omega, lam_apply)`` host fp64 arrays of shape ``(3M,)``.
            All three are empty when ``skip_download=True``.

        References:
            RealSim ``NonSmoothNewton.cpp:102-171`` (Newton assembly),
            ``:343-378`` (frictional FB row), ``:395-403`` (cone clamp).
        """
        from . import kernels as K  # noqa: PLC0415

        if m_contacts is not None:
            M = int(m_contacts)
        elif mu is not None:
            M = int(len(mu))
        elif mu_device is not None:
            M = int(mu_device.shape[0])
        else:
            M = 0
        if M == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        n_rows = 3 * M
        device = self._device

        self._ensure_nsn_inner_buffers(n_rows)
        self._ensure_pcr_solver(n_rows)
        cap = self._nsn_inner_n

        # Stage B mu is a separate per-contact (size-M) array.
        mu_cap = max(M, 16)
        mu_existing = getattr(self, "_nsn_mu_d", None)
        if mu_existing is None or mu_existing.size < mu_cap:
            self._nsn_mu_d = wp.zeros(mu_cap, dtype=wp.float64, device=device)
        if mu_device is not None and mu_device.shape[0] >= M:
            mu_d = mu_device
        else:
            mu_h = np.zeros(self._nsn_mu_d.size, dtype=np.float64)
            mu_h[:M] = mu.astype(np.float64)
            self._nsn_mu_d.assign(mu_h)
            mu_d = self._nsn_mu_d

        # Bottleneck #1 fix: prefer the caller-supplied device W when given,
        # eliminating the cap*cap fp64 upload that previously fired every
        # PD outer iter.
        if W_device is not None and W_device.shape[0] >= n_rows and W_device.shape[1] >= n_rows:
            W_d = W_device
        else:
            W_pad = np.zeros((cap, cap), dtype=np.float64)
            W_pad[:n_rows, :n_rows] = W.astype(np.float64)
            self._nsn_W_d.assign(W_pad)
            W_d = self._nsn_W_d
        # Residual / pene0: prefer device buffers when supplied.
        if r_device is not None and r_device.shape[0] >= n_rows:
            r_d = r_device
        else:
            r_pad = np.zeros(cap, dtype=np.float64)
            r_pad[:n_rows] = r.astype(np.float64)
            self._nsn_r_d.assign(r_pad)
            r_d = self._nsn_r_d
        if pene0_device is not None and pene0_device.shape[0] >= n_rows:
            pene0_d = pene0_device
        else:
            pene0_pad = np.zeros(cap, dtype=np.float64)
            pene0_pad[:n_rows] = pene0.astype(np.float64)
            self._nsn_pene0_d.assign(pene0_pad)
            pene0_d = self._nsn_pene0_d
        # Warm-start.
        if lam_init_device is not None and lam_init_device.shape[0] >= n_rows:
            wp.copy(self._nsn_lam_d, lam_init_device, count=n_rows)
            if cap > n_rows:
                self._nsn_lam_d[n_rows:cap].zero_()
        else:
            lam_h = np.zeros(cap, dtype=np.float64)
            if lam_init is not None and lam_init.shape == (n_rows,):
                lam_h[:n_rows] = lam_init.astype(np.float64)
            self._nsn_lam_d.assign(lam_h)
        if omega_init_device is not None and omega_init_device.shape[0] >= n_rows:
            wp.copy(self._nsn_omega_d, omega_init_device, count=n_rows)
            if cap > n_rows:
                self._nsn_omega_d[n_rows:cap].zero_()
        else:
            omega_h = np.zeros(cap, dtype=np.float64)
            if omega_init is not None and omega_init.shape == (n_rows,):
                omega_h[:n_rows] = omega_init.astype(np.float64)
            self._nsn_omega_d.assign(omega_h)

        dt_w = wp.float64(dt)
        n_rows_w = wp.int32(n_rows)

        # Stage B precond: dt²·W_ii for normal rows; dt·W_ii for tangents.
        wp.launch(
            K.compute_precond_coulomb_kernel,
            dim=n_rows,
            inputs=[W_d, dt_w],
            outputs=[self._nsn_precond_d],
            device=device,
        )

        for _ in range(int(max_iters)):
            wp.launch(
                K.compute_penetration_kernel,
                dim=n_rows,
                inputs=[
                    r_d,
                    W_d,
                    self._nsn_omega_d,
                    self._nsn_lam_d,
                    n_rows_w,
                    dt_w,
                ],
                outputs=[self._nsn_penetration_d],
                device=device,
            )

            wp.launch(
                K.compute_frictional_fb_kernel,
                dim=M,
                inputs=[
                    self._nsn_penetration_d,
                    self._nsn_lam_d,
                    mu_d,
                    self._nsn_precond_d,
                    pene0_d,
                    dt_w,
                ],
                outputs=[
                    self._nsn_omega_d,
                    self._nsn_compliance_d,
                    self._nsn_h_d,
                ],
                device=device,
            )

            wp.launch(
                K.build_a_schur_kernel,
                dim=(n_rows, n_rows),
                inputs=[
                    W_d,
                    self._nsn_omega_d,
                    self._nsn_compliance_d,
                ],
                outputs=[self._nsn_a_schur_d],
                device=device,
            )

            wp.launch(
                K.compute_nsn_rhs_kernel,
                dim=n_rows,
                inputs=[
                    self._nsn_h_d,
                    self._nsn_omega_d,
                    pene0_d,
                    r_d,
                    W_d,
                    self._nsn_lam_d,
                    n_rows_w,
                    dt_w,
                ],
                outputs=[self._nsn_rhs_d],
                device=device,
            )

            a_view = self._nsn_a_schur_d[:n_rows, :n_rows]
            rhs_view = self._nsn_rhs_d[:n_rows]
            dlam_view = self._nsn_dlam_d[:n_rows]
            try:
                self._pcr_solver.solve(a_view, rhs_view, dlam_view)
            except Exception:
                break

            wp.launch(
                K.axpy_lambda_kernel,
                dim=n_rows,
                inputs=[self._nsn_dlam_d],
                outputs=[self._nsn_lam_d],
                device=device,
            )

            # Per-contact signed-cone clamp on tangent lambdas.
            wp.launch(
                K.coulomb_box_clamp_kernel,
                dim=M,
                inputs=[mu_d],
                outputs=[self._nsn_lam_d],
                device=device,
            )

        if self.lambda_cap is not None:
            cap_internal = float(self.lambda_cap) / (dt * dt)
            wp.launch(
                K.lambda_cap_clip_kernel,
                dim=n_rows,
                inputs=[wp.float64(cap_internal)],
                outputs=[self._nsn_lam_d],
                device=device,
            )

        wp.launch(
            K.compute_lam_apply_kernel,
            dim=n_rows,
            inputs=[self._nsn_lam_d, self._nsn_omega_d, dt_w],
            outputs=[self._nsn_lam_apply_d],
            device=device,
        )

        if skip_download:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty

        lam = self._nsn_lam_d.numpy()[:n_rows].astype(np.float64, copy=True)
        omega = self._nsn_omega_d.numpy()[:n_rows].astype(np.float64, copy=True)
        lam_apply = self._nsn_lam_apply_d.numpy()[:n_rows].astype(np.float64, copy=True)
        return lam, omega, lam_apply
