# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

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


def project_coulomb_cone(s: float, v: np.ndarray, mu: float) -> tuple[float, np.ndarray]:
    """Project ``(s, v)`` onto the Coulomb friction cone K = {s' >= 0, |v'| <= mu * s'}.

    Three cases:

    1. Already in K: return ``(s, v)`` unchanged.
    2. In polar cone ``mu * |v| <= -s``: project to origin ``(0, 0, 0)``.
    3. Otherwise: project to cone surface ``|v'| = mu * s'``.

    Args:
        s: Normal component (λ_n scalar).
        v: Tangent components ``(λ_t1, λ_t2)``, shape ``(2,)``.
        mu: Coulomb friction coefficient (>= 0).

    Returns:
        Tuple ``(s_new, v_new)`` on or inside the cone.
    """
    v_norm = float(np.sqrt(v[0] ** 2 + v[1] ** 2))
    # Case 1: already inside cone.
    if v_norm <= mu * s and s >= 0.0:
        return s, v
    # Case 2: in polar cone → project to origin.
    if mu * v_norm <= -s:
        return 0.0, np.zeros(2, dtype=np.float64)
    # Case 3: project to cone surface.
    factor = (s + mu * v_norm) / (1.0 + mu * mu)
    s_new = factor
    if v_norm < 1e-15:
        v_new = np.zeros(2, dtype=np.float64)
    else:
        v_new = (mu * factor / v_norm) * v
    return s_new, v_new


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
            nsn_iterations: Maximum projected Gauss-Seidel iterations for the
                contact NSN solver per PD step. Matches RealSim's per-scene
                ``constraintsolver.iterations`` setting.
            lambda_cap: Optional per-step clamp ``|λ| <= lambda_cap`` applied
                to contact impulses after the NSN solve. ``None`` disables
                clamping; mirrors RealSim's ``constraintsolver.maxforce``.
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
        # is reused by :meth:`_apply_lambda_correction*`; we only need to
        # track whether that buffer is fresh for the current step.
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
            accumulate_vec3_kernel,
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
            project_stretching_neohookean_tet_compute_kernel,
            write_velocity_kernel,
            zero_vec3_kernel,
        )

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
        # shift is applied here.  When ω = 0 everywhere, v_anchor is zero
        # and this is a no-op (offsets equal base).
        if self.friction and self._contact_count > 0 and hasattr(self, "_contact_v_anchor_h"):
            M_kin = self._contact_count
            v_anchor = self._contact_v_anchor_h[:M_kin]
            t1 = self._contact_tangent1_d.numpy()[:M_kin].astype(np.float64)
            t2 = self._contact_tangent2_d.numpy()[:M_kin].astype(np.float64)
            shift1 = dt * np.einsum("ij,ij->i", t1, v_anchor)
            shift2 = dt * np.einsum("ij,ij->i", t2, v_anchor)
            self._contact_tangent1_offset_h = self._contact_tangent1_offset_h_base + shift1
            self._contact_tangent2_offset_h = self._contact_tangent2_offset_h_base + shift2

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

        # 2) PD outer iterations.
        for _k in range(self.iterations):
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
                    wp.launch(
                        project_stretching_neohookean_compute_kernel,
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
                    wp.launch(
                        project_stretching_neohookean_tet_compute_kernel,
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
            # Global linear solve: x_unc = A^-1 . rhs  (unconstrained).
            self._linear_solver.solve(self._rhs, self._x_cur)

            if has_contacts:
                # --- Schur-complement NSN contact correction ---
                M = self._contact_count
                ls = self._linear_solver

                if self.friction and hasattr(self, "_contact_tangent1_d"):
                    # Stage B: 3M Schur complement with Coulomb cone projection.
                    # Task L: reuse cached W and ls._A_inv_Jt_d across PD outer
                    # iters — both depend only on (J, A), which are fixed
                    # within a step.
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
                    W = self._cached_W
                    x_unc_np = self._x_cur.numpy()  # (N, 3) float32
                    r = self._compute_contact_residual_friction(x_unc_np)
                    # Build per-row pene0 from cached anchor projections.
                    pene0_b = np.empty(3 * M, dtype=np.float64)
                    pene0_b[0::3] = self._contact_offset_h[:M]
                    pene0_b[1::3] = self._contact_tangent1_offset_h[:M]
                    pene0_b[2::3] = self._contact_tangent2_offset_h[:M]
                    lam, omega_last, lam_apply = self._solve_nsn_coulomb(
                        W,
                        r,
                        self._contact_mu_h[:M],
                        pene0_b,
                        max_iters=self.nsn_iterations,
                        lam_init=self._lam_coulomb_persistent,
                        omega_init=self._omega_coulomb_persistent,
                        dt=dt,
                    )
                    self._lam_coulomb_persistent = lam.copy()
                    self._omega_coulomb_persistent = omega_last.copy()
                    if np.any(np.abs(lam_apply) > 1e-15):
                        correction = self._apply_lambda_correction_friction(lam_apply)
                        wp.launch(
                            accumulate_vec3_kernel,
                            dim=N,
                            inputs=[correction],
                            outputs=[self._x_cur],
                            device=device,
                        )
                else:
                    # Stage A: M Schur complement, unilateral (λ ≥ 0) only.
                    if self._cached_W is None or not self._cached_A_inv_Jt_valid:
                        self._cached_W = ls.build_schur_complement(
                            M,
                            self._contact_particle_d,
                            self._contact_normal_d,
                            self._contact_alpha_d,
                            use_isodof=self.use_isodof,
                        )
                        self._cached_A_inv_Jt_valid = True
                    W = self._cached_W
                    x_unc_np = self._x_cur.numpy()  # (N, 3) float32
                    r = self._compute_contact_residual(x_unc_np)
                    pene0_a = self._contact_offset_h[:M].astype(np.float64, copy=True)
                    lam, omega_last, lam_apply = self._solve_nsn_unilateral(
                        W,
                        r,
                        pene0_a,
                        max_iters=self.nsn_iterations,
                        lam_init=self._lam_unilateral_persistent,
                        omega_init=self._omega_unilateral_persistent,
                        dt=dt,
                    )
                    self._lam_unilateral_persistent = lam.copy()
                    self._omega_unilateral_persistent = omega_last.copy()
                    if np.any(lam_apply > 1e-15):
                        correction = self._apply_lambda_correction(lam_apply)
                        wp.launch(
                            accumulate_vec3_kernel,
                            dim=N,
                            inputs=[correction],
                            outputs=[self._x_cur],
                            device=device,
                        )

        # 3) Write velocity and update state_out.
        wp.copy(state_out.particle_q, self._x_cur)
        wp.launch(
            write_velocity_kernel,
            dim=N,
            inputs=[state_in.particle_q, self._x_cur, model.particle_inv_mass, dt],
            outputs=[state_out.particle_qd],
            device=device,
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
        """
        self._cached_W = None
        self._cached_A_inv_Jt_valid = False
        # λ warm-start: persistent buffer must be resized when contact-set changes.
        self._lam_unilateral_persistent = None
        self._lam_coulomb_persistent = None
        self._omega_unilateral_persistent = None
        self._omega_coulomb_persistent = None

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

        # Pull contact data to host for filtering (M is small in Stage A).
        particle_h = contacts.soft_contact_particle.numpy()[:M_raw]
        shape_h = contacts.soft_contact_shape.numpy()[:M_raw]
        # soft_contact_body_pos convention (from create_soft_contacts kernel):
        #   - body_index >= 0 (dynamic body): contact point in body-local frame.
        #   - body_index  < 0 (static shape): contact point already in world frame
        #     (because X_wb = identity so X_ws = X_bs, and body_pos = X_bs * x_local).
        body_pos_h = contacts.soft_contact_body_pos.numpy()[:M_raw]
        normal_h = contacts.soft_contact_normal.numpy()[:M_raw]  # world frame

        # Access model fields needed for world-frame body_pos conversion.
        model = self.model
        shape_body_np = model.shape_body.numpy() if hasattr(model, "shape_body") else None

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

        # Deterministic contact ordering.  The collision pipeline writes
        # contacts to slots assigned by ``wp.atomic_add(soft_contact_count,
        # 0, 1)`` (see ``newton/_src/geometry/kernels.py``), so the per-frame
        # order depends on GPU thread-scheduling and varies between runs
        # with identical seeds.  Downstream consumers (NSN Gauss-Seidel,
        # atomic ``J^T lambda`` accumulation) are order-sensitive, which
        # causes wildly different trajectories across runs (SqueezingBall
        # demo: ~5 unit spread in ball ``min_y``).  Lexicographically sort
        # the contact arrays by ``(particle, shape, normal)`` so the
        # solver sees an identical permutation every step.  ``M`` is at
        # most ~thousand here, so the sort is negligible.
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

        # Build offset: pene0[c] = dot(normal, world_anchor)
        # world_anchor is the contact-point position in world frame.
        offset_h = np.zeros(M, dtype=np.float64)
        alpha_h = np.ones(M, dtype=np.float32)

        body_q_np = model.body_q.numpy() if hasattr(model, "body_q") and model.body_q is not None else None

        for c in range(M):
            s_idx = int(shape_h[c])
            bpos = body_pos_h[c]  # contact anchor (coordinate frame depends on body_index; see note above)

            # Compute world anchor.
            # For dynamic bodies: bpos is in body-local frame → apply body_q to get world.
            # For static shapes: bpos is already in world frame (body_index=-1, X_wb=identity).
            # Do NOT apply shape_transform to static shapes — that would be a double-transform.
            world_anchor = bpos.copy()
            if shape_body_np is not None and s_idx >= 0:
                b_idx = int(shape_body_np[s_idx])
                if b_idx >= 0 and body_q_np is not None:
                    # Shape attached to a moving body — transform body-local → world.
                    bq = body_q_np[b_idx]  # (7,): [px, py, pz, qx, qy, qz, qw]
                    pos_b = bq[:3]
                    quat_b = bq[3:]  # [qx, qy, qz, qw]
                    world_anchor = _transform_point(pos_b, quat_b, bpos)
                # else: body_index < 0 → bpos is already world frame; keep world_anchor = bpos

            n = normal_h[c]
            offset_h[c] = float(np.dot(n, world_anchor))

        # Upload compact arrays to device.
        self._contact_particle_d.assign(particle_h[:M].astype(np.int32))
        self._contact_normal_d.assign(normal_h[:M].astype(np.float32))
        self._contact_alpha_d.assign(alpha_h[:M])
        self._contact_offset_d.assign(offset_h[:M])

        # Keep host copies for the residual computation (avoids repeated .numpy()).
        self._contact_particle_h = particle_h[:M].astype(np.int32)
        self._contact_normal_h = normal_h[:M].astype(np.float32)
        self._contact_alpha_h = alpha_h[:M]
        self._contact_offset_h = offset_h[:M]
        self._contact_count = M

        # Stage B: compute tangent basis and friction μ for each contact.
        if self.friction:
            t1_h = np.zeros((M, 3), dtype=np.float32)
            t2_h = np.zeros((M, 3), dtype=np.float32)
            # Per-contact tangential offsets: dot(t1, world_anchor) and
            # dot(t2, world_anchor).  These are used in the friction residual
            # r_t = dot(t, anchor) - alpha * dot(t, x_unc), which correctly
            # measures tangential displacement from the contact anchor (not
            # from the world origin).  Without this correction, the residual
            # is ~10x too large for off-origin contacts, driving astronomically
            # large friction impulses.
            tangent1_offset_h = np.zeros(M, dtype=np.float64)
            tangent2_offset_h = np.zeros(M, dtype=np.float64)

            # Precompute per-shape world-frame Z axis (Newton's cylinder
            # extends along local +Z; world axis = shape_q rotated +Z).
            shape_transform_np = (
                model.shape_transform.numpy()
                if hasattr(model, "shape_transform") and model.shape_transform is not None
                else None
            )

            v_anchor_h = np.zeros((M, 3), dtype=np.float64)

            for c in range(M):
                s_idx = int(shape_h[c])
                # Determine whether this contact is on a spinning cylinder
                # shape; if so, we use a structured cylinder-aligned tangent
                # basis (RealSim parity) instead of the arbitrary basis from
                # ``compute_tangent_basis``.  This prevents the t1/t2 Schur
                # off-diagonal coupling from over-correcting v_anchor and
                # pumping energy into the body (see SqueezingBall diagnosis).
                is_spinning_shape = (
                    s_idx >= 0
                    and s_idx < self._shape_omega_h.shape[0]
                    and self._shape_omega_h[s_idx] != 0.0
                    and shape_transform_np is not None
                )

                # Precompute axis_world once if the shape has a transform —
                # used by both the cylinder-aligned basis and v_anchor below.
                axis_world = None
                shape_p = None
                if is_spinning_shape:
                    xf = shape_transform_np[s_idx]
                    shape_p = np.asarray(xf[:3], dtype=np.float64)
                    shape_q = np.asarray(xf[3:], dtype=np.float64)
                    axis_world = _quat_rotate_z_axis(shape_q)

                # Pick tangent basis.
                if is_spinning_shape:
                    n_arr = np.asarray(normal_h[c], dtype=np.float64)
                    # t1 = normalize(normal × axis) — rolling direction.
                    t1_vec = np.cross(n_arr, axis_world)
                    n1 = float(np.linalg.norm(t1_vec))
                    if n1 < 1e-9:
                        # Degenerate: normal parallel to axis (e.g. cap face).
                        # Fall back to arbitrary basis.
                        t1, t2 = compute_tangent_basis(normal_h[c])
                    else:
                        t1 = t1_vec / n1
                        # t2 lies along the cylinder axis projected onto the
                        # tangent plane; ``normal × t1`` produces an axis-
                        # aligned vector with a determined sign.
                        t2_raw = np.cross(n_arr, t1)
                        n2 = float(np.linalg.norm(t2_raw))
                        t2 = t2_raw / max(n2, 1e-30)
                else:
                    t1, t2 = compute_tangent_basis(normal_h[c])

                t1_h[c] = t1.astype(np.float32)
                t2_h[c] = t2.astype(np.float32)
                # world_anchor for this contact (already in world frame).
                # offset_h[c] = dot(n, world_anchor), so to get world_anchor
                # we project body_pos through the body transform (already done
                # in offset_h construction loop above; reuse body_pos_h).
                # We recompute world_anchor here consistently with offset_h.
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
                if is_spinning_shape:
                    r_local = world_anchor - shape_p
                    omega = float(self._shape_omega_h[s_idx])
                    # RealSim sign convention: tangent = normal × axis (see
                    # CudaTests CylinderCollision.cpp), so the anchor velocity
                    # along that tangent is +radius·ω.  Newton's cross is
                    # ``axis × r_radial`` = ``-radius (normal × axis)``, hence
                    # the leading minus sign here.
                    v_anchor_h[c] = -omega * np.cross(axis_world, r_local)

            self._contact_tangent1_d.assign(t1_h)
            self._contact_tangent2_d.assign(t2_h)
            self._contact_tangent1_h = t1_h
            self._contact_tangent2_h = t2_h
            self._contact_v_anchor_h = v_anchor_h
            # Cache the BASE offsets (no dt-shift); step() applies the
            # kinematic dt·dot(t, v_anchor) shift on entry.
            self._contact_tangent1_offset_h_base = tangent1_offset_h.copy()
            self._contact_tangent2_offset_h_base = tangent2_offset_h.copy()
            self._contact_tangent1_offset_h = tangent1_offset_h
            self._contact_tangent2_offset_h = tangent2_offset_h

            # Compute per-contact friction μ via VBD-style sqrt mixing.
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
                        mu_h[c] = float(np.sqrt(particle_mu * float(shape_mat_mu[s_idx])))
                    else:
                        mu_h[c] = particle_mu
            self._contact_mu_h = mu_h[:M]

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
        """RealSim NonSmoothNewton port for unilateral LCP (Stage A).

        Returns ``(lam, omega, lam_apply)``:
          - ``lam`` (force units, for warm-start): RealSim's accumulated λ.
          - ``omega``: per-row weighting from the last NSN iter.
          - ``lam_apply`` = ``dt² · ω · lam`` (position-LCP units): the value to
            pass into :meth:`_apply_lambda_correction` (which adds
            ``A⁻¹·Jᵀ·lam_apply`` to ``x_unc``).

        The two scales differ because RealSim's correction is
        ``Δq = dt² · A⁻¹ · Jᵀ · (ω · λ_force)``, whereas FBA's
        ``_apply_lambda_correction`` adds ``A⁻¹·Jᵀ·λ`` directly. ``omega``
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
            np.maximum(lam, 0.0, out=lam)

        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
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
        """RealSim NonSmoothNewton port for the frictional LCP.

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
                :meth:`_apply_lambda_correction_friction`.
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

            # boundConstraintForces: clamp box per contact.
            for c in range(M):
                lam_n_c = max(0.0, lam[3 * c])
                lam[3 * c] = lam_n_c
                cone = mu[c] * lam_n_c
                lam[3 * c + 1] = float(np.clip(lam[3 * c + 1], -cone, cone))
                lam[3 * c + 2] = float(np.clip(lam[3 * c + 2], -cone, cone))

        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
        lam_apply = (dt * dt) * omega * lam
        return lam, omega, lam_apply

    def _apply_lambda_correction_friction(self, lam: np.ndarray) -> wp.array:
        """Compute ``correction = A⁻¹ · Jᵀ · λ`` for Stage B (3M λ).

        Uses the cached ``_A_inv_Jt_d`` device buffer from
        :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.build_schur_complement`
        which contains one (N,) vec3 row per row (3M total entries for friction).

        Each row r corresponds to contact ``c = r // 3``, axis ``a = r % 3``:

        - a=0: normal contribution
        - a=1: t1 contribution
        - a=2: t2 contribution

        Args:
            lam: Contact impulse vector, shape ``(3M,)``, float64.

        Returns:
            Correction vec3 Warp array of length N.
        """

        M = self._contact_count
        total_rows = 3 * M
        N = self.model.particle_count
        dev = self._device
        ls = self._linear_solver

        correction = wp.zeros(N, dtype=wp.vec3, device=dev)

        # Isodof-mode path: rebuild ``J^T lambda`` and run a single Cholesky
        # solve. This is the default for ``use_isodof=True`` because the
        # ``_A_inv_Jt_d`` cache is not populated by the isodof Schur build.
        if self.use_isodof and hasattr(ls, "_row_particle_d") and ls._row_total == total_rows:
            ls.apply_lambda_correction_isodof(lam, correction)
            return correction

        if hasattr(ls, "_A_inv_Jt_d") and ls._A_inv_Jt_d.shape[0] >= total_rows:
            # Use cached device buffer: accumulate weighted rows entirely on GPU.
            from .kernels import accumulate_lambda_correction_kernel  # noqa: PLC0415

            lam_d = wp.array(lam.astype(np.float32), dtype=wp.float32, device=dev)
            wp.launch(
                accumulate_lambda_correction_kernel,
                dim=N,
                inputs=[lam_d, ls._A_inv_Jt_d, int(total_rows)],
                outputs=[correction],
                device=dev,
            )
            return correction

        # Fallback path: re-solve for each row (only used when the cached
        # ``_A_inv_Jt_d`` buffer is unavailable, e.g. tests that bypass the
        # batched Schur build).
        correction_np = np.zeros((N, 3), dtype=np.float64)

        from .kernels import build_contact_jacobian_dir_kernel, zero_vec3_kernel  # noqa: PLC0415

        tmp = wp.empty(N, dtype=wp.vec3, device=dev)
        work = wp.empty(N, dtype=wp.vec3, device=dev)
        for c in range(M):
            if not hasattr(self, "_contact_tangent1_h"):
                continue
            n = self._contact_normal_h[c].astype(np.float64)
            t1 = self._contact_tangent1_h[c].astype(np.float64)
            t2 = self._contact_tangent2_h[c].astype(np.float64)
            directions = [n, t1, t2]
            for a, direction in enumerate(directions):
                row = 3 * c + a
                if abs(lam[row]) < 1e-15:
                    continue
                wp.launch(zero_vec3_kernel, dim=N, inputs=[work], device=dev)
                wp.launch(
                    build_contact_jacobian_dir_kernel,
                    dim=1,
                    inputs=[
                        N,
                        c,
                        self._contact_particle_d,
                        self._contact_alpha_d,
                        wp.vec3(float(direction[0]), float(direction[1]), float(direction[2])),
                    ],
                    outputs=[work],
                    device=dev,
                )
                ls.solve(work, tmp)
                tmp_np = tmp.numpy().astype(np.float64)
                correction_np += lam[row] * tmp_np

        correction.assign(correction_np.astype(np.float32))
        return correction

    def _apply_lambda_correction(self, lam: np.ndarray) -> wp.array:
        """Compute ``correction = A⁻¹ · Jᵀ · λ`` (vec3 array of length N).

        Reuses the cached ``_A_inv_Jt_d`` device buffer from the last
        :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.build_schur_complement`
        call when available, otherwise re-solves.

        Args:
            lam: Contact impulse vector, shape ``(M,)``, float64.

        Returns:
            Correction vec3 Warp array of length N.
        """

        M = self._contact_count
        N = self.model.particle_count
        dev = self._device
        ls = self._linear_solver

        correction = wp.zeros(N, dtype=wp.vec3, device=dev)

        # Isodof-mode path: rebuild ``J^T lambda`` and run a single Cholesky
        # solve. This is the default for ``use_isodof=True`` because the
        # ``_A_inv_Jt_d`` cache is not populated by the isodof Schur build.
        if self.use_isodof and hasattr(ls, "_row_particle_d") and ls._row_total == M:
            ls.apply_lambda_correction_isodof(lam, correction)
            return correction

        if hasattr(ls, "_A_inv_Jt_d") and ls._A_inv_Jt_d.shape[0] >= M:
            # Reuse cached A⁻¹ · J_c^T columns from build_schur_complement and
            # run the weighted sum entirely on device.
            from .kernels import accumulate_lambda_correction_kernel  # noqa: PLC0415

            lam_d = wp.array(lam.astype(np.float32), dtype=wp.float32, device=dev)
            wp.launch(
                accumulate_lambda_correction_kernel,
                dim=N,
                inputs=[lam_d, ls._A_inv_Jt_d, int(M)],
                outputs=[correction],
                device=dev,
            )
        else:
            # Fallback: re-solve for each contact.
            from .kernels import (  # noqa: PLC0415
                accumulate_vec3_kernel,
                set_lambda_jacobian_vec3_kernel,
                zero_vec3_kernel,
            )

            tmp = wp.empty(N, dtype=wp.vec3, device=dev)
            work = wp.empty(N, dtype=wp.vec3, device=dev)
            for c in range(M):
                if abs(lam[c]) < 1e-15:
                    continue
                wp.launch(zero_vec3_kernel, dim=N, inputs=[work], device=dev)
                wp.launch(
                    set_lambda_jacobian_vec3_kernel,
                    dim=1,
                    inputs=[
                        c,
                        self._contact_particle_d,
                        self._contact_normal_d,
                        self._contact_alpha_d,
                        float(lam[c]),
                        N,
                    ],
                    outputs=[work],
                    device=dev,
                )
                ls.solve(work, tmp)
                wp.launch(accumulate_vec3_kernel, dim=N, inputs=[tmp], outputs=[correction], device=dev)

        return correction
