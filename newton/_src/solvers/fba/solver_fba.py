# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


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
        nsn_iterations: int = 10,
        lambda_cap: float | None = None,
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

        # Per-element device data (filled by _setup_pd_system).
        self._tri_indices_d = None
        self._tri_rest_inv_d = None
        self._tri_weight_d = None
        self._edge_indices_d = None
        self._edge_quad_q_d = None
        self._edge_weight_d = None
        self._pin_indices_d = None

        # Tet device data (filled by _setup_pd_system).
        self._tet_indices_d = None
        self._tet_rest_inv_d = None
        self._tet_weight_d = None

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
            project_bending_kernel,
            project_pin_kernel,
            project_stretching_arap_kernel,
            project_stretching_arap_tet_kernel,
            project_stretching_corotational_kernel,
            project_stretching_corotational_tet_kernel,
            project_stretching_neohookean_kernel,
            project_stretching_neohookean_tet_kernel,
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
            # Stretching projection.
            if self.stretching_model == "arap":
                wp.launch(
                    project_stretching_arap_kernel,
                    dim=model.tri_count,
                    inputs=[
                        self._x_cur,
                        self._tri_indices_d,
                        self._tri_rest_inv_d,
                        self._tri_weight_d,
                    ],
                    outputs=[self._rhs],
                    device=device,
                )
            elif self.stretching_model == "corotational":
                wp.launch(
                    project_stretching_corotational_kernel,
                    dim=model.tri_count,
                    inputs=[
                        self._x_cur,
                        self._tri_indices_d,
                        self._tri_rest_inv_d,
                        self._tri_weight_d,
                        self._mu,
                        self._lam,
                    ],
                    outputs=[self._rhs],
                    device=device,
                )
            elif self.stretching_model == "neohookean":
                wp.launch(
                    project_stretching_neohookean_kernel,
                    dim=model.tri_count,
                    inputs=[
                        self._x_cur,
                        self._tri_indices_d,
                        self._tri_rest_inv_d,
                        self._tri_weight_d,
                        self._mu,
                        self._lam,
                    ],
                    outputs=[self._rhs],
                    device=device,
                )
            # Bending projection.
            if self._edge_indices_d is not None:
                wp.launch(
                    project_bending_kernel,
                    dim=self._edge_indices_d.shape[0],
                    inputs=[
                        self._x_cur,
                        self._x_ref,
                        self._edge_indices_d,
                        self._edge_quad_q_d,
                        self._edge_weight_d,
                    ],
                    outputs=[self._rhs],
                    device=device,
                )
            # Tet ARAP projection.
            if self._tet_indices_d is not None and model.tet_count > 0:
                if self.stretching_model == "arap":
                    wp.launch(
                        project_stretching_arap_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "corotational":
                    wp.launch(
                        project_stretching_corotational_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "neohookean":
                    wp.launch(
                        project_stretching_neohookean_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                            self._mu,
                            self._lam,
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
                        )
                        self._cached_A_inv_Jt_valid = True
                    W = self._cached_W
                    x_unc_np = self._x_cur.numpy()  # (N, 3) float32
                    r = self._compute_contact_residual_friction(x_unc_np)
                    lam = self._solve_nsn_coulomb(W, r, self._contact_mu_h[:M], max_iters=self.nsn_iterations)
                    if np.any(np.abs(lam) > 1e-15):
                        correction = self._apply_lambda_correction_friction(lam)
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
                        )
                        self._cached_A_inv_Jt_valid = True
                    W = self._cached_W
                    x_unc_np = self._x_cur.numpy()  # (N, 3) float32
                    r = self._compute_contact_residual(x_unc_np)
                    lam = self._solve_nsn_unilateral(W, r, max_iters=self.nsn_iterations)
                    if np.any(lam > 1e-15):
                        correction = self._apply_lambda_correction(lam)
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
            for c in range(M):
                t1, t2 = compute_tangent_basis(normal_h[c])
                t1_h[c] = t1.astype(np.float32)
                t2_h[c] = t2.astype(np.float32)
                # world_anchor for this contact (already in world frame).
                # offset_h[c] = dot(n, world_anchor), so to get world_anchor
                # we project body_pos through the body transform (already done
                # in offset_h construction loop above; reuse body_pos_h).
                # We recompute world_anchor here consistently with offset_h.
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
            self._contact_tangent1_d.assign(t1_h)
            self._contact_tangent2_d.assign(t2_h)
            self._contact_tangent1_h = t1_h
            self._contact_tangent2_h = t2_h
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

    def _solve_nsn_unilateral(self, W: np.ndarray, r: np.ndarray, max_iters: int = 20) -> np.ndarray:
        """Solve the unilateral LCP via projected Gauss-Seidel.

        Finds ``λ ≥ 0`` satisfying ``W·λ = r`` (normal contact forces).
        Uses component-wise clamped Gauss-Seidel:

        .. code-block:: text

            λ_c ← max(0, (r_c - Σ_{c'≠c} W[c,c'] λ_{c'}) / W[c,c])

        Args:
            W: Dense ``(M, M)`` Schur complement matrix.
            r: Residual ``J · x_unc - c``, shape ``(M,)``.
            max_iters: Maximum Gauss-Seidel iterations.

        Returns:
            Contact impulse vector ``λ``, shape ``(M,)``, float64.
        """
        M = len(r)
        lam = np.zeros(M, dtype=np.float64)
        for _ in range(max_iters):
            lam_old = lam.copy()
            for c in range(M):
                if W[c, c] <= 1e-12:
                    continue
                off_diag = W[c, :] @ lam - W[c, c] * lam[c]
                lam[c] = max(0.0, (r[c] - off_diag) / W[c, c])
            if np.linalg.norm(lam - lam_old, np.inf) < 1e-8:
                break
        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
        return lam

    def _solve_nsn_coulomb(self, W: np.ndarray, r: np.ndarray, mu: np.ndarray, max_iters: int = 20) -> np.ndarray:
        """Solve the frictional LCP via blocked projected Gauss-Seidel.

        Operates on 3-blocks ``[λ_n, λ_t1, λ_t2]`` per contact.  Each block
        is updated by solving the local 3x3 system (``W_cc``), then projecting
        onto the Coulomb cone.

        Complementarity is enforced per-block: if the effective normal residual
        ``r_eff[0] ≤ 0`` (no penetration for this contact after accounting for
        contributions from all other active contacts), the entire 3-block is
        forced to zero.  This mirrors Stage A's ``max(0, r/W)`` clamp and
        prevents the friction cone projection from generating spurious repulsive
        impulses on contacts where the gap is still open.

        Args:
            W: Dense ``(3M, 3M)`` Schur complement matrix.
            r: Residual vector of shape ``(3M,)``.
            mu: Per-contact friction coefficient, shape ``(M,)``.
            max_iters: Maximum blocked Gauss-Seidel iterations.

        Returns:
            Contact impulse vector ``λ``, shape ``(3M,)``, float64, ordered
            ``[λ_n_0, λ_t1_0, λ_t2_0, λ_n_1, ...]``.
        """
        M = len(mu)
        lam = np.zeros(3 * M, dtype=np.float64)
        for _ in range(max_iters):
            lam_old = lam.copy()
            for c in range(M):
                s = slice(3 * c, 3 * c + 3)
                W_cc = W[s, s]
                # Effective RHS: r_eff = r[s] - sum_{c' != c} W[s, s'] lam[s']
                off_diag = W[s, :] @ lam - W_cc @ lam[s]
                r_eff = r[s] - off_diag
                # Unilateral complementarity: contact is inactive when the
                # normal component of the effective residual is non-positive
                # (gap still open).  Forcing the block to zero matches the
                # Stage A max(0, r/W) clamp and prevents the Coulomb-cone
                # projection from manufacturing spurious outward impulses via
                # Case-3 when tangential residuals are large but r_n < 0.
                if r_eff[0] <= 0.0:
                    lam[3 * c + 0] = 0.0
                    lam[3 * c + 1] = 0.0
                    lam[3 * c + 2] = 0.0
                    continue
                # Solve 3x3: lam_unc = W_cc^{-1} r_eff
                if np.linalg.det(W_cc) < 1e-20:
                    continue
                lam_unc = np.linalg.solve(W_cc, r_eff)
                # Project onto Coulomb cone.
                s_unc = float(lam_unc[0])
                v_unc = lam_unc[1:3]
                s_proj, v_proj = project_coulomb_cone(s_unc, v_unc, float(mu[c]))
                lam[3 * c + 0] = s_proj
                lam[3 * c + 1] = v_proj[0]
                lam[3 * c + 2] = v_proj[1]
            if np.linalg.norm(lam - lam_old, np.inf) < 1e-8:
                break
        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
        return lam

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
