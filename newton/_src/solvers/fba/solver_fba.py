# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


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
    ) -> None:
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
            subtract_vec3_kernel,
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

                # 1. Build W = J A^{-1} J^T  (M x M dense).
                W = ls.build_schur_complement(
                    M,
                    self._contact_particle_d,
                    self._contact_normal_d,
                    self._contact_alpha_d,
                )

                # 2. Compute residual r = c_offset - J·x_unc  (positive = penetrating).
                x_unc_np = self._x_cur.numpy()  # (N, 3) float32
                r = self._compute_contact_residual(x_unc_np)

                # 3. Solve W λ = r with projected Gauss-Seidel (λ ≥ 0).
                lam = self._solve_nsn_unilateral(W, r, max_iters=20)

                # 4. Apply correction: x_cur = x_unc + A^{-1} J^T λ.
                #    (x* = A^{-1}(b + J^T λ) = x_unc + A^{-1} J^T λ)
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

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Ingest the active particle-vs-shape contact set for the next step.

        Reads the soft contact fields from ``contacts`` and builds device-side
        Jacobian arrays (particle index, normal, alpha, signed-distance offset)
        that the Schur-complement path inside :meth:`step` will consume.

        For Stage A all contacts are particle-vs-static-shape so ``α = 1.0``.
        The signed-distance offset is ``dot(normal, body_pos_world)`` where
        ``body_pos_world = wp.transform_point(body_transform, body_pos)``
        (or ``body_pos`` directly when the shape has no body — body index ``-1``).

        Args:
            contacts: :class:`~newton.Contacts` populated by a prior
                :meth:`~newton.CollisionPipeline.collide` call.
            state: Optional current :class:`~newton.State`; unused in Stage A
                (penetration offset derived purely from the contact anchor).
        """
        M_raw = int(contacts.soft_contact_count.numpy()[0])
        if M_raw == 0:
            self._contact_count = 0
            return

        # Pull contact data to host for filtering (M is small in Stage A).
        particle_h = contacts.soft_contact_particle.numpy()[:M_raw]
        shape_h = contacts.soft_contact_shape.numpy()[:M_raw]
        body_pos_h = contacts.soft_contact_body_pos.numpy()[:M_raw]  # shape-local
        normal_h = contacts.soft_contact_normal.numpy()[:M_raw]       # world frame

        # Access model fields needed for world-frame body_pos conversion.
        model = self.model
        shape_body_np = model.shape_body.numpy() if hasattr(model, "shape_body") else None
        shape_transform_np = model.shape_transform.numpy() if hasattr(model, "shape_transform") else None

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
        # world_anchor = transform_point(body_transform, body_pos)
        offset_h = np.zeros(M, dtype=np.float64)
        alpha_h = np.ones(M, dtype=np.float32)

        for c in range(M):
            s_idx = int(shape_h[c])
            bpos = body_pos_h[c]  # shape-local (vec3)

            # Compute world anchor.
            world_anchor = bpos.copy()
            if shape_body_np is not None and shape_transform_np is not None and s_idx >= 0:
                b_idx = int(shape_body_np[s_idx])
                if b_idx >= 0:
                    # Shape attached to a moving body — read body_q.
                    # body_q is a wp.transform (pos + quat).
                    body_q_np = model.body_q.numpy()
                    bq = body_q_np[b_idx]  # (7,): [px, py, pz, qx, qy, qz, qw]
                    pos_b = bq[:3]
                    quat_b = bq[3:]  # [qx, qy, qz, qw]
                    world_anchor = _transform_point(pos_b, quat_b, bpos)
                else:
                    # Static shape (body -1): shape_transform gives the world pose.
                    st = shape_transform_np[s_idx]  # (7,): [px, py, pz, qx, qy, qz, qw]
                    pos_s = st[:3]
                    quat_s = st[3:]
                    world_anchor = _transform_point(pos_s, quat_s, bpos)

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

    # ------------------------------------------------------------------
    # Phase 4 Stage A — Schur-complement NSN helpers
    # ------------------------------------------------------------------

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
            r[c] = self._contact_offset_h[c] - float(self._contact_alpha_h[c]) * float(np.dot(self._contact_normal_h[c], x_np[ip]))
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
        return lam

    def _apply_lambda_correction(self, lam: np.ndarray) -> wp.array:
        """Compute ``correction = A⁻¹ · Jᵀ · λ`` (vec3 array of length N).

        Reuses the cached ``_y_cache`` from the last
        :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.build_schur_complement`
        call when available, otherwise re-solves.

        Args:
            lam: Contact impulse vector, shape ``(M,)``, float64.

        Returns:
            Correction vec3 Warp array of length N.
        """
        from .kernels import accumulate_vec3_kernel  # noqa: PLC0415

        M = self._contact_count
        N = self.model.particle_count
        dev = self._device
        ls = self._linear_solver

        correction = wp.zeros(N, dtype=wp.vec3, device=dev)

        if hasattr(ls, "_y_cache") and len(ls._y_cache) == M:
            # Reuse cached A⁻¹ · J_c^T columns from build_schur_complement.
            correction_np = np.zeros((N, 3), dtype=np.float64)
            for c in range(M):
                if abs(lam[c]) < 1e-15:
                    continue
                correction_np += lam[c] * ls._y_cache[c]
            correction.assign(correction_np.astype(np.float32))
        else:
            # Fallback: re-solve for each contact.
            from .kernels import set_lambda_jacobian_vec3_kernel, zero_vec3_kernel  # noqa: PLC0415

            tmp = wp.empty(N, dtype=wp.vec3, device=dev)
            work = wp.empty(N, dtype=wp.vec3, device=dev)
            for c in range(M):
                if abs(lam[c]) < 1e-15:
                    continue
                wp.launch(zero_vec3_kernel, dim=N, inputs=[work], device=dev)
                wp.launch(
                    set_lambda_jacobian_vec3_kernel,
                    dim=1,
                    inputs=[c, self._contact_particle_d, self._contact_normal_d,
                            self._contact_alpha_d, float(lam[c]), N],
                    outputs=[work],
                    device=dev,
                )
                ls.solve(work, tmp)
                wp.launch(accumulate_vec3_kernel, dim=N, inputs=[tmp], outputs=[correction], device=dev)

        return correction
