# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


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
    - ``"corotational"`` — Corotational linear elasticity: closed-form 2x2
      solve on singular values of F.  Requires ``mu`` and ``lam`` (first and
      second Lame parameters [Pa]).
    - ``"neohookean"`` — Neo-Hookean hyperelasticity: 5-iteration Newton solve
      on singular values of F.  Requires ``mu`` and ``lam`` (first and second
      Lame parameters [Pa]).

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
            add_inertia_to_rhs_kernel,
            compute_inertial_kernel,
            project_bending_kernel,
            project_pin_kernel,
            project_stretching_arap_kernel,
            project_stretching_arap_tet_kernel,
            project_stretching_corotational_kernel,
            project_stretching_neohookean_kernel,
            write_velocity_kernel,
            zero_vec3_kernel,
        )

        if self._linear_solver is None or self._dt_setup is None or abs(self._dt_setup - dt) > 1e-12:
            self._setup_pd_system(dt)

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
                elif self.stretching_model in ("corotational", "neohookean"):
                    raise NotImplementedError(
                        f"stretching_model={self.stretching_model!r} is not yet implemented for tets. "
                        "Only 'arap' is supported for tetrahedral elements."
                    )
            # Global linear solve: x_cur = A^-1 . rhs.
            self._linear_solver.solve(self._rhs, self._x_cur)

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

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        raise NotImplementedError("Contact-aware FBA solver TBD; SolverFBA MVP supports gravity + pin only")
