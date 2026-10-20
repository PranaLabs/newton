# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase


def _element_lame(materials: wp.array | None, count: int) -> tuple[np.ndarray, np.ndarray]:
    """First and second Lamé parameters [Pa] of each element, from its material row."""
    if count == 0:
        return np.zeros(0), np.zeros(0)
    rows = materials.numpy().astype(np.float64)
    return rows[:, 0], rows[:, 1]


def _host_mirror(name: str) -> property:
    """A host copy of a persistent multiplier buffer of :class:`SolverFBA`, read from the device when it is first
    read after a step."""

    def get(solver):
        if solver._persistent_rows is not None:
            solver._read_persistent()
        return solver.__dict__.get(name)

    def put(solver, value):
        solver.__dict__[name] = value

    return property(get, put)


class SolverFBA(SolverBase):
    """Fast But Accurate projective-dynamics solver for cloth and soft bodies.

    .. experimental::

        :class:`SolverFBA`, its constructor options and its solver-specific
        attributes may change without the normal deprecation period. It
        simulates models with one world and no rigid bodies.

    A step depends only on its input state, contacts and control, and on the
    pin targets last set: the contact multipliers and weights start at zero
    in every step, so the solver keeps no history for :meth:`reset` to
    discard.

    With :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F`
    requested, a step writes row ``rigid_contact_max + i`` for each particle
    contact ``i`` of its contacts: the force [N] on the shape by the
    particle and its torque [N·m] about the world origin, in world frame,
    from the forces the step's last solve applied.

    Implements a Projective Dynamics (PD) local-global iteration for cloth
    and tetrahedral soft bodies, with ARAP, corotational or Neo-Hookean
    stretching, isometric bending of cloth and soft pin constraints. The PD
    Hessian is prefactored once (nested dissection ordering, multifrontal
    LDLᵀ); the sparse inverse ``S = L⁻¹`` is computed exactly on the
    elimination tree and uploaded to Warp BSR matrices. The runtime linear
    solve evaluates ``A⁻¹·b = Sᵀ · D⁻¹ · S · b`` via two GPU SpMVs per
    coordinate component, avoiding the inherently sequential GPU triangular
    solve.

    Stretching models:

    - ``"arap"`` (As-Rigid-As-Possible) projects the deformation gradient
      ``F`` to the nearest rotation, its singular values clamped to 1. It
      takes no ``k_lambda``: the stretching weight below sets its stiffness.
    - ``"corotational"`` (corotational linear elasticity) solves in closed
      form on the singular values of ``F``, 2x2 for cloth and 3x3 with
      Sherman-Morrison for tetrahedra. It takes each element's ``k_mu`` and
      ``k_lambda``, the first and second Lamé parameters [Pa].
    - ``"neohookean"`` (Neo-Hookean hyperelasticity) runs L-BFGS with an
      Armijo cubic backtracking line search (memory 8, ``c_1 = 1e-4``,
      tolerance ``1e-6``, at most 50 iterations) on the singular values of
      ``F``. It takes each element's ``k_mu`` and ``k_lambda``, the first and
      second Lamé parameters [Pa].

    Corotational and Neo-Hookean take the true Lamé parameters. Projective
    dynamics drops the dual variable of the local projection, so its fixed
    point reacts to the stress at the projected deformation rather than at
    ``F`` and comes out softer than the material: with the projection weight
    ``2 * mu`` the small-strain modulus falls by ``c = 2 + nu`` for
    tetrahedra, ``nu = lam / (2 * (lam + mu))``, and by
    ``c = (4 * mu + 3 * lam) / (2 * mu + lam)`` for triangles. SolverFBA
    scales ``mu``, ``lam`` and the stretching weights of these models by
    ``c``, which restores the modulus: at every strain for
    corotational elasticity, at small strain for Neo-Hookean. The stretching
    weight of every element is ``2 * k_mu`` times its rest area or volume
    (``tri_ke`` is a triangle's ``k_mu``). ARAP is exact and unscaled.

    Bending follows :class:`~newton.solvers.SolverStyle3D`: each interior edge
    adds the isometric bending energy of Bergou et al. (2006),
    ``1/2 * edge_ke * 3 / (A0 + A1) * |sum_k q_k x_k|^2``, with the cotangent
    weights ``q`` of the edge's two triangles and their rest areas ``A0`` and
    ``A1``; ``edge_rest_angle`` sets its rest curvature. ``edge_ke`` is thus a
    bending modulus [N·m], and the bending energy of a given shape converges
    as the mesh is refined. :class:`~newton.solvers.SolverVBD` scales
    ``edge_ke`` by the edge length instead, ``1/2 * edge_ke * |e| * theta^2``
    for a dihedral angle ``theta``: at small angles the same ``edge_ke``
    bends an FBA cloth ``3 * |e| / (A0 + A1)`` times as stiffly.

    Notes:
        - Float32 ``particle_q`` output: the interior linear solver
          operates in float64, but ``state_out.particle_q`` (vec3) is
          float32. This caps achievable position precision at ~1e-7 m.

        - Stability tracks ``dt^2 * tri_ke / mass``, the standard CFL-like
          factor for implicit elastic dynamics. Realistic cloth parameters
          (tri_ke ~ 1e4 for ~10 kPa Young's modulus) keep arbitrarily
          sized grids stable with 5-10 PD iterations. Pathologically
          soft inputs (tri_ke < 200 for the default test setup) drive
          unphysical large oscillations that eventually overflow.

    See also:
        :class:`~newton.solvers.SolverStyle3D`: Newton's other PD cloth
        solver, with a different sparse-matrix layout (ELL + PCG).
    """

    SUPPORTED_OBSERVABLES = frozenset({SolverBase.ObservableKind.CONTACT_F})

    _lam_unilateral_persistent = _host_mirror("_lam_unilateral_persistent")
    _lam_coulomb_persistent = _host_mirror("_lam_coulomb_persistent")
    _omega_unilateral_persistent = _host_mirror("_omega_unilateral_persistent")
    _omega_coulomb_persistent = _host_mirror("_omega_coulomb_persistent")

    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap", "corotational", "neohookean"] = "arap",
        friction: bool = True,
        lambda_cap: float | None = None,
        shape_angular_velocity: dict[int, float] | None = None,
        nsn_linear_iterations: int | None = None,
    ) -> None:
        """
        Args:
            model: The :class:`~newton.Model` containing cloth/tet attributes to integrate.
            iterations: Number of PD local-global iterations per step. Each
                takes one semismooth Newton step of the contact multipliers,
                warm-started from the previous iteration.
            pin_stiffness: Soft-pin stiffness [N/m].
            stretching_model: The stretching model, ``"arap"``, ``"corotational"`` or ``"neohookean"``.
            friction: If ``True`` (default), contacts carry Coulomb friction;
                otherwise they only push.
            lambda_cap: Optional clamp on the magnitude of each contact
                row's multiplier, in force units [N]. The multipliers are
                forces divided by ``dt²``, so the clamp is divided by ``dt²``
                at each NSN solve. ``None`` disables clamping.
            shape_angular_velocity: Optional ``{shape_index: ω_rad_s}`` mapping.
                Each listed shape rotates about its local +Z axis (Newton's
                cylinder primitive axis) at the given angular speed.  Surfaces
                with nonzero ω contribute a tangential anchor velocity
                ``v_anchor = -ω · cross(axis_world, world_anchor - shape_center)``
                into the residual of the friction rows; the shape geometry
                itself stays static. A positive ``ω`` moves the surface along
                ``cross(normal, axis)``. Defaults to no kinematic motion.
            nsn_linear_iterations: Iteration cap of the conjugate residual
                solve in each NSN step. ``None`` solves to a relative
                residual of ``1e-10``. A capped step is inexact, and the next
                LG iteration's NSN step continues from it.
        """
        super().__init__(model)
        if model.world_count > 1:
            raise ValueError(
                f"SolverFBA simulates one world, but the model has {model.world_count}: its contact solve spans "
                "all particles of the model"
            )

        if model.body_count > 0:
            raise NotImplementedError(f"SolverFBA does not simulate rigid bodies, but the model has {model.body_count}")

        # ---- Validate ----
        if model.particle_count == 0:
            raise ValueError("SolverFBA requires deformable elements")
        if model.tri_count == 0 and model.tet_count == 0:
            raise ValueError("SolverFBA requires at least one cloth triangle or tetrahedral element")
        if stretching_model not in ("arap", "corotational", "neohookean"):
            raise NotImplementedError(
                f"stretching_model={stretching_model!r} not yet implemented; "
                "supported: 'arap', 'corotational', 'neohookean'"
            )

        if pin_stiffness <= 0:
            raise ValueError("pin_stiffness must be positive")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.iterations = int(iterations)
        self.pin_stiffness = float(pin_stiffness)
        self.stretching_model = stretching_model
        # Dropped-dual gauge of the corotational and Neo-Hookean projections (see the class notes), per element.
        self._tri_gauge = self._tet_gauge = 1.0
        if stretching_model in ("corotational", "neohookean"):
            tri_mu, tri_lam = _element_lame(model.tri_materials, model.tri_count)
            tet_mu, tet_lam = _element_lame(model.tet_materials, model.tet_count)
            # Elements with k_mu = 0, such as the surface triangles of a soft body, carry no stretching.
            mu, lam = np.concatenate([tri_mu, tet_mu]), np.concatenate([tri_lam, tet_lam])
            if (mu < 0).any() or (lam[mu > 0] <= 0).any():
                raise ValueError(f"stretching_model={stretching_model!r} needs k_lambda > 0 wherever k_mu > 0")
            with np.errstate(divide="ignore", invalid="ignore"):
                self._tri_gauge = np.where(tri_mu > 0, (4.0 * tri_mu + 3.0 * tri_lam) / (2.0 * tri_mu + tri_lam), 1.0)
                self._tet_gauge = np.where(tet_mu > 0, 2.0 + tet_lam / (2.0 * (tet_lam + tet_mu)), 1.0)
            tri_lame = np.stack([tri_mu * self._tri_gauge, tri_lam * self._tri_gauge], axis=1)
            tet_lame = np.stack([tet_mu * self._tet_gauge, tet_lam * self._tet_gauge], axis=1)
            self._tri_lame_d = wp.array(tri_lame, dtype=wp.vec2, device=model.device)
            self._tet_lame_d = wp.array(tet_lame, dtype=wp.vec2, device=model.device)
        self.friction = bool(friction)
        self.nsn_linear_iterations = None if nsn_linear_iterations is None else int(nsn_linear_iterations)
        self.lambda_cap = lambda_cap

        # Per-shape kinematic angular velocity (rad/s about local +Z).
        n_shapes = int(model.shape_count) if hasattr(model, "shape_count") and model.shape_count else 0
        self._shape_omega_h = np.zeros(max(n_shapes, 1), dtype=np.float64)
        if shape_angular_velocity is not None:
            for s_idx, omega in shape_angular_velocity.items():
                if 0 <= int(s_idx) < n_shapes:
                    self._shape_omega_h[int(s_idx)] = float(omega)
        # Device mirror consumed by the GPU contact kernels; the
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

        # Contact state (0 = no contacts registered yet).
        self._contact_count: int = 0

        # Cache ``W`` and the device buffer of ``A⁻¹ · Jᵀ`` across
        # PD outer iterations within a single ``step()`` call.  Both depend
        # only on (J, A) which are constant within a step (J is set once
        # by :meth:`_ingest_contacts`, A is fixed unless dt or the model
        # changes).  The cache is invalidated whenever either input could
        # change: by :meth:`_ingest_contacts` (J change) and by
        # :meth:`notify_model_changed` / :meth:`_setup_pd_system` (A change).
        # Sentinel: ``True`` once the device-resident ``W`` is up-to-date for
        # the current contact set, ``None`` otherwise.  No host copy of ``W``
        # is kept: the GPU NSN driver reads it on the device via
        # :meth:`FBALinearSolver.W_device_view`, which saves the host pull,
        # the largest single cost of a Schur build on large contact sets.
        self._cached_W: bool | None = None

        # λ persists across the PD outer iterations of a step: it starts at zero in every step, and λ += dλ
        # accumulates over the iterations. The host mirrors read the last step's values when first read, as their
        # step left ``_persistent_rows``.
        self._persistent_rows: int | None = None
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
        self._pin_indices_d = self._inactive_d = None

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

        # Particle-centered CSR adjacency for deterministic PD reductions.
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

        # Cache the local-phase kernel symbols on the solver so the hot path
        # avoids a per-step ``from .kernels import`` lookup. The kernels
        # module never changes after solver construction.
        from . import kernels as _K  # noqa: PLC0415

        self._k_zero_vec3 = _K.zero_vec3_kernel
        self._k_add_inertia = _K.add_inertia_to_rhs_kernel
        self._k_gather = _K.gather_per_particle_kernel
        self._k_pin = _K.project_pin_kernel
        self._k_bending = _K.project_bending_kernel
        self._k_tri_arap = _K.project_stretching_arap_kernel
        self._k_tet_arap = _K.project_stretching_arap_tet_kernel
        self._k_tri_corot = _K.project_stretching_corotational_kernel
        self._k_tet_corot = _K.project_stretching_corotational_tet_kernel
        self._k_tri_nh = _K.project_stretching_neohookean_kernel
        self._k_tet_nh = _K.project_stretching_neohookean_tet_kernel

    @property
    def _contact_v_anchor_h(self) -> np.ndarray:
        """Velocities of the contacts' anchors, read from the device, for tests and diagnostics."""
        return self._contact_v_anchor_d.numpy()[: self._contact_count]

    def _read_persistent(self) -> None:
        """Read the persistent multipliers of the last step into their host mirrors, for tests and diagnostics."""
        M, self._persistent_rows = self._persistent_rows, None
        if self.friction and hasattr(self, "_lam_coulomb_persistent_d"):
            self._lam_coulomb_persistent = self._lam_coulomb_persistent_d.numpy()[: 3 * M].astype(np.float64)
            self._omega_coulomb_persistent = self._omega_coulomb_persistent_d.numpy()[: 3 * M].astype(np.float64)
        elif hasattr(self, "_lam_unilateral_persistent_d"):
            self._lam_unilateral_persistent = self._lam_unilateral_persistent_d.numpy()[:M].astype(np.float64)
            self._omega_unilateral_persistent = self._omega_unilateral_persistent_d.numpy()[:M].astype(np.float64)

    def _contact_row_args(self) -> list:
        """Particle, normal, alpha and tangents of the deformable contacts."""
        return [
            self._contact_particle_d,
            self._contact_normal_d,
            self._contact_alpha_d,
            self._contact_tangent1_d,
            self._contact_tangent2_d,
        ]

    def _contact_offsets(self) -> list:
        """Normal and shifted tangent offsets of the deformable contacts."""
        return [
            self._contact_offset_d,
            self._contact_tangent1_offset_shifted_d,
            self._contact_tangent2_offset_shifted_d,
        ]

    def _launch_contact_residual(self, r: wp.array[wp.float64]) -> None:
        """Residual of the deformable contact rows ``[n, t1, t2]`` at the current particle positions."""
        from . import kernels as K  # noqa: PLC0415

        wp.launch(
            K.compute_contact_residual_coulomb_kernel,
            dim=self._contact_count,
            inputs=[
                *self._contact_offsets(),
                *(self._contact_alpha_d, self._contact_normal_d, self._contact_tangent1_d),
                *(self._contact_tangent2_d, self._contact_particle_d, self._x_cur),
            ],
            outputs=[r],
            device=self._device,
        )

    def _launch_contact_pene0(self, pene0: wp.array[wp.float64]) -> None:
        """Anchor projections of the deformable contact rows ``[n, t1, t2]``."""
        from . import kernels as K  # noqa: PLC0415

        wp.launch(
            K.build_coulomb_pene0_kernel,
            dim=self._contact_count,
            inputs=self._contact_offsets(),
            outputs=[pene0],
            device=self._device,
        )

    def _setup_pd_system(self, dt: float) -> None:
        """Build / rebuild the PD Hessian, factor it, and upload device data."""
        from .linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            build_pd_system,
            factorize_and_sparse_inverse,
            particle_masses,
        )

        A, meta = build_pd_system(
            self.model,
            dt=dt,
            pin_stiffness=self.pin_stiffness,
            tri_weight_scale=self._tri_gauge,
            tet_weight_scale=self._tet_gauge,
        )
        fs = factorize_and_sparse_inverse(A, self.model.particle_q.numpy())
        self._linear_solver = FBALinearSolver(fs, device=self._device)
        device = self._device
        # Particles with zero inverse mass hold at their pin targets, so a mass they keep must not pull them toward
        # their inertial predictions; inactive ones with mass hold where the input state has them, as Newton's
        # integrators hold inactive particles.
        mass, inv_mass = particle_masses(self.model)
        self._mass_d = wp.array(mass.astype(np.float32), dtype=wp.float32, device=device)
        self._inv_mass_d = wp.array(inv_mass.astype(np.float32), dtype=wp.float32, device=device)
        inactive = (self.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0
        inactive = np.flatnonzero(inactive & (self.model.particle_inv_mass.numpy() > 0.0))
        self._inactive_d = wp.array(inactive.astype(np.int32), dtype=wp.int32, device=device) if inactive.size else None
        self._meta = meta
        self._dt_setup = dt
        # A changed → any cached W / A^{-1} J^T is stale.
        self._invalidate_schur_cache()

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
            # Rest curvature magnitude per edge: ``‖q·x‖`` of the hinge at its
            # rest angle ``model.edge_rest_angle``. The local projection pulls
            # ``q·x_cur`` back to this magnitude (not all the way to a flat
            # configuration). ``q·x``
            # vanishes on a flat hinge, and folding it by ``θ`` about the shared
            # edge moves the opposite vertex v3 by ``2 h3 sin(θ / 2)``, with h3
            # its distance from the edge.
            ei = self.model.edge_indices.numpy()
            rest_angle = self.model.edge_rest_angle.numpy()[(ei[:, 0] >= 0) & (ei[:, 1] >= 0)].astype(np.float64)
            x_rest = self.model.particle_q.numpy().astype(np.float64)
            v0, v1, v3 = (x_rest[meta["edge_indices"][:, k]] for k in (0, 1, 3))
            e_len = np.maximum(np.linalg.norm(v1 - v0, axis=1), 1.0e-20)
            h3 = np.linalg.norm(np.cross(v1 - v0, v3 - v0), axis=1) / e_len
            edge_norm = np.abs(meta["edge_quad_q"][:, 3]) * 2.0 * h3 * np.abs(np.sin(0.5 * rest_angle))
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

        # Particle-centered CSR adjacency for deterministic PD reductions.
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
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance the deformables by one implicit Euler step using PD.

        Args:
            state_in: Input simulation state. Particle positions and velocities
                are read; ``particle_f`` provides external forces [N].
            state_out: Output simulation state. Particle positions and
                velocities are written; other fields are untouched.
            control: Unused: the solver has no actuators.
            contacts: The step's contacts, detected on ``state_in``, or ``None``
                to step without contacts.
            dt: Time step [s]. The first step and a change of ``dt`` factor the
                system again.
            observables: Optional observables from :meth:`observables`; see the
                class notes for :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F`.
        """
        from .kernels import (  # noqa: PLC0415
            compute_inertial_kernel,
            copy_particles_kernel,
            write_velocity_kernel,
        )

        self.validate_observables(observables, contacts)
        contact_f = None
        if observables is not None and observables.is_requested(SolverBase.ObservableKind.CONTACT_F):
            contact_f = observables.contact_f
        if self._linear_solver is None or self._dt_setup is None or abs(self._dt_setup - dt) > 1e-12:
            self._setup_pd_system(dt)

        # Ingest contact data once per step (if provided).
        if contacts is not None:
            self._ingest_contacts(contacts)
        else:
            # Drop the previous step's contacts.
            self._contact_count = 0
            self._invalidate_schur_cache()

        has_contacts = self._contact_count > 0

        # Apply kinematic anchor advancement: tangent_offset = base + dt · dot(t, v_anchor).
        # _ingest_contacts cached base offsets + v_anchor; the dt-dependent
        # shift is applied here on the device, so the step downloads no
        # tangents or anchor velocities. When no contact's shape moves,
        # v_anchor is zero and this is a no-op (offsets equal base).
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
                    self._contact_tangent1_offset_d,  # base (set in _ingest_contacts)
                    self._contact_tangent2_offset_d,  # base
                    wp.float64(dt),
                ],
                outputs=[
                    self._contact_tangent1_offset_shifted_d,
                    self._contact_tangent2_offset_shifted_d,
                ],
                device=self._device,
            )

        # λ and ω start at zero in every step, then carry across its PD outer iterations below.
        M = self._contact_count
        if M > 0:
            self._persistent_rows = None
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

        if self._inactive_d is not None:
            inputs = [self._inactive_d, state_in.particle_q]
            wp.launch(copy_particles_kernel, self._inactive_d.shape[0], inputs, [self._x_ref], device=device)

        # 1) Compute inertial prediction x_inertia.
        wp.launch(
            compute_inertial_kernel,
            dim=N,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self._inv_mass_d,
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
        # Capture the PD local phase and linear solve into a CUDA graph once per step: all energies and pins
        # are fixed by ``_setup_pd_system`` and the buffers are stable, so one ``wp.capture_launch`` per PD
        # iteration replaces ~25-30 launches.
        _use_lg_graph = wp.get_device(device).is_cuda
        _lg_graph = self._get_or_build_lg_iter_graph(dt) if _use_lg_graph else None

        for _k in range(self.iterations):
            # ---- Local phase (energy projection + RHS assembly) ----
            if _lg_graph is not None:
                # The graph replays the local phase and the linear solve; per-iteration state lives in
                # ``self._x_cur`` (the previous iteration's result) and the same buffers.
                wp.capture_launch(_lg_graph)
            else:
                self._run_pd_local_phase_eager(N, dt)

            if has_contacts:
                # --- Schur-complement NSN contact correction ---
                M = self._contact_count
                ls = self._linear_solver

                if self.friction and hasattr(self, "_contact_tangent1_d"):
                    # Frictional contacts: 3M Schur complement with Coulomb cone
                    # projection. Reuse the cached W across PD outer iters: it
                    # depends only on (J, A), which are fixed within a step.
                    if self._cached_W is None:
                        # ``download=False``: the GPU NSN driver consumes ``W``
                        # via :meth:`FBALinearSolver.W_device_view`, so the
                        # device-to-host pull at the end of ``build_schur_complement``
                        # is dead work; skipping it removes the dominant cost on
                        # large contact sets, where the download took ~70% of the
                        # Schur build.
                        ls.build_schur_complement(
                            M,
                            *self._contact_row_args(),
                            download=False,
                        )
                        self._cached_W = True
                    # Residual + pene0 + warm-start all device-resident: the PD
                    # outer iter does no host<->device transfers around the NSN
                    # call.
                    self._ensure_nsn_inner_buffers(3 * M)

                    self._launch_contact_residual(self._nsn_r_d)
                    self._launch_contact_pene0(self._nsn_pene0_d)
                    self._solve_nsn_coulomb_gpu(
                        None,
                        None,
                        None,
                        None,
                        max_iters=1,
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
                    # Apply the contact forces in combined form, with one solve over the incremented right-hand
                    # side. It runs even when the forces vanish: the solve costs less than reading them back to
                    # check.
                    ls.apply_lambda_correction_combined(self._nsn_lam_apply_d, self._rhs, self._x_cur)
                else:
                    # Frictionless contacts: M Schur complement, unilateral (λ ≥ 0) only.
                    if self._cached_W is None:
                        # ``download=False``: see the frictional branch above; the
                        # GPU NSN driver reads ``W`` from device via
                        # :meth:`FBALinearSolver.W_device_view`.
                        ls.build_schur_complement(
                            M,
                            self._contact_particle_d,
                            self._contact_normal_d,
                            self._contact_alpha_d,
                            download=False,
                        )
                        self._cached_W = True
                    # Residual + pene0 + warm-start all device-resident.
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
                    # Frictionless pene0 == contact_offset (no kinematic shift).
                    # Reuse the device buffer directly.
                    self._solve_nsn_unilateral_gpu(
                        None,
                        None,
                        None,
                        max_iters=1,
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
                    # Apply the contact forces in combined form, as the frictional branch above does.
                    ls.apply_lambda_correction_combined(self._nsn_lam_apply_d, self._rhs, self._x_cur)

        # 3) Write velocity and update state_out.
        if self._inactive_d is not None:
            inputs = [self._inactive_d, state_in.particle_q]
            wp.launch(copy_particles_kernel, self._inactive_d.shape[0], inputs, [self._x_cur], device=device)
        wp.copy(state_out.particle_q, self._x_cur)
        wp.launch(
            write_velocity_kernel,
            dim=N,
            inputs=[state_in.particle_q, self._x_cur, self._inv_mass_d, dt],
            outputs=[state_out.particle_qd],
            device=device,
        )

        if contact_f is not None:
            self._write_contact_forces(contact_f, contacts)

        # The host mirrors of the persistent buffers read them when first read, so the step ends without waiting
        # for the device. When M==0 the host arrays were left at None / unchanged.
        if has_contacts:
            self._persistent_rows = M

    def _run_pd_local_phase_eager(self, N: int, dt: float) -> None:
        """Issue the PD local-phase launches on the active stream.

        Runs the body of one PD outer iter (RHS zero + inertia + pin +
        stretching + bending, followed by the linear solve).
        :meth:`step` calls it eagerly, and :meth:`_get_or_build_lg_iter_graph`
        captures it into a CUDA graph, so both run the same launch sequence.
        All branches use cached kernel refs and device arrays, with no host
        syncs and no allocations, so the function is safe to call inside
        ``wp.ScopedCapture``.

        Args:
            N: Particle count (used for the per-particle kernel launch
                dimensions).  Equal to ``model.particle_count``.
            dt: Timestep [s], passed into the inertia kernel.
        """
        model = self.model
        device = self._device
        # Zero RHS.
        wp.launch(self._k_zero_vec3, dim=N, inputs=[self._rhs], device=device)
        # Inertia term.
        wp.launch(
            self._k_add_inertia,
            dim=N,
            inputs=[self._x_inertia, self._mass_d, dt],
            outputs=[self._rhs],
            device=device,
        )
        # Pin projection.
        if self._pin_indices_d is not None:
            wp.launch(
                self._k_pin,
                dim=self._pin_indices_d.shape[0],
                inputs=[self._pin_indices_d, self._x_ref, self.pin_stiffness],
                outputs=[self._rhs],
                device=device,
            )
        # Tri (cloth) stretching projection: deterministic compute+gather.
        if self._tri_indices_d is not None and model.tri_count > 0:
            if self.stretching_model == "arap":
                wp.launch(
                    self._k_tri_arap,
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
                    self._k_gather,
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
                    self._k_tri_corot,
                    dim=model.tri_count,
                    inputs=[
                        self._x_cur,
                        self._tri_indices_d,
                        self._tri_rest_inv_d,
                        self._tri_weight_d,
                        self._tri_lame_d,
                    ],
                    outputs=[self._tri_contrib_d],
                    device=device,
                )
                wp.launch(
                    self._k_gather,
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
                    self._k_tri_nh,
                    dim=model.tri_count,
                    inputs=[
                        self._x_cur,
                        self._tri_indices_d,
                        self._tri_rest_inv_d,
                        self._tri_weight_d,
                        self._tri_lame_d,
                    ],
                    outputs=[self._tri_contrib_d],
                    device=device,
                )
                wp.launch(
                    self._k_gather,
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
        # Bending projection: deterministic (compute → gather) scatter.
        if self._edge_indices_d is not None:
            wp.launch(
                self._k_bending,
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
                self._k_gather,
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
                wp.launch(
                    self._k_tet_arap,
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
                    self._k_gather,
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
                wp.launch(
                    self._k_tet_corot,
                    dim=model.tet_count,
                    inputs=[
                        self._x_cur,
                        self._tet_indices_d,
                        self._tet_rest_inv_d,
                        self._tet_weight_d,
                        self._tet_lame_d,
                    ],
                    outputs=[self._tet_contrib_d],
                    device=device,
                )
                wp.launch(
                    self._k_gather,
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
                wp.launch(
                    self._k_tet_nh,
                    dim=model.tet_count,
                    inputs=[
                        self._x_cur,
                        self._tet_indices_d,
                        self._tet_rest_inv_d,
                        self._tet_weight_d,
                        self._tet_lame_d,
                    ],
                    outputs=[self._tet_contrib_d],
                    device=device,
                )
                wp.launch(
                    self._k_gather,
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
        # Global linear solve: x_unc = A^-1 . rhs.
        self._linear_solver.solve(self._rhs, self._x_cur)

    def _get_or_build_lg_iter_graph(self, dt: float) -> wp.Graph | None:
        """Look up (or capture lazily) the PD local-phase + linear-solve graph.

        Cached keyed by the active ``dt`` and the linear solver instance.
        Both change rarely (PD setup rebuild flips ``_dt_setup`` and
        nulls ``_linear_solver``).  The graph captures buffer pointers
        (``_rhs``, ``_x_cur``, ``_x_inertia``, ``_tri_contrib_d``, ...) but
        not their contents; replays observe whatever the buffers hold at
        replay time, which is exactly the next PD iter's input.

        Returns ``None`` on capture failure so the eager path remains the
        fallback.
        """
        device = self._device
        cache_key = (float(dt), id(self._linear_solver))
        cached = getattr(self, "_lg_graph_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        # Re-record on shape / setup change.
        N = self.model.particle_count
        try:
            with wp.ScopedCapture(device=device) as cap:
                self._run_pd_local_phase_eager(N, dt)
            graph = cap.graph
        except RuntimeError:
            graph = None
        self._lg_graph_cache = (cache_key, graph)
        return graph

    def _write_contact_forces(self, contact_f: wp.array, contacts: Contacts) -> None:
        """Fill the rows of ``contact_f`` for the step's particle contacts and zero the others."""
        from .kernels import soft_contact_force_kernel  # noqa: PLC0415

        contact_f.zero_()
        M, device = self._contact_count, self._device
        if M == 0:
            return
        inputs = [self._contact_record_d, self._contact_normal_d, self._contact_tangent1_d]
        inputs += [self._contact_tangent2_d, 3 if self.friction else 1, self._nsn_lam_apply_d]
        inputs += [self._contact_world_anchor_d, contacts.rigid_contact_max]
        wp.launch(soft_contact_force_kernel, M, inputs, [contact_f], device=device)

    def set_pin_targets(self, target_positions) -> None:
        """Update reference positions for pinned particles (dynamic pin).

        For pinned particles (``inv_mass == 0``), the soft Pin energy pulls them
        toward ``target_positions[i]``; they start at their model positions.
        Inactive particles with mass hold where the input state has them
        instead. For free particles the value is unused but the array must
        cover all particles.

        Args:
            target_positions: Per-particle target positions [particle_count, 3].
                Accepts either a :class:`warp.array` (``dtype=wp.vec3``) or a
                NumPy array of shape ``(particle_count, 3)`` in float32.
        """
        if isinstance(target_positions, wp.array):
            self._x_ref.assign(target_positions)
        else:
            self._x_ref.assign(np.asarray(target_positions, dtype=np.float32))

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        # On any geometry change, force a full re-setup at the next step.
        if flags & ModelFlags.SHAPE_PROPERTIES:
            self._linear_solver = None
            self._dt_setup = None
            self._invalidate_schur_cache()

    def _invalidate_schur_cache(self) -> None:
        """Drop the per-step ``W`` cache.

        Called whenever either input (``J`` from
        :meth:`_ingest_contacts` or the factorization of ``A`` from
        :meth:`_setup_pd_system` / :meth:`notify_model_changed`) might
        change so the next :meth:`step` rebuilds the Schur complement.

        The persistent λ/ω warm-start buffers
        (``_lam_unilateral_persistent``, ``_lam_coulomb_persistent``,
        ``_omega_unilateral_persistent``, ``_omega_coulomb_persistent``)
        are intentionally left untouched here. Per-step entry in
        :meth:`step` reallocates them when the contact count ``M``
        changes and otherwise zero-fills them in place, a per-step concern
        that is decoupled from Schur factor invalidation.
        """
        self._cached_W = None

    # ------------------------------------------------------------------
    # Contact state (allocated lazily on first call).
    # ------------------------------------------------------------------

    def _ensure_contact_buffers(self, max_contacts: int) -> None:
        """Lazily allocate per-step contact buffers sized to ``max_contacts``."""
        if hasattr(self, "_contact_buf_max") and self._contact_buf_max >= max_contacts:
            return
        device = self._device
        cap = max(max_contacts, 16)
        self._contact_buf_max = cap
        self._contact_particle_d = wp.empty(cap, dtype=wp.int32, device=device)
        # The index in Contacts of the particle contact each row comes from.
        self._contact_record_d = wp.empty(cap, dtype=wp.int32, device=device)
        self._contact_normal_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_alpha_d = wp.empty(cap, dtype=wp.float32, device=device)
        self._contact_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
        # Friction: tangent directions and per-contact friction mu.
        self._contact_tangent1_d = wp.empty(cap, dtype=wp.vec3, device=device)
        self._contact_tangent2_d = wp.empty(cap, dtype=wp.vec3, device=device)
        # Device-resident scratch buffers consumed by the
        # ``_ingest_contacts`` GPU kernels (world anchor, tangent offsets,
        # spinning flag, v_anchor). Sized to ``cap`` once here so kernel
        # launches in :meth:`_ingest_contacts` never re-allocate.
        self._contact_shape_d = wp.empty(cap, dtype=wp.int32, device=device)
        self._contact_body_pos_d = wp.empty(cap, dtype=wp.vec3, device=device)
        # Static shapes store their contact points in world frame: the points are the contacts' anchors.
        self._contact_world_anchor_d = self._contact_body_pos_d
        self._contact_is_spinning_d = wp.empty(cap, dtype=wp.int32, device=device)
        self._contact_tangent1_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
        self._contact_tangent2_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
        self._contact_v_anchor_d = wp.empty(cap, dtype=wp.vec3d, device=device)
        # Per-step kinematic-shifted tangent offsets (base + dt·t·v_anchor).
        # The frictional residual and interleaved pene0 read these directly,
        # so the ``dt`` shift does not round-trip through host arrays.
        self._contact_tangent1_offset_shifted_d = wp.empty(cap, dtype=wp.float64, device=device)
        self._contact_tangent2_offset_shifted_d = wp.empty(cap, dtype=wp.float64, device=device)
        # Per-contact friction mu, device-resident fp64. Populated in
        # :meth:`_ingest_contacts`, which avoids a host->device upload in
        # every PD iteration of the frictional NSN driver.
        self._contact_mu_d = wp.empty(cap, dtype=wp.float64, device=device)

    def _ingest_contacts(self, contacts: Contacts) -> None:
        """Ingest the active particle-vs-shape contact set for the next step.

        Reads the soft contact fields from ``contacts`` and builds device-side
        Jacobian arrays (particle index, normal, alpha, signed-distance offset)
        that the Schur-complement path inside :meth:`step` will consume.

        Particle-vs-shape contacts have ``alpha = 1.0``.
        The signed-distance offset is ``dot(normal, anchor)`` with the anchor
        ``soft_contact_body_pos``, which is in world frame for the static
        shapes (``X_wb = identity``, so ``X_ws = X_bs``, and the collision
        pipeline stores ``wp.transform_point(X_bs, x_local)``).

        Args:
            contacts: :class:`~newton.Contacts` populated by a prior
                :meth:`~newton.CollisionPipeline.collide` call.
        """
        contacts._assert_particle_only_soft_contacts("SolverFBA")
        # Contact data is about to change → drop the Schur cache so the
        # next ``step()`` rebuilds ``W`` / ``A⁻¹·Jᵀ`` with fresh J.
        self._invalidate_schur_cache()

        M_raw = int(contacts.soft_contact_count.numpy()[0])
        if M_raw == 0:
            self._contact_count = 0
            return

        # Pull contact data to host for the lexsort, for a deterministic
        # contact order (M is small so this is fast).
        # The collision pipeline writes contacts to slots assigned by
        # ``wp.atomic_add(soft_contact_count, 0, 1)`` so per-frame order
        # depends on GPU thread scheduling. The order of the rows sets the
        # order of the NSN's sums, hence we sort, so that a step is deterministic.
        particle_h = contacts.soft_contact_particle.numpy()[:M_raw]
        shape_h = contacts.soft_contact_shape.numpy()[:M_raw]
        # soft_contact_body_pos of a static shape is the contact point in world frame
        # (X_wb = identity, so X_ws = X_bs, and body_pos = X_bs * x_local).
        body_pos_h = contacts.soft_contact_body_pos.numpy()[:M_raw]
        normal_h = contacts.soft_contact_normal.numpy()[:M_raw]  # world frame

        # Filter out sentinel entries (particle == -1).
        valid_mask = particle_h >= 0
        particle_h = particle_h[valid_mask]
        shape_h = shape_h[valid_mask]
        body_pos_h = body_pos_h[valid_mask]
        normal_h = normal_h[valid_mask]
        record_h = np.flatnonzero(valid_mask).astype(np.int32)
        M = int(particle_h.shape[0])

        if M == 0:
            self._contact_count = 0
            return

        # Deterministic contact ordering (host-side lexsort, see comment above).
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
        record_h = record_h[order]

        alpha_h = np.ones(M, dtype=np.float32)
        self._ensure_contact_buffers(M)

        # Per-contact bookkeeping (n·anchor offset, tangent basis, t·anchor
        # tangent offsets, v_anchor for rolling cylinders) runs on GPU.
        from .kernels import (  # noqa: PLC0415
            compute_normal_offset_kernel,
            compute_tangent_basis_kernel,
            compute_tangent_offsets_kernel,
            compute_v_anchor_kernel,
        )

        model = self.model
        device = self._device

        # Upload (sorted) compact contact arrays to device once.
        self._contact_particle_d.assign(particle_h.astype(np.int32))
        self._contact_normal_d.assign(normal_h.astype(np.float32))
        self._contact_alpha_d.assign(alpha_h)
        self._contact_shape_d.assign(shape_h.astype(np.int32))
        self._contact_body_pos_d.assign(body_pos_h.astype(np.float32))
        self._contact_record_d.assign(record_h)

        # Resolve model fields used by the GPU kernels.  When a field is
        # absent or empty we still pass a non-null dummy buffer through
        # the kernel and gate the lookup with a ``has_*`` flag.
        shape_transform_arr = (
            model.shape_transform if hasattr(model, "shape_transform") and model.shape_transform is not None else None
        )
        has_shape_transform = wp.int32(1 if shape_transform_arr is not None and shape_transform_arr.size > 0 else 0)
        if shape_transform_arr is None or shape_transform_arr.size == 0:
            shape_transform_arr = wp.empty(1, dtype=wp.transform, device=device)

        has_shape_omega = wp.int32(1 if self._shape_omega_d.size > 0 else 0)

        # Normal offset n·anchor.
        wp.launch(
            compute_normal_offset_kernel,
            dim=M,
            inputs=[self._contact_normal_d, self._contact_world_anchor_d],
            outputs=[self._contact_offset_d],
            device=device,
        )
        # Particle contacts stop the particle radius plus the shape margin outside the shape surface, as Newton's
        # contact margins place them.
        from .kernels import add_contact_thickness_kernel  # noqa: PLC0415

        inputs = [self._contact_particle_d, self._contact_shape_d, model.particle_radius, model.shape_margin]
        wp.launch(add_contact_thickness_kernel, M, inputs, [self._contact_offset_d], device=device)

        self._contact_count = M

        # Friction: tangent basis, tangent offsets, kinematic anchor velocity.
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

            # Per-contact friction coefficients from the shape and particle materials.
            particle_mu = float(getattr(model, "particle_mu", 0.5))
            self._contact_mu_d.zero_()
            from .kernels import contact_mu_kernel  # noqa: PLC0415

            shape_mu = getattr(model, "shape_material_mu", None)
            if shape_mu is None:
                shape_mu = wp.empty(0, dtype=wp.float32, device=device)
            inputs = [self._contact_shape_d, shape_mu, wp.float64(particle_mu)]
            wp.launch(contact_mu_kernel, M, inputs, [self._contact_mu_d], device=device)

    # ------------------------------------------------------------------
    # Schur-complement NSN helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # NSN inner-loop GPU drivers
    # ------------------------------------------------------------------
    #
    # The drivers below run the iterations of ``solve_nsn_unilateral`` and ``solve_nsn_coulomb``
    # (``newton/tests/fba/_reference.py``) on the device: residual → penetration → FB rows → Schur build →
    # linear solve → lambda update → box clamp → cap clip. They also accept host-side ``W`` / ``r`` /
    # ``lam_init`` / ``omega_init`` arguments, with which the tests compare them to the host reference, and
    # return host-side ``(lam, omega, lam_apply)`` unless told to skip the download.

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
                max_iter=max(1000, 2 * max_n) if self.nsn_linear_iterations is None else self.nsn_linear_iterations,
            )

    def _ensure_nsn_inner_buffers(self, n_rows: int) -> None:
        """Allocate or resize the device-resident NSN inner-loop scratch.

        Buffers grow with a 1.5x headroom factor so a slowly climbing contact
        count does not realloc the dense ``(cap, cap)`` matrices every frame.
        Contact counts can ramp by tens of contacts between frames; a
        per-frame realloc otherwise pegs the device allocator and stalls.

        Args:
            n_rows: Minimum capacity needed for the buffers.
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
        """GPU version of ``solve_nsn_unilateral`` (``newton/tests/fba/_reference.py``), the frictionless NSN inner.

        Sequence per FB-Newton iter (mirrors the CPU loop body 1:1):

            1. ``penetration = -r + dt²·W·(ω·λ)``
            2. ``(ω, c, h) = fb_unilateral_row_wp(...)`` per row
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
            max_iters: FB-Newton steps; the solver takes one per local-global iteration.
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
                arrays: the device-resident ``_nsn_lam_d`` / ``_nsn_omega_d``
                / ``_nsn_lam_apply_d`` buffers hold the authoritative result
                and the caller is expected to consume them on device.
            m_rows: Override row count when ``r_device`` is supplied (needed
                because device arrays carry capacity, not active size).
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
        # When the caller supplies a device-resident W view
        # (the hot solver path), use it directly: the kernels read only the
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
        # Residual / pene0: prefer device buffers when supplied, which avoids
        # a ``cap*8`` host upload in every PD outer iter.
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
        # host arrays, as the tests pass them.
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
        # Pad M up to the next multiple of the NSN matvec tile width so the
        # tiled ``compute_penetration_kernel`` / ``compute_nsn_rhs_kernel``
        # K-loop runs on whole tiles (the bounds-checked ``tile_load`` zero-
        # pads the partial last tile).
        tile_k = int(K._NSN_MATVEC_TILE_K)
        n_rows_pad_w = wp.int32(((M + tile_k - 1) // tile_k) * tile_k)
        block_dim_nsn = int(K._NSN_MATVEC_BLOCK_DIM)

        # precond = dt² · max(W_ii, 1e-12) (frictionless rows).
        wp.launch(
            K.compute_precond_unilateral_kernel,
            dim=M,
            inputs=[W_d, dt_w],
            outputs=[self._nsn_precond_d],
            device=device,
        )

        for _ in range(int(max_iters)):
            # penetration = -r + dt²·W·(ω·λ).
            wp.launch_tiled(
                K.compute_penetration_kernel,
                dim=M,
                inputs=[
                    r_d,
                    W_d,
                    self._nsn_omega_d,
                    self._nsn_lam_d,
                    n_rows_w,
                    n_rows_pad_w,
                    dt_w,
                ],
                outputs=[self._nsn_penetration_d],
                block_dim=block_dim_nsn,
                device=device,
            )

            # FB rows (unilateral) overwrite omega/compliance/h.
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

            # rhs = (1/dt²) · (h - ω · J·x_corrected).
            wp.launch_tiled(
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
                    n_rows_pad_w,
                    dt_w,
                ],
                outputs=[self._nsn_rhs_d],
                block_dim=block_dim_nsn,
                device=device,
            )

            # PCR Schur solve: ``A_schur · dlam = rhs`` on the active
            # M-sized slice. Slicing a device array gives a view that shares
            # storage with the pre-allocated buffer, so the PCR solver writes
            # pre-allocated buffers (no copy) so the PCR solver writes
            # ``dlam`` straight into ``_nsn_dlam_d``.
            a_view = self._nsn_a_schur_d[:M, :M]
            rhs_view = self._nsn_rhs_d[:M]
            dlam_view = self._nsn_dlam_d[:M]
            self._pcr_solver.solve(a_view, rhs_view, dlam_view)

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
        """GPU version of ``solve_nsn_coulomb`` (``newton/tests/fba/_reference.py``), the frictional NSN inner.

        Identical structure to :meth:`_solve_nsn_unilateral_gpu` with row
        count ``3M`` and the frictional FB block replacing the unilateral
        per-row evaluation. After the Newton update,
        :func:`coulomb_box_clamp_kernel` clamps the tangent multipliers of
        each contact to ``±mu`` times its normal multiplier, and zeroes
        them where the normal multiplier is not positive.

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
                of ``(lam, omega, lam_apply)``. The caller is expected to
                read the device buffers directly.
            m_contacts: Override active contact count when only device
                arrays are provided.

        Returns:
            ``(lam, omega, lam_apply)`` host fp64 arrays of shape ``(3M,)``.
            All three are empty when ``skip_download=True``.
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
        n_rows = 3 * M
        if n_rows == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        device = self._device

        self._ensure_nsn_inner_buffers(n_rows)
        self._ensure_pcr_solver(n_rows)
        cap = self._nsn_inner_n

        # Friction mu is a separate per-contact (size-M) array.
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

        # Prefer the caller-supplied device W when given, which avoids a
        # cap*cap fp64 upload in every PD outer iter.
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
        tile_k = int(K._NSN_MATVEC_TILE_K)
        n_rows_pad_w = wp.int32(((n_rows + tile_k - 1) // tile_k) * tile_k)
        block_dim_nsn = int(K._NSN_MATVEC_BLOCK_DIM)

        # Frictional precond: dt²·W_ii for normal rows; dt·W_ii for tangents.
        wp.launch(
            K.compute_precond_coulomb_kernel,
            dim=n_rows,
            inputs=[W_d, dt_w],
            outputs=[self._nsn_precond_d],
            device=device,
        )

        for _ in range(int(max_iters)):
            wp.launch_tiled(
                K.compute_penetration_kernel,
                dim=n_rows,
                inputs=[
                    r_d,
                    W_d,
                    self._nsn_omega_d,
                    self._nsn_lam_d,
                    n_rows_w,
                    n_rows_pad_w,
                    dt_w,
                ],
                outputs=[self._nsn_penetration_d],
                block_dim=block_dim_nsn,
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
            wp.launch_tiled(
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
                    n_rows_pad_w,
                    dt_w,
                ],
                outputs=[self._nsn_rhs_d],
                block_dim=block_dim_nsn,
                device=device,
            )

            a_view = self._nsn_a_schur_d[:n_rows, :n_rows]
            rhs_view = self._nsn_rhs_d[:n_rows]
            dlam_view = self._nsn_dlam_d[:n_rows]
            self._pcr_solver.solve(a_view, rhs_view, dlam_view)

            wp.launch(
                K.axpy_lambda_kernel,
                dim=n_rows,
                inputs=[self._nsn_dlam_d],
                outputs=[self._nsn_lam_d],
                device=device,
            )

            # Per-contact box clamp of the tangent multipliers; no friction without a positive normal multiplier.
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
                dim=3 * M,
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
