# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase
from .kernels import host_head

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _element_lame(materials: wp.array | None, count: int) -> tuple[np.ndarray, np.ndarray]:
    """First and second Lamé parameters [Pa] of each element, from its material row."""
    if count == 0:
        return np.zeros(0), np.zeros(0)
    rows = materials.numpy().astype(np.float64)
    return rows[:, 0], rows[:, 1]


def _damping_coefficients(name: str, value, count: int) -> np.ndarray | None:
    """One Rayleigh damping coefficient per particle from a scalar or a per-particle array, or ``None`` when all
    are zero."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(count, float(values))
    if values.shape != (count,):
        raise ValueError(f"{name} takes a scalar or one value per particle ({count}), got shape {values.shape}")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"{name} must be finite and >= 0")
    return values if values.any() else None


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
    from the forces the step's last solve applied. The other distance minima
    of a particle on a mesh add to its contact's row. Face contacts and self
    contacts have no rows in the contacts and are not exported.

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
      ``F`` to the nearest rotation, its singular values set to 1. It
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

    Rayleigh damping, ``rayleigh_alpha`` and ``rayleigh_beta``, applies the
    forces ``-(alpha M + beta K) v`` by implicit Euler, with ``M`` the
    particle masses and ``K`` the stretching and bending part of the PD
    Hessian, the weights above included: the Hessian gains
    ``(alpha M + beta K) / dt``, and each step's right-hand side
    ``(alpha M + beta K) x_prev / dt`` for the step's start positions
    ``x_prev``.

    Notes:
        - Float32 ``particle_q`` output: the interior linear solver
          operates in float64, but ``state_out.particle_q`` (vec3) is
          float32. This caps the precision of positions at about 1e-7 of
          their magnitude.

        - Stability tracks ``dt^2 * tri_ke / mass``, the standard CFL-like
          factor for implicit elastic dynamics. Realistic cloth parameters
          (tri_ke ~ 1e4 for ~10 kPa Young's modulus) keep arbitrarily
          sized grids stable with 5-10 PD iterations. Pathologically
          soft inputs drive
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
        nsn_schur_mode: Literal["full", "lite"] = "full",
        contact_regularization: float = 1.0e-2,
        face_contacts: bool = False,
        face_contact_edge_radius: float = 3.0e-3,
        nsn_linear_iterations: int | None = None,
        nsn_lite_accumulate: bool = False,
        face_contact_gap: float | None = None,
        particle_enable_self_contact: bool = False,
        particle_self_contact_mode: Literal["vertex_face", "vertex_vertex"] = "vertex_face",
        particle_self_contact_margin: float = 2.0e-3,
        particle_self_contact_gap: float | None = None,
        particle_topological_contact_filter_threshold: int = 2,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        rayleigh_alpha: float | ArrayLike = 0.0,
        rayleigh_beta: float | ArrayLike = 0.0,
        rayleigh_corotated: bool = False,
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
                at each NSN solve, and the tangent multipliers are clamped
                again into the cone of the capped normal one. A row applies
                its multiplier weighted by its Fischer-Burmeister weight in
                [0, 2], so its force can reach twice the cap. ``None``
                disables clamping.
            shape_angular_velocity: Optional ``{shape_index: ω}`` mapping of
                angular speeds [rad/s], right-handed about each listed shape's
                local +Z axis (Newton's cylinder primitive axis). The surface
                velocity ``ω · cross(axis_world, world_anchor - shape_center)``
                enters the residual of the friction rows; the shape geometry
                itself stays static. Defaults to no kinematic motion.
            nsn_schur_mode: Contact operator ``W`` of the NSN steps.
                ``"full"`` (default) builds ``J A⁻¹ Jᵀ`` from the prefactored
                system matrix ``A``. ``"lite"`` replaces ``A⁻¹`` by
                ``dt² M⁻¹``, the inverse of the mass term ``M / dt²`` of ``A``
                (``(1 + alpha dt) M / dt²`` with ``rayleigh_alpha``):
                the rows of different particles decouple, unless self contacts
                couple them. ``W`` is built, sparse, only for coupled rows or
                more than four contacts on a particle. Each local-global
                iteration solves for the contact forces whose motion,
                predicted through ``W``, closes the gaps of its contact-free
                positions. The lite ``W`` overrates that motion
                where the material spreads a force to the particle's
                neighbors, so a pressed contact settles inside its shape, by
                up to ``dt² F / m`` for a normal force ``F`` [N] on a particle
                of mass ``m`` [kg], and more iterations do not reduce that.
                Face contacts need ``"full"``.
            contact_regularization: Relative regularization ``gamma`` of the
                contact rows in the NSN Newton steps that involve face or self
                contacts, whose rows span several particles: each step solves
                with ``A_cc + gamma diag(A_cc)``. Face contacts act on
                particles that also carry particle contacts, which makes the
                Newton matrix nearly singular, and an unregularized step
                answers a sub-micrometer inconsistency there with forces far
                above the weights involved. The term bounds the gain along
                contact directions by ``1 / gamma`` and leaves the
                complementarity conditions unchanged. Steps with particle
                contacts only are not regularized, and neither is the sparse
                self-contact solve of ``nsn_schur_mode="lite"``.
            face_contacts: If ``True``, a triangle next to a particle contact
                also gets a contact at its deepest point inside the shape when
                that point lies between its particles, so shape edges and
                corners cannot pass through the triangle. Needs
                ``friction=True`` and ``nsn_schur_mode="full"``. Mesh
                and convex hull shapes must be closed. Warp's sign query
                treats faces within 1e-3 of the mean edge length as equally
                close, so parts thinner than that need a finer mesh. A
                particle contact on a mesh also gets a contact at each other
                local minimum of the particle's distance to the mesh inside a
                face or on an edge, within the particle radius plus the shape
                margin plus ``face_contact_gap``: in a concave corner, such
                as a V notch, the closest point alone holds the particle
                against one side, and it slides through the other.
            face_contact_edge_radius: Radius [m] by which box and cylinder
                edges are rounded in the signed distance the face contacts
                minimize, at most half the shape's smallest half extent (of a
                cylinder, its radius or half height). Inside a
                sharp edge the gradient jumps between the two faces, so a
                triangle point that sinks into the edge flips its normal and
                pushes the cloth sideways; the rounding keeps the normal
                continuous.
            nsn_linear_iterations: Iteration cap of the conjugate residual
                solve in each NSN step. ``None`` solves to a relative
                residual of ``1e-10``. A capped step is inexact, and the next
                LG iteration's NSN step continues from it.
            nsn_lite_accumulate: If ``True``, the steps that take the sparse
                lite NSN, those with self contacts or with more than four
                contacts on a particle, keep the contact forces of the step's
                earlier iterations on all their rows: each local-global
                iteration moves its contact-free positions by the displacement
                that those forces caused through ``A``, and its NSN step
                models only their change through the lite ``W``. The
                iterations then approach the forces whose exact motion closes
                the gaps, rather than those that the lite ``W`` predicts would,
                so stacked layers that press on each other keep their margin
                where the material spreads the force. Needs
                ``nsn_schur_mode="lite"`` and ``friction=True``.
            face_contact_gap: Distance [m] within which every triangle is
                also tested for face contacts.
                ``None`` tests only the triangles next to particle contacts,
                which misses a shape tip or edge poking into a triangle whose
                particles all lie beyond the reach of the particle contacts.
            particle_enable_self_contact: If ``True``, the triangles collide with
                themselves through contacts found once per step on its start
                positions, solved with the other contacts. Needs
                ``friction=True``; friction takes ``model.particle_mu``.
                Solids collide through their surface triangles: the interior
                particles of tetrahedral meshes take no vertex-face contact.
            particle_self_contact_mode: ``"vertex_face"`` keeps each particle
                ``particle_self_contact_margin`` from its nearest triangle, which
                covers crossing edges only when the margin reaches half of the
                longest edge. ``"vertex_vertex"`` keeps particles that margin
                apart, which covers the surface when it reaches the longest edge
                (twice the largest circumradius of acute triangles), and leaves
                a bumpy contact surface.
            particle_self_contact_margin: Self-contact distance [m]: the
                distance at which a particle and a triangle or particle stop.
            particle_self_contact_gap: Additional detection distance [m]: self
                contacts are found within the margin plus the gap, which bounds
                the relative motion per step that cannot pass a contact.
                ``None`` takes the margin.
            particle_topological_contact_filter_threshold: Mesh edges within
                which particles do not collide with each other or with
                triangles that have such a particle.
            particle_rest_shape_contact_exclusion_radius: Distance [m] in
                ``model.particle_q`` within which a particle does not collide
                with a triangle or particle, so that a fold can close only
                between parts that start apart.
            rayleigh_alpha: Mass-proportional Rayleigh damping coefficient
                [1/s]: a particle of mass ``m`` moving at ``v`` feels the force
                ``-alpha m v``. A scalar for every particle, or one value per
                particle.
            rayleigh_beta: Stiffness-proportional Rayleigh damping coefficient
                [s]: the particles feel the forces ``-beta K v``, with ``K`` the
                stretching and bending part of the solver's constant system
                matrix. ``K`` measures every change of an element's
                deformation gradient, rotations included, so ``beta`` also
                damps elements that turn, and a slender part's bending more
                than its bending stiffness alone suggests. A scalar for every
                element, or one value per particle, of which an element takes
                the mean. Both coefficients are integrated implicitly with the
                rest of the step: the damping matrix enters the prefactored
                system matrix, and its product with the step's start positions
                the right-hand side, so a change of ``dt`` refactors both
                together.
            rayleigh_corotated: Damp the stretch rate of the triangles in
                place of the rate of their deformation gradient, so that
                ``rayleigh_beta`` spares their rotation: a flap falls and a
                bulge flattens as fast as without damping, while stretch
                waves stay damped. The system matrix is unchanged; each
                iteration's right-hand side turns the triangle's stretch at
                the step's start with its current rotation. Bending and
                tetrahedra keep the constant damping. Needs
                ``stretching_model="arap"`` and a nonzero ``rayleigh_beta``.
        """
        super().__init__(model)
        if model.world_count > 1:
            raise ValueError(
                f"SolverFBA simulates one world, but the model has {model.world_count}: its contact solve and "
                "self contacts span all particles of the model"
            )

        if model.body_count > 0:
            raise NotImplementedError(f"SolverFBA does not simulate rigid bodies, but the model has {model.body_count}")

        # ---- Validate ----
        if model.particle_count == 0:
            raise ValueError("SolverFBA requires deformable elements")
        if model.tri_count == 0 and model.tet_count == 0:
            raise ValueError("SolverFBA requires at least one cloth triangle or tetrahedral element")
        if stretching_model not in ("arap", "corotational", "neohookean"):
            raise ValueError(
                f"stretching_model={stretching_model!r} not supported; choose 'arap', 'corotational' or 'neohookean'"
            )
        if nsn_schur_mode not in ("full", "lite"):
            raise ValueError(f"nsn_schur_mode={nsn_schur_mode!r} not supported; choose 'full' or 'lite'")
        self.nsn_schur_mode = nsn_schur_mode

        if pin_stiffness <= 0:
            raise ValueError("pin_stiffness must be positive")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.iterations = int(iterations)
        self.pin_stiffness = float(pin_stiffness)
        # Rayleigh damping coefficients per particle, None when zero.
        self._rayleigh_alpha = _damping_coefficients("rayleigh_alpha", rayleigh_alpha, model.particle_count)
        self._rayleigh_beta = _damping_coefficients("rayleigh_beta", rayleigh_beta, model.particle_count)
        if rayleigh_corotated and stretching_model != "arap":
            raise ValueError("rayleigh_corotated needs stretching_model='arap'")
        if rayleigh_corotated and self._rayleigh_beta is None:
            raise ValueError("rayleigh_corotated needs a nonzero rayleigh_beta")
        self.rayleigh_corotated = bool(rayleigh_corotated)
        self.stretching_model = stretching_model
        self._update_materials()
        self.friction = bool(friction)
        if nsn_linear_iterations is not None and nsn_linear_iterations < 1:
            raise ValueError(f"nsn_linear_iterations must be >= 1 or None, got {nsn_linear_iterations}")
        self.nsn_linear_iterations = None if nsn_linear_iterations is None else int(nsn_linear_iterations)
        if nsn_lite_accumulate and (nsn_schur_mode != "lite" or not friction):
            raise ValueError("nsn_lite_accumulate needs nsn_schur_mode='lite' and friction=True")
        self.nsn_lite_accumulate = bool(nsn_lite_accumulate)
        # nsn_lite_accumulate: the displacement of the contact forces applied so far, the contact-free iterate, and
        # the weighted multipliers with their lite motion, allocated at the first step that needs them.
        self._acc_d = self._acc_free = self._acc_omega_lam = self._acc_w = None
        if lambda_cap is not None and not lambda_cap > 0.0:
            raise ValueError(f"lambda_cap must be > 0 or None, got {lambda_cap}")
        self.lambda_cap = lambda_cap
        for name, value in (
            ("contact_regularization", contact_regularization),
            ("face_contact_edge_radius", face_contact_edge_radius),
            ("face_contact_gap", 0.0 if face_contact_gap is None else face_contact_gap),
        ):
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        self.contact_regularization = float(contact_regularization)
        self.face_contacts = bool(face_contacts)
        self.face_contact_edge_radius = float(face_contact_edge_radius)
        self.face_contact_gap = None if face_contact_gap is None else float(face_contact_gap)
        self._face_query, self._contact_extra, self._mesh_shapes = None, None, False
        self._self_contact, self._lite_W = None, None
        # Graph of the step's sparse lite contact correction, False once a capture failed: it would fail again.
        self._lite_iter_graph = None
        if particle_enable_self_contact:
            if not self.friction:
                raise ValueError("self contact needs friction=True")
            gap = particle_self_contact_margin if particle_self_contact_gap is None else particle_self_contact_gap
            for name, value in (
                ("particle_self_contact_margin", particle_self_contact_margin),
                ("particle_self_contact_gap", gap),
                ("particle_rest_shape_contact_exclusion_radius", particle_rest_shape_contact_exclusion_radius),
            ):
                if not value >= 0.0:
                    raise ValueError(f"{name} must be >= 0, got {value}")
            from .self_contact import SelfContact  # noqa: PLC0415

            self.particle_self_contact_margin = float(particle_self_contact_margin)
            query_radius = particle_self_contact_margin + gap
            filter_rings = particle_topological_contact_filter_threshold
            rest_radius = particle_rest_shape_contact_exclusion_radius
            self._self_contact = SelfContact(model, particle_self_contact_mode, query_radius, filter_rings, rest_radius)
        if self.face_contacts:
            if not self.friction or nsn_schur_mode != "full":
                raise ValueError("face_contacts needs friction=True and nsn_schur_mode='full'")
            if model.tri_count:
                from ...geometry.types import GeoType  # noqa: PLC0415
                from .face_contact import FaceContactQuery  # noqa: PLC0415

                self._face_query = FaceContactQuery(model.tri_indices.numpy(), model.particle_count, model.device)
                self._mesh_shapes = bool(model.shape_count) and bool((model.shape_type.numpy() == GeoType.MESH).any())

        # Per-shape kinematic angular velocity (rad/s about local +Z).
        n_shapes = model.shape_count
        self._shape_omega_h = np.zeros(max(n_shapes, 1), dtype=np.float64)
        for s_idx, omega in (shape_angular_velocity or {}).items():
            if not 0 <= int(s_idx) < n_shapes:
                raise ValueError(f"shape_angular_velocity names shape {s_idx}, but the model has {n_shapes} shapes")
            self._shape_omega_h[int(s_idx)] = float(omega)
        # Device mirror consumed by the GPU contact kernels; the
        # host array stays the source of truth (small, rarely modified).
        self._shape_omega_d = wp.array(self._shape_omega_h, dtype=wp.float64, device=model.device)

        # PD setup is dt-dependent; we cache the assembly at a reference dt and
        # rebuild lazily inside `step` if the dt changes.
        self._dt_setup: float | None = None
        self._lg_iter_graph = None  # the local-global iteration's graph, False where its capture failed
        self._linear_solver = None
        self._damping_d = None  # the Rayleigh damping matrix in CSR form, set up with the system
        self._tri_beta_d = None  # the triangles' coefficients of the corotated stiffness damping

        # Device buffers (allocated lazily once we know dt - see _setup_pd_system).
        N = model.particle_count
        device = model.device
        self._device = device
        self._x_ref = wp.clone(model.particle_q)  # the pin targets, which start at the model positions
        self._x_inertia = wp.empty(N, dtype=wp.vec3, device=device)
        self._x_cur = wp.empty(N, dtype=wp.vec3, device=device)
        self._rhs = wp.empty(N, dtype=wp.vec3, device=device)

        # Contact state (0 = no contacts registered yet).
        self._contact_count: int = 0

        # ``W`` serves every local-global iteration of a step: it depends only on J, which
        # :meth:`_ingest_contacts` sets once per step, and on A, which changes only with dt or the model. Both
        # invalidate it. ``True`` once the device-resident ``W`` is current. No host copy of ``W`` is kept: the
        # NSN driver reads it on the device through :meth:`FBALinearSolver.W_device_view`.
        self._cached_W = False

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
        self._k_damping_rhs = _K.damping_rhs_kernel
        self._k_add_vec3 = _K.accumulate_vec3_kernel
        self._k_gather = _K.gather_per_particle_kernel
        self._k_pin = _K.project_pin_kernel
        self._k_bending = _K.project_bending_kernel
        self._k_tri_arap = _K.project_stretching_arap_kernel
        self._k_tri_stretch_state = _K.tri_stretch_state_kernel
        self._k_tri_arap_damped = _K.project_stretching_arap_damped_kernel
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

    def _contact_extra_args(self) -> dict:
        """Further particles of the face and self contacts, for ``build_schur_complement`` and
        ``setup_lite_row_meta``."""
        if self._contact_extra is None:
            return {}
        return {"j_extra_indices": self._contact_extra[0], "j_extra_alpha": self._contact_extra[1]}

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
        if self._contact_extra is not None:
            wp.launch(
                K.subtract_slot_residual_kernel,
                dim=self._contact_count,
                inputs=[
                    *self._contact_extra,
                    *(self._contact_normal_d, self._contact_tangent1_d, self._contact_tangent2_d, self._x_cur),
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

    def _update_materials(self) -> None:
        """Check the elements' materials, and set the gauge of the corotational and Neo-Hookean projections (see
        the class notes) and the gauged Lamé parameters of each element."""
        model, stretching_model = self.model, self.stretching_model
        # The dropped-dual gauge of the corotational and Neo-Hookean projections (see the class notes), per element.
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

    def _setup_pd_system(self, dt: float) -> None:
        """Build / rebuild the PD Hessian, factor it, and upload device data."""
        self._update_materials()
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
            rayleigh_alpha=self._rayleigh_alpha,
            rayleigh_beta=self._rayleigh_beta,
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
        # The lite NSN kernels and the dense lite Schur build read the inverse masses; convert them once here
        # rather than in every step. Mass-proportional damping adds alpha m / dt to the mass term
        # m / dt^2 of A, which the lite operator inverts.
        if self.nsn_schur_mode == "lite":
            lite_inv_mass = self._inv_mass_d
            if self._rayleigh_alpha is not None:
                lite_inv_mass = wp.array(
                    (inv_mass / (1.0 + self._rayleigh_alpha * dt)).astype(np.float32), dtype=wp.float32, device=device
                )
            self._linear_solver.setup_lite_mass(lite_inv_mass, dt)
        # The damping matrix C in CSR form, whose product with the step's start positions over dt joins the
        # right-hand side of every iteration.
        damping = meta["damping"]
        self._damping_d = None
        if damping is not None:
            self._damping_d = (
                wp.array(damping.indptr.astype(np.int32), dtype=wp.int32, device=device),
                wp.array(damping.indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.array(damping.data, dtype=wp.float64, device=device),
            )
            self._damping_rhs = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        # Corotated stiffness damping of the triangles: each triangle's coefficient, and its deformation gradient and
        # stretch at the step's start.
        self._tri_beta_d = None
        tri = meta["tri_indices"]
        if self.rayleigh_corotated and self._rayleigh_beta is not None and len(tri):
            from .kernels import mat32  # noqa: PLC0415

            self._tri_beta_d = wp.array(self._rayleigh_beta[tri].mean(axis=1).astype(np.float32), device=device)
            self._tri_F_prev_d = wp.zeros(len(tri), dtype=mat32, device=device)
            self._tri_S_prev_d = wp.zeros(len(tri), dtype=wp.mat22, device=device)
        self._dt_setup = dt
        self._lg_iter_graph = None
        # A changed, so W is stale.
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
            edge_w = meta["edge_weight"] * meta["edge_quad_scale"]
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
                to step without them. With self contact on, the step still
                finds its self contacts.
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

        # Ingest the step's contacts once; with self contact on, a step without contacts still has self contacts.
        if contacts is not None or self._self_contact is not None:
            self._ingest_contacts(contacts, state_in)
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
        # Implicit Rayleigh damping: C x_prev / dt joins the right-hand side of every iteration.
        if self._damping_d is not None:
            inputs = [*self._damping_d, state_in.particle_q, 1.0 / dt]
            wp.launch(self._k_damping_rhs, dim=N, inputs=inputs, outputs=[self._damping_rhs], device=device)
        if self._tri_beta_d is not None:
            inputs = [state_in.particle_q, self._tri_indices_d, self._tri_rest_inv_d]
            outputs = [self._tri_F_prev_d, self._tri_S_prev_d]
            wp.launch(self._k_tri_stretch_state, model.tri_count, inputs, outputs, device=device)
        # Initialize current iterate x_cur = x_inertia (copy).
        wp.copy(self._x_cur, self._x_inertia)
        # The contact forces of the step's sparse lite rows accumulate from zero.
        if self.nsn_lite_accumulate and M > 0:
            self._ensure_accumulation_buffers(3 * M)
            self._acc_d.zero_()

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

                    use_lite_block_diag = self.nsn_schur_mode == "lite"
                    from . import kernels as K_step  # noqa: PLC0415

                    if not self._cached_W:
                        # ``download=False``: the GPU NSN driver consumes ``W``
                        # via :meth:`FBALinearSolver.W_device_view`, so the
                        # device-to-host pull at the end of ``build_schur_complement``
                        # is dead work; skipping it removes the dominant cost on
                        # large contact sets, where the download took ~70% of the
                        # Schur build.
                        if use_lite_block_diag:
                            # LiteNSN block-diagonal-by-particle path: no dense
                            # W is materialized. The fused per-contact kernel
                            # below builds each ``3x3`` local W on-the-fly.
                            # Only the row-metadata CSR is needed so the
                            # ``apply_lambda_correction_combined`` gather can run.
                            ls.setup_lite_row_meta(
                                M,
                                self._contact_particle_d,
                                self._contact_normal_d,
                                self._contact_alpha_d,
                                self._contact_tangent1_d,
                                self._contact_tangent2_d,
                                **self._contact_extra_args(),
                                host=None
                                if self._contact_particle_h is None
                                else (self._contact_particle_h, self._contact_alpha_h),
                            )
                            # Rows on several particles, and more rows on a particle than the per-particle
                            # Newton step holds, take the sparse lite NSN.
                            many = ls._isodof_max_rows > K_step._MAX_ROWS_PER_PARTICLE
                            sparse = self._contact_extra is not None or many
                            self._lite_W = self._build_lite_delassus(M, ls) if sparse else None
                            if self._lite_iter_graph is not False:
                                self._lite_iter_graph = None
                        else:
                            ls.build_schur_complement(
                                M,
                                *self._contact_row_args(),
                                download=False,
                                **self._contact_extra_args(),
                            )
                        self._cached_W = True
                    # Residual + pene0 + warm-start all device-resident: the PD
                    # outer iter does no host<->device transfers around the NSN
                    # call.
                    self._ensure_nsn_inner_buffers(3 * M, alloc_2d=not use_lite_block_diag)
                    if use_lite_block_diag and self._lite_W is not None:
                        # Rows on several particles couple them: the sparse lite NSN, whose whole correction runs
                        # from one graph per step where it can.
                        self._lite_sparse_iteration(M, dt, _use_lg_graph)
                        continue

                    self._launch_contact_residual(self._nsn_r_d)
                    self._launch_contact_pene0(self._nsn_pene0_d)
                    if use_lite_block_diag:
                        # Warm-start ω + λ from the persistent device buffers
                        # (same convention as the dense path).
                        wp.copy(self._nsn_omega_d, self._omega_coulomb_persistent_d, count=3 * M)
                        wp.copy(self._nsn_lam_d, self._lam_coulomb_persistent_d, count=3 * M)
                        # Lambda cap (folded into both per-contact and per-particle kernels).
                        if self.lambda_cap is not None:
                            cap_internal = float(self.lambda_cap) / (dt * dt)
                            use_cap = 1
                        else:
                            cap_internal = 0.0
                            use_cap = 0
                        # Dispatch: the per-particle kernel whenever any
                        # particle has more than 3 contact rows (i.e. >1
                        # frictional contact, since each contact owns 3 rows).
                        # Otherwise stick with the per-contact
                        # kernel: it's the k_p==1 specialisation and avoids
                        # the per-particle kernel's 12x12 register footprint
                        # for what is the only common case of cloth on shapes.
                        use_per_particle = ls._isodof_max_rows > 3
                        if use_per_particle:
                            wp.launch(
                                K_step.fb_newton_lite_coulomb_per_particle_kernel,
                                dim=ls._n_isodofs,
                                inputs=[
                                    ls._isodof_unique_d,
                                    ls._isodof_row_offsets_d,
                                    ls._isodof_row_indices_d,
                                    ls._row_particle_d,
                                    ls._row_dir_d,
                                    ls._row_alpha_d,
                                    self._contact_mu_d,
                                    self._nsn_r_d,
                                    self._nsn_pene0_d,
                                    ls._inv_mass_d,
                                    wp.float64(dt),
                                    wp.float64(cap_internal),
                                    wp.int32(use_cap),
                                ],
                                outputs=[
                                    self._nsn_lam_d,
                                    self._nsn_omega_d,
                                    self._nsn_lam_apply_d,
                                ],
                                device=device,
                            )
                        else:
                            wp.launch(
                                K_step.fb_newton_lite_coulomb_kernel,
                                dim=M,
                                inputs=[
                                    self._contact_particle_d,
                                    self._contact_normal_d,
                                    self._contact_tangent1_d,
                                    self._contact_tangent2_d,
                                    self._contact_alpha_d,
                                    self._contact_mu_d,
                                    self._nsn_r_d,
                                    self._nsn_pene0_d,
                                    ls._inv_mass_d,
                                    wp.float64(dt),
                                    wp.float64(cap_internal),
                                    wp.int32(use_cap),
                                ],
                                outputs=[
                                    self._nsn_lam_d,
                                    self._nsn_omega_d,
                                    self._nsn_lam_apply_d,
                                ],
                                device=device,
                            )
                    else:
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
                            # Face contacts act on particles that also carry particle contacts.
                            contact_regularization=0.0 if self._contact_extra is None else self.contact_regularization,
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
                    if not self._cached_W:
                        # ``download=False``: see the frictional branch above; the
                        # GPU NSN driver reads ``W`` from device via
                        # :meth:`FBALinearSolver.W_device_view`.
                        if self.nsn_schur_mode == "lite":
                            ls.build_schur_lite_dense(
                                M,
                                self._contact_particle_d,
                                self._contact_normal_d,
                                self._contact_alpha_d,
                            )
                        else:
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
        # Rayleigh damping term of the step.
        if self._damping_d is not None:
            wp.launch(self._k_add_vec3, dim=N, inputs=[self._damping_rhs], outputs=[self._rhs], device=device)
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
            if self.stretching_model == "arap" and self._tri_beta_d is not None:
                inputs = [self._x_cur, self._tri_indices_d, self._tri_rest_inv_d, self._tri_weight_d, self._tri_beta_d]
                inputs += [1.0 / dt, self._tri_F_prev_d, self._tri_S_prev_d]
                wp.launch(self._k_tri_arap_damped, model.tri_count, inputs, [self._tri_contrib_d], device=device)
                inputs = [self._tri_contrib_d, self._particle_tri_offsets_d, self._particle_tri_element_d]
                inputs += [self._particle_tri_local_d]
                wp.launch(self._k_gather, dim=N, inputs=inputs, outputs=[self._rhs], device=device)
            elif self.stretching_model == "arap":
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
        # Tet stretching projection: deterministic compute+gather.
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
        """The graph of the PD local phase and linear solve, captured at its first use after
        :meth:`_setup_pd_system`, which drops it with the buffers it captures. The graph holds the buffers'
        pointers, not their contents, so each replay reads what they hold then: the next iteration's input.

        Returns ``None`` where capture failed, so that the eager path remains the fallback.
        """
        if self._lg_iter_graph is None:
            try:
                with wp.ScopedCapture(device=self._device) as cap:
                    self._run_pd_local_phase_eager(self.model.particle_count, dt)
                self._lg_iter_graph = cap.graph
            except RuntimeError:
                self._lg_iter_graph = False
        return self._lg_iter_graph or None

    def _write_contact_forces(self, contact_f: wp.array, contacts: Contacts) -> None:
        """Fill the rows of ``contact_f`` for the step's particle contacts and zero the others."""
        from .kernels import soft_contact_force_kernel  # noqa: PLC0415

        contact_f.zero_()
        M, device = self._contact_count, self._device
        if M == 0:
            return
        inputs = [self._contact_record_d, M, self._contact_normal_d, self._contact_tangent1_d]
        inputs += [self._contact_tangent2_d, 3 if self.friction else 1, self._nsn_lam_apply_d]
        inputs += [self._contact_world_anchor_d, contacts.rigid_contact_max]
        wp.launch(soft_contact_force_kernel, M, inputs, [contact_f], device=device)

    def set_pin_targets(self, target_positions: wp.array | np.ndarray) -> None:
        """Update reference positions for pinned particles (dynamic pin).

        For pinned particles (``inv_mass == 0``), the soft Pin energy pulls them
        toward ``target_positions[i]``; they start at their model positions.
        Inactive particles with mass hold where the input state has them
        instead. For free particles the value is unused but the array must
        cover all particles.

        Args:
            target_positions: Target positions [m], shape ``[particle_count, 3]``:
                a :class:`warp.array` of ``wp.vec3`` or a NumPy array.
        """
        if isinstance(target_positions, wp.array):
            self._x_ref.assign(target_positions)
        else:
            self._x_ref.assign(np.asarray(target_positions, dtype=np.float32))

    def _build_lite_delassus(self, M: int, ls):
        """The sparse lite Delassus matrix of the ``M`` contacts for the sparse lite NSN, which holds for the step."""
        from . import kernels as K  # noqa: PLC0415

        n, device = 3 * M, self._device
        triplets = [wp.empty(4 * n, dtype=t, device=device) for t in (wp.int32, K._mat13d)]
        extra = self._contact_extra or [wp.empty((M, 0), dtype=t, device=device) for t in (wp.int32, wp.float32)]
        assert extra[0].shape[1] <= 3, "the lite Jacobian holds four particles of a row"
        inputs = [self._contact_particle_d, self._contact_alpha_d, *extra, ls._row_dir_d]
        wp.launch(K.lite_row_triplets_kernel, n, inputs, triplets, device=device)
        W = ls.build_schur_lite_triplets(n, *triplets)
        # The NSN's graph captures its buffers: they take their sizes for the step here.
        self._ensure_lite_bsr_buffers(n)
        ls.reserve_a_schur_lite_bsr(W)
        return W

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Rebuild and factor the system matrix at the next step.

        ``ModelFlags.SHAPE_PROPERTIES`` stands for changes of the particles and elements, whose masses,
        materials and rest shapes the factored system matrix holds; Newton has no flag of its own for them.
        """
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
        self._cached_W = False

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
        # The index in Contacts of the particle contact each row comes from, -1 for face and self contacts.
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

    def _ingest_contacts(self, contacts: Contacts | None, state: State | None = None) -> None:
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
                :meth:`~newton.CollisionPipeline.collide` call, or ``None``
                for a step with self contacts only.
            state: Current :class:`~newton.State`, the one passed to
                ``collide``. Supplies the particle positions of the face and
                self contact queries; falls back to the model when ``None``.
        """
        if contacts is not None:
            contacts._assert_particle_only_soft_contacts("SolverFBA")
        # The contacts change, so the next solve rebuilds ``W`` from the new J.
        self._invalidate_schur_cache()
        self._contact_extra = None

        # With a face contact margin, triangles can touch shapes that no particle touches.
        other_contacts = self._face_query is not None and self.face_contact_gap is not None
        other_contacts = other_contacts or self._self_contact is not None
        M_raw = 0 if contacts is None else int(contacts.soft_contact_count.numpy()[0])
        if M_raw == 0 and not other_contacts:
            self._contact_count = 0
            return

        # Pull contact data to host for the lexsort, for a deterministic
        # contact order (M is small so this is fast).
        # The collision pipeline writes contacts to slots assigned by
        # ``wp.atomic_add(soft_contact_count, 0, 1)`` so per-frame order
        # depends on GPU thread scheduling. The order of the rows sets the
        # order of the NSN's sums, hence we sort, so that a step is deterministic.
        # The pipeline gives one contact per particle and shape, and a particle one vertex-face self contact at
        # most: with these alone, the contacts are sorted and assembled on the device. Steps with vertex-face self
        # contacts take that path, as their many contacts make the host round trip costly; the others keep the
        # host path.
        on_device = contacts is not None and self._face_query is None
        if on_device and getattr(self._self_contact, "mode", None) == "vertex_face":
            M, shape_contacts = self._assemble_contacts_on_device(contacts, state, M_raw)
            particle_h = normal_h = alpha_h = shape_h = body_pos_h = record_h = extra = None
        else:
            if contacts is None:
                particle_h, shape_h = np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
                body_pos_h, normal_h = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
            else:
                particle_h = host_head(contacts.soft_contact_particle, M_raw)
                shape_h = host_head(contacts.soft_contact_shape, M_raw)
                # soft_contact_body_pos of a static shape is the contact point in world frame
                # (X_wb = identity, so X_ws = X_bs, and body_pos = X_bs * x_local).
                body_pos_h = host_head(contacts.soft_contact_body_pos, M_raw)
                normal_h = host_head(contacts.soft_contact_normal, M_raw)  # world frame

            # Filter out sentinel entries (particle == -1).
            valid_mask = particle_h >= 0
            particle_h = particle_h[valid_mask]
            shape_h = shape_h[valid_mask]
            body_pos_h = body_pos_h[valid_mask]
            normal_h = normal_h[valid_mask]
            record_h = np.flatnonzero(valid_mask).astype(np.int32)
            M = int(particle_h.shape[0])

            if M == 0 and not other_contacts:
                self._contact_count = 0
                return

            # A particle in a concave corner of a mesh touches every face of the corner, not only the closest.
            if self._face_query is not None and self._mesh_shapes and M:
                src = state if state is not None else self.model
                alarm = 0.0 if self.face_contact_gap is None else self.face_contact_gap
                more = self._face_query.mesh_minima(
                    self.model, src.particle_q, particle_h, shape_h, body_pos_h, normal_h, alarm
                )
                *more, parent = more
                contact = (particle_h, shape_h, body_pos_h, normal_h)
                particle_h, shape_h, body_pos_h, normal_h = (np.concatenate(a) for a in zip(contact, more, strict=True))
                record_h = np.concatenate([record_h, record_h[parent]])
                M = int(particle_h.shape[0])

            # Deterministic contact ordering (host-side lexsort, see comment above); the anchor orders the contacts
            # of one particle and shape.
            sort_keys = (
                *(body_pos_h[:, k].astype(np.float64) for k in (2, 1, 0)),
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

            # Face contacts follow the particle contacts: slot 0 holds their first particle.
            alpha_h, extra = np.ones(M, dtype=np.float32), None
            if self._face_query is not None:
                src = state if state is not None else self.model
                found = self._face_query.query(
                    self.model,
                    src.particle_q,
                    particle_h,
                    shape_h,
                    self.face_contact_edge_radius,
                    self.face_contact_gap,
                )
                if len(found[0]):
                    tri, bary = self._face_query.tri[found[0]], found[2]
                    particle_h, alpha_h = np.concatenate([particle_h, tri[:, 0]]), np.concatenate([alpha_h, bary[:, 0]])
                    shape_h, body_pos_h = np.concatenate([shape_h, found[1]]), np.concatenate([body_pos_h, found[3]])
                    normal_h = np.concatenate([normal_h, found[4]])
                    record_h = np.concatenate([record_h, np.full(len(found[0]), -1, dtype=np.int32)])
                    extra = [np.full((M, 2), -1, dtype=np.int32), np.zeros((M, 2), dtype=np.float32)]
                    extra = [np.concatenate([extra[0], tri[:, 1:]]), np.concatenate([extra[1], bary[:, 1:]])]
                    M = len(particle_h)
            # Self contacts follow, with a static anchor at the contact margin along their normal: their rows hold
            # n . sum_k alpha_k x_k >= margin and start without tangential offset.
            shape_contacts = M
            if self._self_contact is not None:
                src = state if state is not None else self.model
                sp, sa, sn = self._self_contact.detect(src.particle_q)
                if len(sp):
                    particle_h, alpha_h = np.concatenate([particle_h, sp[:, 0]]), np.concatenate([alpha_h, sa[:, 0]])
                    shape_h, normal_h = np.concatenate([shape_h, np.full(len(sp), -1)]), np.concatenate([normal_h, sn])
                    body_pos_h = np.concatenate([body_pos_h, self.particle_self_contact_margin * sn])
                    record_h = np.concatenate([record_h, np.full(len(sp), -1, dtype=np.int32)])
                    if extra is None:
                        extra = [np.full((M, 3), -1, dtype=np.int32), np.zeros((M, 3), dtype=np.float32)]
                    else:
                        extra = [
                            np.pad(extra[0], ((0, 0), (0, 1)), constant_values=-1),
                            np.pad(extra[1], ((0, 0), (0, 1))),
                        ]
                    extra = [np.concatenate([extra[0], sp[:, 1:]]), np.concatenate([extra[1], sa[:, 1:]])]
                    M = len(particle_h)
        if extra is not None:
            self._contact_extra = [wp.array(a, device=self._device) for a in extra]
        if M == 0:
            self._contact_count = 0
            return

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
        if particle_h is not None:
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
        # Particle and face contacts stop the particle radius plus the shape margin outside the shape surface, as
        # Newton's contact margins place them.
        if shape_contacts:
            from .kernels import add_contact_thickness_kernel  # noqa: PLC0415

            extra = self._contact_extra
            if extra is None:
                extra = [
                    wp.empty((1, 1), dtype=wp.int32, device=device),
                    wp.empty((1, 1), dtype=wp.float32, device=device),
                ]
            inputs = [self._contact_particle_d, *extra, int(self._contact_extra is not None)]
            inputs += [self._contact_shape_d, model.particle_radius, model.shape_margin]
            wp.launch(add_contact_thickness_kernel, shape_contacts, inputs, [self._contact_offset_d], device=device)

        # Host mirrors of the basic contact fields, for the host paths that still read them.
        self._contact_particle_h = None if particle_h is None else particle_h.astype(np.int32)
        self._contact_alpha_h = alpha_h
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

    def _assemble_contacts_on_device(self, contacts: Contacts, state: State | None, M_raw: int) -> tuple[int, int]:
        """Sort the particle-shape contacts by particle and shape and append the vertex-face self contacts in the
        order of their particles, as the host sort orders them, on the device. Returns the number of contacts and
        of particle-shape contacts."""
        from .kernels import assemble_contacts_kernel, soft_contact_keys_kernel  # noqa: PLC0415

        device, sc = self._device, self._self_contact
        Ms = min(M_raw, contacts.soft_contact_particle.shape[0])
        keys = wp.empty(2 * Ms, dtype=wp.int64, device=device)
        order = wp.empty(2 * Ms, dtype=wp.int32, device=device)
        inputs = [contacts.soft_contact_particle, contacts.soft_contact_shape, self.model.shape_count]
        wp.launch(soft_contact_keys_kernel, Ms, inputs, [keys, order], device=device)
        wp.utils.radix_sort_pairs(keys, order, Ms)
        M = Ms + sc.find((state if state is not None else self.model).particle_q)
        self._ensure_contact_buffers(M)
        extra = [wp.empty((M, 3), dtype=t, device=device) for t in (wp.int32, wp.float32)]
        inputs = [order, contacts.soft_contact_particle, contacts.soft_contact_shape, contacts.soft_contact_body_pos]
        inputs += [contacts.soft_contact_normal, sc.found_particles, sc.found_alpha, sc.found_normal, Ms]
        inputs += [self.particle_self_contact_margin]
        outputs = [self._contact_particle_d, self._contact_normal_d, self._contact_alpha_d, self._contact_shape_d]
        outputs += [self._contact_body_pos_d, *extra]
        wp.launch(assemble_contacts_kernel, M, inputs, outputs, device=device)
        wp.copy(self._contact_record_d, order, count=Ms)
        if M > Ms:
            self._contact_extra = extra
            self._contact_record_d[Ms:M].fill_(-1)
        return M, Ms

    # ------------------------------------------------------------------
    # NSN inner-loop GPU drivers
    # ------------------------------------------------------------------
    #
    # The drivers below run the iterations of ``solve_nsn_unilateral`` and ``solve_nsn_coulomb``
    # (``newton/tests/fba/_reference.py``) on the device: residual → penetration → FB rows → Schur build →
    # linear solve → lambda update → box clamp → cap clip. They also accept host-side ``W`` / ``r`` /
    # ``lam_init`` / ``omega_init`` arguments, with which the tests compare them to the host reference, and
    # return host-side ``(lam, omega, lam_apply)`` unless told to skip the download.

    def _solve_nsn_coulomb_lite_bsr(
        self,
        *,
        M_total: int,
        contact_mu_d: wp.array,
        r_d: wp.array,
        pene0_d: wp.array,
        lam_init_d: wp.array,
        omega_init_d: wp.array,
        lam_out_d: wp.array,
        omega_out_d: wp.array,
        lam_apply_out_d: wp.array,
        dt: float,
        cap_internal: float,
        use_cap: bool,
        W_bsr,
    ) -> None:
        """LiteNSN inner driver on the BSR + sparse-PCR path.

        Takes FB-Newton steps on the rows of ``W_bsr``, the sparse lite
        Delassus matrix that :meth:`_build_lite_delassus` builds, whose rows may span
        several particles: it builds each step's Newton matrix with
        ``FBALinearSolver.build_a_schur_lite_bsr`` and solves it with
        ``NSNPCRSolver.solve_sparse_fused``.
        """
        from . import kernels as K  # noqa: PLC0415
        from .nsn_pcr_solver import csr_diag_kernel, csr_mv  # noqa: PLC0415

        n_rows = 3 * int(M_total)
        device = self._device
        dt_w = wp.float64(dt)

        ls = self._linear_solver

        self._ensure_lite_bsr_buffers(n_rows)

        # Warm-start λ, ω from the persistent buffers.
        wp.copy(self._bsr_lam, lam_init_d, count=n_rows)
        wp.copy(self._bsr_omega, omega_init_d, count=n_rows)

        # The scratch arrays sliced to ``n_rows``; the underlying buffers are cap-sized.
        precond_v = self._bsr_precond[:n_rows]
        omega_v = self._bsr_omega[:n_rows]
        lam_v = self._bsr_lam[:n_rows]
        omega_lam_v = self._bsr_omega_lam[:n_rows]
        w_omega_lam_v = self._bsr_w_omega_lam[:n_rows]
        penetration_v = self._bsr_penetration[:n_rows]
        compliance_v = self._bsr_compliance[:n_rows]
        h_v = self._bsr_h[:n_rows]
        rhs_v = self._bsr_rhs[:n_rows]
        dlam_v = self._bsr_dlam[:n_rows]

        # Precond per row from W's diagonal.
        W_diag = self._bsr_w_diag[:n_rows]
        wp.launch(csr_diag_kernel, n_rows, [W_bsr.offsets, W_bsr.columns, W_bsr.values], [W_diag], device=device)
        wp.launch(
            K.precond_from_bsr_diag_coulomb_kernel,
            dim=n_rows,
            inputs=[W_diag, dt_w],
            outputs=[precond_v],
            device=device,
        )

        wp.launch(
            K.compute_omega_times_lam_kernel,
            dim=n_rows,
            inputs=[omega_v, lam_v],
            outputs=[omega_lam_v],
            device=device,
        )
        csr_mv(W_bsr, omega_lam_v, w_omega_lam_v)

        wp.launch(
            K.compute_penetration_sparse_kernel,
            dim=n_rows,
            inputs=[r_d, w_omega_lam_v, dt_w],
            outputs=[penetration_v],
            device=device,
        )

        wp.launch(
            K.compute_frictional_fb_kernel,
            dim=int(M_total),
            inputs=[
                penetration_v,
                lam_v,
                contact_mu_d,
                precond_v,
                pene0_d,
                dt_w,
            ],
            outputs=[omega_v, compliance_v, h_v],
            device=device,
        )

        A_bsr = ls.build_a_schur_lite_bsr(W_bsr, omega_v, compliance_v)
        # The right-hand side takes the constraint value under the updated weights, as the dense driver does.
        inputs = [omega_v, lam_v]
        wp.launch(K.compute_omega_times_lam_kernel, dim=n_rows, inputs=inputs, outputs=[omega_lam_v], device=device)
        csr_mv(W_bsr, omega_lam_v, w_omega_lam_v)
        wp.launch(
            K.compute_nsn_rhs_lite_bsr_kernel,
            dim=n_rows,
            inputs=[h_v, omega_v, pene0_d, r_d, w_omega_lam_v, dt_w],
            outputs=[rhs_v],
            device=device,
        )
        self._pcr_solver.solve_sparse_fused(A_bsr, rhs_v, dlam_v)

        wp.launch(
            K.axpy_lambda_kernel,
            dim=n_rows,
            inputs=[dlam_v],
            outputs=[lam_v],
            device=device,
        )

        # Capping a normal multiplier shrinks its friction cone: clamp the tangent multipliers after it.
        if use_cap:
            wp.launch(
                K.lambda_cap_clip_kernel,
                dim=3 * int(M_total),
                inputs=[wp.float64(cap_internal)],
                outputs=[lam_v],
                device=device,
            )
        wp.launch(
            K.coulomb_box_clamp_kernel,
            dim=int(M_total),
            inputs=[contact_mu_d],
            outputs=[lam_v],
            device=device,
        )
        wp.launch(
            K.compute_lam_apply_kernel,
            dim=n_rows,
            inputs=[lam_v, omega_v, dt_w],
            outputs=[lam_apply_out_d],
            device=device,
        )
        wp.copy(lam_out_d, self._bsr_lam, count=n_rows)
        wp.copy(omega_out_d, self._bsr_omega, count=n_rows)

    def _ensure_lite_bsr_buffers(self, n_rows: int) -> None:
        """Size the PCR solver and the per-row scratch of the sparse lite NSN for ``n_rows`` rows, with headroom,
        before the step's graph captures them."""
        self._ensure_pcr_solver(n_rows)
        size = self._bsr_lam.size if hasattr(self, "_bsr_lam") else 0
        if size >= n_rows:
            return
        # Grow with 1.5x headroom so a contact count that ramps up does not reallocate every step.
        cap = max(n_rows, 48) if size == 0 else max(int(n_rows * 1.5), 2 * size)
        for name in "omega_lam w_omega_lam penetration compliance h rhs dlam precond lam omega w_diag".split():
            setattr(self, f"_bsr_{name}", wp.zeros(cap, dtype=wp.float64, device=self._device))

    def _lite_sparse_iteration(self, M: int, dt: float, use_graph: bool) -> None:
        """Run a local-global iteration's sparse lite contact correction. Its matrices and buffers hold for the time
        step, so its launches are captured in a graph at the step's first iteration and replayed at all of them,
        when the step has several iterations and its linear solve runs a fixed count of iterations without reading
        the device."""
        pcr = self._pcr_solver
        if not (use_graph and self.iterations > 1 and pcr.max_iter <= pcr.check_every):
            self._lite_sparse_correction(M, dt)
            return
        if self._lite_iter_graph is None:
            # The buffers that the correction touches take their sizes for the step before the capture.
            ls = self._linear_solver
            ls.reserve_lambda_correction()
            try:
                with wp.ScopedCapture(device=self._device) as capture:
                    self._lite_sparse_correction(M, dt)
                self._lite_iter_graph = capture.graph
            except RuntimeError:
                self._lite_iter_graph = False
        if self._lite_iter_graph:
            wp.capture_launch(self._lite_iter_graph)
        else:
            self._lite_sparse_correction(M, dt)

    def _lite_sparse_correction(self, M: int, dt: float) -> None:
        """One iteration's sparse lite contact correction: residuals, an NSN step, the multipliers kept for the next
        iteration and the combined correction."""
        ls, n_rows = self._linear_solver, 3 * M
        accumulate = self.nsn_lite_accumulate
        if accumulate:
            self._accumulate_predict()
        self._launch_contact_residual(self._nsn_r_d)
        self._launch_contact_pene0(self._nsn_pene0_d)
        # Warm-start ω and λ from the previous iteration.
        wp.copy(self._nsn_omega_d, self._omega_coulomb_persistent_d, count=n_rows)
        wp.copy(self._nsn_lam_d, self._lam_coulomb_persistent_d, count=n_rows)
        W_bsr = self._lite_W
        if accumulate:
            self._accumulate_shift(W_bsr, M, dt)
        self._solve_nsn_coulomb_lite_bsr(
            M_total=M,
            contact_mu_d=self._contact_mu_d,
            r_d=self._nsn_r_d,
            pene0_d=self._nsn_pene0_d,
            lam_init_d=self._lam_coulomb_persistent_d,
            omega_init_d=self._omega_coulomb_persistent_d,
            lam_out_d=self._nsn_lam_d,
            omega_out_d=self._nsn_omega_d,
            lam_apply_out_d=self._nsn_lam_apply_d,
            dt=dt,
            cap_internal=0.0 if self.lambda_cap is None else float(self.lambda_cap) / (dt * dt),
            use_cap=self.lambda_cap is not None,
            W_bsr=W_bsr,
        )
        # Keep the multipliers for the next iteration and apply the forces in combined form.
        wp.copy(self._lam_coulomb_persistent_d, self._nsn_lam_d, count=n_rows)
        wp.copy(self._omega_coulomb_persistent_d, self._nsn_omega_d, count=n_rows)
        ls.apply_lambda_correction_combined(self._nsn_lam_apply_d, self._rhs, self._x_cur)
        if accumulate:
            self._accumulate_displacement()

    def _ensure_accumulation_buffers(self, n_rows: int) -> None:
        """Size the buffers of ``nsn_lite_accumulate`` for ``n_rows`` lite rows, ahead of a graph capture."""
        device, N = self._device, self.model.particle_count
        if self._acc_d is None:
            self._acc_d = wp.zeros(N, dtype=wp.vec3, device=device)
            self._acc_free = wp.zeros(N, dtype=wp.vec3, device=device)
        if self._acc_omega_lam is None or self._acc_omega_lam.shape[0] < n_rows:
            self._acc_omega_lam = wp.zeros(2 * n_rows, dtype=wp.float64, device=device)
            self._acc_w = wp.zeros(2 * n_rows, dtype=wp.float64, device=device)

    def _accumulate_predict(self) -> None:
        """Keep the iterate that the global solve found without contact forces, and move it by the displacement
        that the contact forces applied so far caused, so that the residuals see their exact motion."""
        from .kernels import accumulate_vec3_kernel  # noqa: PLC0415

        wp.copy(self._acc_free, self._x_cur)
        wp.launch(accumulate_vec3_kernel, self.model.particle_count, [self._acc_d], [self._x_cur], device=self._device)

    def _accumulate_shift(self, W_bsr, M: int, dt: float) -> None:
        """Take the lite motion ``dt² W (ω λ)`` of the forces applied so far off the residual, so that the lite NSN
        models only their change: its penetration ``-r + dt² W (ω λ)`` starts at the exact one."""
        from .kernels import add_scaled_f64_kernel, compute_omega_times_lam_kernel  # noqa: PLC0415
        from .nsn_pcr_solver import csr_mv  # noqa: PLC0415

        device, n_rows = self._device, 3 * M
        omega_lam, w = self._acc_omega_lam[:n_rows], self._acc_w[:n_rows]
        inputs = [self._omega_coulomb_persistent_d, self._lam_coulomb_persistent_d]
        wp.launch(compute_omega_times_lam_kernel, n_rows, inputs, [omega_lam], device=device)
        csr_mv(W_bsr, omega_lam, w)
        wp.launch(add_scaled_f64_kernel, n_rows, [w, wp.float64(dt * dt)], [self._nsn_r_d], device=device)

    def _accumulate_displacement(self) -> None:
        """The displacement that the contact forces applied so far caused, for the next iteration."""
        from .kernels import sub_vec3_kernel  # noqa: PLC0415

        inputs = [self._x_cur, self._acc_free]
        wp.launch(sub_vec3_kernel, self.model.particle_count, inputs, [self._acc_d], device=self._device)

    def _ensure_pcr_solver(self, max_n: int) -> None:
        """Lazily create / resize the NSN-Schur PCR solver."""
        from .nsn_pcr_solver import NSNPCRSolver  # noqa: PLC0415

        existing = getattr(self, "_pcr_solver", None)
        if existing is None or existing.max_n < max_n:
            if existing is not None:
                max_n = max(max_n, int(1.5 * existing.max_n))
            self._pcr_solver = NSNPCRSolver(
                max_n=max_n,
                device=self._device,
                # Tight tolerance so PCR ≈ ``np.linalg.solve`` to fp64 noise.
                tol=1.0e-10,
                max_iter=max(1000, 2 * max_n) if self.nsn_linear_iterations is None else self.nsn_linear_iterations,
            )

    def _ensure_nsn_inner_buffers(self, n_rows: int, alloc_2d: bool = True) -> None:
        """Allocate or resize the device-resident NSN inner-loop scratch.

        Buffers grow with headroom so a slowly climbing contact count does not
        realloc them every frame: the dense ``(cap, cap)`` one by
        :func:`dense_schur_capacity`, the others by half again. Contact counts
        can ramp by tens of contacts between frames; a per-frame realloc
        otherwise pegs the device allocator and stalls.

        Args:
            n_rows: Minimum capacity needed for 1D buffers.
            alloc_2d: When ``True`` (default), allocates the ``(cap, cap)``
                dense ``_nsn_a_schur_d`` scratch of the dense NSN inner; a
                driver given ``W`` on the host allocates its upload buffer
                itself. Set to ``False`` in ``nsn_schur_mode="lite"``
                (block-diagonal-by-particle), where the buffer is not
                touched; skipping the alloc keeps the lite memory cost
                ``O(n_rows)``.
        """
        from .linear_solver import dense_schur_capacity  # noqa: PLC0415

        existing_n = int(getattr(self, "_nsn_inner_n", 0))
        existing_2d = bool(getattr(self, "_nsn_inner_has_2d", False))
        if existing_n >= n_rows and (existing_2d or not alloc_2d):
            return
        # ``max`` against the current capacity ensures monotonic growth and avoids shrinking after a transient
        # contact-count spike.
        if alloc_2d:
            cap = max(dense_schur_capacity(n_rows, existing_n), 16)
        else:
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
        # A dense scratch from an earlier call does not cover a grown capacity.
        self._nsn_a_schur_d = wp.zeros((cap, cap), dtype=wp.float64, device=device) if alloc_2d else None
        self._nsn_inner_has_2d = bool(alloc_2d)

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
            # The upload buffer of a host W, allocated only when a caller passes one.
            if getattr(self, "_nsn_W_d", None) is None or self._nsn_W_d.shape != (cap, cap):
                self._nsn_W_d = wp.empty((cap, cap), dtype=wp.float64, device=device)
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
        contact_regularization: float = 0.0,
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
            contact_regularization: Relative regularization ``gamma`` of the
                ``3M`` contact rows of each Newton system,
                ``A_cc += gamma diag(A_cc)``.

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
            # The upload buffer of a host W, allocated only when a caller passes one.
            if getattr(self, "_nsn_W_d", None) is None or self._nsn_W_d.shape != (cap, cap):
                self._nsn_W_d = wp.empty((cap, cap), dtype=wp.float64, device=device)
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
            if contact_regularization > 0.0:
                wp.launch(
                    K.scale_diagonal_kernel,
                    dim=3 * M,
                    inputs=[wp.float64(1.0 + contact_regularization)],
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
            # Capping a normal multiplier shrinks its friction cone: clamp the tangent multipliers again after it.
            cap_internal = float(self.lambda_cap) / (dt * dt)
            wp.launch(
                K.lambda_cap_clip_kernel,
                dim=3 * M,
                inputs=[wp.float64(cap_internal)],
                outputs=[self._nsn_lam_d],
                device=device,
            )
            wp.launch(K.coulomb_box_clamp_kernel, dim=M, inputs=[mu_d], outputs=[self._nsn_lam_d], device=device)

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
