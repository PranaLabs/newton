# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ...core.reset import reset_state_array
from ...geometry import ParticleFlags
from ...sim import BodyFlags, Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags
from ..solver import SolverBase
from .kernels import host_head, stored_contact_count

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
    """Fast But Accurate projective-dynamics solver for cloth, soft bodies and rigid bodies.

    .. experimental::

        :class:`SolverFBA`, its constructor options and its solver-specific
        attributes may change without the normal deprecation period.

    A step depends on its input state, contacts and control, on the pin
    targets and particle constraints last set, and, with ``friction_anchors``
    or ``contact_warmstart``, on the anchors or the multipliers that the steps
    before kept, which :meth:`reset` drops. Otherwise the contact multipliers
    and weights start at zero in every step.

    With :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F`
    requested, a step writes row ``rigid_contact_max + i`` for each particle
    contact ``i`` of its contacts: the force [N] on the shape's body by the
    particle and its torque [N·m] about the body's center of mass, or about
    the world origin for a static shape, in world frame, from the forces
    the step's last solve applied. The other distance minima of a particle
    on a mesh add to its contact's row. Face contacts and self contacts
    have no rows in the contacts, and the rows of rigid contacts are zero.

    Implements a Projective Dynamics (PD) local-global iteration for cloth
    and tetrahedral soft bodies, with ARAP, corotational or Neo-Hookean
    stretching, isometric bending of cloth, springs whose ADMM local steps
    converge to their implicit Euler step, and soft pin constraints. The PD
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

    Tetrahedral activations, the control's ``tet_activations``, grow each
    tetrahedron to ``1 - a`` times its rest volume for its activation ``a``,
    the volume ratio that the volume constraint of
    :class:`~newton.solvers.SolverXPBD` aims for: a negative activation
    inflates the element and a positive one contracts it. The growth is
    isotropic, unless ``tet_activation_directions`` gives the element a
    direction along which alone it stretches by ``1 - a``, as the chambers of
    a pneumatic bending actuator lengthen along it. The stretching projection
    acts on ``F G^-1`` for the growth ``G``, ``(1 - a)^(1/3) I`` or
    ``I + (1 - a - 1) n n^T`` along the unit direction ``n``, and multiplies
    its result by ``G``, so the prefactored system matrix stays valid and a
    free element comes to rest at ``G`` times its rest shape. The element
    keeps the projection weight of its rest volume: grown isotropically, it
    is ``(1 - a)^(-1/3)`` times as stiff as the same material grown to the
    activated shape. Grown along a direction, ``P(F G^-1) G`` is not the
    projection of ``F`` in the solver's metric, so the element's forces are
    not those of an energy, though a free element still comes to rest at its
    grown shape.

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
        friction_anchors: bool = False,
        lambda_cap: float | None = None,
        shape_angular_velocity: dict[int, float] | None = None,
        shape_particle_radius_override: dict[int, float] | None = None,
        nsn_schur_mode: Literal["full", "lite"] = "full",
        nsn_sparse_storage: bool = False,
        integrate_with_external_rigid_solver: bool = False,
        contact_regularization: float = 1.0e-2,
        face_contacts: bool = False,
        face_contact_edge_radius: float = 3.0e-3,
        nsn_linear_iterations: int | None = None,
        nsn_lite_accumulate: bool | ArrayLike = False,
        face_contact_gap: float | None = None,
        particle_enable_self_contact: bool = False,
        particle_self_contact_mode: Literal["vertex_face", "vertex_vertex"] = "vertex_face",
        particle_self_contact_edge_edge: bool = False,
        particle_self_contact_edge_edge_ratio: float = 0.9,
        particle_self_contact_margin: float = 2.0e-3,
        particle_self_contact_gap: float | None = None,
        particle_topological_contact_filter_threshold: int = 2,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        particle_self_contact_group_ids: ArrayLike | None = None,
        particle_self_contact_group_margins: ArrayLike | None = None,
        rod_contact: bool = False,
        rod_contact_gap: float | None = None,
        rod_shape_contacts: bool = False,
        rod_shape_contact_gap: float | None = None,
        rayleigh_alpha: float | ArrayLike = 0.0,
        rayleigh_beta: float | ArrayLike = 0.0,
        rayleigh_corotated: bool = False,
        tet_activation_directions: ArrayLike | None = None,
        contact_warmstart: bool = False,
        rigid_contact_bin: float = 0.0,
        rigid_contact_friction: str = "mean",
        rigid_iteration_graph: bool = False,
        rigid_contact_time: float = 0.0,
        rigid_contact_damping: float = 1.0,
        tendon_time: float = 0.0,
        tendon_damping: float = 1.0,
    ) -> None:
        """
        Args:
            model: The :class:`~newton.Model` containing cloth/tet attributes to integrate.
            iterations: Number of PD local-global iterations per step. Each
                takes one semismooth Newton step of the contact and constraint
                multipliers, warm-started from the previous iteration.
            pin_stiffness: Soft-pin stiffness [N/m].
            stretching_model: The stretching model, ``"arap"``, ``"corotational"`` or ``"neohookean"``.
            friction: If ``True`` (default), contacts carry Coulomb friction;
                otherwise they only push.
            friction_anchors: If ``True``, a contact keeps the point it
                presses on across steps: a point of the other layer for a
                vertex-face self contact, or a point of a static shape. Its
                friction rows then hold the particle at that point rather than
                at the point the step's contact starts from, so slip that a
                step's solve leaves is pulled back in the next step instead of
                adding up. The lite operator leaves such slip where the
                material spreads a friction force to the particle's neighbors,
                which lets stacked cloth layers creep apart over seconds. The
                friction cone still bounds the force: a contact pulled harder
                than ``mu`` times its normal force slides as without the
                point, which trails it by at most a quarter of the contact
                distance (the self-contact margin, or the particle radius plus
                the shape margin) and then moves to it; with a contact distance
                of zero it stays where it was set. A contact drops its
                point when it carries no normal force or acts on a particle
                that rows hold at a point. Vertex-vertex self contacts,
                contacts with shapes on bodies, face contacts, and the contacts
                of a step in which one touches a body that FBA integrates keep
                the step's point. Needs ``friction=True``.
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
            shape_particle_radius_override: Optional ``{shape_index: radius}``
                mapping of the contact thickness [m] of the particles against
                the listed shapes: their particle and face contacts stop this
                radius plus the shape margin outside the shape surface, in
                place of the particle radius, so that a gripper can close on
                a body whose particle radius holds it off the floor. A radius
                lies between 0 and the smallest particle radius, since the
                collision pipeline still finds the contacts at the particle
                radii. Other shapes and self contacts keep the particle
                radii. Not supported with ``friction_anchors``.
            nsn_schur_mode: Contact operator ``W`` of the NSN steps.
                ``"full"`` (default) builds ``J A⁻¹ Jᵀ`` from the prefactored
                system matrix ``A``. ``"lite"`` replaces ``A⁻¹`` by
                ``dt² M⁻¹``, the inverse of the mass term ``M / dt²`` of ``A``
                (``(1 + alpha dt) M / dt²`` with ``rayleigh_alpha``):
                the rows of different particles decouple, unless self contacts or
                particle constraints couple them. ``W`` is built, sparse, only
                for coupled rows or more than four contacts on a particle. Each
                local-global iteration solves for the contact forces whose
                motion, predicted through ``W``, closes the gaps of its
                contact-free positions. The lite ``W`` overrates that motion
                where the material spreads a force to the particle's
                neighbors, so a pressed contact settles inside its shape, by
                up to ``dt² F / m`` for a normal force ``F`` [N] on a particle
                of mass ``m`` [kg], and more iterations do not reduce that.
                Contacts on integrated bodies and face contacts need ``"full"``.
            nsn_sparse_storage: If ``True``, the full NSN steps of frictional
                contacts keep ``W`` and their Newton matrix in CSR form
                without the entries that are zero, such as those between the
                contacts of separate bodies, and ``W`` takes the entries of
                ``A⁻¹`` it needs from the sparse inverse factor directly,
                without a dense copy of its columns. For many contacts on many
                separate bodies, such as a pile of soft bodies, this saves the
                dense Newton matrix and most of the work of the conjugate
                residual iterations. ``W`` is composed in CSR form once per
                step; the block of ``A⁻¹`` on the contacted particles stays
                dense. Sums run in another order than the dense path's,
                so results differ by rounding. Needs ``friction=True`` and
                ``nsn_schur_mode="full"``.
            integrate_with_external_rigid_solver: If ``True``, rigid bodies are
                integrated by another solver: FBA reads their poses and
                velocities from ``state_in``, at which the step's contacts were
                detected, and writes no body array of ``state_out``. Otherwise
                FBA integrates every dynamic body and passes kinematic bodies
                through.
            contact_regularization: Relative regularization ``gamma`` of the
                contact rows in the NSN Newton steps that involve integrated
                bodies, or face or self contacts: each step solves with
                ``A_cc + gamma diag(A_cc)``. Contacts and joints that close a
                loop, such as a hinged plank leaning on a box that rests on
                the ground, make the Newton matrix nearly singular, and an
                unregularized step answers a sub-micrometer inconsistency
                there with forces far above the bodies' weights. Face
                contacts act on particles that also carry particle contacts
                and do the same. The term bounds the gain along contact
                directions by ``1 / gamma`` and leaves the complementarity
                conditions unchanged. Steps with particle contacts only are
                not regularized, and neither is the sparse self-contact solve
                of ``nsn_schur_mode="lite"``.
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
                where the material spreads the force. Rows that hold particles
                at points take their own step as before. Needs
                ``nsn_schur_mode="lite"`` and ``friction=True``.
                A boolean array of shape ``(model.world_count + 1,)`` selects
                worlds independently; the final entry selects global particles.
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
                longest edge, or with ``particle_self_contact_edge_edge=True``.
                ``"vertex_vertex"`` keeps particles that margin apart, which
                covers the surface when it reaches the longest edge (twice the
                largest circumradius of acute triangles), and leaves a bumpy
                contact surface.
            particle_self_contact_edge_edge: If ``True``, also keep two edges
                ``particle_self_contact_margin`` apart where their closest
                points lie inside both edges and come nearer than the nearest
                triangles of their particles, as at a crease, so that crossing
                edges are covered at any margin. Needs
                ``particle_self_contact_mode="vertex_face"``.
            particle_self_contact_edge_edge_ratio: An edge pair takes a
                contact only where its edges come nearer than this times the
                distance of its particles' nearest triangles. Below 1, between
                flat layers the vertex-face contacts hold the edges apart, and
                only ridges, such as creases, bring edges that near; 1 also
                covers edges of layers whose vertex-face contacts barely hold
                them, at a few percent more contacts.
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
            particle_self_contact_group_ids: Optional group of each particle,
                shape ``(particle_count,)``, such as one group per body of a
                pile; the particles of a triangle share one. A particle and a
                triangle of two groups keep ``particle_self_contact_margin``
                apart, and of one group its margin from
                ``particle_self_contact_group_margins``, or do not collide
                where it is 0 or without it. Needs
                ``particle_enable_self_contact=True`` with
                ``particle_self_contact_mode="vertex_face"``;
                not supported with ``particle_self_contact_edge_edge`` or
                ``friction_anchors``.
            particle_self_contact_group_margins: Self-contact margin [m]
                within each group of ``particle_self_contact_group_ids``; 0
                lets the particles of a group pass through its triangles, and
                ``None`` those of every group.
            rod_contact: If ``True``, the model's springs, such as the
                segments of particle rods (see :meth:`add_particle_rod`),
                collide with each other as capsules around them, whose radius
                runs between their particles' radii: a particle and a segment,
                or two segments where their closest points lie inside both,
                keep the sum of their radii apart, through contacts found once
                per step on its start positions and solved with the other
                contacts. Particles within as many segments of each other along
                the springs as the largest sum of two radii plus the gap spans
                at the shortest spring never collide, so that a straight rod
                does not touch itself. Needs ``friction=True``; friction takes
                ``model.particle_mu``. Not supported with ``friction_anchors``.
            rod_contact_gap: Additional detection distance [m] of the rod
                contacts beyond the capsules' surfaces, which bounds the
                relative motion of two segments per step that cannot pass a
                contact. ``None`` takes the largest particle radius.
            rod_shape_contacts: If ``True``, the springs, such as the
                segments of particle rods, also collide with shapes as capsules
                whose radius runs between their particles' radii: a segment
                takes a contact at its deepest point inside a shape when that
                point lies strictly between its particles and deeper than both,
                a row on its two particles with that point's weights, as a face
                contact is on a triangle's. A thin shape then cannot pass
                between two particles of a rod, and a rod slides over a curve
                without catching on its particles. Needs friction and the full
                operator.
            rod_shape_contact_gap: Detection distance [m] of the segment
                contacts beyond the capsules' surfaces and the shape margins.
                ``None`` takes the largest particle radius of the springs.
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
                together. Rigid bodies are not damped.
            rayleigh_corotated: Damp the stretch rate of the triangles in
                place of the rate of their deformation gradient, so that
                ``rayleigh_beta`` spares their rotation: a flap falls and a
                bulge flattens as fast as without damping, while stretch
                waves stay damped. The system matrix is unchanged; each
                iteration's right-hand side turns the triangle's stretch at
                the step's start with its current rotation. Springs, in the
                same way, damp their length rate alone, with ``beta`` times
                their stiffness, in their local steps, and leave the system
                matrix's damping: a swinging chain of springs swings on,
                while its stretch waves die out. Bending and tetrahedra keep
                the constant damping. Needs ``stretching_model="arap"`` and a
                nonzero ``rayleigh_beta``.
            tet_activation_directions: Direction of each tetrahedron's growth
                in the rest frame, shape ``(tet_count, 3)``: an activation
                ``a`` stretches it by ``1 - a`` along its direction alone,
                which again makes ``1 - a`` its volume ratio, as the chambers
                of a pneumatic bending actuator lengthen along it. A zero row,
                or ``None`` for all, grows the tetrahedron isotropically.
            contact_warmstart: If ``True``, each step's contact NSN starts
                from the multipliers that the previous step ended with on the
                same rows rather than from zero: particle contacts matched by
                particle and shape, rigid contacts by shape pair and nearest
                anchors, joint rows by index, and the particle constraint rows
                while their world's rows stay unchanged, with friction forces
                re-expressed in the new tangents. A resting or held object then starts each step
                at the contact forces that carry it, which a few iterations
                cannot build up from zero. Face and self contacts start from
                zero. Needs ``friction=True`` and ``nsn_schur_mode="full"``.
            rigid_contact_bin: If positive [m], keep only the deepest of the
                rigid contacts of a body pair in each cell of this size around
                the first body, per normal direction (26 directions), before
                they enter the solve: a pair of finely decomposed meshes emits
                thousands of contacts whose rows all couple through the two
                bodies. ``0`` keeps them all.
            rigid_contact_friction: The friction coefficient of a rigid contact
                from its two shapes' ``shape_material_mu``: ``"mean"``, their
                mean, or ``"max"``, the larger, as MuJoCo combines them.
            rigid_iteration_graph: If ``True``, a step whose rigid rows are
                built anew, such as one with particle rows on bodies, captures
                its first rigid local-global iteration, the contact NSN step
                included, in a CUDA graph and replays it at its other
                iterations, bit for bit the eager result, with fewer host
                launches. Steps whose rigid rows keep their padded storage
                replay the NSN step's own graph across steps instead. Needs
                CUDA, ``nsn_sparse_storage`` and ``nsn_linear_iterations`` up
                to the linear solver's check interval.
            rigid_contact_time: With a positive time constant ``tau`` [s],
                the normal rows of rigid contacts are soft: a spring and a
                damper in parallel (Kelvin-Voigt), of stiffness
                ``m / tau²`` and damping ratio ``rigid_contact_damping`` for
                the row's effective mass ``m``, that act while the row holds
                a force, so that a contact sinks ``F tau² / m`` under the
                force ``F`` and an impact spreads over a few steps instead of
                one. Their backward Euler step is the complementarity
                ``0 <= lambda _|_ g + eps b - beta g_start >= 0``, ``b`` the
                row's force times ``W_ii``, ``g_start`` its gap at the step
                start, ``x = dt / tau``, ``eps = 1 / (x² + 2 zeta x)`` and
                ``beta = 2 zeta / (x + 2 zeta)``. ``0`` keeps rigid contacts
                hard. The start gap counts only in penetration (above zero it
                is taken as zero), so that contacts reported before the bodies
                touch, within their detection gap, do not brake them. Particle
                contacts are not softened, nor are tendon rows (see
                ``tendon_time``).
            rigid_contact_damping: The damping ratio ``zeta`` of soft rigid
                contacts (``1``, critical: no bounce).
            tendon_time: With a positive time constant [s], the length limits
                of the tendons (:meth:`set_tendons`) are soft in the same way:
                a taut string stretches ``T tau² / m`` under its tension ``T``
                for the effective mass ``m`` of its row, and stops a load it
                catches over a few steps instead of one. Their damper acts on
                the stretch of a taut string only (the start gap is clamped
                to ``<= 0``), so that a slack string pulls nothing. ``0``
                keeps tendons inextensible.
            tendon_damping: The damping ratio of soft tendon limits.
        """
        super().__init__(model)
        self._particle_world_h = model.particle_world.numpy()
        self._body_world_h = model.body_world.numpy()
        self._shape_world_h = model.shape_world.numpy()
        self._deformable_world_rows = self._rigid_world_rows = None
        self._constraint_world_h = np.empty(0, dtype=np.int32)
        self._joint_world_h = np.empty(0, dtype=np.int32)
        self._two_way_world_h = np.zeros(model.world_count, dtype=np.int32)
        self._two_way_world_d = wp.empty(0, dtype=wp.int32, device=model.device)
        # The rows of a step whose worlds split between the two-way and the deformable NSN, and the Schur rows of the
        # factor of the particle system that the two-way rows take beside the deformable ones.
        self._mixed = self._two_way_ls = None
        # The rows that :meth:`_swap_rows` exchanged into the solver, until it exchanges them back.
        self._swapped_rows = None
        # Host copies of the step's contact rows for the Schur build, from the host contact path.
        self._contact_host_rows = self._contact_extra_h = None
        self._empty_contact_scale = wp.empty(0, dtype=wp.float64, device=model.device)
        self._deformable_contact_scale = self._rigid_contact_scale = self._empty_contact_scale
        self._nsn_contact_scale = self._empty_contact_scale
        self._lite_direct = None
        self._empty_indices = wp.empty(0, dtype=wp.int32, device=model.device)
        if model.world_count > 1:
            if (self._particle_world_h < 0).any():
                raise ValueError("SolverFBA requires particles to belong to local worlds in multi-world models")
            dynamic = model.body_flags.numpy() & int(BodyFlags.DYNAMIC) != 0
            if not integrate_with_external_rigid_solver and (dynamic & (self._body_world_h < 0)).any():
                raise ValueError("SolverFBA cannot integrate a global dynamic body shared by several worlds")
            attached = model.shape_body.numpy()
            attached_shapes = np.flatnonzero(attached >= 0)
            if not integrate_with_external_rigid_solver:
                moving = attached_shapes[dynamic[attached[attached_shapes]]]
                if (self._shape_world_h[moving] != self._body_world_h[attached[moving]]).any():
                    raise ValueError("shapes on integrated bodies must belong to the body's world")
            springs = None if model.spring_indices is None else model.spring_indices.reshape((-1, 2))
            rod_bends = getattr(getattr(model, "fba", None), "rod_bend_particles", None)
            for indices in (model.tri_indices, model.tet_indices, model.edge_indices, springs, rod_bends):
                if indices is not None and indices.size:
                    ids = indices.numpy()
                    worlds = self._particle_world_h[np.maximum(ids, 0)]
                    lo = np.min(np.where(ids >= 0, worlds, model.world_count), axis=1)
                    hi = np.max(np.where(ids >= 0, worlds, -1), axis=1)
                    if (lo != hi).any():
                        raise ValueError("SolverFBA elements must not connect particles in different worlds")
            if model.joint_count:
                parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
                valid = parent >= 0
                if (self._body_world_h[parent[valid]] != self._body_world_h[child[valid]]).any():
                    raise ValueError("SolverFBA joints must not connect bodies in different worlds")

        # ---- Validate ----
        if model.particle_count == 0 and model.body_count == 0:
            raise ValueError("SolverFBA requires deformable elements or rigid bodies")
        if model.particle_count > 0 and model.tri_count == 0 and model.tet_count == 0 and model.spring_count == 0:
            raise ValueError("SolverFBA requires at least one cloth triangle, tetrahedral element or spring")
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
        # Unit growth directions of the tets' activations, zero rows to grow isotropically; None for all.
        self._tet_activation_directions = None
        if tet_activation_directions is not None:
            directions = np.asarray(tet_activation_directions, dtype=np.float64)
            if directions.shape != (model.tet_count, 3) or not np.isfinite(directions).all():
                raise ValueError(
                    f"tet_activation_directions needs {model.tet_count} finite rows of 3, got shape {directions.shape}"
                )
            length = np.linalg.norm(directions, axis=1, keepdims=True)
            directions = np.divide(directions, length, out=np.zeros_like(directions), where=length > 0.0)
            self._tet_activation_directions = directions.astype(np.float32)
        self.stretching_model = stretching_model
        self._update_materials()
        self.friction = bool(friction)
        if friction_anchors and not self.friction:
            raise ValueError("friction_anchors needs friction=True")
        if contact_warmstart and (not self.friction or nsn_schur_mode != "full"):
            raise ValueError("contact_warmstart needs friction=True and nsn_schur_mode='full'")
        self.friction_anchors = bool(friction_anchors)
        # Friction anchors: the step count that stamps the self-contact anchors kept per particle, the number of
        # particle contacts whose keys and anchors the step before left, and the number of contacts this step set up.
        self._anchor_step, self._anchor_count, self._anchor_rows, self._anchor_cap = 0, 0, 0, 0
        # Particles whose contacts keep no friction anchors, as the held ones (set_friction_anchor_exclusion).
        self._anchor_exclude = np.zeros(0, dtype=np.int32)
        if self.friction_anchors and model.particle_count:
            n = model.particle_count
            self._anchor_triangle_d = wp.full(n, wp.vec3i(-1, -1, -1), dtype=wp.vec3i, device=model.device)
            self._anchor_bary_d = wp.zeros(n, dtype=wp.vec3, device=model.device)
            self._anchor_step_d = wp.full(n, -2, dtype=wp.int32, device=model.device)
        if nsn_sparse_storage and (not self.friction or nsn_schur_mode != "full"):
            raise ValueError("nsn_sparse_storage needs friction=True and nsn_schur_mode='full'")
        self.nsn_sparse_storage = bool(nsn_sparse_storage)
        self._nsn_graph = None
        # With rigid_iteration_graph the rigid LG iteration replays from a CUDA graph, captured at the first iteration
        # of a step (the solver's very first iteration runs eagerly: it creates the NSN step's solvers).
        # ``_outer_capture`` makes the NSN step record its launches into that capture rather than capture or replay
        # its own graph.
        self._rigid_iter_graph, self._rigid_iter_count, self._outer_capture = None, 0, False
        self.rigid_iteration_graph = bool(rigid_iteration_graph)
        if not (rigid_contact_time >= 0.0 and rigid_contact_damping >= 0.0):
            raise ValueError("rigid_contact_time and rigid_contact_damping must be >= 0")
        self.rigid_contact_time, self.rigid_contact_damping = float(rigid_contact_time), float(rigid_contact_damping)
        if not (tendon_time >= 0.0 and tendon_damping >= 0.0):
            raise ValueError("tendon_time and tendon_damping must be >= 0")
        # Soft tendon limits: the same spring-damper on the tendons' length rows.
        self.tendon_time, self.tendon_damping = float(tendon_time), float(tendon_damping)
        # The soft rigid contacts' per-contact arrays (eps, beta, start gap) of the step; empty arrays for none.
        self._nsn_soft_off = tuple(wp.empty(0, dtype=wp.float64, device=model.device) for _ in range(3))
        self._nsn_soft = self._rc_soft = self._nsn_soft_off
        if not rigid_contact_bin >= 0.0:
            raise ValueError(f"rigid_contact_bin must be >= 0, got {rigid_contact_bin}")
        self.rigid_contact_bin = float(rigid_contact_bin)
        if rigid_contact_friction not in ("mean", "max"):
            raise ValueError(f"rigid_contact_friction must be 'mean' or 'max', got {rigid_contact_friction!r}")
        self.rigid_contact_friction = rigid_contact_friction
        if nsn_linear_iterations is not None and nsn_linear_iterations < 1:
            raise ValueError(f"nsn_linear_iterations must be >= 1 or None, got {nsn_linear_iterations}")
        self.nsn_linear_iterations = None if nsn_linear_iterations is None else int(nsn_linear_iterations)
        accumulation = np.asarray(nsn_lite_accumulate, dtype=bool)
        if accumulation.ndim and accumulation.shape != (model.world_count + 1,):
            raise ValueError("nsn_lite_accumulate needs a scalar or one boolean per world plus global entities")
        self.nsn_lite_accumulate = bool(accumulation.any())
        if self.nsn_lite_accumulate and (nsn_schur_mode != "lite" or not friction):
            raise ValueError("nsn_lite_accumulate needs nsn_schur_mode='lite' and friction=True")
        self._accumulate_worlds = accumulation.copy() if accumulation.ndim else None
        disabled = [] if self._accumulate_worlds is None else np.flatnonzero(~accumulation[self._particle_world_h])
        self._accumulate_disabled_particles = wp.array(disabled, dtype=wp.int32, device=model.device)
        self._accumulate_disabled_rows = self._empty_indices
        self._accumulate_disabled_zeros = self._empty_contact_scale
        # nsn_lite_accumulate: the displacement of the contact forces applied so far, the contact-free iterate, and
        # the weighted multipliers with their lite motion, allocated at the first step that needs them.
        self._acc_d = self._acc_free = self._acc_omega_lam = self._acc_w = None
        if lambda_cap is not None and not lambda_cap > 0.0:
            raise ValueError(f"lambda_cap must be > 0 or None, got {lambda_cap}")
        self.lambda_cap = lambda_cap
        self.integrate_with_external_rigid_solver = bool(integrate_with_external_rigid_solver)
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
        self._constraint_rows = None  # set_particle_constraints: particles, coefficients, directions, targets
        self._constraint_lengths = False  # the rows hold |sum_k c_k x_k|, along its direction at each iteration
        # set_particle_constraints(bodies=...): the body of each row (-1 for none) and its point in the body frame.
        self._constraint_body_h = self._constraint_body_d = self._constraint_point_d = None
        # set_tendons: the step's tendon rows (body, point, target, middle, rest arrays) and their count.
        self._tendons, self._tendon_count, self._tendon_ws = None, 0, None
        # set_tendons: each tendon's friction coefficient and force, and its tension at the end of the last step.
        self._tendon_friction = self._tendon_tension = None
        self._self_contact, self._lite_W = None, None
        self._held_rows = None  # the lite rows that hold particles at points, and their factor
        # Graph of the step's sparse lite contact correction, False once a capture failed: it would fail again.
        self._lite_iter_graph = None
        # Particles that the lite rows hold at points, and a flag per particle on the device.
        self._held_particles, self._held_particle_d = np.zeros(0, dtype=np.int32), None
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

            if particle_self_contact_edge_edge_ratio <= 0.0:
                raise ValueError(
                    f"particle_self_contact_edge_edge_ratio must be > 0, got {particle_self_contact_edge_edge_ratio}"
                )
            self.particle_self_contact_margin = float(particle_self_contact_margin)
            query_radius = particle_self_contact_margin + gap
            filter_rings = particle_topological_contact_filter_threshold
            rest_radius = particle_rest_shape_contact_exclusion_radius
            self._self_contact = SelfContact(
                model,
                particle_self_contact_mode,
                query_radius,
                filter_rings,
                rest_radius,
                edge_edge=particle_self_contact_edge_edge,
                edge_edge_ratio=particle_self_contact_edge_edge_ratio,
            )
        if particle_self_contact_group_margins is not None and particle_self_contact_group_ids is None:
            raise ValueError("particle_self_contact_group_margins needs particle_self_contact_group_ids")
        self._self_contact_groups = particle_self_contact_group_ids is not None
        if self._self_contact_groups:
            if self._self_contact is None or self.friction_anchors:
                raise ValueError("self-contact groups need particle_enable_self_contact and no friction_anchors")
            groups = np.asarray(particle_self_contact_group_ids)
            if not np.issubdtype(groups.dtype, np.integer) or groups.shape != (model.particle_count,):
                raise ValueError("particle_self_contact_group_ids needs one integer group per particle")
            if groups.min() < 0:
                raise ValueError("particle_self_contact_group_ids must be nonnegative")
            margins = particle_self_contact_group_margins
            if margins is not None:
                margins = np.asarray(margins, dtype=np.float64)
                if margins.ndim != 1 or len(margins) <= groups.max():
                    raise ValueError("particle_self_contact_group_margins needs a margin for each group")
                if not np.isfinite(margins).all() or (margins < 0.0).any():
                    raise ValueError("particle_self_contact_group_margins must be nonnegative")
            triangles = model.tri_indices.numpy()
            if (groups[triangles] != groups[triangles[:, :1]]).any():
                raise ValueError("the particles of a triangle must share one self-contact group")
            self._self_contact.set_groups(
                groups, margins, particle_self_contact_margin, query_radius - particle_self_contact_margin
            )
        self._rod_contact = None
        self._segment_query, self.rod_shape_contact_gap = None, None
        if rod_shape_contacts:
            if not self.friction or nsn_schur_mode != "full":
                raise ValueError("rod_shape_contacts needs friction=True and nsn_schur_mode='full'")
            seg = model.spring_indices.numpy().reshape(-1, 2) if model.spring_count else np.zeros((0, 2), np.int32)
            if model.spring_count:
                seg = seg[model.spring_stiffness.numpy() > 0.0]
            if not len(seg):
                raise ValueError("rod_shape_contacts needs springs")
            if rod_shape_contact_gap is None:
                rod_shape_contact_gap = float(model.particle_radius.numpy()[seg].max())
            if not rod_shape_contact_gap >= 0.0:
                raise ValueError(f"rod_shape_contact_gap must be >= 0, got {rod_shape_contact_gap}")
            from .face_contact import SegmentContactQuery  # noqa: PLC0415

            self._segment_query = SegmentContactQuery(seg, model.device)
            self.rod_shape_contact_gap = float(rod_shape_contact_gap)
        if rod_contact:
            if not self.friction or self.friction_anchors:
                raise ValueError("rod_contact needs friction=True and no friction_anchors")
            springs = model.spring_indices.numpy().reshape(-1, 2) if model.spring_count else np.zeros((0, 2))
            if model.spring_count:
                springs = springs[model.spring_stiffness.numpy() > 0.0]
            if not len(springs):
                raise ValueError("rod_contact needs springs")
            if rod_contact_gap is None:
                rod_contact_gap = float(model.particle_radius.numpy()[springs].max())
            if not rod_contact_gap >= 0.0:
                raise ValueError(f"rod_contact_gap must be >= 0, got {rod_contact_gap}")
            from .rod_contact import RodContact  # noqa: PLC0415

            self._rod_contact = RodContact(model, springs, rod_contact_gap)
        if self.face_contacts:
            if not self.friction or nsn_schur_mode != "full":
                raise ValueError("face_contacts needs friction=True and nsn_schur_mode='full'")
            if model.tri_count:
                from ...geometry.types import GeoType  # noqa: PLC0415
                from .face_contact import FaceContactQuery  # noqa: PLC0415

                self._face_query = FaceContactQuery(model.tri_indices.numpy(), model.particle_count, model.device)
                self._mesh_shapes = bool(model.shape_count) and bool((model.shape_type.numpy() == GeoType.MESH).any())
        self._rigid_dt: float | None = None
        self._rigid_linear_solver = None

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
        # Per-shape particle contact thickness, -1 where the particle radius holds; None without overrides.
        self._shape_radius_d = None
        if shape_particle_radius_override:
            if self.friction_anchors:
                raise ValueError("shape_particle_radius_override is not supported with friction_anchors")
            smallest = float(model.particle_radius.numpy().min()) if model.particle_count else 0.0
            radii = np.full(max(n_shapes, 1), -1.0, dtype=np.float32)
            for shape, radius in shape_particle_radius_override.items():
                if not isinstance(shape, (int, np.integer)) or not 0 <= shape < n_shapes:
                    raise ValueError(f"shape_particle_radius_override names shape {shape}, which the model lacks")
                if not np.isfinite(radius) or not 0.0 <= radius <= smallest:
                    raise ValueError(
                        f"shape_particle_radius_override radius {radius} of shape {shape} is not between 0 and the "
                        f"smallest particle radius {smallest}"
                    )
                radii[shape] = radius
            self._shape_radius_d = wp.array(radii, dtype=wp.float32, device=model.device)

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
        # The previous step's NSN rows, which start the next step's with contact_warmstart, and the particles,
        # coefficients and directions of the particle constraint rows that they hold.
        self._warm_start = self._constraint_key = None
        if contact_warmstart:
            from .warm_start import ContactWarmStart  # noqa: PLC0415

            self._warm_start = ContactWarmStart(device)
        self._x_ref = wp.clone(model.particle_q)  # the pin targets, which start at the model positions
        self._x_inertia = wp.empty(N, dtype=wp.vec3, device=device)
        self._x_cur = wp.empty(N, dtype=wp.vec3, device=device)
        self._rhs = wp.empty(N, dtype=wp.vec3, device=device)

        # Contact state (0 = no contacts registered yet).
        self._contact_count: int = 0
        # Host mirrors of the contact particles and weights, which the lite rows of particle constraints read
        # before the first contacts arrive.
        self._contact_particle_h = None
        self._contact_alpha_h = None

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
        # Springs, ADMM stretch elements (filled by _setup_pd_system): their particles, rest lengths, stiffnesses and
        # penalties, their duals, splits and last directions kept across steps, and the flag of a step's first local
        # step.
        self._spring_indices_d = None

        # Tet device data (filled by _setup_pd_system).
        self._tet_indices_d = None
        self._tet_rest_inv_d = None
        self._tet_weight_d = None
        # The step's tet activations, copied from its control (zero without one), and their growth directions.
        self._tet_activation_d: wp.array | None = None
        self._tet_activation_direction_d: wp.array | None = None
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
        self._k_spring = _K.project_spring_admm_kernel
        self._k_spring_dual = _K.update_spring_dual_kernel
        self._k_spring_chain_system = _K.spring_chain_system_kernel
        self._k_spring_chain_reduction = _K.spring_chain_reduction_kernel
        self._k_spring_chain = _K.spring_chain_tension_kernel
        self._k_clear_flag = _K.clear_flag_kernel
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

    def _setup_rigid_system(self, dt: float) -> None:
        """Factor the rigid block for the dynamic bodies this solver integrates."""
        from .linear_solver import build_rigid_system, rigid_block_inverses  # noqa: PLC0415
        from .rigid import RigidBlockSolver  # noqa: PLC0415

        self._rigid_dt = dt
        if self._warm_start is not None:
            # The multipliers are forces divided by dt^2: those of another dt do not carry over.
            self._warm_start.clear()
        self._rigid_linear_solver = None
        self._joint_rows = None
        model = self.model
        if self.integrate_with_external_rigid_solver or model.body_count == 0:
            return
        bodies = np.flatnonzero(model.body_flags.numpy() & int(BodyFlags.DYNAMIC)).astype(np.int32)
        if bodies.size == 0:
            return
        A = build_rigid_system(model, bodies, dt)
        m = bodies.size
        device = self._device
        self._rigid_bodies_d = wp.array(bodies, dtype=wp.int32, device=device)
        self._slot_inv_mass_d = wp.array(rigid_block_inverses(A), dtype=wp.spatial_matrixd, device=device)
        # The blocks do not couple: each body's block inverse solves its DOFs.
        self._rigid_linear_solver = RigidBlockSolver(self._slot_inv_mass_d)
        self._rigid_q_pin_d = wp.empty(m, dtype=wp.quatd, device=device)
        self._rigid_x_prev_d = wp.empty(m, dtype=wp.vec3d, device=device)
        self._rigid_b_d = wp.empty(6 * m, dtype=wp.float64, device=device)
        self._rigid_u_d = wp.empty(6 * m, dtype=wp.float64, device=device)
        self._rigid_x_live_d = wp.empty(m, dtype=wp.vec3d, device=device)
        self._rigid_q_live_d = wp.empty(m, dtype=wp.quatd, device=device)
        self._rigid_turn_d = wp.empty(m, dtype=wp.vec3d, device=device)
        self._rigid_u_free_d = wp.empty(6 * m, dtype=wp.float64, device=device)
        self._rigid_b_iter_d = wp.empty(6 * m, dtype=wp.float64, device=device)
        self._body_slot_h = np.full(model.body_count, -1, dtype=np.int32)
        self._body_slot_h[bodies] = np.arange(m)
        self._body_slot_d = wp.array(self._body_slot_h, dtype=wp.int32, device=device)
        self._setup_joints()

    def _setup_joints(self) -> None:
        """Collect the joint rows on the integrated bodies and their Newton system."""
        from .joints import BilateralSystem, build_joint_rows  # noqa: PLC0415

        device, slot = self._device, self._body_slot_h
        self._joint_rows = build_joint_rows(self.model, slot, device)
        if self._joint_rows is None:
            return
        R = int(self._joint_rows.block_start[-1])
        self._joint_system = BilateralSystem(self._joint_rows, slot, self._slot_inv_mass_d, device)
        # Independent block of each joint row and of each slot it touches.
        rows = self._joint_rows
        self._joint_world_h = self._body_world_h[rows.child.numpy()[rows.row_joint.numpy()]]
        self._joint_row_block = np.repeat(np.arange(len(rows.block_start) - 1), np.diff(rows.block_start))
        start, entry_row = self._joint_system.slot_entry_start.numpy(), self._joint_system.entry_row.numpy()
        self._slot_block = np.full(len(start) - 1, -1)
        jointed = np.diff(start) > 0
        self._slot_block[jointed] = self._joint_row_block[entry_row[start[:-1][jointed]]]
        self._joint_row_slot_d = wp.empty((R, 2), dtype=wp.int32, device=device)
        self._joint_row_jac_d = wp.empty((R, 2), dtype=wp.spatial_vectord, device=device)
        self._joint_residual_d = wp.empty(R, dtype=wp.float64, device=device)
        self._joint_force_d = wp.empty(R, dtype=wp.float64, device=device)

    def _begin_rigid_step(self, state_in: State, state_out: State, dt: float) -> bool:
        """Start the step of the integrated bodies at their free-flight solution.

        Kinematic bodies pass through. The joint Jacobian and ``W`` are
        evaluated once here, at the pinned frame.

        Returns:
            Whether there are integrated bodies to iterate on.
        """
        from .joints import joint_jacobian_kernel  # noqa: PLC0415
        from .rigid import prepare_rigid_step_kernel  # noqa: PLC0415

        # The step's rows are new: its rigid iterations take a new graph.
        self._rigid_iter_graph, self._rigid_iter_count = None, 0
        if self._rigid_dt is None or abs(self._rigid_dt - dt) > 1e-12:
            self._setup_rigid_system(dt)
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        ls = self._rigid_linear_solver
        if ls is None:
            return False
        model = self.model
        m = self._rigid_bodies_d.shape[0]
        wp.launch(
            prepare_rigid_step_kernel,
            dim=m,
            inputs=[
                self._rigid_bodies_d,
                state_in.body_q,
                state_in.body_qd,
                state_in.body_f,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                model.body_world,
                model.gravity,
                wp.float64(dt),
            ],
            outputs=[self._rigid_q_pin_d, self._rigid_x_prev_d, self._rigid_b_d],
            device=self._device,
        )
        ls.solve_scalar(self._rigid_b_d, self._rigid_u_d)
        wp.copy(self._rigid_u_free_d, self._rigid_u_d)
        self._retract_rigid(state_in, state_out, dt)
        if self._joint_rows is not None:
            wp.launch(
                joint_jacobian_kernel,
                dim=self._joint_residual_d.shape[0],
                inputs=[
                    *self._joint_row_args(),
                    self._rigid_x_prev_d,
                    self._rigid_q_pin_d,
                    state_in.body_q,
                    model.body_com,
                ],
                outputs=[self._joint_row_slot_d, self._joint_row_jac_d],
                device=self._device,
            )
            self._joint_system.factor(self._joint_row_slot_d, self._joint_row_jac_d)
            self._joint_force_d.zero_()
        return True

    def _begin_rigid_contacts(self, contacts: Contacts | None, state_in: State, dt: float) -> None:
        """Collect this step's rigid contacts and assemble their rows and ``W`` at the pinned frame.

        The rows' levers are the anchors' at the step-start poses of ``state_in``; the parts of the rows' values
        that stay fixed through the step are formed here, once.

        The joint rows of every articulation a contact touches join the contact
        rows, so one NSN solve couples them; other articulations keep their
        Cholesky-factored joint system.
        Deformable contact rows lead when one of them touches an integrated body; with several worlds, a step whose
        other worlds keep the deformable NSN passes the rows of the touching worlds alone.
        """
        from .joints import select_rows_kernel  # noqa: PLC0415
        from .rigid import row_incidence, slot_entries  # noqa: PLC0415
        from .rigid_contact import (  # noqa: PLC0415
            gather_rigid_contacts,
            rigid_contact_anchor_kernel,
            rigid_contact_jacobian_kernel,
        )

        model, device = self.model, self._device
        if getattr(self, "_rc_gather", None) is None:
            from .rigid_contact import RigidContactGather  # noqa: PLC0415

            self._rc_gather = RigidContactGather(device)
        rc = None
        if contacts is not None:
            rc = gather_rigid_contacts(
                model, contacts, self._body_slot_d, device, self._rc_gather, self.rigid_contact_friction == "max"
            )
        if rc is not None and self.rigid_contact_bin > 0.0:
            rc = self._reduce_rigid_contacts(rc, state_in)
        self._rigid_contacts = rc
        m_s, m_r = 0, 0 if rc is None else rc.count
        slots = np.empty((0, 2), dtype=np.int32) if rc is None else rc.slot
        # Tendon rows follow the rigid contacts as frictionless contacts of their own Jacobian.
        m_t = self._tendon_count
        if m_t:
            tb = self._tendon_body_h
            slots = np.concatenate(
                [slots, np.where(tb >= 0, self._body_slot_h[np.maximum(tb, 0)], -1).astype(np.int32)]
            )
        # Particle constraint rows on integrated bodies join the body rows, and so do all deformable rows with them.
        attach = self._constraint_rows is not None and self._constraint_body_h is not None
        if model.particle_count > 0 and self._contact_count > 0:
            shape = host_head(self._contact_shape_d, self._contact_count)
            body = np.where(shape >= 0, self._shape_body_host()[np.maximum(shape, 0)], -1)
            soft_slot = np.where(body >= 0, self._body_slot_h[np.maximum(body, 0)], -1)
            if (soft_slot >= 0).any() or attach:
                m_s = self._contact_count
                if model.world_count > 1:
                    self._two_way_world_h[self._contact_world_h[soft_slot >= 0]] = 1
                slots = np.concatenate([np.stack([soft_slot, np.full(m_s, -1)], axis=1), slots])
        self._two_way, self._rc_counts, self._rc_n = m_s > 0 or attach, (m_s, m_r), 0
        self._rc_tendons = m_t
        self._rc_padded, self._rc_n_pad, self._rc_n_b = None, 0, 0
        if m_s + m_r + m_t == 0 and not attach:
            return
        if not self.friction or self.nsn_schur_mode != "full":
            raise NotImplementedError(
                "SolverFBA solves the contacts of the bodies it integrates only with friction=True and "
                "nsn_schur_mode='full'"
            )
        joint_rows = np.empty(0, dtype=np.int32)
        if self._joint_rows is not None:
            blocks = self._slot_block[slots[slots >= 0]]
            joint_rows = np.flatnonzero(np.isin(self._joint_row_block, blocks[blocks >= 0])).astype(np.int32)
        self._rc_joint_rows_h = joint_rows
        # Particle constraint rows follow the joint rows when the deformable rows join this solve.
        n_b = self._rc_n_b = (
            0 if self._constraint_rows is None or not (m_s or attach) else self._constraint_rows[0].shape[0]
        )
        n_s, n_c = 3 * m_s, 3 * (m_r + m_t)
        n = self._rc_n = n_s + n_c + joint_rows.size + n_b
        # Rows of rigid contacts and joint rows keep their storage across steps, the contacts padded to their
        # capacity with inert rows and the joint rows to all joint rows, so that the NSN step's graph replays from
        # step to step.
        storage = None
        if self.nsn_sparse_storage and not m_s and not attach and model.world_count == 1:
            if wp.get_device(device).is_cuda:
                from .rigid_contact import RigidRowStorage  # noqa: PLC0415

                R = 0 if self._joint_rows is None else int(self._joint_rows.block_start[-1])
                if getattr(self, "_rc_storage", None) is None or self._rc_storage.joint_rows != R:
                    self._rc_storage = RigidRowStorage(device, R)
                storage = self._rc_storage
                storage.reserve(m_r + m_t)
                storage.begin()
                if R:
                    storage.select_joint_rows(joint_rows)
        self._rc_padded, self._rc_blocked = storage, False
        if storage is None:
            self._rc_joint_rows_d = wp.array(joint_rows, dtype=wp.int32, device=device)
            self._rc_joint_offset, self._rc_n_pad = n_s + n_c, n
        else:
            self._rc_joint_rows_d = storage.joint_select
            self._rc_joint_offset = 3 * storage.contacts
            self._rc_n_pad = 3 * storage.contacts + storage.joint_section
            # Padded rows whose PCR runs fused keep W in 3x3 blocks of three rows, which it multiplies with fewer
            # column reads than scalar entries.
            self._ensure_pcr_solver(self._rc_n_pad)
            self._rc_blocked = self._pcr_solver.fuses_single(self._rc_n_pad)
        if storage is None:
            self._rc_row_slot_d = wp.full((n, 2), -1, dtype=wp.int32, device=device)
            self._rc_row_jac_d = wp.zeros((n, 2), dtype=wp.spatial_vectord, device=device)
            self._rc_pene0_d = wp.zeros(n, dtype=wp.float64, device=device)
            self._rc_mu_d = wp.zeros(m_s + m_r + m_t, dtype=wp.float64, device=device)
            soft = (wp.zeros(m_s + m_r + m_t, dtype=wp.float64, device=device) for _ in range(3))
            self._rc_soft = (
                tuple(soft) if self.rigid_contact_time > 0.0 or self.tendon_time > 0.0 else self._nsn_soft_off
            )
        else:
            self._rc_row_slot_d, self._rc_row_jac_d = storage.row_slot, storage.row_jac
            self._rc_pene0_d, self._rc_mu_d = storage.pene0, storage.mu
            self._rc_soft = (
                storage.soft if self.rigid_contact_time > 0.0 or self.tendon_time > 0.0 else self._nsn_soft_off
            )
        self._rc_lam_soft_d = wp.empty(n_s + n_b, dtype=wp.float64, device=device) if n_b else None
        if m_s or attach:
            from .rigid_contact import particle_contact_body_jacobian_kernel  # noqa: PLC0415

            if m_s:
                wp.launch(
                    particle_contact_body_jacobian_kernel,
                    dim=(m_s, 3),
                    inputs=[
                        *self._particle_contact_args(),
                        *(self._body_slot_d, model.body_com, self._rigid_q_pin_d, state_in.body_q),
                    ],
                    outputs=[self._rc_row_slot_d, self._rc_row_jac_d],
                    device=device,
                )
            if not m_s:
                self._ensure_contact_buffers(1)
            ls = self._linear_solver
            ls.build_schur_complement(
                m_s,
                *self._contact_row_args(),
                download=False,
                **self._contact_extra_args(),
                **(self._constraint_args() if n_b else {}),
            )
            self._cached_W = True
            if m_s:
                self._launch_contact_pene0(self._rc_pene0_d)
                wp.copy(self._rc_mu_d, self._contact_mu_d, count=m_s)
        if m_r:
            wp.launch(
                rigid_contact_jacobian_kernel,
                dim=(m_r, 3),
                inputs=[rc.body, rc.anchor, rc.normal, self._body_slot_d, self._rigid_q_pin_d, n_s, state_in.body_q],
                outputs=[self._rc_row_slot_d, self._rc_row_jac_d],
                device=device,
            )
            wp.copy(self._rc_mu_d, rc.mu, dest_offset=m_s, count=m_r)
            anchors = getattr(self, "_rc_anchors", None)
            if anchors is None or anchors[0].shape[0] < m_r:
                cap = max(m_r, 64 if anchors is None else int(1.5 * anchors[0].shape[0]))
                anchors = self._rc_anchors = tuple(wp.empty((cap, 2), dtype=wp.vec3d, device=device) for _ in range(4))
            wp.launch(
                rigid_contact_anchor_kernel,
                dim=m_r,
                inputs=[
                    *(rc.body, rc.anchor, rc.normal, self._body_slot_d, self._rigid_x_prev_d, state_in.body_q),
                    *(state_in.body_qd, model.body_com, wp.float64(dt)),
                ],
                outputs=list(anchors),
                device=device,
            )
            if self.rigid_contact_time > 0.0:
                from .rigid_contact import rigid_contact_soft_kernel  # noqa: PLC0415

                # Kelvin-Voigt rows: x = dt / tau, eps = 1 / (x² + 2 zeta x), beta = 2 zeta / (x + 2 zeta).
                x, zeta = self._rigid_dt / self.rigid_contact_time, self.rigid_contact_damping
                wp.launch(
                    rigid_contact_soft_kernel,
                    dim=m_r,
                    inputs=[
                        *(rc.body, rc.anchor, rc.normal, self._body_slot_d, self._rigid_x_live_d),
                        *(self._rigid_q_live_d, self._rigid_x_prev_d, state_in.body_q, state_in.body_qd),
                        *(model.body_com, wp.float64(self._rigid_dt), wp.float64(1.0 / (x * x + 2.0 * zeta * x))),
                        *(wp.float64(2.0 * zeta / (x + 2.0 * zeta)), m_s),
                    ],
                    outputs=list(self._rc_soft),
                    device=device,
                )
        if m_t:
            from .rigid_contact import (  # noqa: PLC0415
                tendon_anchor_kernel,
                tendon_friction_kernel,
                tendon_jacobian_kernel,
            )

            # The step's friction coefficient bounds the slide row by mu times the tension (the normal row's
            # multiplier): the tendon's coefficient plus its friction force over last step's tension, at most 1.
            body, point, target, _middle, _rest, mu = self._tendons
            wp.launch(
                tendon_friction_kernel,
                dim=m_t,
                inputs=[*self._tendon_friction, self._tendon_tension, m_s + m_r],
                outputs=[mu, self._rc_mu_d],
                device=device,
            )
            if self.tendon_time > 0.0:
                from .rigid_contact import tendon_soft_kernel  # noqa: PLC0415

                x_t, zeta_t = self._rigid_dt / self.tendon_time, self.tendon_damping
                wp.launch(
                    tendon_soft_kernel,
                    dim=m_t,
                    inputs=[
                        *(body, point, target, self._tendons[3], self._tendons[4], model.body_com, state_in.body_q),
                        *(
                            wp.float64(1.0 / (x_t * x_t + 2.0 * zeta_t * x_t)),
                            wp.float64(2.0 * zeta_t / (x_t + 2.0 * zeta_t)),
                        ),
                        m_s + m_r,
                    ],
                    outputs=list(self._rc_soft),
                    device=device,
                )
            wp.launch(
                tendon_jacobian_kernel,
                dim=m_t,
                inputs=[
                    *(body, point, target, mu, self._body_slot_d, model.body_com, self._rigid_q_pin_d),
                    *(state_in.body_q, n_s + 3 * m_r),
                ],
                outputs=[self._rc_row_slot_d, self._rc_row_jac_d],
                device=device,
            )
            ends = getattr(self, "_tendon_ends", None)
            if ends is None or ends[0].shape[0] != m_t:
                ends = self._tendon_ends = tuple(wp.empty((m_t, 2), dtype=wp.vec3d, device=device) for _ in range(3))
            wp.launch(
                tendon_anchor_kernel,
                dim=m_t,
                inputs=[
                    *(body, point, target, self._body_slot_d, self._rigid_x_prev_d, state_in.body_q),
                    *(state_in.body_qd, model.body_com, wp.float64(dt)),
                ],
                outputs=list(ends),
                device=device,
            )
        if joint_rows.size:
            wp.launch(
                select_rows_kernel,
                dim=joint_rows.size,
                inputs=[self._rc_joint_rows_d, self._joint_row_slot_d, self._joint_row_jac_d, self._rc_joint_offset],
                outputs=[self._rc_row_slot_d, self._rc_row_jac_d],
                device=device,
            )
        if attach:
            from .rigid_contact import attachment_body_jacobian_kernel  # noqa: PLC0415

            wp.launch(
                attachment_body_jacobian_kernel,
                dim=n_b,
                inputs=[
                    *(self._constraint_body_d, self._constraint_point_d, self._constraint_rows[2]),
                    *(self._body_slot_d, model.body_com, self._rigid_q_pin_d, state_in.body_q, n - n_b),
                ],
                outputs=[self._rc_row_slot_d, self._rc_row_jac_d],
                device=device,
            )
        if self.nsn_sparse_storage:
            n_pad = self._rc_n_pad
            self._rc_W_pattern = self._two_way_pattern(n_pad, n_s, n_pad - n_b)
        else:
            self._rc_W_d = wp.empty((n, n), dtype=wp.float64, device=device)
        self._compose_rigid_delassus()
        if attach:
            # Every row's body sides but the joint rows', whose forces the joint system applies.
            row_slot = self._rc_row_slot_d.numpy()[:n].copy()
            row_slot[self._rc_joint_offset : self._rc_joint_offset + joint_rows.size] = -1
            incidence = row_incidence(row_slot, self._rigid_bodies_d.shape[0])
            self._rc_incidence = [wp.array(a, dtype=wp.int32, device=device) for a in incidence]
        elif m_s:
            incidence = row_incidence(np.repeat(slots, 3, axis=0), self._rigid_bodies_d.shape[0])
            self._rc_incidence = [wp.array(a, dtype=wp.int32, device=device) for a in incidence]
        else:
            self._rc_incidence = list(slot_entries(self._rc_row_slot_d, n_c, self._rigid_bodies_d.shape[0]))
        if storage is None:
            self._rc_value_d = wp.empty(n, dtype=wp.float64, device=device)
            self._rc_r_d = wp.empty(n, dtype=wp.float64, device=device)
            self._rc_lam_d = wp.zeros(n, dtype=wp.float64, device=device)
            self._rc_omega_d = wp.zeros(n, dtype=wp.float64, device=device)
        else:
            self._rc_value_d, self._rc_r_d = storage.value, storage.r
            self._rc_lam_d, self._rc_omega_d = storage.lam, storage.omega
        if model.world_count > 1:
            from .nsn_pcr_solver import group_world_rows  # noqa: PLC0415

            soft = self._particle_world_h[host_head(self._contact_particle_d, m_s)] if m_s else np.empty(0, int)
            rigid = np.max(self._shape_world_h[rc.shape], axis=1) if m_r else np.empty(0, int)
            # Without two-way coupling, retain the world's separate deformable and rigid recurrences.
            rigid_group = rigid + model.world_count * (self._two_way_world_h[rigid] == 0)
            joint_world = self._joint_world_h[joint_rows]
            joint_group = joint_world + model.world_count * (self._two_way_world_h[joint_world] == 0)
            worlds = np.concatenate(
                [
                    np.repeat(soft, 3),
                    np.repeat(rigid_group, 3),
                    joint_group,
                    self._constraint_world_h[:n_b],
                ]
            )
            self._rigid_world_rows = group_world_rows(worlds, device)
            soft_scale = (self._regularized_world_h[soft] | self._two_way_world_h[soft].astype(bool)) if m_s else []
            scale = 1.0 + self.contact_regularization * np.r_[soft_scale, np.ones(m_r + m_t)]
            self._rigid_contact_scale = wp.array(scale, dtype=wp.float64, device=device)

    def _shape_body_host(self) -> np.ndarray:
        """The model's shape-to-body map on the host, read once (the model's, fixed after finalize)."""
        if getattr(self, "_shape_body_h", None) is None:
            self._shape_body_h = self.model.shape_body.numpy()
        return self._shape_body_h

    def _reduce_rigid_contacts(self, rc, state_in: State):
        """Keep one contact per bin of each body pair: the contacts of a pair binned by their side-0 anchor, in cells
        of ``rigid_contact_bin`` [m], and by their normal, rounded to the 26 directions of a 3x3x3 grid, and in each
        bin the deepest at the step-start poses. A body resting on finely tessellated geometry, such as a thin-walled
        cup of many prisms on a bolt of many convex parts, otherwise takes thousands of contacts, whose rows all
        couple in ``W`` through the body: its cost grows with their square. The kept contacts keep their order."""
        from .rigid_contact import RigidContacts  # noqa: PLC0415

        m = rc.count
        if m <= 1:
            return rc
        body = rc.body.numpy()
        anchor_h, normal_h, mu_h = rc.anchor.numpy(), rc.normal.numpy(), rc.mu.numpy()
        anchor, normal = anchor_h.astype(np.float64), normal_h.astype(np.float64)
        q = state_in.body_q.numpy().astype(np.float64)
        if getattr(self, "_body_com_h", None) is None:
            self._body_com_h = self.model.body_com.numpy().astype(np.float64)  # the model's, fixed
        com = self._body_com_h
        world = np.empty((m, 2, 3))
        for k in range(2):
            b = body[:, k]
            dyn = b >= 0
            world[~dyn, k] = anchor[~dyn, k]
            if dyn.any():
                bq = q[b[dyn]]
                x, y, z, w = bq[:, 3], bq[:, 4], bq[:, 5], bq[:, 6]
                R = np.stack(
                    [
                        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
                    ],
                    1,
                )
                c_world = bq[:, :3] + np.einsum("nij,nj->ni", R, com[b[dyn]])
                world[dyn, k] = c_world + np.einsum("nij,nj->ni", R, anchor[dyn, k])
        gap = np.einsum("ij,ij->i", world[:, 1] - world[:, 0], normal)
        cell = np.floor(anchor[:, 0] / self.rigid_contact_bin).astype(np.int64)
        direction = np.rint(normal * 1.4).astype(np.int64)  # 0.36 from the axes: 26 directions
        keys = np.concatenate([body.astype(np.int64), cell, direction], axis=1)
        # One contact per bin, the deepest; ties keep the earlier contact (np.lexsort is stable).
        order = np.lexsort((np.arange(m), gap, *keys.T[::-1]))
        k_sorted = keys[order]
        first = np.ones(m, dtype=bool)
        first[1:] = (k_sorted[1:] != k_sorted[:-1]).any(axis=1)
        keep = np.sort(order[first])
        if len(keep) == m:
            return rc
        device = self._device
        return RigidContacts(
            slot=np.ascontiguousarray(rc.slot[keep]),
            shape=np.ascontiguousarray(rc.shape[keep]),
            body=wp.array(body[keep], dtype=wp.int32, device=device),
            anchor=wp.array(anchor_h[keep], dtype=wp.vec3, device=device),
            normal=wp.array(normal_h[keep], dtype=wp.vec3, device=device),
            mu=wp.array(mu_h[keep], dtype=wp.float64, device=device),
        )

    _CONTACT_ROW_ARRAYS = (
        "_contact_particle_d",
        "_contact_record_d",
        "_contact_normal_d",
        "_contact_alpha_d",
        "_contact_offset_d",
        "_contact_tangent1_d",
        "_contact_tangent2_d",
        "_contact_shape_d",
        "_contact_body_pos_d",
        "_contact_world_anchor_d",
        "_contact_is_spinning_d",
        "_contact_tangent1_offset_d",
        "_contact_tangent2_offset_d",
        "_contact_v_anchor_d",
        "_contact_tangent1_offset_shifted_d",
        "_contact_tangent2_offset_shifted_d",
        "_contact_mu_d",
    )

    def _group_deformable_worlds(self) -> None:
        """The worlds of the step's particle contacts and the deformable NSN's rows grouped by world."""
        from .nsn_pcr_solver import group_world_rows  # noqa: PLC0415

        M = self._contact_count
        particles = host_head(self._contact_particle_d, M) if M else np.empty(0, dtype=np.int32)
        worlds = np.repeat(self._particle_world_h[particles], 3 if self.friction else 1)
        self._contact_world_h = self._particle_world_h[particles]
        self._regularized_world_h = np.zeros(self.model.world_count, dtype=bool)
        if M and self._contact_extra is not None:
            extra = host_head(self._contact_extra[0], M)
            self._regularized_world_h[self._contact_world_h[(extra >= 0).any(axis=1)]] = True
        scale = 1.0 + self.contact_regularization * self._regularized_world_h[self._contact_world_h]
        self._deformable_contact_scale = wp.array(scale, dtype=wp.float64, device=self._device)
        self._deformable_world_rows = group_world_rows(np.r_[worlds, self._constraint_world_h], self._device)

    def _split_two_way_contacts(self) -> None:
        """Split the step's rows when some worlds' particles touch an integrated body and other worlds have particle
        contacts or constraint rows: the former worlds solve all of theirs with the body rows, the others keep the
        deformable NSN. ``_mixed`` then holds the two-way rows' selection, the deformable ones' follows the friction
        anchors; it stays ``None`` when one of the two NSN solves all rows."""
        M = self._contact_count
        if self.model.world_count == 1 or M == 0:
            return
        shape = host_head(self._contact_shape_d, M)
        body = np.where(shape >= 0, self._shape_body_host()[np.maximum(shape, 0)], -1)
        coupled = np.zeros(self.model.world_count, dtype=bool)
        coupled[self._contact_world_h[np.where(body >= 0, self._body_slot_h[np.maximum(body, 0)], -1) >= 0]] = True
        contact, row = coupled[self._contact_world_h], coupled[self._constraint_world_h]
        if not contact.any() or (contact.all() and row.all()):
            return
        selection = (np.flatnonzero(contact), np.flatnonzero(row), np.flatnonzero(~contact), np.flatnonzero(~row))
        two_way, rows = self._select_rows(*selection[:2])
        base = self._linear_solver
        if self._two_way_ls is None or self._two_way_ls[0] is not base:
            self._two_way_ls = (base, base.schur_view())
        # The two-way rows take their own Schur rows, ``W`` cache and NSN graph.
        two_way.update(_linear_solver=self._two_way_ls[1], _cached_W=False, _nsn_graph=None)
        self._mixed = SimpleNamespace(selection=selection, two_way=two_way, two_way_rows=rows)

    def _select_rows(self, contacts: np.ndarray, rows: np.ndarray) -> tuple[dict, wp.array]:
        """The per-contact arrays and constraint rows of the selected contacts and rows, in their order, with their
        multipliers, and the rows of those multipliers in the step's ``[3 contacts | constraint rows]``."""
        from .joints import gather_rows_2d_kernel, gather_rows_kernel  # noqa: PLC0415

        device, M = self._device, self._contact_count

        def gather(a, index):
            out = wp.empty((max(index.shape[0], 1), *a.shape[1:]), dtype=a.dtype, device=device)
            if index.shape[0]:
                kernel = gather_rows_kernel if a.ndim == 1 else gather_rows_2d_kernel
                wp.launch(kernel, (index.shape[0], *a.shape[1:]), [a, index], [out], device=device)
            return out

        index = wp.array(contacts, dtype=wp.int32, device=device)
        view = {name: gather(getattr(self, name), index) for name in self._CONTACT_ROW_ARRAYS}
        view["_contact_count"] = len(contacts)
        view["_contact_world_h"] = self._contact_world_h[contacts]
        view["_contact_extra"] = (
            None if self._contact_extra is None else [gather(a, index) for a in self._contact_extra]
        )
        view["_contact_particle_h"] = view["_contact_alpha_h"] = None
        rows_d = wp.array(rows, dtype=wp.int32, device=device)
        constraints = self._constraint_rows
        view["_constraint_rows"] = [gather(a, rows_d) for a in constraints] if len(rows) else None
        view["_constraint_world_h"] = self._constraint_world_h[rows]
        selected = np.r_[(3 * contacts[:, None] + np.arange(3)).reshape(-1), 3 * M + rows].astype(np.int32)
        selected = wp.array(selected, dtype=wp.int32, device=device)
        for name in ("_lam_coulomb_persistent_d", "_omega_coulomb_persistent_d"):
            view[name] = gather(getattr(self, name), selected)
        return view, selected

    def _swap_rows(self, view: dict) -> None:
        """Exchange the solver's rows with those of ``view``; a second call exchanges them back, as :meth:`step`
        does when it raises in between."""
        for name, value in view.items():
            view[name] = getattr(self, name, None)
            setattr(self, name, value)
        self._swapped_rows = None if self._swapped_rows is view else view

    def _begin_mixed_rows(self) -> int:
        """Start the two-way rows of a split step from the step's multipliers and select the deformable rows, after
        the friction anchors moved their offsets; returns the deformable constraint row count."""
        from .joints import gather_values_kernel  # noqa: PLC0415

        mixed, device = self._mixed, self._device
        n_s, n, n_b = 3 * self._rc_counts[0], self._rc_n, self._rc_n_b
        rows = mixed.two_way_rows
        for src, dst in (
            (self._lam_coulomb_persistent_d, self._rc_lam_d),
            (self._omega_coulomb_persistent_d, self._rc_omega_d),
        ):
            if n_s:
                wp.launch(gather_values_kernel, n_s, [src, rows[:n_s], 0], [dst], device=device)
            if n_b:
                wp.launch(gather_values_kernel, n_b, [src, rows[n_s:], n - n_b], [dst], device=device)
        mixed.deformable, mixed.deformable_rows = self._select_rows(*mixed.selection[2:])
        mixed.deformable.update(_deformable_world_rows=None, _deformable_contact_scale=None, _regularized_world_h=None)
        self._swap_rows(mixed.deformable)
        self._group_deformable_worlds()
        self._swap_rows(mixed.deformable)
        # Both NSN solves of every iteration share the NSN buffers: sized for both at once, neither's graph loses them.
        size = max(3 * len(mixed.selection[2]) + len(mixed.selection[3]), n)
        self._ensure_nsn_inner_buffers(size, alloc_2d=not self.nsn_sparse_storage)
        self._ensure_pcr_solver(size)
        if self.nsn_sparse_storage:
            self._ensure_nsn_sparse_scratch(size)
        return len(mixed.selection[3])

    def _end_mixed_rows(self) -> None:
        """Return the multipliers of both NSN solves of a split step to the step's rows, in the contacts' order."""
        from .joints import scatter_values_kernel  # noqa: PLC0415

        mixed, device = self._mixed, self._device
        n_s, n, n_b = 3 * self._rc_counts[0], self._rc_n, self._rc_n_b
        rows, deformable = mixed.deformable_rows, mixed.deformable
        for name, two_way in (
            ("_lam_coulomb_persistent_d", self._rc_lam_d),
            ("_omega_coulomb_persistent_d", self._rc_omega_d),
        ):
            dst = getattr(self, name)
            wp.launch(scatter_values_kernel, rows.shape[0], [deformable[name], 0, rows], [dst], device=device)
            if n_s:
                wp.launch(scatter_values_kernel, n_s, [two_way, 0, mixed.two_way_rows[:n_s]], [dst], device=device)
            if n_b:
                inputs = [two_way, n - n_b, mixed.two_way_rows[n_s:]]
                wp.launch(scatter_values_kernel, n_b, inputs, [dst], device=device)

    def _warm_start_rows(self) -> tuple:
        """This step's NSN rows by kind, as ``(lam, omega, first row, count)`` for the particle contacts, the rigid
        contacts, the joint rows and the particle constraint rows, ``None`` for a kind the step does not solve."""
        particle = rigid = joint = bilateral = None
        if self._two_way and self._mixed is None:
            (m_s, m_r), rows = self._rc_counts, (self._rc_lam_d, self._rc_omega_d)
            particle, rigid = (*rows, 0, m_s), (*rows, 3 * m_s, m_r)
            joint = (*rows, self._rc_joint_offset, len(self._rc_joint_rows_h))
            bilateral = (*rows, self._rc_n - self._rc_n_b, self._rc_n_b)
            return particle, rigid, joint, bilateral
        M = self._contact_count
        n_b = 0 if self._constraint_rows is None else self._constraint_rows[0].shape[0]
        if M or n_b:
            rows = (self._lam_coulomb_persistent_d, self._omega_coulomb_persistent_d)
            particle, bilateral = (*rows, 0, M), (*rows, 3 * M, n_b)
        if self._rc_n:
            # Beside the deformable rows, the two-way particle contact rows lead the rigid rows.
            (m_s, m_r), rows = self._rc_counts, (self._rc_lam_d, self._rc_omega_d)
            rigid, joint = (*rows, 3 * m_s, m_r), (*rows, self._rc_joint_offset, len(self._rc_joint_rows_h))
        return particle, rigid, joint, bilateral

    def _restore_warm_start(self) -> None:
        """Start this step's NSN rows from the rows of the same contacts and constraints that the previous step
        ended with."""
        from .rigid_contact import rigid_contact_tangents_kernel  # noqa: PLC0415
        from .warm_start import index_keys_kernel, particle_contact_keys_kernel, rigid_contact_keys  # noqa: PLC0415

        ws, device = self._warm_start, self._device
        particle, rigid, joint, bilateral = self._warm_start_rows()

        def particle_keys(keys, order):
            extra = self._contact_extra
            first = extra[0] if extra is not None else wp.full((1, 1), -1, dtype=wp.int32, device=device)
            inputs = [self._contact_particle_d, self._contact_shape_d, first, int(extra is not None)]
            inputs += [self.model.shape_count]
            wp.launch(particle_contact_keys_kernel, self._contact_count, inputs, [keys, order], device=device)

        def given_keys(values):
            def write(keys, order):
                wp.launch(
                    index_keys_kernel,
                    len(values),
                    [wp.array(values, dtype=wp.int64, device=device)],
                    [keys, order],
                    device=device,
                )

            return write

        ws.particle.begin(0 if particle is None else particle[3], particle_keys)
        if particle is not None:
            ws.particle.restore(*particle[:3], (self._contact_tangent1_d, self._contact_tangent2_d))
        self._rc_tangents = None
        if rigid is not None and rigid[3]:
            rc = self._rigid_contacts
            self._rc_tangents = [wp.empty(rc.count, dtype=wp.vec3, device=device) for _ in range(2)]
            wp.launch(rigid_contact_tangents_kernel, rc.count, [rc.normal], self._rc_tangents, device=device)
            ws.rigid.begin(rc.count, given_keys(rigid_contact_keys(rc.shape, self.model.shape_count)))
            ws.rigid.restore(*rigid[:3], self._rc_tangents, rc.anchor, rc.normal)
        else:
            ws.rigid.begin(0, None)
        if joint is not None and joint[3]:
            ws.joint.begin(joint[3], given_keys(self._rc_joint_rows_h.astype(np.int64)))
            ws.joint.restore(*joint[:3])
        else:
            ws.joint.begin(0, None)
        if bilateral is not None:
            ws.restore_bilateral(*bilateral)
        m_t = getattr(self, "_rc_tendons", 0)
        if m_t and self._rc_n and self._tendon_ws is not None and self._tendon_ws[2] == m_t:
            first = 3 * (self._rc_counts[0] + self._rc_counts[1])
            wp.copy(self._rc_lam_d, self._tendon_ws[0], dest_offset=first, count=3 * m_t)
            wp.copy(self._rc_omega_d, self._tendon_ws[1], dest_offset=first, count=3 * m_t)

    def _store_tendon_tension(self) -> None:
        """Keep each tendon's tension at the end of the step, for the next step's friction coefficient."""
        m_t = getattr(self, "_rc_tendons", 0)
        if not m_t or not self._rc_n or self._tendon_tension is None:
            return
        from .rigid_contact import tendon_tension_kernel  # noqa: PLC0415

        wp.launch(
            tendon_tension_kernel,
            dim=m_t,
            inputs=[self._rc_lam_d, 3 * (self._rc_counts[0] + self._rc_counts[1]), wp.float64(self._rigid_dt)],
            outputs=[self._tendon_tension],
            device=self._device,
        )

    def _store_warm_start(self) -> None:
        """Keep this step's final NSN rows for the next step's warm start."""
        ws = self._warm_start
        particle, rigid, joint, bilateral = self._warm_start_rows()
        if particle is not None:
            ws.particle.store(*particle[:3], (self._contact_tangent1_d, self._contact_tangent2_d))
        else:
            ws.particle.store(None, None, 0)
        if rigid is not None and rigid[3]:
            rc = self._rigid_contacts
            ws.rigid.store(*rigid[:3], self._rc_tangents, rc.anchor, rc.normal)
        else:
            ws.rigid.store(None, None, 0)
        if joint is not None and joint[3]:
            ws.joint.store(*joint[:3])
        else:
            ws.joint.store(None, None, 0)
        ws.store_bilateral(*(bilateral if bilateral is not None else (None, None, 0, 0)))
        m_t = getattr(self, "_rc_tendons", 0)
        if m_t and self._rc_n:
            # The tendons keep their rows from step to step: their forces carry over by tendon.
            first = 3 * (self._rc_counts[0] + self._rc_counts[1])
            if self._tendon_ws is None or self._tendon_ws[2] != m_t:
                self._tendon_ws = (
                    wp.empty(3 * m_t, dtype=wp.float64, device=self._device),
                    wp.empty(3 * m_t, dtype=wp.float64, device=self._device),
                    m_t,
                )
            wp.copy(self._tendon_ws[0], self._rc_lam_d, src_offset=first, count=3 * m_t)
            wp.copy(self._tendon_ws[1], self._rc_omega_d, src_offset=first, count=3 * m_t)

    def _two_way_pattern(self, n: int, n_s: int, soft_tail: int) -> tuple:
        """CSR pattern of the two-way ``W`` of ``nsn_sparse_storage``: rows couple where their deformable rows'
        particles share a component of the system, or where they share a body slot."""
        from .linear_solver import group_pattern, pattern_from_groups  # noqa: PLC0415
        from .rigid import slot_entries  # noqa: PLC0415

        if not n_s and soft_tail >= n:
            # Rows on bodies alone: the body slots are the groups, sorted on the device. Padded rows keep the pattern
            # in their storage, of blocks of three rows that couple where any of their rows share a slot.
            slots = self._rigid_bodies_d.shape[0]
            if self._rc_blocked:
                from .rigid import BLOCK_GROUPS, block_slot_groups_kernel, group_entries  # noqa: PLC0415

                groups = wp.empty((n // 3, BLOCK_GROUPS), dtype=wp.int32, device=self._device)
                wp.launch(block_slot_groups_kernel, n // 3, [self._rc_row_slot_d], [groups], device=self._device)
                start, rows = group_entries(groups, slots)
                return pattern_from_groups(groups, start, rows, self._rc_padded.blocks)
            start, rows, _ = slot_entries(self._rc_row_slot_d, n, slots, distinct=True)
            storage = None if self._rc_padded is None else self._rc_padded.csr
            return pattern_from_groups(self._rc_row_slot_d, start, rows, storage)
        slots = self._rc_row_slot_d.numpy()
        width = self._linear_solver._row_groups.shape[1]
        groups = np.full((n, width + 2), -1, dtype=np.int64)
        if n_s or soft_tail < n:
            # The deformable rows: the contact rows, and after the rigid and joint rows the constraint rows.
            soft = np.concatenate([np.arange(n_s), np.arange(soft_tail, n)])
            groups[soft, :width] = self._linear_solver._row_groups[: len(soft)]
        # Body slots number after every particle, which numbers the components.
        groups[:, width:] = np.where(slots >= 0, self.model.particle_count + slots, -1)
        return group_pattern(groups, self._device)

    def _compose_rigid_delassus(self) -> None:
        """Compose the two-way ``W`` from the deformable block and the body slots of the rows."""
        from .rigid import rigid_delassus_kernel  # noqa: PLC0415

        n, n_s, device = self._rc_n_pad, 3 * self._rc_counts[0], self._device
        if self.nsn_sparse_storage:
            from .linear_solver import csr_without_zeros  # noqa: PLC0415
            from .rigid import DELASSUS_LANES, rigid_delassus_rows_kernel, rigid_row_minv_jac_kernel  # noqa: PLC0415

            offsets, columns, nnz = self._rc_W_pattern
            padded = self._rc_padded
            has_soft = bool(n_s or self._rc_n_b)
            soft = self._linear_solver.W_sparse[0] if has_soft else None
            soft = (
                [soft.offsets, soft.columns, soft.values]
                if has_soft
                else [wp.zeros(1, dtype=wp.int32, device=device)] * 2
            )
            if not has_soft:
                soft.append(wp.empty(1, dtype=wp.float64, device=device))
            # Each row side's A⁻¹ Jᵀ once, then each row's entries, or each block row's blocks, by a block of lanes.
            minv_jac = wp.empty_like(self._rc_row_jac_d)
            inputs = [self._rc_row_slot_d, self._rc_row_jac_d, self._slot_inv_mass_d]
            wp.launch(rigid_row_minv_jac_kernel, (n, 2), inputs=inputs, outputs=[minv_jac], device=device)
            if self._rc_blocked:
                from .rigid import rigid_delassus_blocks_kernel  # noqa: PLC0415

                W = padded.blocks.matrix
                wp.launch_tiled(
                    rigid_delassus_blocks_kernel,
                    dim=n // 3,
                    inputs=[offsets, columns, self._rc_row_slot_d, self._rc_row_jac_d, minv_jac],
                    outputs=[W.values],
                    block_dim=DELASSUS_LANES,
                    device=device,
                )
                self._rc_W_sparse = (W, None)
                return
            # Padded rows, on bodies alone, take W at their pattern in their storage, its few zero entries
            # included; the other rows drop the zero entries of their pattern.
            values = (
                padded.csr.matrices[0].values
                if padded is not None
                else wp.empty(max(nnz, 1), dtype=wp.float64, device=device)
            )
            inputs = [offsets, columns, *soft, n_s, n - self._rc_n_b, self._rc_row_slot_d, self._rc_row_jac_d, minv_jac]
            wp.launch_tiled(
                rigid_delassus_rows_kernel,
                dim=n,
                inputs=inputs,
                outputs=[values],
                block_dim=DELASSUS_LANES,
                device=device,
            )
            self._rc_W_sparse = (
                padded.csr.matrices if padded is not None else csr_without_zeros(offsets, columns, values, n, nnz)
            )
            return
        W_soft = (
            self._linear_solver.W_device_view()
            if n_s or self._rc_n_b
            else wp.empty((0, 0), dtype=wp.float64, device=device)
        )
        wp.launch(
            rigid_delassus_kernel,
            dim=(n, n),
            inputs=[W_soft, n_s, n - self._rc_n_b, self._rc_row_slot_d, self._rc_row_jac_d, self._slot_inv_mass_d],
            outputs=[self._rc_W_d],
            device=device,
        )

    def _point_length_rows(self, x: wp.array) -> None:
        """Point the length rows of :meth:`set_particle_constraints` along their combination at ``x``."""
        from .kernels import particle_constraint_direction_kernel  # noqa: PLC0415

        particles, coefficients, directions, _ = self._constraint_rows
        wp.launch(
            particle_constraint_direction_kernel,
            dim=directions.shape[0],
            inputs=[particles, coefficients, x],
            outputs=[directions],
            device=self._device,
        )

    def _solve_rigid_contacts(self, state_in: State, dt: float) -> wp.array[wp.float64]:
        """Solve the NSN of the rigid contact rows from their live residual; returns the row forces [N].

        The residual ``-g + J (u - u_free)`` makes the NSN prediction
        ``-r + dt² W (omega λ)`` reproduce the live gap ``g`` for the forces
        that produced ``u``, while ``W`` and ``J`` stay at the pinned frame. The
        joint rows solved here get their forces written back to the joint forces.
        Deformable rows also correct the particle positions.
        """
        from .joints import gather_values_kernel  # noqa: PLC0415
        from .kernels import particle_constraint_residual_kernel  # noqa: PLC0415
        from .rigid import rigid_row_residual_kernel  # noqa: PLC0415
        from .rigid_contact import rigid_contact_gap_kernel  # noqa: PLC0415

        rc, device, n, n_pad = self._rigid_contacts, self._device, self._rc_n, self._rc_n_pad
        (m_s, m_r), joint_rows, n_b = self._rc_counts, self._rc_joint_rows_d, self._rc_n_b
        m_t = self._rc_tendons
        n_s, n_c, n_j, j0 = 3 * m_s, 3 * m_s + 3 * (m_r + m_t), len(self._rc_joint_rows_h), self._rc_joint_offset
        if n_b and self._constraint_lengths:
            self._point_length_rows(self._x_cur)
            self._linear_solver.update_row_directions(self._constraint_rows[2], n_s)
            self._compose_rigid_delassus()
        live = [self._rigid_x_live_d, self._rigid_q_live_d]
        if m_s:
            from .rigid_contact import particle_contact_body_residual_kernel  # noqa: PLC0415

            self._launch_contact_residual(self._rc_r_d)
            wp.launch(
                particle_contact_body_residual_kernel,
                dim=(m_s, 3),
                inputs=[
                    *self._particle_contact_args()[:5],
                    *(self._contact_world_anchor_d, self._contact_v_anchor_d, self.model.shape_body),
                    *(self._body_slot_d, self.model.body_com, *live, state_in.body_q, self._rc_row_jac_d),
                    *(self._rigid_u_d, self._rigid_u_free_d, wp.float64(dt)),
                ],
                outputs=[self._rc_r_d],
                device=device,
            )
        if m_r:
            wp.launch(
                rigid_contact_gap_kernel,
                dim=m_r,
                inputs=[
                    *(rc.body, rc.normal, *self._rc_anchors, self._body_slot_d, self._rigid_x_live_d),
                    *(self._rigid_turn_d, n_s),
                ],
                outputs=[self._rc_value_d],
                device=device,
            )
        if m_t:
            from .rigid_contact import tendon_gap_kernel  # noqa: PLC0415

            body, _point, target, middle, rest, mu = self._tendons
            wp.launch(
                tendon_gap_kernel,
                dim=m_t,
                inputs=[
                    *(body, target, middle, rest, mu, *self._tendon_ends, self._body_slot_d),
                    *(self._rigid_x_live_d, self._rigid_turn_d, n_s + 3 * m_r),
                ],
                outputs=[self._rc_value_d],
                device=device,
            )
        if n_j:
            wp.launch(
                gather_values_kernel,
                dim=n_j,
                inputs=[self._joint_residual_d, joint_rows, j0],
                outputs=[self._rc_value_d],
                device=device,
            )
        # The rigid contact rows and the joint rows, which follow the contacts' capacity in padded storage.
        spans = [(n_s, n_c + n_j)] if j0 == n_c else [(n_s, n_c), (j0, j0 + n_j)]
        for lo, hi in spans:
            if hi > lo:
                wp.launch(
                    rigid_row_residual_kernel,
                    dim=hi - lo,
                    inputs=[
                        *(self._rc_value_d[lo:], self._rc_row_slot_d[lo:], self._rc_row_jac_d[lo:]),
                        *(self._rigid_u_d, self._rigid_u_free_d),
                    ],
                    outputs=[self._rc_r_d[lo:]],
                    device=device,
                )
        if n_b:
            wp.launch(
                particle_constraint_residual_kernel,
                dim=n_b,
                inputs=[*self._constraint_rows, self._x_cur, wp.int32(n - n_b)],
                outputs=[self._rc_r_d, self._rc_pene0_d],
                device=device,
            )
            if self._constraint_body_h is not None:
                from .rigid_contact import attachment_body_residual_kernel  # noqa: PLC0415

                wp.launch(
                    attachment_body_residual_kernel,
                    dim=n_b,
                    inputs=[
                        *(self._constraint_body_d, self._constraint_point_d, self._constraint_rows[2]),
                        *(self._body_slot_d, self.model.body_com, *live, state_in.body_q, self._rc_row_jac_d),
                        *(self._rigid_u_d, self._rigid_u_free_d, wp.int32(n - n_b)),
                    ],
                    outputs=[self._rc_r_d, self._rc_pene0_d],
                    device=device,
                )
        if m_s or n_b:
            self._linear_solver.reserve_lambda_correction()
        tail_objects = (self._linear_solver, self._rc_lam_d, self._rc_omega_d, self._rc_lam_soft_d, self._rhs)
        tail_objects += tuple(
            getattr(self, name, None)
            for name in ("_x_cur", "_lam_coulomb_persistent_d", "_omega_coulomb_persistent_d", "_joint_force_d")
        )
        self._solve_nsn_coulomb_gpu(
            None,
            None,
            None,
            None,
            max_iters=1,
            dt=dt,
            W_device=None if self.nsn_sparse_storage else self._rc_W_d,
            W_sparse=self._rc_W_sparse if self.nsn_sparse_storage else None,
            graph="two-way",
            world_rows=self._rigid_world_rows,
            contact_scale=self._rigid_contact_scale,
            graph_tail=functools.partial(
                self._two_way_contact_tail, m_s, n_pad, n_s, n_b, j0, joint_rows if n_j else None
            ),
            graph_tail_objects=tail_objects,
            r_device=self._rc_r_d,
            pene0_device=self._rc_pene0_d,
            mu_device=self._rc_mu_d,
            lam_init_device=self._rc_lam_d,
            omega_init_device=self._rc_omega_d,
            skip_download=True,
            # Padding rows of the storage follow the contact rows as contacts, and the joint rows as joint rows.
            m_contacts=j0 // 3,
            n_bilateral=n_pad - j0,
            # The joint rows' block of A_schur is their W: Cholesky-factored per articulation.
            block_precond=functools.partial(self._joint_system.solve_rows, joint_rows, j0) if n_j else None,
            contact_regularization=self.contact_regularization,
            soft_rows=self._rc_soft,
        )
        return self._nsn_lam_apply_d

    def _two_way_contact_tail(
        self, m_s: int, n: int, n_s: int, n_b: int, joint_offset: int, joint_rows: wp.array | None
    ) -> None:
        """Keep the two-way NSN step's multipliers, correct the particle positions and write the forces of the joint
        rows ``joint_rows``, which start at row ``joint_offset``."""
        from .joints import scatter_values_kernel  # noqa: PLC0415

        wp.copy(self._rc_lam_d, self._nsn_lam_d, count=n)
        wp.copy(self._rc_omega_d, self._nsn_omega_d, count=n)
        if m_s or n_b:
            # W_soft orders the deformable rows as the contact rows, then the particle constraint rows.
            lam_soft = self._nsn_lam_apply_d
            if n_b:
                # ``count=0`` would copy whole arrays: the rows on bodies may come without contact rows.
                lam_soft = self._rc_lam_soft_d
                if n_s:
                    wp.copy(lam_soft, self._nsn_lam_apply_d, count=n_s)
                wp.copy(lam_soft, self._nsn_lam_apply_d, dest_offset=n_s, src_offset=n - n_b, count=n_b)
            self._linear_solver.apply_lambda_correction_combined(lam_soft, self._rhs, self._x_cur)
            if n_s:
                wp.copy(self._lam_coulomb_persistent_d, self._rc_lam_d, count=n_s)
                wp.copy(self._omega_coulomb_persistent_d, self._rc_omega_d, count=n_s)
        if joint_rows is not None:
            wp.launch(
                scatter_values_kernel,
                dim=joint_rows.shape[0],
                inputs=[self._nsn_lam_apply_d, joint_offset, joint_rows],
                outputs=[self._joint_force_d],
                device=self._device,
            )

    def _retract_rigid(self, state_in: State, state_out: State, dt: float) -> None:
        from .rigid import retract_rigid_kernel  # noqa: PLC0415

        wp.launch(
            retract_rigid_kernel,
            dim=self._rigid_bodies_d.shape[0],
            inputs=[
                self._rigid_bodies_d,
                self._rigid_u_d,
                self._rigid_q_pin_d,
                self._rigid_x_prev_d,
                state_in.body_q,
                self.model.body_com,
                wp.float64(dt),
            ],
            outputs=[
                *(state_out.body_q, state_out.body_qd),
                *(self._rigid_x_live_d, self._rigid_q_live_d, self._rigid_turn_d),
            ],
            device=self._device,
        )

    def _joint_row_args(self) -> list:
        rows = self._joint_rows
        return [
            rows.row_joint,
            rows.row_angular,
            rows.row_dir,
            rows.parent,
            rows.child,
            rows.X_pj,
            rows.X_cj,
            self._body_slot_d,
        ]

    def _rigid_iteration(self, state_in: State, state_out: State, dt: float) -> None:
        """One LG iteration of the integrated bodies, with ``rigid_iteration_graph`` from a CUDA graph where it can.

        The first iteration of a step is captured, launches included of the contact NSN step (which then records
        into this capture rather than its own graph) and of its tail, and runs from the graph; the step's other
        iterations replay it. The iteration reads nothing back to the host, so its launches are the same at every
        iteration of the step. The solver's very first rigid iteration runs eagerly, as it creates the NSN step's
        solvers; so do the iterations of a step whose rows keep their padded storage (the NSN step's own graph then
        replays across steps), without the sparse NSN storage (whose NSN stops by reading the device), with length
        rows (whose directions are set on the host) or whose capture failed.
        """
        capturable = (
            self.rigid_iteration_graph
            and wp.get_device(self._device).is_cuda
            and self.nsn_sparse_storage
            and getattr(self, "_rc_padded", None) is None
            and not (self._rc_n_b and self._constraint_lengths)
            and getattr(self, "_pcr_solver", None) is not None
            and self._pcr_solver.max_iter <= self._pcr_solver.check_every
        )
        first = not getattr(self, "_rigid_iter_warm", False)
        if not capturable or first or self._rigid_iter_graph is False:
            self._rigid_iteration_eager(state_in, state_out, dt)
            self._rigid_iter_count += 1
            self._rigid_iter_warm = True
            return
        if self._rigid_iter_graph is None:
            # Grow the buffers the step uses first, so that nothing allocates inside the capture.
            if self._rc_n:
                n = self._rc_n_pad
                self._ensure_nsn_inner_buffers(n, alloc_2d=False)
                self._ensure_pcr_solver(n)
                self._ensure_nsn_sparse_scratch(n)
                if self._rc_counts[0] or self._rc_n_b:
                    self._linear_solver.reserve_lambda_correction()
            self._outer_capture = True
            try:
                with wp.ScopedCapture(device=self._device) as capture:
                    self._rigid_iteration_eager(state_in, state_out, dt)
                self._rigid_iter_graph = capture.graph
            except RuntimeError:
                self._rigid_iter_graph = False
            finally:
                self._outer_capture = False
            if self._rigid_iter_graph is False:
                self._rigid_iteration_eager(state_in, state_out, dt)
                self._rigid_iter_count += 1
                return
        wp.capture_launch(self._rigid_iter_graph)
        self._rigid_iter_count += 1

    def _rigid_iteration_eager(self, state_in: State, state_out: State, dt: float) -> None:
        """One LG iteration of the integrated bodies.

        Takes a Newton step on the joint forces and solves the contact NSN, both
        from the residual at the live pose, then re-solves ``A u = b + Jᵀ f``
        with all row forces ``f`` and retracts.
        """
        from .joints import joint_residual_kernel  # noqa: PLC0415
        from .rigid import ROW_FORCE_LANES, row_force_rhs_kernel  # noqa: PLC0415

        b = self._rigid_b_d
        if self._joint_rows is not None:
            wp.launch(
                joint_residual_kernel,
                dim=self._joint_residual_d.shape[0],
                inputs=[
                    *self._joint_row_args(),
                    *(self._rigid_x_live_d, self._rigid_q_live_d, state_in.body_q, self.model.body_com),
                ],
                outputs=[self._joint_residual_d],
                device=self._device,
            )
            self._joint_system.update(self._joint_residual_d, self._joint_force_d)
        force = self._solve_rigid_contacts(state_in, dt) if self._rc_n else None
        if self._joint_rows is not None:
            self._joint_system.apply_force(self._joint_row_jac_d, self._joint_force_d, b, self._rigid_b_iter_d)
            b = self._rigid_b_iter_d
        if force is not None:
            wp.launch_tiled(
                row_force_rhs_kernel,
                dim=self._rigid_bodies_d.shape[0],
                inputs=[*self._rc_incidence, self._rc_row_jac_d, force, b],
                outputs=[self._rigid_b_iter_d],
                block_dim=ROW_FORCE_LANES,
                device=self._device,
            )
            b = self._rigid_b_iter_d
        if b is self._rigid_b_d:
            return
        self._rigid_linear_solver.solve_scalar(b, self._rigid_u_d)
        self._retract_rigid(state_in, state_out, dt)

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

    def _constraint_args(self) -> dict:
        """Bilateral particle rows of :meth:`set_particle_constraints`, for ``build_schur_complement``."""
        if self._constraint_rows is None:
            return {}
        particles, coefficients, directions, _ = self._constraint_rows
        return {"bilateral_particles": particles, "bilateral_alpha": coefficients, "bilateral_dir": directions}

    def _contact_offsets(self) -> list:
        """Normal and shifted tangent offsets of the deformable contacts."""
        return [
            self._contact_offset_d,
            self._contact_tangent1_offset_shifted_d,
            self._contact_tangent2_offset_shifted_d,
        ]

    def _particle_contact_args(self) -> list:
        """Shape, anchor and row directions of the deformable contacts, then the shape bodies."""
        return [
            self._contact_shape_d,
            self._contact_body_pos_d,
            self._contact_normal_d,
            self._contact_tangent1_d,
            self._contact_tangent2_d,
            self.model.shape_body,
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

    def _spring_bending_stiffness(self) -> np.ndarray | None:
        """Bending stiffness [N m^2] of the rod each of the model's springs belongs to, for the springs' ADMM
        penalties: the largest of the bending elements that share the spring's segment, zero for a spring outside
        rods, or ``None`` when the model has no rod bending elements."""
        from .linear_solver import rod_bending_elements  # noqa: PLC0415

        model = self.model
        bends, stiffness = rod_bending_elements(model)
        if bends.shape[0] == 0 or not model.spring_count:
            return None
        n = np.int64(max(model.particle_count, 1))

        def segment_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)

        keys = np.concatenate([segment_key(bends[:, 0], bends[:, 1]), segment_key(bends[:, 1], bends[:, 2])])
        values = np.concatenate([stiffness, stiffness])
        order = np.argsort(keys, kind="stable")
        unique, first = np.unique(keys[order], return_index=True)
        largest = np.maximum.reduceat(values[order], first)
        springs = model.spring_indices.numpy().reshape(-1, 2)
        spring_keys = segment_key(springs[:, 0], springs[:, 1])
        at = np.minimum(np.searchsorted(unique, spring_keys), unique.size - 1)
        return np.where(unique[at] == spring_keys, largest[at], 0.0)

    def _setup_springs(self, meta: dict, dt: float) -> None:
        """Upload the springs of :func:`~newton._src.solvers.fba.linear_solver.build_pd_system`'s ``meta`` and
        start their ADMM state: zero duals and splits, and the rest directions; with corotated damping, also each
        spring's damping ``beta k / dt`` of its length rate; and the chains of the tension solve,
        :func:`~newton._src.solvers.fba.kernels.spring_chain_tension_kernel`."""
        from .linear_solver import build_particle_element_csr, particle_masses, spring_chains  # noqa: PLC0415

        springs = meta["spring_indices"]
        if springs.shape[0] == 0:
            self._spring_indices_d = None
            return
        device = self._device
        x0 = self.model.particle_q.numpy().astype(np.float64)
        rest = x0[springs[:, 1]] - x0[springs[:, 0]]
        length = np.linalg.norm(rest, axis=1)
        direction = np.where(length[:, None] > 0.0, rest / np.maximum(length, 1.0e-30)[:, None], [1.0, 0.0, 0.0])
        self._spring_indices_d = wp.array(springs.astype(np.int32), dtype=wp.int32, device=device)
        f32 = (meta["spring_rest_length"], meta["spring_stiffness"], meta["spring_penalty"])
        self._spring_rest_d, self._spring_k_d, self._spring_rho_d = (
            wp.array(a.astype(np.float32), dtype=wp.float32, device=device) for a in f32
        )
        damping = meta["spring_beta"] * meta["spring_stiffness"] / dt
        self._spring_damped = bool((damping > 0.0).any())
        self._spring_c_d = wp.array(damping.astype(np.float32), dtype=wp.float32, device=device)
        self._spring_len0_d = wp.zeros(springs.shape[0], dtype=wp.float32, device=device)
        self._spring_dual_d = wp.zeros(springs.shape[0], dtype=wp.vec3, device=device)
        self._spring_split_d = wp.zeros(springs.shape[0], dtype=wp.vec3, device=device)
        self._spring_dir_d = wp.array(direction.astype(np.float32), dtype=wp.vec3, device=device)
        self._spring_first_d = wp.ones(1, dtype=wp.int32, device=device)
        self._spring_contrib_d = wp.zeros((springs.shape[0], 2), dtype=wp.vec3, device=device)
        csr = build_particle_element_csr(springs, int(self.model.particle_count), 2)
        self._particle_spring_csr_d = tuple(wp.array(a, dtype=wp.int32, device=device) for a in csr)
        # The tension solve: per place along each chain, the diagonal h^2 (w_a + w_b) + 1 / (k + c - rho) and the
        # coupling h^2 w sign to the next spring; springs no stiffer than their penalty take no correction.
        offsets, order, link, sign, scale = spring_chains(springs, int(self.model.particle_count))
        _, inv_mass = particle_masses(self.model)
        stiffness = meta["spring_stiffness"] + damping
        excess = stiffness - meta["spring_penalty"]
        compliance = np.where(excess > 1.0e-6 * stiffness, 1.0 / np.maximum(excess, 1.0e-30), 1.0e30)
        h2 = dt * dt
        diagonal = h2 * (inv_mass[springs] * scale).sum(axis=1) + compliance
        coupling = np.where(link >= 0, h2 * inv_mass[np.maximum(link, 0)] * sign, 0.0)
        # The chains' systems are solved together by parallel cyclic reduction, ceil(log2(L)) levels for the longest
        # chain's L springs, each from one set of rows to the other.
        self._spring_chain_levels = int(np.ceil(np.log2(max(int(np.diff(offsets).max(initial=1)), 1))))
        self._spring_chain_order_d = wp.array(order, dtype=wp.int32, device=device)
        self._spring_chain_diagonal_d = wp.array(diagonal[order].astype(np.float32), dtype=wp.float32, device=device)
        self._spring_chain_link_d = wp.array(coupling.astype(np.float32), dtype=wp.float32, device=device)
        # The rods' bends between consecutive springs of a chain: kappa = EI / (v l_- l_+), with the coupling's signs.
        bend = np.zeros(springs.shape[0], dtype=np.float64)
        rod_bend, rod_weight = meta.get("rod_bend_indices"), meta.get("rod_bend_weight")
        if rod_bend is not None and rod_bend.shape[0] > 0:
            x_rest = self.model.particle_q.numpy().astype(np.float64)
            l_minus = np.linalg.norm(x_rest[rod_bend[:, 1]] - x_rest[rod_bend[:, 0]], axis=1)
            l_plus = np.linalg.norm(x_rest[rod_bend[:, 2]] - x_rest[rod_bend[:, 1]], axis=1)
            kappa = {
                (min(a, c), b, max(a, c)): w / (lm * lp)
                for (a, b, c), w, lm, lp in zip(rod_bend.tolist(), rod_weight, l_minus, l_plus, strict=True)
            }
            # A spring between held particles has no stretch to solve for: its bends couple nothing.
            held_spring = (inv_mass[springs] == 0.0).all(axis=1)
            for place in np.flatnonzero(link >= 0):
                e, f, p = int(order[place]), int(order[place + 1]), int(link[place])
                if held_spring[e] or held_spring[f]:
                    continue
                a = int(springs[e, 0] if springs[e, 1] == p else springs[e, 1])
                c = int(springs[f, 0] if springs[f, 1] == p else springs[f, 1])
                bend[place] = kappa.get((min(a, c), p, max(a, c)), 0.0) * sign[place]
        # A spring held by its bends as firmly as by its own stiffness takes their stiffness in series with its
        # compliance: each place's row of the bends' term scales by 1 / (1 + L_ii C).
        diagonal_bend = np.abs(bend) + np.abs(np.r_[0.0, bend[:-1]] * (np.r_[-1, link[:-1]] >= 0))
        bend_scale = 1.0 / (1.0 + diagonal_bend * compliance[order])
        self._spring_chain_bend_d = wp.array(bend.astype(np.float32), dtype=wp.float32, device=device)
        self._spring_chain_bend_scale_d = wp.array(bend_scale.astype(np.float32), dtype=wp.float32, device=device)
        self._spring_residual_d = wp.zeros(springs.shape[0], dtype=wp.float32, device=device)
        # The rows (lower, diagonal, upper, right-hand side) of the chains' systems, two sets for the reduction.
        self._spring_chain_rows_d = [
            [wp.zeros(springs.shape[0], dtype=wp.float32, device=device) for _ in range(4)] for _ in range(2)
        ]

    def _setup_pd_system(self, dt: float) -> None:
        """Build / rebuild the PD Hessian, factor it, and upload device data."""
        self._update_materials()
        from .linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            build_pd_system,
            connected_components,
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
            spring_bending_stiffness=self._spring_bending_stiffness(),
            spring_corotated=self.rayleigh_corotated,
        )
        worlds = self._particle_world_h if self.model.world_count > 1 else None
        fs = factorize_and_sparse_inverse(A, self.model.particle_q.numpy(), worlds)
        self._linear_solver = FBALinearSolver(fs, device=self._device)
        # The two-way Schur rows of the previous factor go with it.
        self._two_way_ls = None
        self._linear_solver.exact_inverse_entries = self.nsn_sparse_storage
        self._linear_solver.sparse_W = self.nsn_sparse_storage
        if self.nsn_sparse_storage:
            self._linear_solver.components = connected_components(A)
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
        if self._warm_start is not None:
            # The multipliers are forces divided by dt^2: those of another dt do not carry over.
            self._warm_start.clear()
        self._lg_iter_graph = None
        # A changed, so W is stale.
        self._invalidate_schur_cache()
        if self._held_rows is not None:
            self._held_rows[1] = None

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
        self._setup_springs(meta, dt)
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
            self._tet_activation_d = wp.zeros(int(self.model.tet_count), dtype=wp.float32, device=device)
            directions = self._tet_activation_directions
            if directions is None:
                directions = np.zeros((int(self.model.tet_count), 3), dtype=np.float32)
            self._tet_activation_direction_d = wp.array(directions, dtype=wp.vec3, device=device)
        else:
            self._particle_tet_offsets_d = None
            self._particle_tet_element_d = None
            self._particle_tet_local_d = None
            self._tet_contrib_d = None
            self._tet_activation_d = None
            self._tet_activation_direction_d = None

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
        """Advance deformables and rigid bodies by one implicit Euler step using PD.

        Args:
            state_in: Input simulation state. Particle and body positions and
                velocities are read; ``particle_f`` and ``body_f`` provide
                external forces [N] and torques [N·m].
            state_out: Output simulation state. Particle positions and
                velocities are written, and those of the bodies unless another
                solver integrates them; other fields are untouched.
            control: The step's ``tet_activations``, which grow or shrink the
                tetrahedra (see the class notes), or ``None`` to step without
                activations.
            contacts: The step's contacts, detected on ``state_in``, or ``None``
                to step without them. With self contact on, the step still
                finds its self contacts.
            dt: Time step [s]. The first step and a change of ``dt`` factor the
                system again.
            observables: Optional observables from :meth:`observables`; see the
                class notes for :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F`.
        """
        try:
            self._step(state_in, state_out, control, contacts, dt, observables)
        finally:
            # A split step that raises with rows of a world selection in place returns the solver's own.
            if self._swapped_rows is not None:
                self._swap_rows(self._swapped_rows)

    def _step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        observables: SolverBase.Observables | None,
    ) -> None:
        """The body of :meth:`step`."""
        from .kernels import (  # noqa: PLC0415
            compute_inertial_kernel,
            copy_particles_kernel,
            write_velocity_kernel,
        )

        self.validate_observables(observables, contacts)
        contact_f = None
        if observables is not None and observables.is_requested(SolverBase.ObservableKind.CONTACT_F):
            contact_f = observables.contact_f
        self._two_way, self._rc_n, self._mixed = False, 0, None
        self._two_way_world_h.fill(0)
        rigid = self.model.body_count > 0 and not self.integrate_with_external_rigid_solver
        rigid = rigid and self._begin_rigid_step(state_in, state_out, dt)
        if self.model.particle_count == 0:
            if rigid:
                self._begin_rigid_contacts(contacts, state_in, dt)
                if self._warm_start is not None:
                    self._restore_warm_start()
            for _ in range(self.iterations if rigid else 0):
                self._rigid_iteration(state_in, state_out, dt)
            if rigid:
                self._store_tendon_tension()
            if rigid and self._warm_start is not None:
                self._store_warm_start()
            if contact_f is not None:
                contact_f.zero_()
            return

        if self._linear_solver is None or self._dt_setup is None or abs(self._dt_setup - dt) > 1e-12:
            self._setup_pd_system(dt)

        # The step's tet activations, which every local-global iteration's tet projections read.
        if self._tet_activation_d is not None:
            activations = None if control is None else control.tet_activations
            if activations is None:
                self._tet_activation_d.zero_()
            elif activations.shape != (self.model.tet_count,):
                raise ValueError(
                    f"control.tet_activations needs one value per tetrahedron, {self.model.tet_count}, "
                    f"got shape {activations.shape}"
                )
            else:
                wp.copy(self._tet_activation_d, activations)

        # Ingest the step's contacts once; with self contact on, a step without contacts still has self contacts.
        if contacts is not None or self._self_contact is not None or self._rod_contact is not None:
            self._ingest_contacts(contacts, state_in)
        else:
            # Drop the previous step's contacts.
            self._contact_count = 0
            self._invalidate_schur_cache()

        has_contacts = self._contact_count > 0
        if self.model.world_count > 1:
            self._group_deformable_worlds()

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

        # λ and ω start at zero in every step, or from the step before's with contact_warmstart, then carry across
        # its PD outer iterations below.
        M = self._contact_count
        n_b = 0 if self._constraint_rows is None else self._constraint_rows[0].shape[0]
        if self.nsn_lite_accumulate and self._accumulate_worlds is not None:
            particles = host_head(self._contact_particle_d, M) if M else np.empty(0, dtype=np.int32)
            worlds = np.r_[np.repeat(self._particle_world_h[particles], 3), self._constraint_world_h]
            disabled = np.flatnonzero(~self._accumulate_worlds[worlds])
            self._accumulate_disabled_rows = wp.array(disabled, dtype=wp.int32, device=self._device)
            self._accumulate_disabled_zeros = wp.zeros(len(disabled), dtype=wp.float64, device=self._device)
        if M > 0 or n_b:
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
            n_rows_max = 3 * M + n_b
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

        # Rigid contact rows are assembled after the deformable contacts they may share bodies with.
        if rigid:
            self._split_two_way_contacts()
            mixed = self._mixed
            if mixed is not None:
                self._swap_rows(mixed.two_way)
            self._begin_rigid_contacts(contacts, state_in, dt)
            if mixed is not None:
                self._swap_rows(mixed.two_way)
        if self._warm_start is not None:
            self._restore_warm_start()
        if self.friction_anchors:
            if model.world_count > 1:
                self._two_way_world_d = wp.array(self._two_way_world_h, dtype=wp.int32, device=device)
            self._apply_friction_anchors(state_in)
        mixed = self._mixed
        if mixed is not None:
            n_b = self._begin_mixed_rows()

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
        if self._spring_indices_d is not None:
            self._spring_first_d.fill_(1)
            if self._spring_damped:
                from .kernels import spring_length_kernel  # noqa: PLC0415

                inputs = [state_in.particle_q, self._spring_indices_d]
                S = self._spring_indices_d.shape[0]
                wp.launch(spring_length_kernel, S, inputs, [self._spring_len0_d], device=device)
        # The contact forces of the step's sparse lite rows accumulate from zero.
        if self.nsn_lite_accumulate and (M > 0 or n_b):
            self._ensure_accumulation_buffers(3 * M + n_b)
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

            if (has_contacts or n_b) and (not self._two_way or mixed is not None):
                # --- Schur-complement NSN contact correction ---
                if mixed is not None:
                    # The rows of the worlds without two-way contacts.
                    self._swap_rows(mixed.deformable)
                M = self._contact_count
                ls = self._linear_solver
                if n_b:
                    self._ensure_contact_buffers(1)

                if self.friction and hasattr(self, "_contact_tangent1_d"):
                    # Frictional contacts: 3M Schur complement with Coulomb cone
                    # projection. Reuse the cached W across PD outer iters: it
                    # depends only on (J, A), which are fixed within a step.

                    use_lite_block_diag = self.nsn_schur_mode == "lite"
                    from . import kernels as K_step  # noqa: PLC0415

                    if n_b and self._constraint_lengths:
                        # Length rows act along the current direction of their combination.
                        self._point_length_rows(self._x_cur)
                        if self._cached_W:
                            ls.update_row_directions(self._constraint_rows[2], 3 * M)
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
                                bilateral=self._constraint_rows_h if n_b else None,
                                **self._contact_extra_args(),
                                host=None
                                if self._contact_particle_h is None
                                else (self._contact_particle_h, self._contact_alpha_h),
                            )
                            # Rows on several particles, and more rows on a particle than the per-particle
                            # Newton step holds, take the sparse lite NSN.
                            many = ls._isodof_max_rows > K_step._MAX_ROWS_PER_PARTICLE
                            sparse = self._contact_extra is not None or many
                            self._prepare_lite_worlds(ls)
                            self._lite_W = self._build_lite_delassus(M, ls, n_b) if sparse else None
                            if self._lite_iter_graph is not False:
                                self._lite_iter_graph = None
                        else:
                            ls.build_schur_complement(
                                M,
                                *self._contact_row_args(),
                                download=False,
                                **self._contact_extra_args(),
                                **self._constraint_args(),
                                # The host copies hold the step's whole contact set, not one that a split swapped in.
                                host_rows=self._contact_host_rows
                                if self._swapped_rows is None and not self._two_way
                                else None,
                            )
                        self._cached_W = True
                    # Residual + pene0 + warm-start all device-resident: the PD
                    # outer iter does no host<->device transfers around the NSN
                    # call.
                    self._ensure_nsn_inner_buffers(
                        3 * M + n_b, alloc_2d=not use_lite_block_diag and not self.nsn_sparse_storage
                    )
                    if use_lite_block_diag and self._lite_W is not None:
                        # Rows on several particles couple them: the sparse lite NSN, whose whole correction runs
                        # from one graph per step where it can.
                        self._lite_sparse_iteration(M, n_b, dt, _use_lg_graph)
                        if rigid:
                            self._rigid_iteration(state_in, state_out, dt)
                        continue

                    self._launch_contact_residual(self._nsn_r_d)
                    self._launch_contact_pene0(self._nsn_pene0_d)
                    if n_b:
                        wp.launch(
                            K_step.particle_constraint_residual_kernel,
                            dim=n_b,
                            inputs=[*self._constraint_rows, self._x_cur, wp.int32(3 * M)],
                            outputs=[self._nsn_r_d, self._nsn_pene0_d],
                            device=device,
                        )
                    if use_lite_block_diag:
                        # Warm-start ω + λ from the persistent device buffers
                        # (same convention as the dense path).
                        wp.copy(self._nsn_omega_d, self._omega_coulomb_persistent_d, count=3 * M + n_b)
                        wp.copy(self._nsn_lam_d, self._lam_coulomb_persistent_d, count=3 * M + n_b)
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
                        use_per_particle = n_b or ls._isodof_max_rows > 3
                        if model.world_count > 1:
                            self._launch_lite_worlds(
                                ls,
                                M,
                                dt,
                                cap_internal,
                                use_cap,
                                self._nsn_r_d,
                                self._nsn_pene0_d,
                                self._nsn_lam_d,
                                self._nsn_omega_d,
                                self._nsn_lam_apply_d,
                            )
                        elif use_per_particle:
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
                                    wp.int32(3 * M),
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
                                    self._empty_indices,
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
                        if self._held_rows is not None:
                            # The contacts of particles held at points carry no force, so the rows that hold them
                            # take their own step after the correction of the other rows.
                            from .joints import scatter_values_kernel  # noqa: PLC0415

                            h = self._held_row_factor(ls)
                            outputs = [self._nsn_lam_apply_d[3 * M :]]
                            wp.launch(
                                scatter_values_kernel, h.rows.shape[0], [h.zeros, 0, h.rows], outputs, device=device
                            )
                    else:
                        # With the sparse storage, the copies and the correction below follow the NSN step in its
                        # graph.
                        tail, tail_objects = None, ()
                        if self.nsn_sparse_storage:
                            ls.reserve_lambda_correction()
                            tail = functools.partial(self._deformable_contact_tail, ls, 3 * M + n_b)
                            tail_objects = (ls, self._lam_coulomb_persistent_d, self._omega_coulomb_persistent_d)
                            tail_objects += (self._rhs, self._x_cur)
                        self._solve_nsn_coulomb_gpu(
                            None,
                            None,
                            None,
                            None,
                            max_iters=1,
                            dt=dt,
                            W_device=None if self.nsn_sparse_storage else ls.W_device_view(),
                            W_sparse=ls.W_sparse if self.nsn_sparse_storage else None,
                            graph="deformable",
                            world_rows=self._deformable_world_rows,
                            contact_scale=self._deformable_contact_scale,
                            graph_tail=tail,
                            graph_tail_objects=tail_objects,
                            r_device=self._nsn_r_d,
                            pene0_device=self._nsn_pene0_d,
                            mu_device=self._contact_mu_d,
                            lam_init_device=self._lam_coulomb_persistent_d,
                            omega_init_device=self._omega_coulomb_persistent_d,
                            skip_download=True,
                            m_contacts=M,
                            n_bilateral=n_b,
                            # Face contacts act on particles that also carry particle contacts.
                            contact_regularization=0.0 if self._contact_extra is None else self.contact_regularization,
                        )
                    if use_lite_block_diag or not self.nsn_sparse_storage:
                        self._deformable_contact_tail(ls, 3 * M + n_b)
                    if use_lite_block_diag and self._held_rows is not None:
                        self._step_held_rows(ls, M, dt)
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
                if mixed is not None:
                    self._swap_rows(mixed.deformable)
            if rigid:
                if mixed is not None:
                    self._swap_rows(mixed.two_way)
                self._rigid_iteration(state_in, state_out, dt)
                if mixed is not None:
                    self._swap_rows(mixed.two_way)

        if mixed is not None:
            self._end_mixed_rows()
        self._store_tendon_tension()
        if self._warm_start is not None:
            self._store_warm_start()
        if self.friction_anchors:
            self._update_friction_anchors()

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
            self._write_contact_forces(contact_f, contacts, state_in, dt)

        # The host mirrors of the persistent buffers read them when first read, so the step ends without waiting
        # for the device. When M==0 the host arrays were left at None / unchanged.
        if has_contacts:
            self._persistent_rows = self._contact_count

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
        # Spring stretching projection, ADMM: the dual steps, the chains' tension solve, then the prox, a
        # deterministic (compute -> gather) scatter. The first local step of a time step clears the flag that keeps
        # the duals from taking the previous step's splits.
        if self._spring_indices_d is not None:
            S = self._spring_indices_d.shape[0]
            inputs = [self._x_cur, self._spring_indices_d, self._spring_rest_d, self._spring_split_d]
            outputs = [self._spring_dual_d, self._spring_dir_d, self._spring_residual_d]
            wp.launch(self._k_spring_dual, S, [*inputs, self._spring_first_d], outputs, device=device)
            rows, other = self._spring_chain_rows_d
            inputs = [self._spring_chain_order_d, self._spring_chain_diagonal_d, self._spring_chain_link_d]
            inputs += [self._spring_residual_d, self._spring_dir_d, self._spring_first_d]
            wp.launch(self._k_spring_chain_system, S, inputs, rows, device=device)
            for level in range(self._spring_chain_levels):
                inputs = [1 << level, self._spring_first_d, *rows]
                wp.launch(self._k_spring_chain_reduction, S, inputs, other, device=device)
                rows, other = other, rows
            inputs = [self._spring_chain_order_d, self._spring_chain_bend_d, self._spring_chain_bend_scale_d]
            inputs += [self._spring_residual_d, self._spring_dir_d, self._spring_rho_d, self._spring_first_d]
            inputs += [rows[1], rows[3]]
            wp.launch(self._k_spring_chain, S, inputs, [self._spring_dual_d], device=device)
            inputs = [self._x_cur, self._spring_indices_d, self._spring_rest_d, self._spring_k_d, self._spring_rho_d]
            inputs += [self._spring_c_d, self._spring_len0_d, self._spring_dual_d, self._spring_dir_d]
            outputs = [self._spring_split_d, self._spring_contrib_d]
            wp.launch(self._k_spring, S, inputs, outputs, device=device)
            wp.launch(self._k_clear_flag, 1, [self._spring_first_d], device=device)
            inputs = [self._spring_contrib_d, *self._particle_spring_csr_d]
            wp.launch(self._k_gather, dim=N, inputs=inputs, outputs=[self._rhs], device=device)
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
                        self._tet_activation_d,
                        self._tet_activation_direction_d,
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
                        self._tet_activation_d,
                        self._tet_activation_direction_d,
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
                        self._tet_activation_d,
                        self._tet_activation_direction_d,
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

    def _write_contact_forces(self, contact_f: wp.array, contacts: Contacts, state: State, dt: float) -> None:
        """Fill the rows of ``contact_f`` for the step's particle contacts and zero the others."""
        from .kernels import compute_lam_apply_kernel, soft_contact_force_kernel  # noqa: PLC0415

        contact_f.zero_()
        M, model, device = self._contact_count, self.model, self._device
        if M == 0:
            return
        forces = self._nsn_lam_apply_d
        if self._rc_n and (not self._two_way or self._mixed is not None):
            # The rigid rows took their NSN step after the particle contacts' in every iteration, in the same buffer;
            # the particle contacts' forces come from the multipliers their own NSN kept, in the contacts' order.
            forces = wp.empty(3 * M, dtype=wp.float64, device=device)
            inputs = [self._lam_coulomb_persistent_d, self._omega_coulomb_persistent_d, wp.float64(dt)]
            wp.launch(compute_lam_apply_kernel, 3 * M, inputs, [forces], device=device)
        no_bodies = model.body_count == 0 or state.body_q is None
        body_q = wp.empty(1, dtype=wp.transform, device=device) if no_bodies else state.body_q
        body_com = wp.empty(1, dtype=wp.vec3, device=device) if no_bodies else model.body_com
        inputs = [self._contact_record_d, M, self._contact_normal_d, self._contact_tangent1_d]
        inputs += [self._contact_tangent2_d, 3 if self.friction else 1, forces]
        inputs += [self._contact_world_anchor_d, self._contact_shape_d, model.shape_body, body_q, body_com]
        inputs += [contacts.rigid_contact_max]
        wp.launch(soft_contact_force_kernel, M, inputs, [contact_f], device=device)

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register the custom attributes of SolverFBA: the bending elements of particle rods.

        Registers the custom frequency ``fba:rod_bend``, one entry per bending element, with the attributes
        ``fba:rod_bend_particles`` (``vec3i``), the rod's particles before, at and after the bend, and
        ``fba:rod_bend_stiffness`` (``float32``), the rod's bending stiffness ``EI`` [N·m²]. The model holds them
        as ``model.fba.rod_bend_particles`` and ``model.fba.rod_bend_stiffness``. :meth:`add_particle_rod`
        registers them itself.

        Args:
            builder: Model builder to register the attributes on.
        """
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="rod_bend", namespace="fba"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rod_bend_particles",
                frequency="fba:rod_bend",
                dtype=wp.vec3i,
                namespace="fba",
                references="particle",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rod_bend_stiffness",
                frequency="fba:rod_bend",
                dtype=wp.float32,
                namespace="fba",
            )
        )

    @staticmethod
    def add_particle_rod(
        builder: ModelBuilder,
        points: np.ndarray | list,
        radius: float,
        youngs_modulus: float,
        density: float | None = None,
        mass: float | None = None,
        bending_stiffness: float | ArrayLike | None = None,
    ) -> list[int]:
        """Add an elastic rod of circular cross-section along a polyline, as particles that SolverFBA simulates.

        Each vertex becomes a particle of radius ``radius`` whose mass is the rod's line density times half the
        length of its segments; each segment, a spring of stiffness ``E pi r^2 / l``, the rod's axial stiffness
        over the segment's rest length ``l``; and each interior vertex, a bending element, an entry of
        ``fba:rod_bend`` (see :meth:`register_custom_attributes`), of energy

            EI / (2 v) |(x_+ - x) / l_+ - (x - x_-) / l_-|^2,

        with ``x_-``, ``x`` and ``x_+`` the vertex and its neighbors, ``l_-`` and ``l_+`` the rest lengths of its
        two segments and ``v = (l_- + l_+) / 2``: for small turns ``phi`` of the rod at the vertex,
        ``EI / 2 kappa^2 v`` with the curvature ``kappa = phi / v``. The rod is straight at rest, whatever the
        polyline, and has no twist. Rest lengths come from ``points``. Contacts with shapes act at the particles,
        so a rod whose segments are longer than about twice its radius can pass thin shapes between its
        particles; rods collide with each other as chains of capsules with ``rod_contact=True``.

        A particle rod is not the rod of :meth:`~newton.ModelBuilder.add_rod`, a chain of rigid capsules joined
        by rod joints. To hold an end, set its particles' mass to zero, which pins them.

        Args:
            builder: Model builder to add the rod to.
            points: Vertices of the polyline [m], shape ``(N, 3)``, ``N >= 2``.
            radius: Radius of the cross-section [m], also the particles' contact radius.
            youngs_modulus: Young's modulus ``E`` [Pa].
            density: Density [kg/m³] of the rod's material, for the line density ``density pi r^2``. Give
                exactly one of ``density`` and ``mass``.
            mass: Total mass [kg] of the rod, spread evenly over its length.
            bending_stiffness: Bending stiffness ``EI`` [N·m²], for the whole rod or one per interior vertex,
                shape ``(N - 2,)``, as a cable stiffens toward a plug; ``E pi r^4 / 4``, that of a solid circular
                section, if ``None``. A rope takes one far below that.

        Returns:
            The indices of the rod's particles, in the order of ``points``.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("points must have shape (N, 3) with N >= 2")
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not (segment > 0.0).all():
            raise ValueError("the rod's consecutive points must be distinct")
        if not radius > 0.0 or not youngs_modulus > 0.0:
            raise ValueError("radius and youngs_modulus must be positive")
        if (density is None) == (mass is None):
            raise ValueError("give exactly one of density and mass")
        area = np.pi * radius * radius
        line_density = density * area if density is not None else mass / segment.sum()
        if not line_density > 0.0:
            raise ValueError("the rod's density or mass must be positive")
        if bending_stiffness is None:
            bending_stiffness = youngs_modulus * np.pi * radius**4 / 4.0
        bending = np.asarray(bending_stiffness, dtype=np.float64)
        if bending.ndim not in (0, 1) or (bending.ndim == 1 and bending.shape != (points.shape[0] - 2,)):
            raise ValueError("bending_stiffness must be a scalar or one value per interior vertex, shape (N - 2,)")
        if (bending < 0.0).any():
            raise ValueError("bending_stiffness must not be negative")
        bending = np.broadcast_to(bending, (max(points.shape[0] - 2, 0),))
        masses = 0.5 * line_density * (np.r_[segment, 0.0] + np.r_[0.0, segment])
        SolverFBA.register_custom_attributes(builder)
        particles = [
            builder.add_particle(pos=tuple(p), vel=(0.0, 0.0, 0.0), mass=float(m), radius=radius)
            for p, m in zip(points, masses, strict=True)
        ]
        for k, length in enumerate(segment):
            stiffness = youngs_modulus * area / length
            builder.add_spring(particles[k], particles[k + 1], ke=float(stiffness), kd=0.0, control=0.0)
        for k in range(1, len(particles) - 1):
            builder.add_custom_values(
                **{
                    "fba:rod_bend_particles": tuple(particles[k - 1 : k + 2]),
                    "fba:rod_bend_stiffness": float(bending[k - 1]),
                }
            )
        return particles

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

    def set_particle_self_contact(self, *, margin: float | ArrayLike, gap: float | ArrayLike | None = None) -> None:
        """Change the self-contact distance and detection gap for the following steps.

        A cloth dropped onto a surface needs a margin and gap that its layers cannot pass in one step; once it lies
        still, a thinner margin lets folded layers stack closer. Thinner than half the longest edge, vertex-face
        contacts need ``particle_self_contact_edge_edge=True`` to keep crossing edges apart.

        Args:
            margin: Positive self-contact distance [m]. A scalar applies to all worlds. An array of shape
                ``(model.world_count + 1,)`` sets each world's distance independently; the last entry is for
                global particles. Per-world distances require self-contact groups to be disabled.
            gap: Nonnegative additional detection distance [m], as a scalar or an array of shape
                ``(model.world_count + 1,)``. ``None`` takes the margin. Scalars broadcast independently
                for either parameter. Neither parameter is restored by :meth:`reset`.
        """
        sc = self._self_contact
        if sc is None:
            raise ValueError("set_particle_self_contact needs particle_enable_self_contact=True")
        margin = np.asarray(margin, dtype=np.float64)
        gap = margin if gap is None else np.asarray(gap, dtype=np.float64)
        shape = (self.model.world_count + 1,)
        if any(a.shape not in ((), shape) for a in (margin, gap)):
            raise ValueError(f"margin and gap must be scalars or arrays of shape {shape}")
        if not np.isfinite(margin).all() or not np.isfinite(gap).all() or (margin <= 0.0).any() or (gap < 0.0).any():
            raise ValueError("set_particle_self_contact needs finite margin > 0 and gap >= 0")
        per_world = margin.ndim != 0 or gap.ndim != 0
        if per_world and self._self_contact_groups:
            raise ValueError("per-world self-contact distances require self-contact groups to be disabled")
        self.particle_self_contact_margin = float(margin.max())
        sc._world_margins_h = np.broadcast_to(margin, shape).astype(np.float32).copy() if per_world else None
        values = (sc._world_margins_h, np.broadcast_to(margin + gap, shape)) if per_world else ([], [])
        for name, value in zip(("world_margins", "world_radii"), values, strict=True):
            current = getattr(sc, name)
            if current.shape == (len(value),):
                current.assign(np.asarray(value, dtype=np.float32))
            else:
                setattr(sc, name, wp.array(value, dtype=wp.float32, device=self._device))
        if self._self_contact_groups:
            sc.set_margin(float(margin), float(gap))
        else:
            sc.query_radius = float((margin + gap).max())

    def set_tendons(
        self, bodies, points, targets, middle_lengths, rest_lengths, friction_coefficients=None, friction_forces=None
    ) -> None:
        """Hold tendons, inextensible strings between two bodies, taut or slack, in the following steps.

        Tendon ``t`` runs from ``points[t, 0]`` on body ``bodies[t, 0]`` to ``points[t, 1]`` on ``bodies[t, 1]``
        (points in the body frames; on body ``-1`` a fixed world point). Between its ends its path is fixed for the
        step: from each end it runs straight to ``targets[t, k]`` (world), its next path point, such as where it
        meets a pulley, and between the two targets it has the length ``middle_lengths[t]``. Its row holds the slack
        ``rest_lengths[t] - L >= 0`` with ``L`` the middle length plus each end's run to its target, as a
        frictionless contact: the tension is the row's force, zero while the string is slack. Within a step an end's
        run is measured along the direction from its start-of-step position to its target, so that the row is linear
        in the step, as a contact's normal gap is, also for an end at its target. An end on an
        integrated body takes the force at its point along the string; an end on the world or on a body that
        SolverFBA does not integrate is fixed for the step. The caller updates the paths before each step (the
        targets move as the ends do round wrapped objects); the length changes only to second order as a target
        slides along a wrapped surface, so the row needs no term for it.

        A tendon with a friction coefficient ``mu = friction_coefficients[t]`` resists the string running over its
        pulleys with a friction that grows with the load, as a linearized capstan or a bearing whose friction grows
        with the tension: a second row holds the slide of its first end along its run at zero with a Coulomb force
        of at most ``mu T``, ``T`` the tension, solved with the other rows. The first end then feels a tension
        between ``T (1 - mu)`` and ``T (1 + mu)``, and ``mu <= 1`` keeps it nonnegative; a slack tendon carries no
        friction. A friction force ``F0 = friction_forces[t]`` [N] adds a part that does not grow with the load (a
        bearing's seal drag, the string's bending over its pulleys): the bound becomes ``F0 + mu T``, at most ``T``,
        so that the string never pushes its first end, and still none while slack. Within a step it is the
        coefficient ``min(1, mu + F0 / T_prev)`` times the tension, ``T_prev`` the tension at the end of the
        previous step; after a step without tension (before the first, or after a slack step) the coefficient is
        ``mu`` alone, so that the ``F0`` part starts at the step after the string holds a tension and the step in
        which a slack string snaps taut does not hold its slide with the whole catch impulse.

        The rows are solved with the rigid contacts (``friction=True``, ``nsn_schur_mode="full"``, one world).
        ``None`` removes the tendons.

        Args:
            bodies: The two bodies of each tendon, shape ``(T, 2)``, ``-1`` for the world.
            points: The ends' points in their bodies' frames [m], shape ``(T, 2, 3)``.
            targets: Each end's next path point in the world [m], shape ``(T, 2, 3)``.
            middle_lengths: The path's length between the two targets [m], shape ``(T,)``.
            rest_lengths: The tendons' lengths [m], shape ``(T,)``.
            friction_coefficients: Each tendon's friction coefficient ``0 <= mu <= 1`` on its slide over its pulleys,
                shape ``(T,)``; ``None`` for none.
            friction_forces: Each tendon's friction force ``F0 >= 0`` on its slide that does not grow with the load
                [N], shape ``(T,)``; ``None`` for none.
        """
        if bodies is None or len(bodies) == 0:
            self._tendons, self._tendon_count, self._tendon_ws = None, 0, None
            self._tendon_friction = self._tendon_tension = None
            return
        b = np.asarray(bodies, dtype=np.int32)
        T = b.shape[0]
        if b.shape != (T, 2) or np.shape(points) != (T, 2, 3) or np.shape(targets) != (T, 2, 3):
            raise ValueError(f"bodies need shape ({T}, 2), points and targets ({T}, 2, 3)")
        if np.shape(middle_lengths) != (T,) or np.shape(rest_lengths) != (T,):
            raise ValueError(f"middle_lengths and rest_lengths need shape ({T},)")
        mu = np.zeros(T) if friction_coefficients is None else np.asarray(friction_coefficients, dtype=np.float64)
        if mu.shape != (T,) or (mu < 0.0).any() or (mu > 1.0).any():
            raise ValueError(f"friction_coefficients need shape ({T},) and values in [0, 1]")
        f0 = np.zeros(T) if friction_forces is None else np.asarray(friction_forces, dtype=np.float64)
        if f0.shape != (T,) or (f0 < 0.0).any():
            raise ValueError(f"friction_forces need shape ({T},) and no negative value")
        if (b < -1).any() or (b >= self.model.body_count).any():
            raise ValueError("body indices out of range")
        if not self.friction or self.nsn_schur_mode != "full" or self.model.world_count > 1:
            raise NotImplementedError("tendons need friction=True, nsn_schur_mode='full' and one world")
        if self.integrate_with_external_rigid_solver:
            raise ValueError("tendons need the bodies that SolverFBA integrates")
        device = self._device
        arrays = (
            (b, wp.int32),
            (np.asarray(points, dtype=np.float32), wp.vec3),
            (np.asarray(targets, dtype=np.float32), wp.vec3),
            (np.asarray(middle_lengths, dtype=np.float64), wp.float64),
            (np.asarray(rest_lengths, dtype=np.float64), wp.float64),
            (mu, wp.float64),  # the step's coefficient, set from the two below at the step
        )
        if self._tendons is not None and self._tendon_count == T:
            for dst, (a, _) in zip(self._tendons, arrays, strict=True):
                dst.assign(a)  # the same arrays from step to step
            self._tendon_friction[0].assign(mu)
            self._tendon_friction[1].assign(f0)
        else:
            self._tendons = tuple(wp.array(a, dtype=t, device=device) for a, t in arrays)
            self._tendon_friction = (
                wp.array(mu, dtype=wp.float64, device=device),
                wp.array(f0, dtype=wp.float64, device=device),
            )
            self._tendon_tension = wp.zeros(T, dtype=wp.float64, device=device)
            self._tendon_ws = None
        self._tendon_count = T
        self._tendon_body_h = b

    def set_particle_constraints(
        self, particles, coefficients, targets, *, directions=None, exact=False, bodies=None, body_points=None
    ) -> None:
        """Hold lengths or linear combinations of particle positions in the following steps.

        Row ``r`` combines ``v_r = sum_k coefficients[r, k] * x[particles[r, k]]``. Without ``directions``
        it holds the length ``|v_r| == targets[r]``; with them, ``dot(directions[r], v_r) == targets[r]``.
        The rows are bilateral and solved with the contacts in each NSN step, so their multipliers carry
        the forces that hold them. A length row acts along the direction of ``v_r`` at every iteration, so
        at tension ``T`` it keeps the transverse stiffness ``T / |v_r|`` of a taut segment; a direction
        fixed for the step drops it, and a body that chained rows compress beyond its buckling load then
        flutters. Unused slots have particle ``-1``; slot 0 must be used. ``None`` removes the rows.

        With ``nsn_schur_mode="lite"`` every row needs a direction and one particle. Three rows along
        independent directions hold a particle at a point, and its contacts then carry no force. These rows
        take their Newton step after the correction of the other rows, with the Cholesky factor of their own
        Delassus matrix ``J A^-1 J^T`` as the joint rows of articulations do, and so reach their targets in every
        iteration; at most 64 of them in each world. Other rows are solved with the particle's contacts, which see its mass
        alone, ``dt^2 / m`` in place of its block of ``A^-1``, and lag their targets by the share of the stiffness
        in that block.

        Args:
            particles: Particle indices, shape ``(R, K)``.
            coefficients: Their coefficients, shape ``(R, K)``.
            targets: Row values [m], shape ``(R,)``.
            directions: Row directions, shape ``(R, 3)``, or ``None`` for length rows.
            exact: With ``nsn_schur_mode="lite"``, give every row the exact step of the rows that hold particles at
                points, also rows that hold a particle along one or two directions, whose contacts keep their
                forces: such rows then reach their targets in every iteration too. At most 64 rows per world. The full NSN
                solves every row with ``A`` already.
            bodies: Body of each row, shape ``(R,)``, ``-1`` for a row on particles alone. A row on body ``b`` holds
                ``dot(d_r, v_r) - dot(d_r, X_b p_r) == targets[r]``, with ``X_b`` the body's pose and ``p_r`` its
                point ``body_points[r]`` in the body frame: three rows along independent directions on one particle
                and target 0 tie the particle to that point of the body, as a rope tied to a rigid part. The rows are
                solved with the body's rows, so the body takes their forces. The body must be one that SolverFBA
                integrates (dynamic, not left to an external rigid solver); the rows need ``directions``, the full
                NSN and one world.
            body_points: The rows' points in their bodies' frames [m], shape ``(R, 3)``.
        """
        self._invalidate_schur_cache()
        self._constraint_body_h = self._constraint_body_d = self._constraint_point_d = None
        if particles is None or len(particles) == 0:
            self._constraint_world_h = np.empty(0, dtype=np.int32)
            self._constraint_rows = self._held_rows = self._constraint_key = None
            if self._warm_start is not None:
                self._warm_start.bilateral_count = 0
            self._held_particles, self._held_particle_d = np.zeros(0, dtype=np.int32), None
            return
        if not self.friction:
            raise ValueError("particle constraints need friction=True")
        p = np.asarray(particles)
        R = len(targets)
        if p.ndim != 2 or p.shape[0] != R or np.shape(coefficients) != p.shape:
            raise ValueError(f"particles and coefficients need shape (R, K) for {R} targets")
        if directions is not None and np.shape(directions) != (R, 3):
            raise ValueError(f"directions need shape ({R}, 3)")
        if (p[:, 0] < 0).any() or (p >= self.model.particle_count).any():
            raise ValueError("particle indices out of range, or slot 0 unused")
        worlds = self._particle_world_h[np.maximum(p, 0)]
        if self.model.world_count > 1 and ((p >= 0) & (worlds != worlds[:, :1])).any():
            raise ValueError("particle constraint rows must not connect different worlds")
        if self.nsn_schur_mode == "lite" and (directions is None or (p[:, 1:] >= 0).any()):
            raise ValueError("the lite NSN holds only rows with a direction on one particle")
        body_rows = None
        if bodies is not None:
            body_rows = np.asarray(bodies, dtype=np.int32).reshape(-1)
            if body_rows.shape != (R,) or np.shape(body_points) != (R, 3):
                raise ValueError(f"bodies need shape ({R},) and body_points ({R}, 3)")
            if (body_rows >= 0).any():
                if directions is None or self.nsn_schur_mode != "full" or self.model.world_count > 1:
                    raise NotImplementedError("rows on bodies need directions, nsn_schur_mode='full' and one world")
                if self.integrate_with_external_rigid_solver:
                    raise ValueError("rows on bodies need the bodies that SolverFBA integrates")
                if (body_rows >= self.model.body_count).any():
                    raise ValueError("body indices out of range")
                flags = self.model.body_flags.numpy()[body_rows[body_rows >= 0]]
                if not (flags & int(BodyFlags.DYNAMIC)).all():
                    raise ValueError("rows on bodies need dynamic bodies, which SolverFBA integrates")
            else:
                body_rows = None
        if self.nsn_schur_mode == "lite":
            self._hold_at_points(p[:, 0], np.asarray(coefficients)[:, 0], np.asarray(directions), exact)
        lengths = directions is None
        if directions is None:
            directions = np.zeros((len(targets), 3))
        arrays = [(particles, np.int32, wp.int32), (coefficients, np.float32, wp.float32)]
        arrays += [(directions, np.float32, wp.vec3), (targets, np.float64, wp.float64)]
        constraint_rows = [wp.array(np.asarray(a, dtype=n), dtype=t, device=self._device) for a, n, t in arrays]
        if self._warm_start is not None:
            key = [np.array(a, copy=True) for a in (particles, coefficients, () if lengths else directions)]
            previous = self._constraint_key
            if self._warm_start.bilateral_count and not all(map(np.array_equal, key, previous)):

                def by_world(world, values):
                    order = np.argsort(world, kind="stable")
                    for rows in np.split(order, np.flatnonzero(np.diff(world[order])) + 1):
                        columns = (values[0][rows] >= 0).any(axis=0)
                        data = [a[np.ix_(rows, columns)] for a in values[:2]]
                        data.append(values[2][rows] if values[2].size else values[2])
                        yield int(world[rows[0]]), (rows, data)

                old = dict(by_world(self._constraint_world_h, previous))
                retained = np.full(R, -1, dtype=np.int32)
                for world, (rows, data) in by_world(worlds[:, 0], key):
                    before = old.get(world)
                    if before is not None and all(map(np.array_equal, data, before[1])):
                        retained[rows] = before[0]
                self._warm_start.remap_bilateral(retained)
            self._constraint_key = key
        self._constraint_lengths = lengths
        self._constraint_rows = constraint_rows
        self._constraint_world_h = worlds[:, 0]
        if body_rows is not None:
            self._constraint_body_h = body_rows
            self._constraint_body_d = wp.array(body_rows, dtype=wp.int32, device=self._device)
            points = np.asarray(body_points, dtype=np.float32).reshape(R, 3)
            self._constraint_point_d = wp.array(points, dtype=wp.vec3, device=self._device)
        if self.nsn_schur_mode == "lite":
            # The lite rows' particles, directions and coefficients, which hold until the rows change.
            rows_h = (p[:, 0].astype(np.int32), np.asarray(directions, dtype=np.float32))
            self._constraint_rows_h = (*rows_h, np.asarray(coefficients, dtype=np.float32)[:, 0])

    def _hold_at_points(self, p: np.ndarray, c: np.ndarray, d: np.ndarray, exact: bool = False) -> None:
        """Keep the lite rows that hold particles at points, or all rows when ``exact``, and their factor while they
        keep their particles, coefficients and directions."""
        held = np.array([q for q in np.unique(p) if np.linalg.matrix_rank(d[p == q], tol=1e-3) >= 3], dtype=np.int32)
        rows = np.arange(len(p), dtype=np.int32) if exact else np.flatnonzero(np.isin(p, held)).astype(np.int32)
        if np.bincount(p[rows]).max(initial=3) > 3:
            raise ValueError("a particle held at a point takes three rows")
        counts = np.unique(self._held_row_world(p[rows]), return_counts=True)[1]
        if counts.max(initial=0) > 64:
            raise NotImplementedError(
                f"SolverFBA's lite NSN takes the exact step of at most 64 rows per world, got {counts.max()}"
            )
        if self._held_particle_d is None or not np.array_equal(held, self._held_particles):
            flags = np.zeros(self.model.particle_count, dtype=np.int32)
            flags[held] = 1
            self._held_particles, self._held_particle_d = held, wp.array(flags, device=self._device)
        key = (rows, p[rows], c[rows].astype(np.float64), d[rows].astype(np.float64))
        if not len(rows):
            self._held_rows = None
        elif self._held_rows is None or not all(map(np.array_equal, key, self._held_rows[0])):
            self._held_rows = [key, None]

    def _held_row_world(self, p: np.ndarray) -> np.ndarray:
        """The block of the held rows on particles ``p``: their world in a model of several worlds, which do not
        couple, and one block in a model of one, whose global particles may couple to its others."""
        return self._particle_world_h[p] if self.model.world_count > 1 else np.zeros(len(p), dtype=np.int32)

    def _held_row_factor(self, ls) -> SimpleNamespace:
        """The rows that hold particles at points, in one block per world, with the Cholesky factors ``L`` of
        the blocks' Delassus matrices ``J A^-1 J^T``, each padded with identity to ``tile`` rows, the entry
        ``slot`` of each row in the blocks, and the columns of ``A^-1`` at the held particles of each block over
        the block's particles only: the worlds do not couple, so these hold all of ``A^-1 J^T`` and the blocks of
        ``J A^-1 J^T`` are its only nonzero ones. ``A`` holds for the run, so these hold while the rows do."""
        from .joints import gather_values_kernel  # noqa: PLC0415

        if self._held_rows[1] is None:
            rows, p, c, d = self._held_rows[0]
            # Order the rows by world, keeping their order within each.
            worlds, block = np.unique(self._held_row_world(p), return_inverse=True)
            order = np.argsort(block, kind="stable")
            rows, p, c, d, block = rows[order], p[order], c[order], d[order], block[order]
            row_start = np.searchsorted(block, np.arange(len(worlds) + 1))
            local = np.arange(len(rows)) - row_start[block]
            # The particles of each block, ascending, and the place of each particle in its block; -1 off them.
            world = self._held_row_world(np.arange(ls.n))
            particle_block = np.searchsorted(worlds, world)
            particle_block[particle_block == len(worlds)] = 0
            particle_block = np.where(worlds[particle_block] == world, particle_block, -1)
            particles = np.argsort(particle_block, kind="stable")[np.count_nonzero(particle_block < 0) :]
            counts = np.bincount(particle_block[particles], minlength=len(worlds))
            particle_local = np.full(ls.n, -1, dtype=np.int64)
            particle_local[particles] = (
                np.arange(len(particles)) - np.r_[0, np.cumsum(counts)][particle_block[particles]]
            )
            # The unique held particles of each block, ascending, and the column of each row among them.
            unique, index = np.unique(np.stack([block, p]), axis=1, return_inverse=True)
            columns = np.bincount(unique[0], minlength=len(worlds))
            column = index.ravel() - np.r_[0, np.cumsum(columns)][block]
            column_start = np.r_[0, np.cumsum(columns * counts)]
            if column_start[-1] >= 2**31:
                raise NotImplementedError("SolverFBA's lite NSN takes the exact rows of fewer held particles")
            # Each block's columns of A^-1 at its held particles, over its particles, from one scratch column.
            device, f64 = self._device, wp.float64
            e = wp.zeros(ls.n, dtype=f64, device=device)
            solution = wp.empty(ls.n, dtype=f64, device=device)
            particles_d = wp.array(particles, dtype=wp.int32, device=device)
            response = wp.empty(int(column_start[-1]), dtype=f64, device=device)
            first = np.r_[0, np.cumsum(counts)]
            for u, (b, q) in enumerate(unique.T):
                e.zero_()
                e[int(q) : int(q) + 1].fill_(1.0)
                ls.solve_scalar(e, solution)
                offset = int(column_start[b] + (u - np.searchsorted(unique[0], b)) * counts[b])
                inputs = [solution, particles_d[first[b] : first[b + 1]], offset]
                wp.launch(gather_values_kernel, int(counts[b]), inputs, [response], device=device)
            tile = max(8, 1 << int(np.ceil(np.log2(local.max() + 1))))
            L = np.tile(np.eye(tile), (len(worlds), 1, 1))
            for b in range(len(worlds)):
                k = np.arange(row_start[b], row_start[b + 1])
                Ainv = response[int(column_start[b]) : int(column_start[b + 1])].numpy().reshape(columns[b], -1)
                Ainv = Ainv[np.ix_(column[k], particle_local[p[k]])]
                W = Ainv * (d[k] @ d[k].T) * np.outer(c[k], c[k])
                L[b, : len(k), : len(k)] = np.linalg.cholesky(W)
            i32 = wp.int32
            self._held_rows[1] = SimpleNamespace(
                tile=tile,
                L=wp.array(L, dtype=f64, device=device),
                rhs=wp.zeros((len(worlds), tile), dtype=f64, device=device),
                x=wp.zeros((len(worlds), tile), dtype=f64, device=device),
                residual=wp.zeros(len(rows), dtype=f64, device=device),
                slot=wp.array(block * tile + local, dtype=i32, device=device),
                zeros=wp.zeros(len(rows), dtype=f64, device=device),
                rows=wp.array(rows, dtype=i32, device=device),
                particle_block=wp.array(particle_block, dtype=i32, device=device),
                particle_local=wp.array(particle_local, dtype=i32, device=device),
                row_start=wp.array(row_start, dtype=i32, device=device),
                column_start=wp.array(column_start[:-1], dtype=i32, device=device),
                block_particles=wp.array(counts, dtype=i32, device=device),
                column=wp.array(column, dtype=i32, device=device),
                response=response,
                coefficient=wp.array(c, dtype=f64, device=device),
                direction=wp.array(d, dtype=wp.vec3d, device=device),
            )
        return self._held_rows[1]

    def _step_held_rows(self, ls, M: int, dt: float) -> None:
        """Newton step of the rows that hold particles at points, at the positions the other rows corrected."""
        from . import kernels as K  # noqa: PLC0415
        from .joints import _tile_kernels, gather_values_kernel, scatter_values_kernel  # noqa: PLC0415

        h, device, r = self._held_row_factor(ls), self._device, self._nsn_r_d
        inputs = [*self._constraint_rows, self._x_cur, wp.int32(3 * M)]
        n_b = self._constraint_rows[0].shape[0]
        wp.launch(K.particle_constraint_residual_kernel, n_b, inputs, [r, self._nsn_pene0_d], device=device)
        n_h = h.rows.shape[0]
        wp.launch(gather_values_kernel, n_h, [r[3 * M :], h.rows, 0], [h.residual], device=device)
        wp.launch(scatter_values_kernel, n_h, [h.residual, 0, h.slot], [h.rhs.flatten()], device=device)
        blocks = h.L.shape[0]
        wp.launch_tiled(_tile_kernels(h.tile)[1], dim=blocks, inputs=[h.L, h.rhs, h.x], block_dim=64, device=device)
        outputs = [a[3 * M :] for a in (self._nsn_lam_d, self._nsn_omega_d, self._nsn_lam_apply_d)]
        inputs = [h.rows, h.slot, h.x.flatten(), wp.float64(dt)]
        wp.launch(K.held_row_update_kernel, n_h, inputs, outputs, device=device)
        inputs = [h.particle_block, h.particle_local, h.row_start, h.column_start, h.block_particles, h.column]
        inputs += [h.response, h.coefficient, h.direction, h.slot, h.x.flatten()]
        wp.launch(K.held_row_correction_kernel, ls.n, inputs, [self._x_cur], device=device)

    def _prepare_lite_worlds(self, ls) -> None:
        """Keep each world's standalone choice of cofactor, Cholesky or sparse PCR."""
        from . import kernels as K  # noqa: PLC0415

        device = self._device
        self._lite_direct = None
        if self.model.world_count > 1:
            # A complex neighbor must not move a world's small exact blocks onto truncated PCR.
            counts = np.diff(ls._isodof_row_offsets_d.numpy())
            maximum = np.zeros(self.model.world_count, dtype=np.int32)
            np.maximum.at(maximum, self._particle_world_h, counts)
            direct = ~self._regularized_world_h & (maximum <= K._MAX_ROWS_PER_PARTICLE)
            # rows built on the device, with rows on several particles, keep no list of their particles
            unique = ls._isodof_unique_d
            particles = np.unique(ls._row_particle_d.numpy()) if unique is None else unique.numpy()
            particles = particles[direct[self._particle_world_h[particles]]]
            if len(particles):
                per_contact = direct & (maximum <= 3)
                per_contact[self._constraint_world_h] = False
                contacts = np.flatnonzero(per_contact[self._contact_world_h])
                particles = particles[~per_contact[self._particle_world_h[particles]]]
                rows = np.flatnonzero(direct[self._particle_world_h[ls._row_particle_d.numpy()]])
                all_particles = np.flatnonzero(direct[self._particle_world_h])
                self._lite_direct = SimpleNamespace(
                    particles=wp.array(particles, dtype=wp.int32, device=device),
                    contacts=wp.array(contacts, dtype=wp.int32, device=device),
                    rows=wp.array(rows, dtype=wp.int32, device=device),
                    all_particles=wp.array(all_particles, dtype=wp.int32, device=device),
                    zeros=wp.zeros(len(rows), dtype=wp.float64, device=device),
                )

    def _launch_lite_worlds(self, ls, M, dt, cap_internal, use_cap, r, pene0, lam, omega, lam_apply) -> None:
        """Advance the direct subsets once with their original single-world kernels."""
        from . import kernels as K  # noqa: PLC0415

        direct = self._lite_direct
        if direct is None:
            return
        outputs = [lam, omega, lam_apply]
        tail = [ls._inv_mass_d, wp.float64(dt), wp.float64(cap_internal), int(use_cap)]
        if direct.particles.shape[0]:
            inputs = [direct.particles, ls._isodof_row_offsets_d, ls._isodof_row_indices_d]
            inputs += [ls._row_particle_d, ls._row_dir_d, ls._row_alpha_d, self._contact_mu_d, r, pene0]
            wp.launch(
                K.fb_newton_lite_coulomb_per_particle_kernel,
                direct.particles.shape[0],
                [*inputs, *tail, 3 * M],
                outputs,
                device=self._device,
            )
        if direct.contacts.shape[0]:
            inputs = [
                direct.contacts,
                self._contact_particle_d,
                self._contact_normal_d,
                self._contact_tangent1_d,
                self._contact_tangent2_d,
                self._contact_alpha_d,
                self._contact_mu_d,
                r,
                pene0,
            ]
            wp.launch(
                K.fb_newton_lite_coulomb_kernel,
                direct.contacts.shape[0],
                [*inputs, *tail],
                outputs,
                device=self._device,
            )

    def _build_lite_delassus(self, M: int, ls, n_b: int) -> tuple:
        """The sparse lite Delassus matrix of the ``M`` contacts and ``n_b`` bilateral rows, which holds for the
        step, and the contacts on a particle that the bilateral rows hold at a point, for the sparse lite NSN."""
        from . import kernels as K  # noqa: PLC0415

        n, device = 3 * M + n_b, self._device
        if self._held_particle_d is None:
            self._held_particle_d = wp.zeros(self.model.particle_count, dtype=wp.int32, device=device)
        if n_b:
            bilateral = self._constraint_rows[:2]
        else:
            bilateral = [wp.empty((0, 1), dtype=t, device=device) for t in (wp.int32, wp.float32)]
        triplets = [wp.empty(4 * n, dtype=t, device=device) for t in (wp.int32, K._mat13d)]
        held_d = wp.empty(M, dtype=wp.int32, device=device)
        extra = self._contact_extra or [wp.empty((M, 0), dtype=t, device=device) for t in (wp.int32, wp.float32)]
        assert extra[0].shape[1] <= 3, "the lite Jacobian holds four particles of a row"
        inputs = [self._contact_particle_d, self._contact_alpha_d, *extra, *bilateral, ls._row_dir_d]
        inputs += [self._held_particle_d, M]
        wp.launch(K.lite_row_triplets_kernel, n, inputs, [*triplets, held_d], device=device)
        W = ls.build_schur_lite_triplets(n, *triplets)
        # The NSN's graph captures its buffers: they take their sizes for the step here.
        self._ensure_lite_bsr_buffers(n)
        ls.reserve_a_schur_lite_bsr(W)
        return W, held_d if n_b else None

    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Restore selected body and particle state and clear the selected worlds' solver history.

        Requested fields are copied from the model defaults. Missing State arrays are skipped.
        With ``flags=0``, the supplied physical state is preserved; friction anchors, warm-start
        multipliers and the springs' ADMM duals are cleared regardless of flags. Bodies are left untouched when
        ``integrate_with_external_rigid_solver=True``.

        FBA uses maximal body coordinates, so ``JOINT_Q`` and ``JOINT_QD`` are ignored. To reset
        from joint coordinates, run :func:`~newton.eval_fk` after reset. Force flags are also ignored;
        external forces, control inputs, pin targets and particle constraint targets remain caller-owned.

        This does not reset a CollisionPipeline or actuator state. After a discontinuous state change,
        call :meth:`~newton.CollisionPipeline.reset_contact_matching` with the same world mask and
        regenerate contacts before stepping. Reset any actuator history through its own state reset.

        Args:
            state: The simulation state to modify in place.
            world_mask: Boolean mask of shape ``(world_count + 1,)``; the last entry selects global
                entities. ``None`` clears every world's history.
            flags: State fields to restore: ``BODY_Q``, ``BODY_QD``, ``PARTICLE_Q`` and ``PARTICLE_QD``.
                ``None`` requests all supported fields; ``0`` clears only solver history.
        """
        if state is None:
            raise ValueError("'state' argument is required.")
        world_mask = self._normalize_reset_world_mask(world_mask)
        model, device = self.model, self._device
        flags_value = int(StateFlags.ALL if flags is None else flags)
        fields = [
            ("particle_q", StateFlags.PARTICLE_Q, model.particle_world),
            ("particle_qd", StateFlags.PARTICLE_QD, model.particle_world),
        ]
        if not self.integrate_with_external_rigid_solver:
            fields += [
                ("body_q", StateFlags.BODY_Q, model.body_world),
                ("body_qd", StateFlags.BODY_QD, model.body_world),
            ]
        copies = []
        # Validate every requested field before modifying state or history.
        for name, flag, world in fields:
            if not flags_value & int(flag):
                continue
            current, initial = getattr(state, name), getattr(model, name)
            if current is None or initial is None:
                continue
            if current.device != device or current.shape != initial.shape or current.dtype != initial.dtype:
                raise ValueError(f"state.{name} must match the model's shape, dtype and device")
            copies.append((world, initial, current))
        for world, initial, current in copies:
            if world_mask is None:
                wp.copy(current, initial)
            elif current.size:
                wp.launch(
                    reset_state_array,
                    current.shape[0],
                    [world, world_mask, model.world_count, initial],
                    [current],
                    device=device,
                )
        if world_mask is not None:
            from .warm_start import (  # noqa: PLC0415
                clear_anchor_contacts_kernel,
                clear_anchor_particles_kernel,
                clear_bilateral_worlds_kernel,
                clear_group_worlds_kernel,
            )

            if self.friction_anchors and model.particle_count:
                inputs = [model.particle_world, world_mask]
                wp.launch(
                    clear_anchor_particles_kernel, model.particle_count, inputs, [self._anchor_step_d], device=device
                )
                if self._anchor_count:
                    inputs = [self._anchor_previous_key_d, model.particle_world, max(model.shape_count, 1), world_mask]
                    wp.launch(
                        clear_anchor_contacts_kernel,
                        self._anchor_count,
                        inputs,
                        [self._anchor_previous_pressed_d],
                        device=device,
                    )
            ws = self._warm_start
            if ws is not None:
                joint_world = wp.array(self._joint_world_h, dtype=wp.int32, device=device)
                worlds = (model.particle_world, model.shape_world, joint_world)
                for kind, (group, world) in enumerate(zip((ws.particle, ws.rigid, ws.joint), worlds, strict=True)):
                    if group.prev_count:
                        inputs = [
                            group.prev_sorted,
                            group.prev_order,
                            world,
                            kind,
                            model.shape_count,
                            group.rows,
                            world_mask,
                        ]
                        wp.launch(
                            clear_group_worlds_kernel,
                            group.prev_count,
                            inputs,
                            [group.prev_lam, group.prev_omega],
                            device=device,
                        )
                if ws.bilateral_count:
                    world = wp.array(self._constraint_world_h, dtype=wp.int32, device=device)
                    wp.launch(
                        clear_bilateral_worlds_kernel,
                        ws.bilateral_count,
                        [world, world_mask],
                        [ws.bilateral_lam, ws.bilateral_omega],
                        device=device,
                    )
            if self._spring_indices_d is not None:
                from .kernels import clear_spring_state_kernel  # noqa: PLC0415

                inputs = [self._spring_indices_d, model.particle_world, world_mask]
                outputs = [self._spring_dual_d, self._spring_split_d]
                wp.launch(clear_spring_state_kernel, self._spring_indices_d.shape[0], inputs, outputs, device=device)
            return
        if self._spring_indices_d is not None:
            self._spring_dual_d.zero_()
            self._spring_split_d.zero_()
        self._anchor_count = 0
        if self.friction_anchors and self.model.particle_count:
            self._anchor_step_d.fill_(-2)
        if self._warm_start is not None:
            self._warm_start.clear()

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Rebuild and factor the system matrices at the next step.

        ``ModelFlags.SHAPE_PROPERTIES`` stands for changes of the particles and elements, whose masses,
        materials and rest shapes the factored system matrix holds; Newton has no flag of its own for them.
        ``BODY_PROPERTIES``, ``BODY_INERTIAL_PROPERTIES`` and ``JOINT_PROPERTIES`` rebuild the system of the
        integrated bodies and their joint rows.
        """
        if flags & ModelFlags.SHAPE_PROPERTIES:
            self._linear_solver = self._two_way_ls = self._mixed = None
            self._dt_setup = None
            self._invalidate_schur_cache()
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.JOINT_PROPERTIES):
            self._rigid_dt = None

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
        # The graph of the sparse NSN step stays: its key holds the linear solver's array generation, the sizes and
        # the identities of what it reads, and the Schur build writes its arrays in place while their shapes hold.

    # ------------------------------------------------------------------
    # Contact state (allocated lazily on first call).
    # ------------------------------------------------------------------

    def _anchor_extra(self) -> tuple[list, int]:
        """The further particles and coefficients of the step's contacts, or a stand-in when no contact has any, and
        whether they are the contacts'."""
        if self._contact_extra is not None:
            return self._contact_extra, 1
        if not hasattr(self, "_anchor_no_extra"):
            self._anchor_no_extra = [
                wp.full((1, 3), -1, dtype=wp.int32, device=self._device),
                wp.zeros((1, 3), dtype=wp.float32, device=self._device),
            ]
        return self._anchor_no_extra, 0

    def _anchor_shapes(self) -> tuple[wp.array, wp.array]:
        """The model's shape bodies and margins, or empty stand-ins for a model without shapes."""
        model = self.model
        if model.shape_count:
            return model.shape_body, model.shape_margin
        if not hasattr(self, "_anchor_no_shapes"):
            self._anchor_no_shapes = (
                wp.zeros(1, dtype=wp.int32, device=self._device),
                wp.zeros(1, dtype=float, device=self._device),
            )
        return self._anchor_no_shapes

    def _ensure_anchor_buffers(self, count: int) -> None:
        """Size the per-contact friction anchor buffers to ``count`` contacts, keeping the step before's anchors."""
        if count <= self._anchor_cap:
            return
        cap, device = max(int(count * 1.5), 64), self._device
        previous = None
        if self._anchor_cap:
            previous = (self._anchor_previous_key_d, self._anchor_previous_point_d, self._anchor_previous_pressed_d)
        self._anchor_flag_d = wp.zeros(cap, dtype=wp.int32, device=device)
        self._anchor_key_d = wp.zeros(cap, dtype=wp.int64, device=device)
        self._anchor_point_d = wp.zeros(cap, dtype=wp.vec3, device=device)
        self._anchor_previous_key_d = wp.zeros(cap, dtype=wp.int64, device=device)
        self._anchor_previous_point_d = wp.zeros(cap, dtype=wp.vec3, device=device)
        self._anchor_previous_pressed_d = wp.zeros(cap, dtype=wp.int32, device=device)
        self._anchor_next_key_d = wp.zeros(cap, dtype=wp.int64, device=device)
        self._anchor_next_point_d = wp.zeros(cap, dtype=wp.vec3, device=device)
        self._anchor_next_pressed_d = wp.zeros(cap, dtype=wp.int32, device=device)
        if previous is not None and self._anchor_count:
            current = (self._anchor_previous_key_d, self._anchor_previous_point_d, self._anchor_previous_pressed_d)
            for dst, src in zip(current, previous, strict=True):
                wp.copy(dst, src, count=self._anchor_count)
        self._anchor_cap = cap

    def _apply_friction_anchors(self, state_in: State) -> None:
        """Move the tangent offsets of the contacts that kept a friction anchor from the step before, so that their
        friction rows hold the particle at the anchor."""
        from .kernels import friction_anchor_apply_kernel  # noqa: PLC0415

        self._anchor_step += 1
        self._anchor_rows = 0
        M = self._contact_count
        # The contacts of a step that solves rigid contact rows with them keep the step's points.
        if M == 0 or (self._two_way and self.model.world_count == 1):
            return
        self._ensure_anchor_buffers(M)
        (extra_particle, extra_alpha), has_extra = self._anchor_extra()
        shape_body, shape_margin = self._anchor_shapes()
        margin = self.particle_self_contact_margin if self._self_contact is not None else 0.0
        inputs = [self._contact_shape_d, self._contact_particle_d, extra_particle, extra_alpha, wp.int32(has_extra)]
        inputs += [self._contact_normal_d, self._contact_tangent1_d, self._contact_tangent2_d]
        inputs += [self._contact_world_anchor_d, state_in.particle_q, self.model.particle_radius]
        inputs += [shape_body, shape_margin, wp.int64(max(self.model.shape_count, 1)), float(margin)]
        inputs += [
            self._self_contact.world_margins
            if self._self_contact is not None
            else wp.empty(0, dtype=wp.float32, device=self._device)
        ]
        inputs += [wp.int32(self._anchor_step), self._anchor_triangle_d, self._anchor_bary_d, self._anchor_step_d]
        inputs += [self._anchor_previous_key_d, self._anchor_previous_point_d, self._anchor_previous_pressed_d]
        inputs += [wp.int32(self._anchor_count)]
        inputs += [self.model.particle_world, self._two_way_world_d]
        outputs = [self._contact_tangent1_offset_shifted_d, self._contact_tangent2_offset_shifted_d]
        outputs += [self._anchor_flag_d, self._anchor_key_d, self._anchor_point_d]
        wp.launch(friction_anchor_apply_kernel, M, inputs, outputs, device=self._device)
        self._anchor_rows = M

    def set_friction_anchor_exclusion(self, particles: ArrayLike | None) -> None:
        """Keep no friction anchors on the contacts that ``particles`` make from the following steps on, with shapes
        or as the vertex of a vertex-face self contact, as on particles that rows hold at points; ``None`` frees none.

        A grasp that gathers cloth between fingers frees the cloth around them, so that it slides toward the
        fingers instead of being pulled back to where it lay.

        Args:
            particles: Particle indices, or ``None`` for none.
        """
        if not self.friction_anchors:
            raise ValueError("set_friction_anchor_exclusion needs friction_anchors=True")
        self._anchor_exclude = (
            np.zeros(0, np.int32) if particles is None else np.unique(np.asarray(particles, np.int32))
        )

    def _update_friction_anchors(self) -> None:
        """Keep or set the friction anchors of the contacts that pressed in the step, from its last multipliers."""
        from .kernels import friction_anchor_update_kernel  # noqa: PLC0415

        M = self._anchor_rows
        if M == 0:
            self._anchor_count = 0
            return
        (extra_particle, extra_alpha), has_extra = self._anchor_extra()
        shape_body, _ = self._anchor_shapes()
        held = self._held_particle_d
        if len(self._anchor_exclude):
            mask = np.zeros(self.model.particle_count, dtype=np.int32)
            mask[self._held_particles] = 1
            mask[self._anchor_exclude] = 1
            held = wp.array(mask, dtype=wp.int32, device=self._device)
        inputs = [self._contact_shape_d, self._contact_particle_d, extra_particle, extra_alpha, wp.int32(has_extra)]
        inputs += [self._lam_coulomb_persistent_d]
        inputs += [self._anchor_step_d if held is None else held, wp.int32(held is not None), shape_body]
        inputs += [wp.int32(self._anchor_step), self._anchor_flag_d, self._anchor_key_d, self._anchor_point_d]
        inputs += [self.model.particle_world, self._two_way_world_d]
        outputs = [self._anchor_triangle_d, self._anchor_bary_d, self._anchor_step_d]
        outputs += [self._anchor_next_key_d, self._anchor_next_point_d, self._anchor_next_pressed_d]
        wp.launch(friction_anchor_update_kernel, M, inputs, outputs, device=self._device)
        self._anchor_previous_key_d, self._anchor_next_key_d = self._anchor_next_key_d, self._anchor_previous_key_d
        self._anchor_previous_point_d, self._anchor_next_point_d = (
            self._anchor_next_point_d,
            self._anchor_previous_point_d,
        )
        self._anchor_previous_pressed_d, self._anchor_next_pressed_d = (
            self._anchor_next_pressed_d,
            self._anchor_previous_pressed_d,
        )
        self._anchor_count = M

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
        self._contact_world_anchor_d = wp.empty(cap, dtype=wp.vec3, device=device)
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
        The signed-distance offset is ``dot(normal, world_anchor)`` where
        ``world_anchor`` is:

        - For dynamic bodies (``body_index >= 0``): ``wp.transform_point(body_q, body_pos)``,
          since ``soft_contact_body_pos`` is in body-local frame.
        - For static shapes (``body_index < 0``): ``body_pos`` directly, since
          ``soft_contact_body_pos`` is already in world frame (``X_wb = identity`` so
          ``X_ws = X_bs``, and the kernel stores ``wp.transform_point(X_bs, x_local)``).

        Args:
            contacts: :class:`~newton.Contacts` populated by a prior
                :meth:`~newton.CollisionPipeline.collide` call, or ``None``
                for a step with self contacts only.
            state: Current :class:`~newton.State`, the one passed to
                ``collide``. Supplies the particle positions of the face and
                self contact queries, and ``body_q``/``body_qd`` for contacts
                on moving bodies; falls back to the model when ``None``.
        """
        if contacts is not None:
            contacts._assert_particle_only_soft_contacts("SolverFBA")
        # The contacts change, so the next solve rebuilds ``W`` from the new J.
        self._invalidate_schur_cache()
        self._contact_extra = None
        self._contact_host_rows = None

        # With a face contact margin, triangles can touch shapes that no particle touches.
        other_contacts = self._face_query is not None and self.face_contact_gap is not None
        other_contacts = other_contacts or self._self_contact is not None or self._rod_contact is not None
        other_contacts = other_contacts or self._segment_query is not None
        M_raw = 0
        if contacts is not None:
            capacity = contacts.soft_contact_particle.shape[0]
            M_raw = stored_contact_count(contacts.soft_contact_count, capacity, "soft")
        if M_raw == 0 and not other_contacts:
            self._contact_count = 0
            return

        # Pull contact data to host for the lexsort, for a deterministic
        # contact order (M is small so this is fast).
        # The collision pipeline writes contacts to slots assigned by
        # ``wp.atomic_add(soft_contact_count, 0, 1)`` so per-frame order
        # depends on GPU thread scheduling. The order of the rows sets the
        # order of the NSN's sums, hence we sort, so that a step is deterministic.
        # The pipeline gives one contact per particle and shape, a particle one vertex-face self contact at most,
        # and a pair of edges one edge-edge contact after them: with these alone, the contacts are sorted and
        # assembled on the device. Steps with vertex-face self
        # contacts take that path, as their many contacts make the host round trip costly; the others keep the
        # host path.
        on_device = contacts is not None and self._face_query is None and self._rod_contact is None
        on_device = on_device and self._segment_query is None
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
                # soft_contact_body_pos convention (from create_soft_contacts kernel):
                #   - body_index >= 0 (dynamic body): contact point in body-local frame.
                #   - body_index  < 0 (static shape): contact point already in world frame
                #     (because X_wb = identity so X_ws = X_bs, and body_pos = X_bs * x_local).
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
                    self.model, src.particle_q, src.body_q, particle_h, shape_h, body_pos_h, normal_h, alarm
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
                    src.body_q,
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
            # Segment contacts of the rods follow: slot 0 holds the segment's first particle, the first extra slot
            # the second.
            if self._segment_query is not None:
                src = state if state is not None else self.model
                found = self._segment_query.query(
                    self.model, src.particle_q, src.body_q, self.face_contact_edge_radius, self.rod_shape_contact_gap
                )
                if len(found[0]):
                    seg, w = self._segment_query.seg[found[0]], found[2]
                    particle_h, alpha_h = np.concatenate([particle_h, seg[:, 0]]), np.concatenate([alpha_h, w[:, 0]])
                    shape_h = np.concatenate([shape_h, found[1].astype(shape_h.dtype)])
                    body_pos_h = np.concatenate([body_pos_h, found[3].astype(body_pos_h.dtype)])
                    normal_h = np.concatenate([normal_h, found[4].astype(normal_h.dtype)])
                    record_h = np.concatenate([record_h, np.full(len(found[0]), -1, dtype=np.int32)])
                    width = 1 if extra is None else extra[0].shape[1]
                    if extra is None:
                        extra = [np.full((M, 1), -1, dtype=np.int32), np.zeros((M, 1), dtype=np.float32)]
                    seg_extra = np.full((len(seg), width), -1, dtype=np.int32)
                    seg_alpha = np.zeros((len(seg), width), dtype=np.float32)
                    seg_extra[:, 0], seg_alpha[:, 0] = seg[:, 1], w[:, 1]
                    extra = [np.concatenate([extra[0], seg_extra]), np.concatenate([extra[1], seg_alpha])]
                    M = len(particle_h)
            # Self contacts follow, with a static anchor at the contact margin along their normal: their rows hold
            # n . sum_k alpha_k x_k >= margin and start without tangential offset.
            shape_contacts = M
            found = []
            if self._self_contact is not None:
                src = state if state is not None else self.model
                sp, sa, sn = self._self_contact.detect(src.particle_q)
                if len(sp):
                    margin = self.particle_self_contact_margin
                    if self._self_contact._world_margins_h is not None:
                        worlds = self.model.particle_world.numpy()[sp[:, 0]]
                        margin = self._self_contact._world_margins_h[worlds, None]
                    if self._self_contact_groups and self._self_contact.within_groups:
                        margin = self._self_contact.contact_margins(sp)[:, None]
                    found.append((sp, sa, sn, margin))
            # Rod contacts follow, each with its own margin, the sum of its capsules' radii.
            if self._rod_contact is not None:
                src = state if state is not None else self.model
                rp, ra, rn, rm = self._rod_contact.detect(src.particle_q)
                if len(rp):
                    found.append((rp, ra, rn, rm[:, None]))
            for sp, sa, sn, margin in found:
                particle_h, alpha_h = np.concatenate([particle_h, sp[:, 0]]), np.concatenate([alpha_h, sa[:, 0]])
                shape_h, normal_h = np.concatenate([shape_h, np.full(len(sp), -1)]), np.concatenate([normal_h, sn])
                body_pos_h = np.concatenate([body_pos_h, margin * sn])
                record_h = np.concatenate([record_h, np.full(len(sp), -1, dtype=np.int32)])
                if extra is None:
                    extra = [np.full((M, 3), -1, dtype=np.int32), np.zeros((M, 3), dtype=np.float32)]
                elif extra[0].shape[1] < 3:
                    extra = [
                        np.pad(extra[0], ((0, 0), (0, 1)), constant_values=-1),
                        np.pad(extra[1], ((0, 0), (0, 1))),
                    ]
                extra = [np.concatenate([extra[0], sp[:, 1:]]), np.concatenate([extra[1], sa[:, 1:]])]
                M = len(particle_h)
        self._contact_extra_h = None
        if extra is not None:
            self._contact_extra = [wp.array(a, device=self._device) for a in extra]
            self._contact_extra_h = extra
        if M == 0:
            self._contact_count = 0
            return

        self._ensure_contact_buffers(M)

        # Per-contact bookkeeping (world_anchor, n·anchor offset, tangent
        # basis, t·anchor tangent offsets, v_anchor for rolling cylinders)
        # runs on GPU.
        from .kernels import (  # noqa: PLC0415
            compute_normal_offset_kernel,
            compute_tangent_basis_kernel,
            compute_tangent_offsets_kernel,
            compute_v_anchor_kernel,
            compute_world_anchor_kernel,
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
        # The pose must match the one ``collide`` used to emit
        # ``soft_contact_body_pos``, i.e. the current state, not the model.
        body_src = state if state is not None else model
        body_q_arr = body_src.body_q if getattr(body_src, "body_q", None) is not None else None
        has_body_q = wp.int32(1 if body_q_arr is not None and body_q_arr.size > 0 else 0)
        if body_q_arr is None or body_q_arr.size == 0:
            body_q_arr = wp.empty(1, dtype=wp.transform, device=device)

        shape_body_arr = model.shape_body if hasattr(model, "shape_body") and model.shape_body is not None else None
        has_shape_body = wp.int32(1 if shape_body_arr is not None and shape_body_arr.size > 0 else 0)
        if shape_body_arr is None or shape_body_arr.size == 0:
            shape_body_arr = wp.empty(1, dtype=wp.int32, device=device)

        shape_transform_arr = (
            model.shape_transform if hasattr(model, "shape_transform") and model.shape_transform is not None else None
        )
        has_shape_transform = wp.int32(1 if shape_transform_arr is not None and shape_transform_arr.size > 0 else 0)
        if shape_transform_arr is None or shape_transform_arr.size == 0:
            shape_transform_arr = wp.empty(1, dtype=wp.transform, device=device)

        has_shape_omega = wp.int32(1 if self._shape_omega_d.size > 0 else 0)

        # World-frame anchor: static-shape contacts pass body_pos through,
        # dynamic contacts apply body_q.
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

        # Normal offset n·anchor.
        wp.launch(
            compute_normal_offset_kernel,
            dim=M,
            inputs=[self._contact_normal_d, self._contact_world_anchor_d],
            outputs=[self._contact_offset_d],
            device=device,
        )
        # Particle and face contacts stop the particle radius plus the shape margin outside the shape surface, as
        # Newton's contact margins place them, or the shape's radius from shape_particle_radius_override.
        if shape_contacts:
            from .kernels import add_contact_thickness_kernel  # noqa: PLC0415

            extra = self._contact_extra
            if extra is None:
                extra = [
                    wp.empty((1, 1), dtype=wp.int32, device=device),
                    wp.empty((1, 1), dtype=wp.float32, device=device),
                ]
            shape_radius = self._shape_radius_d
            if shape_radius is None:
                shape_radius = wp.empty(1, dtype=wp.float32, device=device)
            inputs = [self._contact_particle_d, *extra, int(self._contact_extra is not None)]
            inputs += [self._contact_shape_d, model.particle_radius, model.shape_margin]
            inputs += [shape_radius, int(self._shape_radius_d is not None)]
            wp.launch(add_contact_thickness_kernel, shape_contacts, inputs, [self._contact_offset_d], device=device)

        # Host mirrors of the basic contact fields, for the host paths that still read them.
        self._contact_particle_h = None if particle_h is None else particle_h.astype(np.int32)
        self._contact_alpha_h = alpha_h
        # The rows' particles and coefficients on the host, for the Schur build without a read back.
        if particle_h is not None:
            extra_h = (None, None) if self._contact_extra_h is None else tuple(self._contact_extra_h)
            self._contact_host_rows = (self._contact_particle_h, alpha_h, *extra_h)
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
            # When the contact's parent shape is attached to a moving rigid
            # body, body-driven 6-DoF velocity supersedes the
            # per-shape ``shape_angular_velocity``.  This kernel only writes
            # contacts whose ``shape_body[s] >= 0``; static-shape contacts keep
            # the value computed above by ``compute_v_anchor_kernel``.
            body_qd_arr = body_src.body_qd if getattr(body_src, "body_qd", None) is not None else None
            if (
                has_shape_body == wp.int32(1)
                and body_qd_arr is not None
                and body_qd_arr.size > 0
                and has_body_q == wp.int32(1)
            ):
                from .kernels import compute_v_anchor_from_body_kernel  # noqa: PLC0415

                wp.launch(
                    compute_v_anchor_from_body_kernel,
                    dim=M,
                    inputs=[
                        self._contact_shape_d,
                        shape_body_arr,
                        body_q_arr,
                        body_qd_arr,
                        model.body_com,
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
        """Sort the particle-shape contacts by particle and shape and append the self contacts, the vertex-face ones
        in the order of their particles and the edge-edge ones after them, as the host sort orders them, on the
        device. Returns the number of contacts and of particle-shape contacts."""
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
        groups = [sc.groups, sc.group_margins, 1]
        # Without contacts within groups, every self contact joins two groups and keeps the margin between groups.
        if not self._self_contact_groups or not sc.within_groups:
            groups = [wp.empty(1, dtype=wp.int32, device=device), wp.empty(1, dtype=wp.float32, device=device), 0]
        inputs += [self.particle_self_contact_margin, self.model.particle_world, sc.world_margins, *groups]
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
        n_bilateral: int = 0,
        held_d: wp.array | None = None,
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

        n_rows = 3 * int(M_total) + int(n_bilateral)
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
                *self._nsn_soft_off,
            ],
            outputs=[omega_v, compliance_v, h_v],
            device=device,
        )
        # Bilateral rows follow the contact rows; contacts on a particle they hold at a point carry no force.
        if n_bilateral:
            inputs = [pene0_d, wp.int32(3 * int(M_total))]
            law_out = [omega_v, compliance_v, h_v]
            wp.launch(K.bilateral_law_kernel, dim=int(n_bilateral), inputs=inputs, outputs=law_out, device=device)
        if held_d is not None:
            law_out = [omega_v, compliance_v, h_v]
            wp.launch(K.held_contact_law_kernel, dim=int(M_total), inputs=[held_d], outputs=law_out, device=device)

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
        self._pcr_solver.world_rows = self._deformable_world_rows
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
        if self._lite_direct is not None:
            direct = self._lite_direct
            wp.launch(
                K.restore_lite_direct_rows_kernel,
                direct.rows.shape[0],
                [direct.rows, lam_init_d, omega_init_d],
                [lam_out_d, omega_out_d],
                device=device,
            )
            self._launch_lite_worlds(
                ls, M_total, dt, cap_internal, use_cap, r_d, pene0_d, lam_out_d, omega_out_d, lam_apply_out_d
            )

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

    def _lite_sparse_iteration(self, M: int, n_b: int, dt: float, use_graph: bool) -> None:
        """Run a local-global iteration's sparse lite contact correction. Its matrices and buffers hold for the time
        step, so its launches are captured in a graph at the step's first iteration and replayed at all of them,
        when the step has several iterations and its linear solve runs a fixed count of iterations without reading
        the device."""
        pcr = self._pcr_solver
        if not (use_graph and self.iterations > 1 and pcr.max_iter <= pcr.check_every):
            self._lite_sparse_correction(M, n_b, dt)
            return
        if self._lite_iter_graph is None:
            # The buffers that the correction touches take their sizes for the step before the capture.
            ls = self._linear_solver
            ls.reserve_lambda_correction()
            if self._held_rows is not None:
                self._held_row_factor(ls)
            try:
                with wp.ScopedCapture(device=self._device) as capture:
                    self._lite_sparse_correction(M, n_b, dt)
                self._lite_iter_graph = capture.graph
            except RuntimeError:
                self._lite_iter_graph = False
        if self._lite_iter_graph:
            wp.capture_launch(self._lite_iter_graph)
        else:
            self._lite_sparse_correction(M, n_b, dt)

    def _lite_sparse_correction(self, M: int, n_b: int, dt: float) -> None:
        """One iteration's sparse lite contact correction: residuals, an NSN step, the multipliers kept for the next
        iteration, the combined correction, and the step of the rows that hold particles at points."""
        from . import kernels as K  # noqa: PLC0415

        ls, device, n_rows = self._linear_solver, self._device, 3 * M + n_b
        accumulate = self.nsn_lite_accumulate
        if accumulate:
            self._accumulate_predict()
        self._launch_contact_residual(self._nsn_r_d)
        self._launch_contact_pene0(self._nsn_pene0_d)
        if n_b:
            inputs = [*self._constraint_rows, self._x_cur, wp.int32(3 * M)]
            outputs = [self._nsn_r_d, self._nsn_pene0_d]
            wp.launch(K.particle_constraint_residual_kernel, dim=n_b, inputs=inputs, outputs=outputs, device=device)
        # Warm-start ω and λ from the previous iteration.
        wp.copy(self._nsn_omega_d, self._omega_coulomb_persistent_d, count=n_rows)
        wp.copy(self._nsn_lam_d, self._lam_coulomb_persistent_d, count=n_rows)
        W_bsr, held_d = self._lite_W
        if accumulate:
            self._accumulate_shift(W_bsr, M, n_b, dt)
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
            n_bilateral=n_b,
            held_d=held_d,
        )
        if self._held_rows is not None:
            # The contacts of particles held at points carry no force, so the rows that hold them take their own
            # step after the correction of the other rows.
            from .joints import scatter_values_kernel  # noqa: PLC0415

            h = self._held_row_factor(ls)
            outputs = [self._nsn_lam_apply_d[3 * M :]]
            wp.launch(scatter_values_kernel, h.rows.shape[0], [h.zeros, 0, h.rows], outputs, device=device)
        self._deformable_contact_tail(ls, n_rows)
        if accumulate:
            self._accumulate_displacement()
        if self._held_rows is not None:
            self._step_held_rows(ls, M, dt)

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

    def _accumulate_shift(self, W_bsr, M: int, n_b: int, dt: float) -> None:
        """Take the lite motion ``dt² W (ω λ)`` of the forces applied so far off the residual, so that the lite NSN
        models only their change: its penetration ``-r + dt² W (ω λ)`` starts at the exact one."""
        from .kernels import add_scaled_f64_kernel, compute_omega_times_lam_kernel  # noqa: PLC0415
        from .nsn_pcr_solver import csr_mv  # noqa: PLC0415

        device, n_rows = self._device, 3 * M + n_b
        omega_lam, w = self._acc_omega_lam[:n_rows], self._acc_w[:n_rows]
        inputs = [self._omega_coulomb_persistent_d, self._lam_coulomb_persistent_d]
        wp.launch(compute_omega_times_lam_kernel, n_rows, inputs, [omega_lam], device=device)
        if self._accumulate_disabled_rows.size:
            from .joints import scatter_values_kernel  # noqa: PLC0415

            rows = self._accumulate_disabled_rows
            wp.launch(
                scatter_values_kernel, rows.size, [self._accumulate_disabled_zeros, 0, rows], [omega_lam], device=device
            )
        if self._lite_direct is not None:
            from .joints import scatter_values_kernel  # noqa: PLC0415

            direct = self._lite_direct
            wp.launch(
                scatter_values_kernel, direct.rows.shape[0], [direct.zeros, 0, direct.rows], [omega_lam], device=device
            )
        # The rows that hold particles at points take their own exact step and apply nothing here.
        if self._held_rows is not None:
            from .joints import scatter_values_kernel  # noqa: PLC0415

            h = self._held_row_factor(self._linear_solver)
            wp.launch(scatter_values_kernel, h.rows.shape[0], [h.zeros, 0, h.rows], [omega_lam[3 * M :]], device=device)
        csr_mv(W_bsr, omega_lam, w)
        wp.launch(add_scaled_f64_kernel, n_rows, [w, wp.float64(dt * dt)], [self._nsn_r_d], device=device)

    def _accumulate_displacement(self) -> None:
        """The displacement that the contact forces applied so far caused, for the next iteration."""
        from .kernels import sub_vec3_kernel  # noqa: PLC0415

        inputs = [self._x_cur, self._acc_free]
        wp.launch(sub_vec3_kernel, self.model.particle_count, inputs, [self._acc_d], device=self._device)
        if self._accumulate_disabled_particles.size:
            from .kernels import clear_lite_direct_displacement_kernel  # noqa: PLC0415

            particles = self._accumulate_disabled_particles
            wp.launch(
                clear_lite_direct_displacement_kernel, particles.size, [particles], [self._acc_d], device=self._device
            )
        if self._lite_direct is not None:
            from .kernels import clear_lite_direct_displacement_kernel  # noqa: PLC0415

            particles = self._lite_direct.all_particles
            wp.launch(
                clear_lite_direct_displacement_kernel,
                particles.shape[0],
                [particles],
                [self._acc_d],
                device=self._device,
            )

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
        self._pcr_solver.world_rows = self._deformable_world_rows

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

    def _deformable_contact_tail(self, ls, rows: int) -> None:
        """Keep the NSN step's multipliers for the next local-global iteration and correct the positions."""
        # Persist device-resident lam / omega for next PD iter.
        wp.copy(self._lam_coulomb_persistent_d, self._nsn_lam_d, count=rows)
        wp.copy(self._omega_coulomb_persistent_d, self._nsn_omega_d, count=rows)
        # Apply the contact forces in combined form, with one solve over the incremented right-hand side. It runs
        # even when the forces vanish: the solve costs less than reading them back to check.
        ls.apply_lambda_correction_combined(self._nsn_lam_apply_d, self._rhs, self._x_cur)

    def _ensure_nsn_sparse_scratch(self, n_rows: int) -> None:
        """Allocate the vectors of the sparse NSN steps for ``n_rows`` rows."""
        if getattr(self, "_nsn_sparse_scratch", None) is None or self._nsn_sparse_scratch[0].shape[0] < n_rows:
            cap = max(self._nsn_inner_n, n_rows)
            self._nsn_sparse_scratch = [wp.empty(cap, dtype=wp.float64, device=self._device) for _ in range(5)]

    def _nsn_sparse_graph_step(
        self, name: str, W_sparse: tuple, inputs: tuple, sizes: tuple, block_precond, tail: tuple
    ) -> None:
        """The sparse NSN step of :meth:`_solve_nsn_coulomb_gpu` on the device inputs ``(r, pene0, mu, lam_init,
        omega_init)``, its launches captured in a graph at the first call and replayed while the matrices, the
        buffers, the sizes ``(M, n_rows, n_bilateral, dt, max_iters, contact_regularization)`` and the block
        preconditioner stay those of the capture, followed by the launches of ``tail = (function, objects)``, keyed
        by the objects. The graph holds the objects whose identities key it.

        It runs eagerly where the linear solve reads the device to stop, or the block preconditioner is not the
        joint rows' solve."""
        from . import kernels as K  # noqa: PLC0415

        r_d, pene0_d, mu_d, lam_d, omega_d = inputs
        M, n_rows, n_bilateral, dt, max_iters, regularization = sizes
        device, cap = self._device, self._nsn_inner_n
        self._ensure_nsn_sparse_scratch(n_rows)

        def run():
            for src, dst in ((lam_d, self._nsn_lam_d), (omega_d, self._nsn_omega_d)):
                wp.copy(dst, src, count=n_rows)
                if cap > n_rows:
                    dst[n_rows:cap].zero_()
            args = (W_sparse, r_d, pene0_d, mu_d, M, n_rows, n_bilateral, dt, max_iters)
            self._nsn_coulomb_sparse_steps(*args, block_precond, regularization)
            if self.lambda_cap is not None:
                limit = wp.float64(float(self.lambda_cap) / (dt * dt))
                wp.launch(K.lambda_cap_clip_kernel, 3 * M, inputs=[limit], outputs=[self._nsn_lam_d], device=device)
            inputs = [self._nsn_lam_d, self._nsn_omega_d, wp.float64(dt)]
            wp.launch(K.compute_lam_apply_kernel, n_rows, inputs=inputs, outputs=[self._nsn_lam_apply_d], device=device)
            if tail[0] is not None:
                tail[0]()

        if self._outer_capture:
            # The rigid iteration's graph is being captured: record this step's launches into it.
            run()
            return
        pcr = self._pcr_solver
        precond_key = None
        if block_precond is not None:
            # The joint rows' solve: its system and rows.
            func = getattr(block_precond, "func", None)
            owner = getattr(func, "__self__", None)
            if owner is None or getattr(block_precond, "keywords", None):
                run()
                return
            precond_key = (id(owner), *(id(a) if isinstance(a, wp.array) else a for a in block_precond.args))
        if pcr.max_iter > pcr.check_every or not wp.get_device(device).is_cuda:
            run()
            return
        ls = self._linear_solver
        key = (
            name,
            # The linear solver's arrays that the step reads: a new one raises their generation.
            getattr(ls, "_graph_generation", 0),
            getattr(ls, "_row_slots", None) is not None,
            # The entries of W, which launches of the step count; a rigid step's pattern matrix may be None.
            tuple(int(getattr(m, "nnz", 0)) for m in W_sparse),
            *(id(a) for a in (*W_sparse, *inputs, *tail[1])),
            sizes,
            precond_key,
            id(pcr),
            id(pcr.world_rows),
            id(self._nsn_contact_scale),
            *(id(a) for a in self._nsn_soft),
            # The step's buffers, which grow with the rows of another NSN solve.
            self._nsn_inner_n,
            id(self._nsn_sparse_scratch[0]),
        )
        cached = self._nsn_graph
        if cached is None or cached[0] != key:
            try:
                with wp.ScopedCapture(device=device) as capture:
                    run()
                graph = capture.graph
            except RuntimeError:
                graph = None
            # The objects of the key stay alive with the graph, so that no new object takes one of their identities.
            cached = self._nsn_graph = (
                key,
                graph,
                (
                    W_sparse,
                    inputs,
                    block_precond,
                    pcr,
                    tail[1],
                    pcr.world_rows,
                    self._nsn_contact_scale,
                    self._nsn_soft,
                ),
                self._nsn_sparse_scratch,
            )
            if graph is None:
                run()
                return
        if cached[1] is None:
            run()
        else:
            wp.capture_launch(cached[1])

    def _nsn_coulomb_sparse_steps(
        self,
        W_sparse: tuple,
        r_d: wp.array[wp.float64],
        pene0_d: wp.array[wp.float64],
        mu_d: wp.array[wp.float64],
        M: int,
        n_rows: int,
        n_bilateral: int,
        dt: float,
        max_iters: int,
        block_precond: Callable[[wp.array, wp.array], None] | None,
        contact_regularization: float,
    ) -> None:
        """The Newton steps of :meth:`_solve_nsn_coulomb_gpu` on ``W`` in CSR form, ``W_sparse``, with a matrix of
        its pattern for the Newton matrix: the same expressions, with the products by ``W`` and the Newton matrix
        sparse."""
        import warp.sparse as wps  # noqa: PLC0415

        from . import kernels as K  # noqa: PLC0415

        device, dt_w = self._device, wp.float64(dt)
        from .nsn_pcr_solver import block3_diag, block3_mv  # noqa: PLC0415

        W_sp, A_sp = W_sparse
        self._ensure_nsn_sparse_scratch(n_rows)
        ol, wol, diag, shift, a_diag = (a[:n_rows] for a in self._nsn_sparse_scratch)
        pcr = self._pcr_solver
        lam, omega = self._nsn_lam_d[:n_rows], self._nsn_omega_d[:n_rows]
        # W in 3x3 blocks, of padded rigid rows, comes with a fused PCR, which applies the Newton matrix factored.
        blocked = getattr(W_sp, "block3", False)
        if blocked and not pcr.fuses_alone(n_rows):
            raise RuntimeError("SolverFBA keeps W in 3x3 blocks only for a fused PCR")
        from .nsn_pcr_solver import _LONG_ROW, _TILED_MV  # noqa: PLC0415

        # Long rows of W take the tiled product, as the conjugate residual iterations' products do.
        tile = _TILED_MV if int(W_sp.nnz) > _LONG_ROW * n_rows else -1
        product = block3_mv if blocked else functools.partial(wps.bsr_mv, tile_size=tile)
        # Frictional precond: dt²·W_ii for normal rows; dt·W_ii for tangents.
        if blocked:
            block3_diag(W_sp, diag)
        else:
            wps.bsr_get_diag(W_sp, out=diag)
        wp.launch(K.precond_from_bsr_diag_coulomb_kernel, n_rows, [diag, dt_w], [self._nsn_precond_d], device=device)
        for _ in range(max_iters):
            wp.launch(K.compute_omega_times_lam_kernel, n_rows, [omega, lam], [ol], device=device)
            product(W_sp, ol, wol)
            outputs = [self._nsn_penetration_d]
            wp.launch(K.compute_penetration_sparse_kernel, n_rows, [r_d, wol, dt_w], outputs, device=device)
            wp.launch(
                K.compute_frictional_fb_kernel,
                dim=M,
                inputs=[
                    *(self._nsn_penetration_d, self._nsn_lam_d, mu_d, self._nsn_precond_d, pene0_d, dt_w),
                    *self._nsn_soft,
                ],
                outputs=[self._nsn_omega_d, self._nsn_compliance_d, self._nsn_h_d],
                device=device,
            )
            if n_bilateral > 0:
                wp.launch(
                    K.bilateral_law_kernel,
                    dim=n_bilateral,
                    inputs=[pene0_d, wp.int32(3 * M)],
                    outputs=[self._nsn_omega_d, self._nsn_compliance_d, self._nsn_h_d],
                    device=device,
                )
            # The Newton matrix, with the diagonal of the contact rows regularized. A fused PCR applies it as its
            # factors Ω W Ω + diag(shift), Ω = diag(ω); the other runs take it formed.
            regularized = 3 * M if contact_regularization > 0.0 else 0
            factored = pcr.fuses_alone(n_rows)
            if factored:
                inputs = [diag, self._nsn_omega_d, self._nsn_compliance_d, regularized]
                inputs += [wp.float64(1.0 + contact_regularization), self._nsn_contact_scale]
                wp.launch(K.newton_matrix_diagonal_kernel, n_rows, inputs, [shift, a_diag], device=device)
            else:
                inputs = [W_sp.offsets, W_sp.columns, W_sp.values, self._nsn_omega_d, self._nsn_compliance_d]
                inputs += [regularized, wp.float64(1.0 + contact_regularization)]
                inputs += [self._nsn_contact_scale, int(W_sp.nrow)]
                wp.launch(K.build_a_schur_csr_kernel, int(W_sp.nnz), inputs, [A_sp.values], device=device)
            wp.launch(K.compute_omega_times_lam_kernel, n_rows, [omega, lam], [ol], device=device)
            product(W_sp, ol, wol)
            wp.launch(
                K.compute_nsn_rhs_lite_bsr_kernel,
                dim=n_rows,
                inputs=[self._nsn_h_d, self._nsn_omega_d, pene0_d, r_d, wol, dt_w],
                outputs=[self._nsn_rhs_d],
                device=device,
            )
            rhs, dlam = self._nsn_rhs_d[:n_rows], self._nsn_dlam_d[:n_rows]
            # Fixed-count runs take the fused iteration: four launches per iteration instead of a dozen.
            if factored:
                pcr.solve_sparse_scaled(W_sp, omega, shift, a_diag, rhs, dlam, block_precond=block_precond)
            else:
                pcr.solve_sparse_fused(A_sp, rhs, dlam, block_precond=block_precond)
            wp.launch(K.axpy_lambda_kernel, n_rows, inputs=[self._nsn_dlam_d], outputs=[self._nsn_lam_d], device=device)
            # Per-contact box clamp of the tangent multipliers; no friction without a positive normal multiplier.
            wp.launch(K.coulomb_box_clamp_kernel, M, inputs=[mu_d], outputs=[self._nsn_lam_d], device=device)

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
        W_sparse: tuple | None = None,
        graph: str | None = None,
        world_rows: tuple | None = None,
        contact_scale: wp.array | None = None,
        graph_tail: Callable[[], None] | None = None,
        graph_tail_objects: tuple = (),
        r_device: wp.array[wp.float64] | None = None,
        pene0_device: wp.array[wp.float64] | None = None,
        mu_device: wp.array[wp.float64] | None = None,
        lam_init_device: wp.array[wp.float64] | None = None,
        omega_init_device: wp.array[wp.float64] | None = None,
        skip_download: bool = False,
        m_contacts: int | None = None,
        n_bilateral: int = 0,
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
        contact_regularization: float = 0.0,
        soft_rows: tuple | None = None,
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
            W_sparse: With ``nsn_sparse_storage``, ``W`` in CSR form and a
                matrix of its pattern for the Newton matrix, in place of
                ``W`` and ``W_device``.
            graph: With ``W_sparse``, device inputs and ``skip_download``, a
                name under which the step's launches are captured in a CUDA
                graph at the first call and replayed at the following calls
                with the same matrices and buffers, as the local-global
                iterations of a time step make them.
            graph_tail: Launches that follow the step, captured with it,
                such as the correction of the positions; the objects whose
                identities key them are ``graph_tail_objects``. Without the
                graph, they follow the step eagerly.
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
            n_bilateral: Number of bilateral rows after the ``3M`` contact
                rows; all row arrays then hold ``3M + n_bilateral`` rows.
                They follow the identity law of :func:`bilateral_law_kernel`
                and skip the cone clamp and ``lambda_cap``.
            block_precond: Optional block preconditioner of the bilateral
                rows, passed to :meth:`NSNPCRSolver.solve`.
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
        n_rows = 3 * M + int(n_bilateral)
        if n_rows == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        device = self._device

        self._ensure_nsn_inner_buffers(n_rows, alloc_2d=not self.nsn_sparse_storage)
        self._ensure_pcr_solver(n_rows)
        cap = self._nsn_inner_n
        self._pcr_solver.world_rows = world_rows
        self._nsn_contact_scale = self._empty_contact_scale if contact_scale is None else contact_scale
        # Soft rigid contacts (eps, beta, start gap per contact) for the law of this solve.
        self._nsn_soft = self._nsn_soft_off if soft_rows is None else soft_rows
        if graph is not None and W_sparse is not None and skip_download:
            inputs = (r_device, pene0_device, mu_device, lam_init_device, omega_init_device)
            if all(a is not None for a in inputs):
                sizes = (M, n_rows, int(n_bilateral), float(dt), int(max_iters), float(contact_regularization))
                tail = (graph_tail, graph_tail_objects)
                self._nsn_sparse_graph_step(graph, W_sparse, inputs, sizes, block_precond, tail)
                empty = np.zeros(0, dtype=np.float64)
                return empty, empty, empty

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
        if W_sparse is not None:
            W_d = None
        elif W_device is not None and W_device.shape[0] >= n_rows and W_device.shape[1] >= n_rows:
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

        if self.nsn_sparse_storage:
            sparse = (W_sparse, r_d, pene0_d, mu_d, M, n_rows, n_bilateral, dt)
            self._nsn_coulomb_sparse_steps(*sparse, int(max_iters), block_precond, contact_regularization)
        else:
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
                        *self._nsn_soft,
                    ],
                    outputs=[
                        self._nsn_omega_d,
                        self._nsn_compliance_d,
                        self._nsn_h_d,
                    ],
                    device=device,
                )
                if n_bilateral > 0:
                    wp.launch(
                        K.bilateral_law_kernel,
                        dim=n_bilateral,
                        inputs=[pene0_d, wp.int32(3 * M)],
                        outputs=[self._nsn_omega_d, self._nsn_compliance_d, self._nsn_h_d],
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
                        inputs=[wp.float64(1.0 + contact_regularization), self._nsn_contact_scale],
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
                self._pcr_solver.solve(a_view, rhs_view, dlam_view, block_precond=block_precond)

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

        if graph_tail is not None:
            graph_tail()
        if skip_download:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty

        lam = self._nsn_lam_d.numpy()[:n_rows].astype(np.float64, copy=True)
        omega = self._nsn_omega_d.numpy()[:n_rows].astype(np.float64, copy=True)
        lam_apply = self._nsn_lam_apply_d.numpy()[:n_rows].astype(np.float64, copy=True)
        return lam, omega, lam_apply
