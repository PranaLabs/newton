# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase


class SolverFBA(SolverBase):
    """Fast But Accurate projective-dynamics cloth solver.

    .. experimental::

        :class:`SolverFBA`, its constructor options and its solver-specific
        attributes may change without the normal deprecation period. It
        simulates cloth in models with one world and no rigid bodies.

    A step depends only on its input state and control, and on the pin
    targets last set, so the solver keeps no history for :meth:`reset` to
    discard.

    Implements a Projective Dynamics (PD) local-global iteration for cloth
    with ARAP stretching, isometric bending and soft pin constraints. The PD
    Hessian is prefactored once (nested dissection ordering, multifrontal
    LDLᵀ); the sparse inverse ``S = L⁻¹`` is computed exactly on the
    elimination tree and uploaded to Warp BSR matrices. The runtime linear
    solve evaluates ``A⁻¹·b = Sᵀ · D⁻¹ · S · b`` via two GPU SpMVs per
    coordinate component, avoiding the inherently sequential GPU triangular
    solve.

    Stretching is As-Rigid-As-Possible (ARAP): the local step projects each
    triangle's deformation gradient to the nearest rotation, its singular
    values clamped to 1, with the stretching weight ``2 * tri_ke`` times the
    triangle's rest area.

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

    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
    ) -> None:
        """
        Args:
            model: The :class:`~newton.Model` containing the cloth to integrate.
            iterations: Number of PD local-global iterations per step.
            pin_stiffness: Soft-pin stiffness [N/m].
        """
        super().__init__(model)
        if model.world_count > 1:
            raise ValueError(f"SolverFBA simulates one world, but the model has {model.world_count}")

        if model.body_count > 0:
            raise NotImplementedError(f"SolverFBA does not simulate rigid bodies, but the model has {model.body_count}")

        if model.tet_count > 0:
            raise NotImplementedError(f"SolverFBA simulates cloth only, but the model has {model.tet_count} tetrahedra")

        # ---- Validate ----
        if model.particle_count == 0:
            raise ValueError("SolverFBA requires deformable elements")
        if model.tri_count == 0:
            raise ValueError("SolverFBA requires at least one cloth triangle")

        if pin_stiffness <= 0:
            raise ValueError("pin_stiffness must be positive")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.iterations = int(iterations)
        self.pin_stiffness = float(pin_stiffness)
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
        self._edge_norm_d = None
        self._pin_indices_d = self._inactive_d = None

        # Per-(tri, local vertex) scratch for deterministic (compute → gather)
        # PD cloth-stretching scatter; allocated in _setup_pd_system, reused per step.
        self._tri_contrib_d: wp.array | None = None
        # Per-(edge, local vertex) scratch for deterministic (compute → gather)
        # PD isometric bending scatter; allocated in _setup_pd_system, reused per step.
        self._edge_contrib_d: wp.array | None = None

        # Particle-centered CSR adjacency for deterministic PD reductions.
        # Lets the PD scatter use (compute → gather) instead of atomic_add.
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
        # Particle-centered CSR adjacency for deterministic PD reductions.
        # Lets the PD scatter use (compute → gather) instead of atomic_add.
        from .linear_solver import build_particle_element_csr  # noqa: PLC0415

        n_p = int(self.model.particle_count)
        tri_indices_np = meta["tri_indices"]
        edge_indices_np = meta["edge_indices"]

        if tri_indices_np is not None and tri_indices_np.size > 0:
            offs, elem_idx, local_v = build_particle_element_csr(tri_indices_np, n_p, 3)
            self._particle_tri_offsets_d = wp.array(offs, dtype=wp.int32, device=device)
            self._particle_tri_element_d = wp.array(elem_idx, dtype=wp.int32, device=device)
            self._particle_tri_local_d = wp.array(local_v, dtype=wp.int32, device=device)
            # Per-(tri, local vertex) scratch buffer for deterministic
            # (compute → gather) PD reduction.
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
            contacts: ``None``, or contacts without soft contacts: SolverFBA
                does not solve contacts, and it raises
                :class:`NotImplementedError` when ``contacts`` holds any.
            dt: Time step [s]. The first step and a change of ``dt`` factor the
                system again.
            observables: Optional observables from :meth:`observables`; the
                solver exports none.
        """
        from .kernels import (  # noqa: PLC0415
            compute_inertial_kernel,
            copy_particles_kernel,
            write_velocity_kernel,
        )

        self.validate_observables(observables, contacts)
        if contacts is not None and int(contacts.soft_contact_count.numpy()[0]) > 0:
            raise NotImplementedError("SolverFBA does not solve contacts, but the contacts hold soft contacts")
        if self._linear_solver is None or self._dt_setup is None or abs(self._dt_setup - dt) > 1e-12:
            self._setup_pd_system(dt)

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
