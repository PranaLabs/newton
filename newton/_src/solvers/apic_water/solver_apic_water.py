# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incompressible free-surface water: APIC particles on a MAC grid with a cut-cell pressure projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry.flags import MeshProperties, MeshSignMethod, ShapeFlags
from ...geometry.types import GeoType
from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State
from ..solver import SolverBase
from . import kernels as K

# Shape types that the colliders take, by their signed distance.
_COLLIDER_TYPES = (
    GeoType.PLANE,
    GeoType.SPHERE,
    GeoType.CAPSULE,
    GeoType.ELLIPSOID,
    GeoType.CYLINDER,
    GeoType.BOX,
    GeoType.MESH,
    GeoType.CONE,
    GeoType.CONVEX_MESH,
)

# The liquid level set's particle radius per particle spacing: a lattice of particles puts the surface half a spacing
# above its top layer, as a block of water of the particles' volume has it.
_SURFACE_RADIUS = 0.882
# Particles closer than this part of their spacing are pushed apart.
_SEPARATION_SPACING = 0.8
# How far outside the solids [cells] the particles are kept.
_SOLID_MARGIN = 0.05
# The node spacing of the bodies' distance volumes [cells].
_VOLUME_SPACING = 0.5


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the rows of ``v`` by the quaternion ``q = (x, y, z, w)``."""
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _shape_half_extents(model: Model, shape: int, geo: GeoType, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The box that bounds a shape in its own frame, as its lower and upper corners."""
    if geo == GeoType.SPHERE:
        half = np.full(3, scale[0])
    elif geo in (GeoType.BOX, GeoType.ELLIPSOID):
        half = np.abs(scale)
    elif geo == GeoType.CAPSULE:
        half = np.array([scale[0], scale[0], scale[1] + scale[0]])
    elif geo in (GeoType.CYLINDER, GeoType.CONE):
        half = np.array([scale[0], scale[0], scale[1]])
    elif geo in (GeoType.MESH, GeoType.CONVEX_MESH):
        points = np.asarray(model.shape_source[shape].vertices, dtype=np.float64) * scale
        return points.min(axis=0), points.max(axis=0)
    else:
        raise ValueError(f"shape {shape} of type {geo.name} on a body cannot be a water collider")
    return -half, half


class SolverAPICWater(SolverBase):
    """Incompressible water with a free surface, by APIC particles on a MAC grid with a cut-cell pressure solve.

    The water is the active particles of the model (:attr:`ParticleFlags.ACTIVE <newton.ParticleFlags.ACTIVE>`),
    each carrying ``particle_mass / density`` of volume. Each step transfers them to the faces of a fixed MAC grid by
    the affine particle-in-cell method (APIC, quadratic B-splines), adds gravity, solves the pressure of the fluid
    cells so that the velocity is divergence-free on the parts of the faces outside the solids, and transfers the
    velocity back to the particles, which then move, leave the solids, and keep apart:

    - The solids are the shapes of a rigid model taken by :meth:`setup_colliders`: their signed distances at the
      grid's nodes give each face's open fraction (the variational cut-cell method of Batty, Bertails and Bridson,
      2007), the static shapes' once, the bodies' at each step from their poses. A closed part of a face moves with
      its body.
    - The free surface is the particles' level set (Zhu and Bridson, 2005); the pressure vanishes on it through the
      ghost fluid, and cells with particles are always fluid.
    - The pressure solve takes a fixed number of conjugate gradient iterations with a Jacobi preconditioner, one
      reduction per iteration (Chronopoulos and Gear), its sums in a fixed order. A global correction keeps the
      liquid's volume at the particles' volume, and particles closer than their rest spacing are pushed apart.
    - The pressure's impulse on each body over the step, :attr:`body_impulse`, couples the water to a rigid solver
      explicitly: :meth:`add_body_forces` applies it in the body solver's next step.
    - Particles that leave the grid's box, or turn non-finite, leave the water: they become inactive. Streams
      (:meth:`add_stream`) bring inactive particles back in.

    A step reads nothing back to the host and allocates nothing, so that it records into a CUDA graph. It needs a
    CUDA device. The time step should keep the particles within a cell or two per step. Sums of particle quantities
    onto the grid use atomic additions, so that steps differ from run to run by rounding.

    Call :meth:`register_custom_attributes` on the :class:`~newton.ModelBuilder` of the particles before building
    the model: it adds the per-particle APIC matrix ``apic_water:particle_qd_grad`` to the states.

    The bodies' poses and velocities come from ``state_in.body_q`` and ``state_in.body_qd``, which the caller sets to
    the rigid model's state arrays, as for :class:`~newton.solvers.SolverImplicitMPM`'s colliders.

    Args:
        model: The model of the water particles.
        config: The solver's configuration, see :class:`SolverAPICWater.Config`.

    Raises:
        ValueError: On a CPU device, a model without the custom attributes, or an empty grid box.
    """

    @dataclass
    class Config:
        """Configuration of :class:`SolverAPICWater`."""

        voxel_size: float = 0.01
        """Cell size [m] of the grid; two particles per cell edge give the water its best surface."""
        domain_lo: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Lower corner [m] of the grid's box. Particles that leave the box leave the water."""
        domain_hi: tuple[float, float, float] = (1.0, 1.0, 1.0)
        """Upper corner [m] of the grid's box, rounded up to whole cells."""
        density: float = 1000.0
        """Density [kg/m^3] of the water: each particle carries ``particle_mass / density`` of volume."""
        pressure_iterations: int = 60
        """Conjugate gradient iterations of the pressure solve per step."""
        max_fluid_cells: int | None = None
        """Fluid cells that the pressure solve holds; ``None`` takes the particle count, at most the grid's cells.
        Fluid cells past it stay air; :meth:`SolverAPICWater.dropped_fluid_cells` reports them."""
        volume_correction: float = 0.05
        """Part of the liquid's relative volume error, against the particles' volume, that a step corrects."""
        separation: float = 0.5
        """Part of the overlap of two particles closer than 0.8 times their spacing that a step removes; 0 turns
        the separation off."""
        extrapolation_layers: int = 4
        """Layers of faces into the air that take the water's velocity for the transfer back."""

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register the solver's per-particle state ``apic_water:particle_qd_grad`` on ``builder``: the APIC matrix
        [1/s] whose row ``a`` is the affine vector of velocity component ``a``."""
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="particle_qd_grad",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.mat33,
                default=wp.mat33(0.0),
                namespace="apic_water",
            )
        )

    def __init__(self, model: Model, config: SolverAPICWater.Config | None = None):
        super().__init__(model)
        config = config or SolverAPICWater.Config()
        self.config = config
        device = model.device
        if not wp.get_device(device).is_cuda:
            raise ValueError("SolverAPICWater needs a CUDA device")
        if not hasattr(model, "apic_water"):
            raise ValueError(
                "the model lacks the 'apic_water' custom attributes: call SolverAPICWater.register_custom_attributes "
                "on its ModelBuilder before finalize()"
            )
        dx = float(config.voxel_size)
        lo = np.asarray(config.domain_lo, dtype=np.float64)
        n = np.ceil((np.asarray(config.domain_hi, dtype=np.float64) - lo) / dx - 1.0e-9).astype(int)
        if dx <= 0.0 or np.any(n < 1):
            raise ValueError(f"empty grid box {config.domain_lo} .. {config.domain_hi} at voxel_size {dx}")
        self.dx = dx
        self.lo = lo
        self.n = n
        self.hi = lo + n * dx
        nx, ny, nz = (int(a) for a in n)
        self._far = K.FAR_CELLS * dx

        def floats(shape):
            return wp.zeros(shape, dtype=float, device=device)

        def ints(shape):
            return wp.zeros(shape, dtype=wp.int32, device=device)

        # Faces: per axis the transferred weight and momentum, the velocity and its extrapolation's other buffer, the
        # validity flags, the open and the bodies' fractions, and the closing solids' normal velocity.
        faces = [(nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1)]
        self._faces = faces
        self._m = [floats(s) for s in faces]
        self._mv = [floats(s) for s in faces]
        self._vel = [floats(s) for s in faces]
        self._vel_tmp = [floats(s) for s in faces]
        self._valid = [ints(s) for s in faces]
        self._valid_tmp = [ints(s) for s in faces]
        self._open = [floats(s) for s in faces]
        self._body_frac = [floats(s) for s in faces]
        self._solid_vel = [floats(s) for s in faces]
        # Nodes: the solids' signed distances, and the body nearest to each node.
        nodes = (nx + 1, ny + 1, nz + 1)
        self._nodes = nodes
        self._phi_static = wp.full(nodes, self._far, dtype=float, device=device)
        self._phi_body = wp.full(nodes, self._far, dtype=float, device=device)
        self._owner = wp.full(nodes, -1, dtype=wp.int32, device=device)
        # The static solids' distances at each cell's eight sub-cell centers, for the cells' parts outside them.
        self._static_sub = wp.full((nx, ny, nz, 8), self._far, dtype=float, device=device)
        # Cells.
        cells = nx * ny * nz
        self._cells_shape = (nx, ny, nz)
        self._cell_count = ints((nx, ny, nz))
        self._cell_type = ints((nx, ny, nz))
        self._liquid_weight = floats((nx, ny, nz))
        self._liquid_center = wp.zeros((nx, ny, nz), dtype=wp.vec3, device=device)
        self._liquid_phi = floats((nx, ny, nz))
        self._fluid_flags = ints(cells)
        self._fluid_position = ints(cells)
        self._fluid_index = ints(cells)
        self._fluid_count = ints(1)
        self._dropped_cells = ints(1)
        # The fluid cells of the pressure solve, in blocks of K.BLOCK rows.
        particles = max(int(model.particle_count), 1)
        capacity = config.max_fluid_cells if config.max_fluid_cells is not None else particles
        capacity = max(min(int(capacity), cells), 1)
        blocks = (capacity + int(K.BLOCK) - 1) // int(K.BLOCK)
        self._rows = blocks * int(K.BLOCK)
        self._row_blocks = blocks
        self._fluid_cells = ints(self._rows)
        self._cg = {name: floats(self._rows) for name in ("x", "r", "u", "w", "p", "s", "rhs", "diag", "liquid")}
        # Per-block partial sums and their fixed-size totals.
        particle_blocks = (particles + int(K.BLOCK) - 1) // int(K.BLOCK)
        self._particle_blocks = particle_blocks
        self._total_size = max(256, 1 << max(blocks - 1, particle_blocks - 1, 1).bit_length())
        self._cg_partials = wp.zeros(self._total_size, dtype=wp.vec2, device=device)
        self._cg_scalars = wp.zeros(2, dtype=wp.vec3, device=device)
        self._liquid_partials = floats(self._total_size)
        self._particle_partials = floats(self._total_size)
        self._liquid_volume = floats(1)
        self._particle_volume = floats(1)
        # Particles: the spacing from their mean volume, the separation's hash grid and the streams' free list.
        mass = model.particle_mass.numpy() if model.particle_count else np.ones(1)
        self._spacing = float(np.cbrt(max(float(np.mean(mass)), 1.0e-30) / config.density))
        self._grid_points = wp.zeros(particles, dtype=wp.vec3, device=device)
        self._shift = wp.zeros(particles, dtype=wp.vec3, device=device)
        self._hash_grid = wp.HashGrid(128, 128, 128, device=device)
        self._free = ints(particles)
        self._free_position = ints(particles)
        self._free_list = ints(particles)
        self._free_count = ints(2)
        self._clock = wp.zeros(1, dtype=wp.float64, device=device)
        self._streams = []
        # Colliders: none until setup_colliders.
        self._rigid_model = None
        self._static_shapes = None
        self._body_shapes = None
        self._shape_sign = None
        self._body_com = wp.zeros(1, dtype=wp.vec3, device=device)
        self._no_body_q = wp.zeros(1, dtype=wp.transform, device=device)
        self._no_body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self._impulse = wp.zeros(1, dtype=wp.spatial_vector, device=device)

    @property
    def body_impulse(self) -> wp.array:
        """The water's pressure impulse on each body of the colliders' rigid model over the last step, shape
        ``(body_count,)``, dtype :class:`spatial_vector`: the linear impulse [N s] and the angular impulse about the
        body's center of mass [N m s], in the world frame."""
        return self._impulse

    def setup_colliders(self, model: Model, shapes: list[int] | np.ndarray | None = None) -> None:
        """Take the shapes ``shapes`` of the rigid model ``model``, by default those that collide with particles
        (:attr:`ShapeFlags.COLLIDE_PARTICLES <newton.ShapeFlags.COLLIDE_PARTICLES>`), as the water's solids.

        The static shapes' signed distances at the grid's nodes are found now. Each body's shapes give a distance
        volume in the body's frame, found now on a grid of half a cell, from which each step reads the nodes' distances
        at the body's pose in ``state_in.body_q``. A later call replaces the colliders.

        Args:
            model: The rigid model whose shapes the water meets, on the water's device.
            shapes: The indices of its shapes to take, or ``None`` for those that collide with particles.

        Raises:
            ValueError: For a shape type without a signed distance here (heightfields, Gaussians, planes on bodies).
        """
        device = self.model.device
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        if shapes is None:
            flags = model.shape_flags.numpy()
            shapes = np.flatnonzero(flags & int(ShapeFlags.COLLIDE_PARTICLES))
        shapes = np.asarray(shapes, dtype=np.int32)
        for s in shapes:
            if int(shape_type[s]) not in _COLLIDER_TYPES:
                raise ValueError(f"shape {s} of type {GeoType(int(shape_type[s])).name} cannot be a water collider")
        properties = getattr(model, "_shape_mesh_properties", None)
        properties = properties.numpy() if properties is not None else np.zeros(model.shape_count, dtype=np.int32)
        sign = np.where(
            properties & int(MeshProperties.WATERTIGHT), int(MeshSignMethod.PARITY), int(MeshSignMethod.NORMAL)
        ).astype(np.int32)
        self._rigid_model = model
        self._shape_sign = wp.array(sign, dtype=wp.int32, device=device)
        static = shapes[shape_body[shapes] < 0]
        on_bodies = shapes[shape_body[shapes] >= 0]
        self._static_shapes = wp.array(static, dtype=wp.int32, device=device)
        self._body_shapes = None
        if len(on_bodies):
            self._body_shapes = wp.array(on_bodies, dtype=wp.int32, device=device)
            self._build_body_volumes(model, on_bodies, shape_type, shape_body)
        self._body_com = model.body_com if model.body_count else wp.zeros(1, dtype=wp.vec3, device=device)
        self._impulse = wp.zeros(max(model.body_count, 1), dtype=wp.spatial_vector, device=device)
        self._phi_static.fill_(self._far)
        self._static_sub.fill_(self._far)
        if len(static):
            inputs = [self._static_shapes, *self._shape_arrays(model), wp.vec3(*self.lo), self.dx, self._far]
            wp.launch(K.static_distance_kernel, self._nodes, inputs, [self._phi_static], device=device)
            wp.launch(K.static_sub_distance_kernel, (*self._cells_shape, 8), inputs, [self._static_sub], device=device)
        self._phi_body.fill_(self._far)
        self._owner.fill_(-1)

    def _build_body_volumes(
        self, model: Model, shapes: np.ndarray, shape_type: np.ndarray, shape_body: np.ndarray
    ) -> None:
        """Each body's signed distance volume in its frame: the distance to the union of its shapes ``shapes`` at the
        nodes of a grid of half a cell over the box that bounds them, widened by the solids' band, found once."""
        device = self.model.device
        spacing = _VOLUME_SPACING * self.dx
        xforms = model.shape_transform.numpy()
        scales = model.shape_scale.numpy()
        bodies = sorted({int(b) for b in shape_body[shapes]})
        layout, total = [], 0
        for b in bodies:
            own = shapes[shape_body[shapes] == b]
            corners = []
            for s in own:
                lo, hi = _shape_half_extents(model, int(s), GeoType(int(shape_type[s])), scales[s].astype(np.float64))
                box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                corners.append(_rotate(xforms[s][3:].astype(np.float64), box) + xforms[s][:3])
            corners = np.concatenate(corners)
            origin = corners.min(axis=0) - self._far
            dims = np.ceil((corners.max(axis=0) + self._far - origin) / spacing).astype(int) + 1
            layout.append((b, own, origin, dims, total))
            total += int(np.prod(dims))
        self._volume = wp.full(total, self._far, dtype=float, device=device)
        for _body, own, origin, dims, offset in layout:
            inputs = [wp.array(own, dtype=wp.int32, device=device), *self._shape_arrays(model), wp.vec3(*origin)]
            inputs += [spacing, wp.vec3i(*(int(d) for d in dims)), self._far, offset]
            wp.launch(K.body_volume_kernel, tuple(int(d) for d in dims), inputs, [self._volume], device=device)
        self._volume_spacing = spacing
        self._volume_body = wp.array([v[0] for v in layout], dtype=wp.int32, device=device)
        self._volume_offset = wp.array([v[4] for v in layout], dtype=wp.int32, device=device)
        self._volume_dims = wp.array([wp.vec3i(*(int(d) for d in v[3])) for v in layout], dtype=wp.vec3i, device=device)
        self._volume_origin = wp.array([wp.vec3(*v[2]) for v in layout], dtype=wp.vec3, device=device)

    def _shape_arrays(self, model: Model) -> list:
        return [
            model.shape_transform,
            model.shape_type,
            model.shape_scale,
            model.shape_source_ptr,
            self._shape_sign,
            model.shape_collision_radius,
        ]

    def add_stream(
        self,
        origin,
        direction,
        flow: float,
        side_count: int = 3,
        start: float = 0.0,
        stop: float | None = None,
    ) -> None:
        """Add a stream of water, as from a spout: from ``start`` until ``stop`` [s] of the solver's time, which its
        steps advance, layers of ``side_count`` x ``side_count`` particles one spacing apart leave ``origin`` along
        ``direction`` at the speed that carries ``flow`` [m^3/s], one layer per spacing of travel. A step's layers
        take inactive particles in index order and join at the step's end, where their flight from ``origin`` puts
        them; the stream waits while too few particles are inactive.

        Args:
            origin: The stream's origin [m].
            direction: The stream's direction; its length does not matter.
            flow: The stream's flow [m^3/s].
            side_count: Particles across each side of the stream's square section.
            start: When the stream starts [s].
            stop: When the stream stops [s], or ``None`` for never.
        """
        along = np.asarray(direction, dtype=np.float64)
        along = along / np.linalg.norm(along)
        side = np.cross(along, (0.0, 0.0, 1.0))
        if np.linalg.norm(side) < 1.0e-6:
            side = np.cross(along, (1.0, 0.0, 0.0))
        side /= np.linalg.norm(side)
        normal = np.cross(along, side)
        spacing = self._spacing
        speed = flow / (side_count * side_count * spacing**2)
        offsets = (np.arange(side_count) - 0.5 * (side_count - 1)) * spacing
        layer = np.array([a * side + b * normal for a in offsets for b in offsets], dtype=np.float32)
        device = self.model.device
        self._streams.append(
            {
                "layer": wp.array(layer, dtype=wp.vec3, device=device),
                "origin": wp.vec3(*np.asarray(origin, dtype=np.float64)),
                "velocity": wp.vec3(*(speed * along)),
                "interval": spacing / speed,
                "start": float(start),
                "stop": math.inf if stop is None else float(stop),
                "next_layer": wp.zeros(1, dtype=wp.int64, device=device),
                "plan": wp.zeros(3, dtype=wp.int64, device=device),
            }
        )

    def add_body_forces(self, body_f: wp.array, dt: float) -> None:
        """Add the water's last impulse on each body, :attr:`body_impulse`, to ``body_f`` as a constant wrench over a
        step of ``dt`` [s]: before the rigid solver's next step, for the explicit coupling."""
        wp.launch(
            K.add_body_forces_kernel,
            min(body_f.shape[0], self._impulse.shape[0]),
            inputs=[self._impulse, 1.0 / dt],
            outputs=[body_f],
            device=self.model.device,
        )

    def dropped_fluid_cells(self) -> int:
        """The most fluid cells that a step since the last call left out of the pressure solve, past
        :attr:`Config.max_fluid_cells`; it reads the device."""
        dropped = int(self._dropped_cells.numpy()[0])
        self._dropped_cells.zero_()
        return max(dropped, 0)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """The particles' flags and masses are read at each step; the colliders change through
        :meth:`setup_colliders`."""

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
        """Advance the water by ``dt`` [s] from ``state_in`` to ``state_out``, which may be the same state.

        Args:
            state_in: The particles at the step's start, with ``body_q`` and ``body_qd`` of the colliders' bodies
                when shapes on bodies collide.
            state_out: The particles at the step's end.
            control: Unused.
            contacts: Unused.
            dt: The time step [s].
            observables: Unused; the solver reports no observables.
        """
        del control, contacts, observables
        device = self.model.device
        model = self.model
        lo, hi, dx = wp.vec3(*self.lo), wp.vec3(*self.hi), self.dx
        count = model.particle_count
        gravity = model.gravity
        grad_in = state_in.apic_water.particle_qd_grad
        grad_out = state_out.apic_water.particle_qd_grad
        body_q, body_qd = self._no_body_q, self._no_body_qd
        if self._body_shapes is not None:
            body_q, body_qd = state_in.body_q, state_in.body_qd
            if body_q is None or body_qd is None:
                raise ValueError("colliders on bodies need state_in.body_q and state_in.body_qd of the rigid model")
            inputs = [self._volume_body, self._volume_offset, self._volume_dims, self._volume_origin]
            inputs += [self._volume_spacing, self._volume, body_q, lo, dx, self._far]
            wp.launch(K.body_distance_kernel, self._nodes, inputs, [self._phi_body, self._owner], device=device)

        # Particles to the faces, with gravity, and the faces' solid parts.
        for a in range(3):
            self._m[a].zero_()
            self._mv[a].zero_()
        self._cell_count.zero_()
        if count:
            inputs = [state_in.particle_q, state_in.particle_qd, grad_in, model.particle_flags, lo, dx]
            outputs = [self._m[0], self._mv[0], self._m[1], self._mv[1], self._m[2], self._mv[2], self._cell_count]
            wp.launch(K.p2g_kernel, count, inputs, outputs, device=device)
        for a in range(3):
            wp.launch(
                K.normalize_kernel, self._faces[a], [self._m[a], self._mv[a], gravity, a, dt], [self._vel[a]],
                device=device,
            )  # fmt: skip
            inputs = [self._phi_static, self._phi_body, self._owner, a, lo, dx, body_q, body_qd, self._body_com]
            outputs = [self._open[a], self._body_frac[a], self._solid_vel[a]]
            wp.launch(K.face_fractions_kernel, self._faces[a], inputs, outputs, device=device)

        # The liquid's level set and the fluid cells.
        self._liquid_weight.zero_()
        self._liquid_center.zero_()
        radius = _SURFACE_RADIUS * self._spacing
        if count:
            inputs = [state_in.particle_q, model.particle_flags, lo, dx, 1.5 * dx]
            wp.launch(K.liquid_splat_kernel, count, inputs, [self._liquid_weight, self._liquid_center], device=device)
        inputs = [self._liquid_weight, self._liquid_center, self._phi_static, self._phi_body, lo, dx, radius]
        inputs += [self._far]
        wp.launch(K.liquid_phi_kernel, self._cells_shape, inputs, [self._liquid_phi], device=device)
        inputs = [self._liquid_phi, self._cell_count, self._phi_static, self._phi_body, dx]
        wp.launch(K.classify_kernel, self._cells_shape, inputs, [self._cell_type, self._fluid_flags], device=device)
        wp.utils.array_scan(self._fluid_flags, self._fluid_position, inclusive=False)
        outputs = [self._fluid_index, self._fluid_cells, self._fluid_count, self._dropped_cells]
        wp.launch(
            K.compact_cells_kernel, self._fluid_flags.shape[0], [self._fluid_flags, self._fluid_position], outputs,
            device=device,
        )  # fmt: skip

        self._solve_pressure(dt)

        # The pressure's impulse on the bodies.
        cg = self._cg
        self._impulse.zero_()
        if self._body_shapes is not None:
            inputs = [self._fluid_cells, self._fluid_count, cg["x"], *self._body_frac, self._phi_body, self._owner]
            inputs += [lo, dx, self.config.density * dx**3, gravity, dt / dx, body_q, self._body_com]
            wp.launch(K.body_impulse_kernel, self._rows, inputs, [self._impulse], device=device)

        # The pressure gradient, and the velocity into the air.
        n_layers = int(self.config.extrapolation_layers)
        for a in range(3):
            inputs = [self._cell_type, self._fluid_index, cg["x"], a, self._open[a], self._solid_vel[a]]
            inputs += [self._liquid_phi, self._far]
            wp.launch(K.project_kernel, self._faces[a], inputs, [self._vel[a], self._valid[a]], device=device)
            src, dst = (self._vel[a], self._valid[a]), (self._vel_tmp[a], self._valid_tmp[a])
            for _ in range(n_layers):
                wp.launch(K.extrapolate_kernel, self._faces[a], [*src], [*dst], device=device)
                src, dst = dst, src
            if src[0] is not self._vel[a]:
                wp.copy(self._vel[a], src[0])

        # Back to the particles, which move, leave the solids, and keep apart.
        margin = _SOLID_MARGIN * dx
        if count:
            inputs = [*self._vel, self._phi_static, self._phi_body, self._owner, state_in.particle_q]
            inputs += [state_in.particle_qd, grad_in, lo, hi, dx, dt, margin, body_q, body_qd, self._body_com]
            outputs = [model.particle_flags, state_out.particle_q, state_out.particle_qd, grad_out]
            wp.launch(K.g2p_kernel, count, inputs, outputs, device=device)
            if self.config.separation > 0.0:
                spacing = _SEPARATION_SPACING * self._spacing
                wp.launch(
                    K.grid_points_kernel, count, [state_out.particle_q, model.particle_flags, lo, dx],
                    [self._grid_points], device=device,
                )  # fmt: skip
                self._hash_grid.build(self._grid_points, spacing)
                inputs = [self._hash_grid.id, state_out.particle_q, model.particle_flags, spacing]
                inputs += [self.config.separation]
                wp.launch(K.separate_kernel, count, inputs, [self._shift], device=device)
                inputs = [self._shift, model.particle_flags, self._phi_static, self._phi_body, self._owner, lo, dx]
                inputs += [margin]
                wp.launch(K.apply_separation_kernel, count, inputs, [state_out.particle_q], device=device)
            self._emit(state_out, dt)
        wp.launch(K.advance_clock_kernel, 1, [self._clock, wp.float64(dt)], device=device)

    def _solve_pressure(self, dt: float) -> None:
        """The pressure unknowns ``x = p dt / (density dx)`` of the fluid cells, by the single-reduction conjugate
        gradient, with the global volume correction."""
        device, model, cg = self.model.device, self.model, self._cg
        dx = self.dx
        blocks = self._row_blocks
        total = K.total_kernel(self._total_size)
        # The liquid's and the particles' volumes, summed in a fixed order.
        self._liquid_partials.zero_()
        inputs = [self._fluid_cells, self._fluid_count, self._cell_count, dx**3 / self._spacing**3, self._static_sub]
        inputs += [self._phi_body, dx, self._cell_type]
        wp.launch_tiled(
            K.liquid_volume_kernel, dim=[blocks], inputs=inputs, outputs=[cg["liquid"], self._liquid_partials],
            block_dim=int(K.BLOCK), device=device,
        )  # fmt: skip
        wp.launch_tiled(
            total, dim=[1], inputs=[self._liquid_partials, self._liquid_volume], block_dim=int(K.BLOCK), device=device
        )
        self._particle_partials.zero_()
        if model.particle_count:
            wp.launch_tiled(
                K.particle_volume_kernel, dim=[self._particle_blocks],
                inputs=[model.particle_mass, model.particle_flags, 1.0 / self.config.density],
                outputs=[self._particle_partials], block_dim=int(K.BLOCK), device=device,
            )  # fmt: skip
        wp.launch_tiled(
            total, dim=[1], inputs=[self._particle_partials, self._particle_volume], block_dim=int(K.BLOCK),
            device=device,
        )  # fmt: skip
        inputs = [self._fluid_cells, self._fluid_count, *self._vel, *self._open, *self._solid_vel, self._cell_type]
        inputs += [self._liquid_phi, self._far, self._liquid_volume, self._particle_volume, dx**3, cg["liquid"]]
        inputs += [self.config.volume_correction * dx / dt]
        wp.launch(K.divergence_kernel, self._rows, inputs, [cg["rhs"], cg["diag"]], device=device)
        cg["x"].zero_()
        wp.copy(cg["r"], cg["rhs"])
        wp.launch(K.precondition_kernel, self._rows, [cg["r"], cg["diag"]], [cg["u"]], device=device)
        step = K.cg_step_kernel(self._total_size)
        dims = wp.vec3i(*(int(a) for a in self.n))
        self._cg_partials.zero_()
        for it in range(int(self.config.pressure_iterations)):
            inputs = [self._fluid_cells, self._fluid_count, self._fluid_index, cg["diag"], *self._open, cg["u"]]
            inputs += [cg["r"], dims]
            wp.launch_tiled(
                K.cg_stencil_kernel, dim=[blocks], inputs=inputs, outputs=[cg["w"], self._cg_partials],
                block_dim=int(K.BLOCK), device=device,
            )  # fmt: skip
            inputs = [self._cg_partials, self._cg_scalars, it % 2, int(it == 0), cg["diag"], cg["w"], cg["u"]]
            inputs += [cg["p"], cg["s"], cg["x"], cg["r"]]
            wp.launch_tiled(step, dim=[blocks], inputs=inputs, block_dim=int(K.BLOCK), device=device)

    def _emit(self, state: State, dt: float) -> None:
        """The streams' layers over the step join ``state`` at the step's end."""
        if not self._streams:
            return
        device, model = self.model.device, self.model
        count = model.particle_count
        wp.launch(K.free_flags_kernel, count, [model.particle_flags], [self._free], device=device)
        wp.utils.array_scan(self._free, self._free_position, inclusive=False)
        wp.launch(
            K.free_list_kernel, count, [self._free, self._free_position], [self._free_list, self._free_count],
            device=device,
        )  # fmt: skip
        for stream in self._streams:
            max_layers = int(math.ceil(dt / stream["interval"])) + 1
            inputs = [self._clock, wp.float64(dt), wp.float64(stream["start"]), wp.float64(stream["stop"])]
            inputs += [wp.float64(stream["interval"]), int(stream["layer"].shape[0]), max_layers]
            inputs += [stream["next_layer"], self._free_count]
            wp.launch(K.stream_schedule_kernel, 1, inputs, [stream["plan"]], device=device)
            inputs = [stream["plan"], self._clock, wp.float64(dt), wp.float64(stream["start"])]
            inputs += [wp.float64(stream["interval"]), stream["layer"], stream["origin"], stream["velocity"]]
            inputs += [model.gravity, self._free_list]
            outputs = [model.particle_flags, state.particle_q, state.particle_qd, state.apic_water.particle_qd_grad]
            wp.launch(K.stream_emit_kernel, max_layers * int(stream["layer"].shape[0]), inputs, outputs, device=device)
