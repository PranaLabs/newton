.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

APIC Water
==========

:class:`~newton.solvers.SolverAPICWater` simulates incompressible water with a
free surface. The water is a set of particles; each step transfers their
velocities to a fixed MAC grid by the affine particle-in-cell method (APIC),
adds gravity, solves the pressure that makes the velocity divergence-free on
the parts of the grid's faces outside the solids, and transfers the velocity
back to the particles, which then move. The solids are shapes of a rigid model,
static or on bodies; the pressure's impulse on the bodies drives a rigid
solver, such as :class:`~newton.solvers.SolverFBA`.

.. experimental::

   :class:`~newton.solvers.SolverAPICWater` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

When to Choose It
-----------------

The solver suits water that is poured, splashes, fills containers and moves
them by its weight and pressure, at a cost of a few milliseconds per step for
tens of thousands of particles:

- The pressure solve keeps the water incompressible at any time step that
  moves it a cell or two, rather than at the sound-speed limit of a weakly
  compressible method, and a step records into a CUDA graph.
- Solids cut the grid's cells (Batty, Bertails and Bridson's variational
  method), so that thin walls and round vessels hold the water without
  stair steps, and the pressure on a body is that of its true surface.
- A global volume correction keeps the water's volume at its particles'.

For granular, viscous or elasto-plastic materials, such as sand or dough, use
:class:`~newton.solvers.SolverImplicitMPM`.

Setting Up
----------

The water is a model of particles, each carrying ``particle_mass / density``
of volume; two particles per cell edge give the best surface. Register the
solver's per-particle state on the builder before building the model, give the
grid's box, and take the solids from a rigid model:

.. code-block:: python

   water_builder = newton.ModelBuilder()
   newton.solvers.SolverAPICWater.register_custom_attributes(water_builder)
   spacing = 0.0025  # two particles per 5 mm cell edge
   water_builder.add_particles(
       pos=points, vel=[(0.0, 0.0, 0.0)] * len(points), mass=[1000.0 * spacing**3] * len(points),
       radius=[0.5 * spacing] * len(points),
   )
   water_model = water_builder.finalize()

   config = newton.solvers.SolverAPICWater.Config(
       voxel_size=0.005, domain_lo=(-0.6, -0.2, -0.3), domain_hi=(0.1, 0.2, 0.2)
   )
   water = newton.solvers.SolverAPICWater(water_model, config)
   water.setup_colliders(rigid_model)

   water_state = water_model.state()
   water_state.body_q = rigid_state.body_q
   water_state.body_qd = rigid_state.body_qd

:meth:`~newton.solvers.SolverAPICWater.setup_colliders` takes the shapes that
collide with particles (``has_particle_collision``) of the rigid model: planes,
spheres, capsules, cylinders, cones, ellipsoids, boxes, meshes and convex
hulls. Their signed distances at the grid's nodes give each face's open part:
the static shapes' found once, and the bodies' read at each step, at the
bodies' poses in ``state_in.body_q``, from a distance volume of each body that
:meth:`~newton.solvers.SolverAPICWater.setup_colliders` samples on a grid of
half a cell in the body's frame. A mesh needs a closed surface for its inside.

Coupling to Rigid Bodies
------------------------

The coupling is explicit: the water's impulse on each body over a step,
:attr:`~newton.solvers.SolverAPICWater.body_impulse`, acts on the body as a
constant wrench over the rigid solver's next step, and the water's next step
sees the bodies where that step left them:

.. code-block:: python

   for _ in range(substeps):
       rigid_state.clear_forces()
       water.add_body_forces(rigid_state.body_f, dt)
       pipeline.collide(rigid_state, contacts)
       rigid_solver.step(rigid_state, next_rigid_state, control, contacts, dt)
       wp.copy(rigid_state.body_q, next_rigid_state.body_q)
       wp.copy(rigid_state.body_qd, next_rigid_state.body_qd)
       water.step(water_state, water_state, None, None, dt)

The explicit coupling is stable when the water a body carries along is lighter
than the body, for a hinged body by their moments of inertia about the hinge:
a bamboo tube on a pivot, a water wheel on its axle, or a fixed vessel. A light
floating object can oscillate unstably.

Particles, Streams and the Grid's Box
-------------------------------------

The water is the active particles of the model
(:attr:`ParticleFlags.ACTIVE <newton.ParticleFlags.ACTIVE>`). A particle that
leaves the grid's box, less a stencil's reach, or turns non-finite becomes
inactive and waits far outside the box at rest, so that water can flow out of
the scene. :meth:`~newton.solvers.SolverAPICWater.add_stream` brings inactive
particles back in, as a spout does: layers of particles, a square of
``side_count`` per side one spacing apart, leave an origin at the speed that
carries the stream's flow, each joining the water at the end of the step in
which it left, where its flight from the origin puts it. A stream waits while
fewer particles than a layer are inactive; give the model a pool of inactive
particles for the water a stream brings in at once.

The grid's box is fixed: its cells cost memory, and the pressure solve holds
at most ``Config.max_fluid_cells`` fluid cells, by default the particle count.
Fluid cells past it stay air, and
:meth:`~newton.solvers.SolverAPICWater.dropped_fluid_cells` reports the most a
step left out.

Parameters
----------

- ``voxel_size`` sets the grid's cells. The time step should move the water
  a cell or two at most.
- ``pressure_iterations`` (default 60) are the conjugate gradient iterations
  of each step's pressure solve, a fixed count, so that a step reads nothing
  back. Fewer iterations leave the pressure softer: a container fills and
  empties more slowly.
- ``volume_correction`` (default 0.05) is the part of the liquid's relative
  volume error that a step corrects. The liquid's volume counts each fluid
  cell's part outside the solids, and a cell at the surface by the particles
  it holds; the particles' volume sums ``particle_mass / density``.
- ``separation`` (default 0.5) pushes particles closer than 0.8 times their
  spacing apart, so that the water cannot pack against walls.
- ``extrapolation_layers`` (default 4) carries the water's velocity into the
  air for the particles at the surface.

CUDA Graphs
-----------

A step allocates nothing and reads nothing back to the host, so that a frame
of coupled steps records into one CUDA graph (:class:`warp.ScopedCapture`) and
replays. A graph replays the arrays it recorded: keep the water's state and
the rigid state's ``body_q`` and ``body_qd`` the same arrays across replays,
copying a rigid solver's output state into its input as above. The sums of
particle quantities onto the grid use atomic additions, so that runs differ by
rounding.

Limitations
-----------

- CUDA devices only, one world.
- Inviscid water without surface tension.
- Explicit coupling to rigid bodies; the water does not push particles of
  other solvers.
- Heightfields and Gaussian shapes are not colliders.

References
----------

- C. Jiang, C. Schroeder, A. Selle, J. Teran and A. Stomakhin, *The affine
  particle-in-cell method*, ACM Transactions on Graphics 34 (4), 2015.
- C. Batty, F. Bertails and R. Bridson, *A fast variational framework for
  accurate solid-fluid coupling*, ACM Transactions on Graphics 26 (3), 2007.
- Y. Zhu and R. Bridson, *Animating sand as a fluid*, ACM Transactions on
  Graphics 24 (3), 2005.
- A. T. Chronopoulos and C. W. Gear, *s-step iterative methods for symmetric
  linear systems*, Journal of Computational and Applied Mathematics 25 (2),
  1989.
- J. C. Martin and W. J. Moyce, *An experimental study of the collapse of
  liquid columns on a rigid horizontal plane*, Philosophical Transactions of
  the Royal Society A 244 (882), 1952.
