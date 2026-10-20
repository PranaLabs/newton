.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

FBA
===

:class:`~newton.solvers.SolverFBA` simulates cloth and volumetric soft bodies
with projective dynamics and resolves their frictional contact as
complementarity conditions. Each step runs a fixed number of local-global
iterations: the local phase projects every element onto its constraint set,
the global phase solves a linear system whose matrix is factored once, and
each iteration takes one semismooth Newton step of the contact multipliers.

.. experimental::

   :class:`~newton.solvers.SolverFBA` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

When to Choose FBA
------------------

FBA suits scenes in which cloth or soft bodies rest on, wrap around, or are
pressed against shapes:

- With the default contact operator, contacts do not penetrate and need no
  contact stiffness: each contact supplies the force that keeps its gap
  closed, within its friction cone, so cloth rests on notched blades, and
  spinning cylinders drag a soft ball between them by friction alone.
- Stiff materials step stably at large time steps. The examples step at
  ``dt = 0.01`` s, including a Neo-Hookean bar with a Young's modulus of
  ``1e9`` Pa.

Choose another solver for models with several worlds or with rigid bodies,
and for differentiable simulation. See the :ref:`Supported Features` matrix
for a comparison with the other solvers. The solver builds on projective
dynamics and a quadratic bending energy:

- Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark
  Pauly. 2014. Projective Dynamics: Fusing Constraint Projections for Fast
  Simulation. *ACM Trans. Graph.* 33, 4.
  https://doi.org/10.1145/2601097.2601116
- Miklós Bergou, Max Wardetzky, David Harmon, Denis Zorin, and Eitan
  Grinspun. 2006. A Quadratic Bending Model for Inextensible Surfaces.
  *Symposium on Geometry Processing*.

Construction and Stepping
-------------------------

Detect contacts with a :class:`~newton.CollisionPipeline` on the input state
of each step and pass them to :meth:`~newton.solvers.SolverFBA.step`:

.. code-block:: python

   import warp as wp

   import newton

   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   builder.add_cloth_grid(
       pos=wp.vec3(-0.5, -0.5, 0.5),
       rot=wp.quat_identity(),
       vel=wp.vec3(0.0),
       dim_x=20,
       dim_y=20,
       cell_x=0.05,
       cell_y=0.05,
       mass=0.005,
       tri_ke=1.0e4,
       edge_ke=1.0e-4,
       particle_radius=0.005,
   )
   model = builder.finalize()

   solver = newton.solvers.SolverFBA(model, iterations=10)
   pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
   contacts = pipeline.contacts()
   state_0, state_1 = model.state(), model.state()

   for _ in range(120):
       pipeline.collide(state_0, contacts)
       solver.step(state_0, state_1, None, contacts, 1.0 / 60.0)
       state_0, state_1 = state_1, state_0

A particle contact keeps the particle its radius plus the shape's margin from
the shape: the cloth above comes to rest 5 mm over the ground. The first step
factors the system matrix on the CPU and takes longer than the others. The
matrix depends on the time step, the particle masses, the pinned and inactive
particles, and the elements and their materials; the solver factors it again
when the time step changes or after
:meth:`~newton.solvers.SolverFBA.notify_model_changed` with
``SHAPE_PROPERTIES``, and other changes to those arrays take effect only then.
A step reads its contact counts back to the host, so it cannot be captured in
a CUDA graph; it captures and replays its own local-global iterations. FBA
runs on the CPU and on CUDA devices.

Model Inputs
------------

- **Cloth**: triangles with a stretching model chosen by ``stretching_model``:
  ``"arap"`` takes ``tri_ke`` as the stiffness, and ``"corotational"`` and
  ``"neohookean"`` take the elements' Lamé parameters; the
  :class:`~newton.solvers.SolverFBA` reference states how projective dynamics
  scales them. Interior edges add the isometric bending energy of Bergou et
  al. with ``edge_ke`` as a bending modulus [N·m] and ``edge_rest_angle`` as
  the rest angle. The same ``edge_ke`` bends an FBA cloth more stiffly than a
  :class:`~newton.solvers.SolverVBD` one, which scales it by the edge length.
- **Soft bodies**: tetrahedra with the same stretching models.
- **Pins**: particles with zero inverse mass follow their pin targets through
  a stiff spring of ``pin_stiffness`` [N/m]. The targets start at the model
  positions and move with :meth:`~newton.solvers.SolverFBA.set_pin_targets`.
  Particles without the ``ACTIVE`` flag that have a mass hold where the input
  state has them.

FBA does not simulate rigid bodies: a model with bodies raises
``NotImplementedError``. It does not simulate springs either, cloth self
contact, or the inputs of :class:`~newton.Control`.

Contacts
--------

- ``nsn_schur_mode`` selects the contact operator of the Newton steps.
  ``"full"`` (default) builds it from the factored system matrix. ``"lite"``
  replaces the inverse system matrix by the inverse masses, which decouples
  the contacts of different particles and costs far less, but lets a pressed
  contact settle inside its shape by up to ``dt² F / m`` for a normal force
  ``F`` on a particle of mass ``m``.
- ``face_contacts=True`` adds a contact at the deepest point of a triangle
  inside a shape, so shape edges and corners cannot pass between the
  particles of coarse cloth, and holds particles in concave corners of
  meshes. It needs the full operator.
- The friction coefficient of a particle contact is its shape's ``mu``.
- :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F` reports the
  force of each particle contact on its shape; see :ref:`solver_observables`.

Accuracy
--------

Positions are float32 and velocities come from position differences, so far
from the origin velocity rounding grows: over a 350 cm free fall a cloth
drifts by 0.08 cm. The bending energy is quadratic in the positions, so a
fold with a large rest angle settles a little away from that angle under
load; in Newton's shared cloth tests the folds of a curved cloth settle
0.11 rad from their rest angles.

Examples
--------

``python -m newton.examples <name>`` runs each of these:

- ``stretching_cloth_fba``: cloth pulled apart by pins on two opposite edges.
- ``cloth_on_knives_fba``: cloth on notched blades with face contacts.
- ``twisting_bar_fba``, ``squeezing_ball_fba``: a soft bar twisted by its
  pinned ends and a soft ball dragged by spinning cylinders.
