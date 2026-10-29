.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

FBA
===

:class:`~newton.solvers.SolverFBA` simulates cloth and volumetric soft bodies
with projective dynamics, together with rigid bodies, and resolves the
frictional contact among them as complementarity conditions. Each step runs a
fixed number of local-global iterations: the local phase projects every
element onto its constraint set, the global phase solves a linear system whose
matrix is factored once, and each iteration takes one semismooth Newton step
of the contact and constraint multipliers.

.. experimental::

   :class:`~newton.solvers.SolverFBA` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

When to Choose FBA
------------------

FBA suits scenes in which cloth or soft bodies rest on, wrap around, or are
held by other objects:

- With the default contact operator, contacts do not penetrate and need no
  contact stiffness: each contact supplies the force that keeps its gap
  closed, within its friction cone, so cloth rests on stacked boxes and on
  notched blades, and a soft ball hangs between two fingertip pads by
  friction alone.
- Stiff materials step stably at large time steps. The examples step at
  ``dt = 0.01`` s, including a Neo-Hookean bar with a Young's modulus of
  ``1e9`` Pa.
- Cloth, tetrahedral soft bodies, and rigid bodies with joints share one
  solve, or rigid bodies come from another solver, such as an arm that
  :class:`~newton.solvers.SolverFeatherstone` integrates.

Choose another solver for differentiable simulation and for articulations that need joint limits, drives, or
actuators: FBA treats joints as bilateral constraints only, so integrate such
an articulation with another solver and pass
``integrate_with_external_rigid_solver=True``. See the :ref:`Supported
Features` matrix and the :ref:`Joint feature support` tables for a comparison
with the other solvers. The solver builds on projective dynamics and a
quadratic bending energy:

- Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark
  Pauly. 2014. Projective Dynamics: Fusing Constraint Projections for Fast
  Simulation. *ACM Trans. Graph.* 33, 4.
  https://doi.org/10.1145/2601097.2601116
- Miklós Bergou, Max Wardetzky, David Harmon, Denis Zorin, and Eitan
  Grinspun. 2006. A Quadratic Bending Model for Inextensible Surfaces.
  *Symposium on Geometry Processing*.

Construction and Stepping
-------------------------

Parallel worlds
^^^^^^^^^^^^^^^

Build independent environments with :meth:`ModelBuilder.add_world` or
:meth:`ModelBuilder.replicate`. Worlds may overlap spatially and have different
geometry and contact counts. FBA filters self contacts by world and runs the
contact linear solver with independent reductions, iteration budgets and
stopping decisions for each world. Gravity uses the model's per-world values.
For large batches, use ``nsn_sparse_storage=True`` with the full frictional
operator to avoid storing a dense matrix across all environments. This does
not share the prefactorization of repeated meshes; each replica retains its
own system data.
With several worlds, vertex-face and edge-edge self-contact queries search
only their own world's triangles and edges, and vertex-vertex queries share one
spatial hash across worlds. If particles change worlds, the queries search all
worlds' geometry and filter by world. Contacts between distinct local worlds
are always excluded.
With several worlds, the particle system and the system of the integrated
bodies factor one world at a time, each world's matrix whole with all of its
components; a model of one world factors each system whole. A world whose
particles touch an integrated body solves all of its particle contacts and
particle constraint rows together with the body rows, while the other worlds
keep the deformable contact solve that they take alone.
Lite's exact particle constraints take at most 64 rows in each world: the
rows of each world form their own Cholesky block, since the worlds do not
couple. A model of one world takes its global and local particles' rows in
one block, within one limit of 64 rows.

Each world solves the same equations with the same algorithms and iteration
rules as it would alone. Batching can still change floating-point results:
the worlds' systems factor separately, reductions sum in a different order,
and the choice among self-contact triangles at equal distances follows the
query order. Batched runs therefore do not promise bitwise equality with
separate runs or with earlier versions, and long nonlinear contact
trajectories can amplify such differences.

Particles and integrated dynamic bodies must belong to local worlds in a
multi-world model. Elements, joints and particle constraint rows must not
connect different worlds. Global static shapes, such as a shared ground
plane, are supported. These are runtime solver rules on Newton's existing
world metadata; no additional USD schema is required.

Call ``solver.reset(state, world_mask=mask)`` to restore selected worlds'
body and particle positions and velocities from the model defaults and clear
their friction anchors and contact warm start. The boolean mask has
``model.world_count + 1`` entries; the final entry selects global entities.
Passing no mask resets all worlds.

Use ``flags`` to select ``StateFlags.BODY_Q``, ``BODY_QD``, ``PARTICLE_Q``
and ``PARTICLE_QD``. ``None`` restores all supported fields. After writing a
custom initial state, pass ``flags=0`` to clear only solver history and retain
that state. Missing state arrays are skipped. When an external solver
integrates rigid bodies, FBA leaves their state unchanged. Joint-coordinate
flags are ignored because FBA uses body coordinates; use :func:`eval_fk`
after reset to initialize bodies from joint coordinates.

Reset preserves external forces, control inputs, pin targets and particle
constraint targets. Their owner must restore them as needed. Collision
matching and actuator histories also retain their own reset interfaces:

.. code-block:: python

   solver.reset(state, world_mask=mask)
   collision_pipeline.reset_contact_matching(world_mask=mask)
   collision_pipeline.collide(state, contacts)

If actuators are used, reset their :meth:`~newton.actuators.Actuator.State.reset`
history with the corresponding actuator-entry selection. Its mask selects
actuator entries, rather than worlds. Regenerate contacts after restoring or
randomizing state and before the next simulation step. This reset is a
runtime operation and requires no additional USD schema.

Single step
^^^^^^^^^^^

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
particles, the elements and their materials, and the Rayleigh damping
coefficients; the solver factors it again
when the time step changes or after
:meth:`~newton.solvers.SolverFBA.notify_model_changed` with
``SHAPE_PROPERTIES`` or ``BODY_INERTIAL_PROPERTIES``, and other changes to
those arrays take effect only then. A step reads its contacts back to
the host, so it cannot be captured in a CUDA graph; it captures and replays
its own local-global iterations. FBA runs on the CPU and on CUDA devices.

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
- **Activations**: the control's ``tet_activations`` grow each tetrahedron
  to ``1 - a`` times its rest volume, so a negative activation inflates it
  and a positive one contracts it: isotropically, or along a direction per
  tetrahedron from ``tet_activation_directions``, as the chambers of a
  pneumatic bending actuator lengthen along it. The local phase projects the
  deformation gradient with the growth taken out and puts the growth back
  into the projection, so the factored system matrix stays valid and the
  activations may change every step. In ``pneunet_gripper_fba`` they
  lengthen the chambers of a soft gripper along its fingers.
- **Damping**: ``rayleigh_alpha`` [1/s] and ``rayleigh_beta`` [s] apply
  Rayleigh damping, the forces ``-(alpha M + beta K) v`` with ``M`` the
  particle masses and ``K`` the stretching and bending part of the constant
  system matrix, implicitly: the damping matrix joins the prefactored system
  matrix. Each coefficient is a scalar or one value per particle. ``K``
  measures every change of an element's deformation gradient, rotations
  included, so ``beta`` damps the bending of a slender part far more than its
  bending stiffness alone suggests: in ``softbody_cantilever_fba``,
  ``rayleigh_beta=2e-5`` brings a 2.4 m beam to rest within seconds. With
  ``rayleigh_corotated=True``, ``beta`` damps the stretch rate of ARAP
  triangles instead and spares their rotation, so a released cloth flap falls
  as fast as without damping.
- **Pins**: particles with zero inverse mass follow their pin targets through
  a stiff spring of ``pin_stiffness`` [N/m]. The targets start at the model
  positions and move with :meth:`~newton.solvers.SolverFBA.set_pin_targets`.
  Particles without the ``ACTIVE`` flag that have a mass hold where the input
  state has them.
- **Particle constraints**:
  :meth:`~newton.solvers.SolverFBA.set_particle_constraints` holds lengths or
  linear combinations of particle positions at targets, solved with the
  contacts. With the lite operator, the rows that hold particles at points,
  or all rows with ``exact=True``, take one exact Newton step of their own.
  The ``cable_gripper_fba`` example builds the rows of cables through
  tetrahedral bodies.
- **Rigid bodies**: FBA integrates the dynamic bodies and passes kinematic
  bodies through, or leaves every body to another solver with
  ``integrate_with_external_rigid_solver=True``. PRISMATIC, REVOLUTE, BALL,
  FIXED, and D6 joints are bilateral constraints, at most 64 rows per
  articulation; DISTANCE and ROD joints, and joints that lock more directions
  of a set of connected bodies than the bodies have, as kinematic loops do,
  raise ``NotImplementedError``. A body turns by at most 0.5 rad per step.

FBA does not simulate springs, equality constraints, mimic joints,
particle-body attachments, joint limits, drives, armature, or joint friction, or
the inputs of :class:`~newton.Control`, and it does not read the damping
coefficients ``tri_kd``, ``edge_kd`` and ``k_damp`` of the elements.

Contacts
--------

- ``nsn_schur_mode`` selects the contact operator of the Newton steps.
  ``"full"`` (default) builds it from the factored system matrix. ``"lite"``
  replaces the inverse system matrix by the inverse masses, which decouples
  the contacts of different particles and costs far less, but lets a pressed
  contact settle inside its shape by up to ``dt² F / m`` for a normal force
  ``F`` on a particle of mass ``m``. In steps with self contacts or more than
  four contacts on a particle, ``nsn_lite_accumulate=True`` keeps the contact
  forces of a step's earlier
  iterations and models only their change, so that the iterations approach
  the forces whose exact motion closes the gaps and stacked layers keep their
  margin at longer steps.
- ``face_contacts=True`` adds a contact at the deepest point of a triangle
  inside a shape, so shape edges and corners cannot pass between the
  particles of coarse cloth, and holds particles in concave corners of
  meshes. It needs friction and the full operator.
- ``particle_enable_self_contact=True`` makes the triangles collide with each
  other, through vertex-face or vertex-vertex contacts. Vertex-face contacts
  keep crossing edges apart when the margin reaches half of the longest edge;
  ``particle_self_contact_edge_edge=True`` adds edge-edge contacts at creases
  for thinner margins, ``particle_self_contact_edge_edge_ratio`` sets how near
  their edges must come compared with their particles' nearest triangles, and
  :meth:`~newton.solvers.SolverFBA.set_particle_self_contact` changes the
  margin between steps, so that a dropped cloth can land with a thick margin
  and fold with a thin one. ``particle_self_contact_group_ids`` gives each
  particle a group, such as one per body of a pile: the particles and
  triangles of two groups keep the margin apart, and those of one group the
  group's margin from ``particle_self_contact_group_margins``, or do not
  collide.
- ``friction_anchors=True`` keeps the point a contact presses on across steps,
  on the other layer for a self contact or on a static shape, so its friction
  rows pull back the slip that a step's solve leaves instead of letting it add
  up. With the lite operator, that slip lets stacked cloth layers creep apart
  over seconds. :meth:`~newton.solvers.SolverFBA.set_friction_anchor_exclusion`
  frees chosen particles of their anchors, so that a grasp can gather cloth.
- Rigid contacts between shapes on bodies that FBA integrates need
  ``friction=True`` and the full operator.
- A contact on a body that FBA integrates acts at the contact point of the
  step start, which moves with the body to first order in its rotation over
  the step, so a ball keeps rolling at its speed on flat ground.
- ``contact_warmstart=True`` starts each step's contact multipliers from the
  previous step's on the same contacts and constraints rather than from zero,
  so that a resting or held object starts its step at the forces that carry
  it, which a few iterations per step cannot build up from zero. It needs
  friction and the full operator.
- ``shape_particle_radius_override`` sets the particles' contact thickness
  against chosen shapes, between zero and the smallest particle radius, so
  that a gripper can close on a body whose particle radius holds it off the
  floor.
- ``nsn_sparse_storage=True`` keeps the contact operator and the Newton
  matrix of the full operator's steps in CSR form without their zero entries,
  such as those between separate bodies, which saves most of their memory and
  work for many contacts on many bodies, as in a pile of soft bodies. It
  needs friction and the full operator.
- The friction coefficient of a particle contact is its shape's ``mu``,
  ``model.particle_mu`` for self contacts, and the mean of the two shapes'
  ``mu`` for rigid contacts.
- :attr:`~newton.solvers.SolverBase.ObservableKind.CONTACT_F` reports the
  force of each particle contact on its shape's body; see
  :ref:`solver_observables`.

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
- ``cloth_on_knives_fba``: cloth on notched blades with face contacts and light
  Rayleigh stiffness damping.
- ``cloth_on_boxes_fba``: cloth on rigid bodies with joints.
- ``twisting_bar_fba``, ``squeezing_ball_fba``: a soft bar twisted by its
  pinned ends and a soft ball dragged by spinning cylinders.
- ``softbody_franka_fba``: a Franka arm integrated by
  :class:`~newton.solvers.SolverFeatherstone` picks up a soft ball.
- ``cable_gripper_fba``: soft fingers curled by cables lift a box.
- ``cloth_dual_franka_fba``: two Franka arms integrated by
  :class:`~newton.solvers.SolverFeatherstone` pinch the corners of a square
  towel and fold it twice into a square, with light Rayleigh mass damping.
  With ``--fully-coupled``, an experiment: FBA integrates the arms too, with
  force-limited joint drives built from body wrenches and friction-only
  pinches, so the towel acts back on the arms; the pinches do not hold
  reliably enough to complete the folds.
  ``--world-count N`` folds N towels in parallel worlds, replicated in place
  and spread out only in the viewer by ``--world-spacing``, with one
  Featherstone and one FBA solver over all worlds and each world's own IK
  problem, fold plan and grasps. ``--action-noise A`` [rad] adds to each
  world's arm joint targets a fixed bias, drawn once per world and joint from
  ``U(-A, A)`` with seed 42, clipped to the joint limits; it leaves the
  fingers, the timing, the initial pose and the IK state unchanged.
- ``shirt_pins_fba``, ``bunny_ears_fba``: two handles move a T-shirt by its
  shoulders, with self-contact and light Rayleigh stiffness damping, and a
  soft bunny by its ears.
- ``shirt_fold_fba``: two Franka arms fold a dropped T-shirt, its sides in and
  its hem up, their fingers squeezing the cloth into a bulge that they hold,
  at two steps per frame with the lite operator accumulating its contact
  forces, a self-contact margin that thins after the landing, edge-edge
  contacts, friction anchors and corotated Rayleigh stiffness damping.
  ``--fully-coupled`` integrates the arms in FBA as well, as for
  ``cloth_dual_franka_fba``, with the full contact operator, whose dense
  block over the contacted particles ran out of GPU memory for the default
  6480-particle shirt, with and without ``nsn_sparse_storage``;
  ``--cloth-edge-length 0.0222`` remeshes the shirt coarser in both modes for
  an experimental comparison at matched resolution. ``--world-count``,
  ``--world-spacing`` and ``--action-noise`` run parallel worlds as for
  ``cloth_dual_franka_fba``.
- ``softbody_cantilever_fba``: two soft cantilevers swing under gravity, one
  with Rayleigh damping.
- ``pneunet_gripper_fba``: a three-finger PneuNet gripper whose chambers,
  lengthened along its fingers by tetrahedral activations, close the fingers
  around a ball and lift it.
- ``claw_machine_fba``: an articulated claw driven by force-limited PD forces
  picks a bear out of a pile of 25 soft toys that collide with each other, and
  only the elephants with themselves, with a thinner contact against its
  fingers and the sparse Schur storage.
