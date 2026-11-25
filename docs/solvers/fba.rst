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

Choose another solver for differentiable simulation and for articulations whose drives need velocity limits or
actuators other than PD drives: FBA treats joints as bilateral constraints, with the limits of revolute and
prismatic joints, passive damping, PD drives bounded by effort limits, armature and joint friction, so integrate
such an articulation with another solver and pass
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
Changing one world's particle constraint rows clears only that world's
constraint warm start; unchanged worlds retain theirs even when their row
indices shift.
``nsn_lite_accumulate`` also accepts one boolean per world plus a final global
entry, so cloths that need accumulated contact forces can share a solver with
cloths that use the original lite correction.
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

The ``cloth_dual_franka_fba`` and ``shirt_fold_fba`` examples compose these
operations through ``example.reset(world_mask=None)`` between frames. They
restore both state buffers, robot joint state, controls, IK, grasp constraints
and the selected episodes' task phases. Unselected episodes continue from
their current state. Each world has its own ``episode_time``; ``sim_time``
remains the viewer clock. The unreplicated single-world examples use global
entities, so select the final mask entry or call ``example.reset()``.

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
``SHAPE_PROPERTIES`` (Newton has no flag of its own for particles and
elements) when those arrays changed, and other changes to them take effect
only then. Edits of shapes alone, such as of their friction, keep the factor
and the contact warm start, and edits of ``edge_rest_angle`` alone refresh the
bending targets without factoring. A step records into the caller's CUDA
graph (:class:`warp.ScopedCapture`), collision included, when it reads
nothing back to the host and allocates nothing: after a step of the same
``dt`` outside the graph, which sets up the system, a step without contacts,
also with particle constraint rows after a step with the same rows outside
the graph, which sets up their ``W``, and a step with particle-shape contacts
with friction in one world, after a step with the same contacts outside the graph, which reserves the
buffers of their capacity: with the lite operator, or with the full operator
and a dense ``W`` (``nsn_sparse_storage=False``) and a fixed count of at most
25 ``nsn_linear_iterations``, its face contacts, the contacts of its rod segments with shapes and its rod contacts
included. Such a step takes
every slot of the contact buffer as a contact row, those past the stored count
and the dropped contacts of held particles inert, sorted on the device. With the lite operator
it gives every particle the per-particle Newton step; a particle with more than
four contacts keeps its multipliers there. With the full operator the step
finds its face contacts and the other distance minima of its particles beside
meshes on the device too, and the contacts of its rod segments with shapes and its rod contacts, at most four per
segment, in the order of the steps outside a graph, counts
its active contacts on the device and takes, by a conditional node of the graph
(``wp.capture_if``), the smallest of the sizes 64, 96, 128, 192, 256, 384, 512,
768, 1024 and 1536 contacts, and the most its contacts can reach up to 2048,
that holds them: it builds the dense ``W`` of three rows per slot of that size
on the device, the rows of inert slots those of the identity, and takes its NSN
steps on them, so that its cost is that of the size rather than of the buffer.
Steps with deformable contacts on the bodies FBA integrates record so too, in
one world, with hard rigid contacts (``rigid_contact_time=0``) and neither
tendons, mimic joints, particle constraint rows, ``contact_warmstart`` nor
``friction_anchors``: the rigid contacts follow the deformable ones in the same
sizes, and the rows of the joints of the articulations that contacts touch,
selected on the device, follow them. A step drops the active contacts past
2048, the rod contacts past four per segment, and a full contact buffer the
contacts past it, silently, as it reads nothing back: :meth:`~newton.solvers.SolverFBA.dropped_contacts`, called
outside the graph, reports the most contacts a recorded step dropped since its
last call. Rigid contacts record too, after a step with the same contacts
outside the graph, in a model without particles, in one world, with
``nsn_sparse_storage``, a fixed count of at most 25 ``nsn_linear_iterations``,
and neither tendons nor mimic joints: every slot of the rigid contact buffer
becomes a contact's three rows, those past the step's contacts inert, the joint
rows of the articulations that contacts touch are selected on the device, and
the pattern of ``W`` takes a storage of 32 blocks per slot beside those of the
joint rows, so that the recorded steps match those outside a graph bit for bit,
``contact_warmstart`` and ``rigid_contact_bin`` included. Their cost grows with
the buffer rather than with the contacts: with a ``rigid_contact_max`` far
above the contacts that the steps find, recorded steps run slower than steps
outside a graph. A step whose pattern outgrows its storage drops its last
contacts, which ``dropped_contacts`` reports. Steps that are GPU-bound gain
little from a graph, which saves host work, while the inert rows of a size add
work on the device. Self contacts, rod contacts with the lite operator, rod contacts and the contacts of rod segments
with shapes in ``nsn_sparse_storage``, particle constraint rows with contacts, particle-body attachments,
several worlds, model tendons, and with particle contacts
``contact_warmstart``, ``friction_anchors``, ``nsn_lite_accumulate`` and the
full operator with ``nsn_sparse_storage`` keep a step out of graphs: during a
capture it then raises ``RuntimeError``, naming the reason. A graph
replays the launches of the steps it recorded: settings the solver keeps on
the host, such as :meth:`~newton.solvers.SolverFBA.set_pin_targets`,
:meth:`~newton.solvers.SolverFBA.set_particle_constraints` and
:meth:`~newton.solvers.SolverFBA.notify_model_changed`, reach the graphs
recorded after them; called inside the capture with targets in a
:class:`warp.array` on the solver's device, ``set_pin_targets`` and
``set_particle_constraints`` with the rows already set record the copies of
their targets, so that a graph's steps follow targets the caller writes to that
array between its launches. Outside a capture a step captures and replays its own
local-global iterations. A step recorded into a graph, and the solver's own
captures, run with Python's garbage collector paused, so that no collection
frees arrays of unreachable reference cycles inside the capture. The solver
keeps the arrays of a step recorded into a graph while it lives: a later step
outside the graph that replaces its buffers, such as one of another ``dt`` or
with more contacts than the recorded step's capacity, frees none of them, so
the graph still replays safely, but it no longer shares the new buffers nor
the state they carry from step to step; the solver then warns once with
:class:`~newton.exceptions.NewtonWarning` that the graph should be recorded
again. FBA runs
on the CPU and on CUDA devices. A step may write its output into its input
state.

Model Inputs
------------

- **Cloth**: triangles with a stretching model chosen by ``stretching_model``:
  ``"arap"`` takes ``tri_ke`` as the stiffness, and ``"corotational"`` and
  ``"neohookean"`` take the elements' Lamé parameters; the
  :class:`~newton.solvers.SolverFBA` reference states how projective dynamics
  scales them. Interior edges add the isometric bending energy of Bergou et
  al., with ``edge_rest_angle`` as the rest angle, its sign included: a hinge
  bent to the other side of its rest fold, or flat, bends back toward it. On
  equilateral triangles
  it is the energy of an isotropic plate of bending stiffness
  ``3 * edge_ke`` [N·m] and Poisson's ratio ``-1/3``, so a plate of
  stiffness ``D`` that bends mostly about one axis takes ``edge_ke = D / 3``;
  on grids of right triangles, bending about a diagonal axis is stiffer, by
  the factors the :class:`~newton.solvers.SolverFBA` reference gives. The
  same ``edge_ke`` bends an FBA cloth more stiffly than a
  :class:`~newton.solvers.SolverVBD` one, which scales it by the edge length.
- **Soft bodies**: tetrahedra with the same stretching models.
  ``"corotational"`` and ``"neohookean"`` need a stable material, of positive
  bulk modulus (``k_lambda > -k_mu`` for triangles, ``k_lambda > -2 k_mu / 3``
  for tetrahedra); a Poisson's ratio of 0, ``k_lambda = 0``, is fine.
- **Particles** without elements fall, collide and rest on their own.
- **Springs**: each spring of the model is a stretch element of energy
  ``spring_stiffness / 2 (|x_b - x_a| - spring_rest_length)^2``. Its local
  step is an ADMM step: a scaled dual, kept from step to step, corrects the
  prox of the energy, so the iterations converge to the implicit Euler step
  of the springs and not to a softened one. The dual keeps its part along
  the spring, a tension, and the prox acts along the spring at its signed
  length with the dual, so a dual that pushes the particles apart harder
  than the penalty times the spring's length shortens the target instead of
  turning the spring over. The element weighs
  ``60 m / dt^2`` in the system matrix, ``m`` the mass of its lighter
  particle, which sets how fast the iterations converge, not where: springs
  far softer than it, whose oscillation the step resolves, converge slowly.
  Stiffer springs converge their own stretch at that rate, but a long chain
  of them carries stretch waves that span many particles, and a dual that
  lags its chain's swing makes a taut rope shake. After the dual steps, each
  chain of springs, a path through particles of two springs, therefore
  solves its tensions with the tridiagonal response of its stretches through
  the particles' inverse masses, and the compliance that the dual steps
  leave out: a rope of 80 springs between two pins settles at 10 iterations
  as at 100, and one step of a chain of 40 springs a hundred times stiffer
  than their penalty reaches the implicit Euler step in 10 iterations. Where
  a chain meets more springs, the particle's inverse mass counts once for
  each of them, which keeps the solve below the exact one.
  ``spring_damping`` and ``spring_control`` are not read.
- **Particle rods**: :meth:`~newton.solvers.SolverFBA.add_particle_rod`
  lays an elastic rod of circular cross-section along a polyline: a particle
  per vertex, its mass lumped by length and the rod's radius as its contact
  radius, a spring per segment of stiffness ``E pi r^2 / l``, and a bending
  element per interior vertex. A bending element is an entry of the custom
  frequency ``fba:rod_bend`` (see
  :meth:`~newton.solvers.SolverFBA.register_custom_attributes`) of energy
  ``EI / (2 v) |(x_+ - x) / l_+ - (x - x_-) / l_-|^2``, with ``l_-`` and
  ``l_+`` the rest lengths of the vertex's two segments and
  ``v = (l_- + l_+) / 2``: ``EI / 2 kappa^2 v`` for small turns, straight at
  rest. It adds to the constant system matrix alone, and raises the penalty
  of the rod's springs by ``0.1 EI / l^3``. In the matrix it also resists a
  difference of the stretches of its two springs with ``EI / (v l_- l_+)``,
  far more than the penalty of a stiff rod's short segments, so the tension
  solve of the rod's springs counts it too: a pin moved through a stiff rod
  drags the rod along within 10 iterations. A 0.3 m rod of radius 5 mm
  clamped at one end sags under its weight to within 0.2% of the
  small-deflection sag of its discrete bending energy at ``E = 1e8`` Pa.
  Rods carry no twist, and their contacts with shapes act at their
  particles, so segments longer than about twice the radius can pass thin
  shapes. With ``rod_contact=True`` the rods collide with each other and
  with themselves as chains of capsules (see below). These rods are
  particles, unlike the rigid capsule chains of
  :meth:`~newton.ModelBuilder.add_rod`.
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
  ``rayleigh_beta=2e-5`` brings a 2.4 m beam to rest within seconds. A
  spring enters ``K`` with its stiffness along every direction, so ``beta``
  also damps the swinging of a chain of springs. With
  ``rayleigh_corotated=True``, ``beta`` damps the stretch rate of ARAP
  triangles and the length rate of springs instead and spares their rotation,
  so a released cloth flap falls as fast as without damping and a chain of
  springs swings on while its stretch waves die out.
- **Pins**: particles with zero inverse mass follow their pin targets through
  a stiff spring of ``pin_stiffness`` [N/m] and end each step at them, with
  their motion over the step as their velocity. The targets are where the
  input state has the particles, as the other solvers hold the particles they
  do not move, so writing ``state.particle_q`` moves them, until
  :meth:`~newton.solvers.SolverFBA.set_pin_targets` sets them. Particles
  without the ``ACTIVE`` flag that have a mass hold where the input state has
  them. Contacts on held particles alone, such as a pinned particle's contact
  with a static shape, are dropped: they cannot move the particles, and
  pressing on a pin would only move it off its target. Those with the shapes
  of the bodies the solver integrates stay and push the bodies, so a free
  body rests on a pinned cloth.
- **Particle constraints**:
  :meth:`~newton.solvers.SolverFBA.set_particle_constraints` holds lengths or
  linear combinations of particle positions at targets, solved with the
  contacts. With the lite operator, the rows that hold particles at points,
  or all rows with ``exact=True``, take one exact Newton step of their own.
  The ``cable_gripper_fba`` example builds the rows of cables through
  tetrahedral bodies. With ``bodies`` and ``body_points``, a row holds its
  combination of particle positions against a point of a body that FBA
  integrates and is solved with the body's rows, so that the body takes the
  row's force: three rows along independent directions with target zero tie a
  particle to that point, as a rope is tied to a rigid part. Such rows need
  ``directions``, the full operator and one world.
- **Particle-body attachments**: the model's attachments
  (:meth:`~newton.ModelBuilder.add_attachment_particle_body`) join these rows,
  three per attachment along the world axes after the rows of
  ``set_particle_constraints``, as soft rows: each holds the backward Euler
  step of its spring ``stiffness`` and damper ``damping`` on the particle's
  offset from the body point, ``f = -k g - c (g - g0) / dt``, solved with the
  body's contacts and joints, so that the forces are equal and opposite. They
  need dynamic bodies that FBA integrates, ``friction=True``, the full
  operator and one world; as :class:`~newton.solvers.SolverVBD` does, FBA
  rejects them with ``integrate_with_external_rigid_solver=True``. Disabled
  attachments and those of zero stiffness and damping take no rows, and
  ``ModelFlags.CONSTRAINT_PROPERTIES`` takes edits of the three.
- **Rigid bodies**: FBA integrates the dynamic bodies and passes kinematic
  bodies through, or leaves every body to another solver with
  ``integrate_with_external_rigid_solver=True``. PRISMATIC, REVOLUTE, BALL,
  FIXED, and D6 joints are bilateral constraints, at most 64 rows per
  articulation, the axis rows below included; DISTANCE and ROD joints, and
  joints that lock more directions of a set of connected bodies than the
  bodies have, as kinematic loops do, raise ``NotImplementedError``. A D6
  joint with two free angular axes ``a0`` and ``a1`` turns as Newton composes
  it, ``q(a0, q0) q(a1, q1)``: it holds the angle between ``a0`` in the
  parent's joint frame and ``a1`` in the child's, as a universal joint does.
  Joints hold a dynamic body to a kinematic one where the kinematic body's
  prescribed velocity ``body_qd`` takes it by the step's end. A step is
  linearized about a frame turned by at most 0.5 rad, and a body turns by at
  most 3 rad per step. The gyroscopic torque is integrated implicitly, so a
  tumbling body loses a little energy rather than gaining it. FBA does not
  write ``State.joint_q`` and
  ``State.joint_qd``: :func:`~newton.eval_ik` computes them from the bodies.
- **Joint limits, damping and drives**: every free axis of a joint, also of
  BALL and D6 joints, with passive damping, a drive, armature or joint
  friction, and the free axis of a REVOLUTE or PRISMATIC joint with limits,
  takes one row in the articulation's joint system on the axis's motion over
  the step, whose force is the axis's total force. Its passive damping
  :attr:`~newton.Model.joint_damping` (the ``damping`` of
  :class:`~newton.ModelBuilder.JointDofConfig`, MJCF's ``damping``) enters as
  the backward Euler step of the force ``-c q_dot``: it slows the axis by the
  factor ``1 / (1 + c dt / I)`` per step for an axis of effective inertia
  ``I``. An angular axis's rate is the relative angular velocity along it in
  the parent's joint frame, or the child's for the second axis of a D6 joint
  with two angular axes, and its motion over a step that of the relative
  rotation's increment. A drive, :attr:`~newton.Model.joint_target_ke` or
  :attr:`~newton.Model.joint_target_kd`, and armature,
  :attr:`~newton.Model.joint_armature`, join the same force: the backward Euler
  step of ``ke (q* - q) + kd (q_dot* - q_dot) - c q_dot - I_a q_ddot`` at the
  step's end, toward the control's :attr:`~newton.Control.joint_target_q` and
  :attr:`~newton.Control.joint_target_qd` (the model's without a control). As
  in :class:`~newton.solvers.SolverFeatherstone`, the gains act whatever the
  DOF's :attr:`~newton.Model.joint_target_mode`. A position drive needs an
  axis with one coordinate: a linear axis, or the angular axis of a joint with
  one, whose angle FBA measures in ``(-pi, pi]`` as
  :class:`~newton.solvers.SolverSemiImplicit` does; a position gain on a BALL
  joint or on the angular axes of a D6 joint with several raises
  ``NotImplementedError`` at the first step, while velocity drives, damping and
  armature act on every free axis. An
  :attr:`~newton.Model.joint_effort_limit` below ``1e6`` (the builder's
  default) bounds the drive's force ``ke (q* - q) + kd (q_dot* - q_dot)`` to
  ``[-E, E]``, as MuJoCo's ``actfrcrange`` does, while the damping and armature
  stay unbounded; :attr:`~newton.Model.joint_friction` ``F`` holds the axis
  still with a force within ``F`` of the others' and lets it slide against
  ``F`` beyond, as MuJoCo's ``frictionloss`` does; and the limits,
  :attr:`~newton.Model.joint_limit_lower` and
  :attr:`~newton.Model.joint_limit_upper`, hold the axis within them with any
  force that pushes it inside, as hard limits. The axis's force is then a
  monotone, piecewise linear function of its motion; at each Newton step, the
  row takes the piece that a step of the row alone reaches, as a compliance and
  a target motion, in the joint system (factored again in the blocks whose
  compliance changes) and in the contact solve when contacts touch the
  articulation, so that a step of saturated drives, sticking or sliding
  friction and reached limits lands on the law. A revolute axis measures its
  angle from the middle of its range and needs both limits: one finite limit
  raises ``NotImplementedError``. FBA knows the angle up to whole turns, so it
  leaves out, with a warning, the limits of a range of a full turn or more,
  such as the ``+-2 pi`` of many arm descriptions.
  :attr:`~newton.Control.joint_f` adds each DOF's generalized force to the
  bodies as :class:`~newton.solvers.SolverSemiImplicit` applies it, and a FREE
  joint's wrench to its child. FBA ignores
  :attr:`~newton.Model.joint_velocity_limit`, with a warning where an
  articulation it integrates sets it. Joints at the default infinite limits,
  zero damping, armature and friction and no drive take no such rows.
- **Mimic joints**: a mimic joint (:attr:`~newton.Model.joint_mimic_joint`
  and :attr:`~newton.Model.joint_mimic_coeffs`) holds each coordinate at
  ``c0 + c1`` times its leader's, as a hard row: a pair of rows on the
  follower's and the leader's axes, combined into one row in the
  articulation's joint system, and in the contact solve when contacts touch
  the articulation, so that its force acts on both joints, equal and
  opposite. FBA holds mimics between PRISMATIC, REVOLUTE and D6 joints of the
  same axes, at most one of them angular, whose leader is a joint on bodies it
  integrates; it warns about mimics of other joint types and leaves them out,
  as the other solvers do. A revolute coordinate is its angle in
  ``(-pi, pi]``. Mimic joints that contacts touch need
  ``nsn_sparse_storage=False``.
- **Tendons**: FBA simulates the model's tendons
  (:meth:`~newton.ModelBuilder.add_tendon`) and routes them itself with
  Newton's routing of anchors, pinholes and rollers, the same as the other
  solvers'. Each tendon splits at its interior anchors into groups, and a group
  is one row on its total length: it holds ``L <= rest``, its rest length the
  group's material (its spans and its rollers' arcs), so that the material
  passes freely through pinholes and rollers, and its anchors take the force.
  The guides between a group's anchors must be on bodies that FBA does not
  integrate (static or kinematic). A tendon pulls, never pushes, and takes no
  force while slack. Its row is a frictionless contact with the anchors' bodies
  in the rigid Newton step, its Jacobian the directions from the anchors toward
  the next points of the path at the step's start (the length's derivative,
  since the path between them is fixed for the step), and its value measures
  the runs along those directions, so that it is linear within the step, as a
  contact's normal gap is, also for an end that reaches its next point.
  Friction acts on the slide of the group's first anchor along its run, solved
  with the contacts: the guides' capstan coefficients ``mu`` give it a Coulomb
  bound of ``min(1, sum mu theta)`` times the tension, ``theta`` the guides'
  wrap angles at the step's start (a linearized capstan), so that a slack
  tendon carries no friction and the first anchor's tension stays nonnegative;
  a guide's ``fba:guide_friction_force`` ``F0`` [N] adds a part that does not
  grow with the load, the bound ``F0 + mu T`` at most ``T``, within a step a
  coefficient from the previous step's tension (``mu`` alone after a step
  without tension, so that a string snapping taut is not held by the catch).
  The spans' compliance and damping, summed over a group as Newton merges
  spans, make its row a spring and damper in parallel on the stretch of a taut
  string, so that a slack string pulls nothing and a caught load stops over a
  few steps; ``fba:tendon_time`` and ``fba:tendon_damping_ratio`` per tendon
  instead take them relative to the row's effective mass, as MuJoCo's
  ``solref`` does, for scenes tuned in MuJoCo.
  :meth:`~newton.solvers.SolverFBA.register_custom_attributes` registers these
  attributes. Tendons need the full operator and one world.

FBA does not simulate equality constraints, the
limits of BALL and D6 joints, the limit stiffness and damping
``joint_limit_ke`` and ``joint_limit_kd`` (its limits are hard), the
velocity limits of joint DOFs, or the inputs of
:class:`~newton.Control` other than ``tet_activations``, ``joint_f``,
``joint_target_q`` and ``joint_target_qd``, and it
does not read the damping coefficients ``tri_kd``, ``edge_kd`` and ``k_damp``
of the elements.

Contacts
--------

- ``rigid_contact_time`` makes the normal rows of rigid contacts soft: a
  spring and a damper in parallel (Kelvin-Voigt) of time constant ``tau`` and
  damping ratio ``rigid_contact_damping``, acting while a row holds a force,
  for the row's effective mass. A contact then sinks ``F tau² / m`` under a
  force ``F``, and an impact of a heavy body on a light one spreads over a few
  steps instead of being settled in one, which a hard contact answers with a
  large rotation of the light body in that step. The backward Euler step of
  the pair is the complementarity ``0 <= lambda _|_ g + eps b - beta g_start
  >= 0`` with ``b`` the row's force times ``W_ii``, ``g_start`` its gap at the
  step start, taken as zero above zero so that contacts reported within the
  detection gap do not brake bodies before they touch, ``x = dt / tau``,
  ``eps = 1 / (x² + 2 zeta x)`` and ``beta = 2 zeta / (x + 2 zeta)``; it adds
  ``eps`` to the row's diagonal of the Newton matrix. The default ``0`` keeps rigid contacts hard; tendon and
  particle contact rows stay hard.
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
  margin at longer steps. Its rows still take the inverse masses as their
  response: a contact that presses a particle which stiff cloth ties to
  particles held at points, as a finger pad beside a pinch, sees far less
  motion than its row expects, and its force grows over many steps while the
  particle stays inside the shape; ``cloth_dual_franka_fba`` keeps it off for
  that reason.
- ``nsn_linear_iterations`` caps the conjugate residual iterations of the
  linear solve in each Newton step. ``None`` (default) solves to a relative
  residual of ``1e-10``. A capped step is inexact, and the next local-global
  iteration's Newton step continues from it; most examples cap it at 10. A
  solve to the tolerance reads its residual back to the host every 25
  iterations and runs up to ``max(1000, 2 n)`` of them, which can cost a step
  many times a capped one's, above all with ``nsn_sparse_storage=True`` or the
  lite operator's self contacts. On CUDA the capped iterations of a system of
  at most 512 rows in 3x3 blocks, such as the rigid contacts of a few bodies
  with the joint rows of at most eight articulations, run in one launch, one
  block of threads for the whole solve, and so do those of a sparse system of
  at most 512 rows that hold at most 64 entries on average, such as the
  contacts of a few rods with ``nsn_sparse_storage=True``, and those of a dense
  system of at most 128 active rows, such as the contacts of a step recorded
  into a graph with the full operator: the launch adds its sums in another
  order than the iterations of larger systems, so that the same rows padded
  past those sizes give results that differ by rounding. The products by the
  dense ``W`` of a step recorded into a graph take its active rows and the
  128-square tiles of their nonzero entries alone, such as the blocks of each
  rod or cloth that its contacts touch.
- ``contact_regularization`` (default ``0.01``) is the relative
  regularization ``gamma`` of the contact rows in the Newton steps that
  involve bodies FBA integrates, face contacts, self contacts or rod contacts: such a step
  solves with ``A_cc + gamma diag(A_cc)``. Contacts and joints that close a
  loop, such as a hinged plank leaning on a box that rests on the ground,
  make the Newton matrix nearly singular, and an unregularized step answers a
  sub-micrometer inconsistency there with forces far above the bodies'
  weights. The term bounds the gain along contact directions by
  ``1 / gamma`` and leaves the complementarity conditions unchanged. Steps
  with particle contacts alone are not regularized, nor is the sparse
  self-contact solve of the lite operator.
- ``face_contacts=True`` adds a contact at the deepest point of a triangle
  inside a shape, so shape edges and corners cannot pass between the
  particles of coarse cloth, and holds particles in concave corners of
  meshes. It needs friction and the full operator. By default only the
  triangles next to particle contacts are tested, which misses a shape's tip
  or edge poking into a triangle whose particles all lie beyond the reach of
  the particle contacts; ``face_contact_gap`` [m] tests every triangle within
  that distance of a shape instead. ``face_contact_edge_radius`` [m]
  (default 3 mm) rounds the edges of boxes and cylinders in the signed
  distance that the face contacts minimize, at most half of the shape's
  smallest half extent, so that a triangle point that sinks into an edge does
  not flip its normal between the two faces; the segment contacts of
  ``rod_shape_contacts`` take the same rounding. It is a length, not a
  fraction of the shape, so scale it with the scene.
- ``particle_enable_self_contact=True`` makes the triangles collide with each
  other, through vertex-face or vertex-vertex contacts, as
  ``particle_self_contact_mode`` selects: ``"vertex_face"`` (default) keeps
  each particle ``particle_self_contact_margin`` [m] (default 2 mm) from its
  nearest triangle, and ``"vertex_vertex"`` keeps the particles that far from
  each other, which covers the surface only when the margin reaches the
  longest edge and leaves a bumpy contact surface. The contacts are found
  once per step, within the margin plus ``particle_self_contact_gap`` [m]
  (by default the margin), which bounds the relative motion over a step that
  cannot pass a contact. Particles within
  ``particle_topological_contact_filter_threshold`` mesh edges of each other
  (default 2) do not collide with each other or with the triangles that have
  such a particle, and ``particle_rest_shape_contact_exclusion_radius`` [m]
  excludes particles and triangles closer than that in ``model.particle_q``,
  so that a fold closes only between parts that start apart; both mean what
  :class:`~newton.solvers.SolverVBD`'s options of the same names mean.
  Vertex-face contacts
  keep crossing edges apart when the margin reaches half of the longest edge;
  ``particle_self_contact_edge_edge=True`` adds edge-edge contacts at creases
  for thinner margins, ``particle_self_contact_edge_edge_ratio`` sets how near
  their edges must come compared with their particles' nearest triangles
  (below 1 by default, since every edge pair of two flat layers lies exactly
  as near as its particles' triangles; folded layers that press on each
  other take 1, at a few percent more contacts), and
  :meth:`~newton.solvers.SolverFBA.set_particle_self_contact` changes the
  margin between steps, so that a dropped cloth can land with a thick margin
  and fold with a thin one: the margin is the solver's contact thickness, not
  the cloth's, and fast layers need a thicker one, since the contacts are
  found once per step and the lite operator lets an impact press them in.
  Pass a scalar to set all worlds, or arrays of shape
  ``(model.world_count + 1,)`` to set ``margin`` and ``gap`` independently per
  world, with the last entry for global particles. Per-world values require
  self-contact groups to be disabled and persist across solver resets; the
  shirt example restores the selected episodes' landing margins itself.
  ``particle_self_contact_group_ids`` gives each
  particle a group, such as one per body of a pile: the particles and
  triangles of two groups keep the margin apart, and those of one group the
  group's margin from ``particle_self_contact_group_margins``, or do not
  collide. Groups do not combine with edge-edge contacts or per-world margins.
- ``rod_contact=True`` makes the model's springs, such as the segments of
  particle rods, collide with each other as capsules whose radius runs
  between their particles' radii. A particle takes a contact with each
  segment whose capsule its sphere comes within ``rod_contact_gap`` of, two
  particles take one where each is the closest point of every segment at the
  other, and two segments take one on their four particles where their
  closest points lie inside both and they are not nearly parallel. Each
  contact keeps the sum of the radii at its points and acts along the
  direction between them at the step's start, so a rod draped over another
  sits a little higher at a vertex, where its capsules meet at an angle.
  Particles within as many segments of each other along the springs as the
  largest sum of two radii plus the gap spans at the shortest spring do not
  collide, so that a straight rod never touches itself. The contacts are found
  on the host each step and take ``model.particle_mu`` as friction.
- ``rod_shape_contacts=True`` makes the springs also collide with shapes as
  capsules: a segment takes a contact at its deepest point inside a shape
  when that point lies strictly between its particles and deeper than both,
  a row on its two particles with that point's weights, as a face contact is
  on a triangle's. A shape thinner than the gap between two particles' spheres
  then cannot pass between them, and a rod slides over a curve without
  catching on its particles. ``rod_shape_contact_gap`` sets the detection
  distance. It needs friction and the full operator. In the
  ``rope_on_knives_fba`` example, a rope rests on blades thinner than the gaps
  between its particles' spheres.
- ``friction_anchors=True`` keeps the point a contact presses on across steps,
  on the other layer for a vertex-face self contact or on a static shape, so
  its friction rows pull back the slip that a step's solve leaves instead of
  letting it add up. With the lite operator, that slip lets stacked cloth
  layers creep apart over seconds; with the full operator, a cloth resting
  on a surface gathers in-plane stress from it until its contacts slide,
  after some hundred steps, sooner on some processors. The anchors work with either operator,
  self-contact groups, rod contacts and ``shape_particle_radius_override``;
  edge-edge, vertex-vertex and rod contacts, face contacts, contacts with
  shapes on bodies, and the contacts of a world whose particles touch a body
  that FBA integrates in the step keep the step's point.
  :meth:`~newton.solvers.SolverFBA.set_friction_anchor_exclusion` frees
  chosen particles of their anchors, so that a grasp can gather cloth.
- Rigid contacts between shapes on bodies that FBA integrates need
  ``friction=True`` and the full operator.
- A contact on a body that FBA integrates acts at the contact point of the
  step start, which moves with the body to first order in its rotation over
  the step, so a ball keeps rolling at its speed on flat ground.
- ``rigid_contact_bin`` [m], if positive, keeps only the deepest rigid
  contact of a body pair in each cell of that size around the first body and
  each of 26 normal directions before the solve, so that two finely
  decomposed meshes do not bring thousands of contacts whose rows all couple
  through the two bodies. ``0`` (default) keeps every contact.
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
- ``rigid_iteration_graph=True`` runs the rigid local-global iterations of a
  step whose rows are built anew, such as one with particle contacts on
  bodies, from a CUDA graph captured at the step's first iteration, bit for
  bit the eager result with fewer launches from the host. Steps whose rigid
  rows keep their padded storage replay the contact step's own graph across
  steps instead. It needs CUDA, ``nsn_sparse_storage=True`` and a capped
  ``nsn_linear_iterations`` of at most 25, whose iterations read nothing back,
  and leaves eager the steps with length rows of particle constraints, whose
  directions are set on the host.
- The friction coefficient of a particle contact is its shape's ``mu``,
  ``model.particle_mu`` for self contacts, and the mean of the two shapes'
  ``mu`` for rigid contacts, or the larger of the two with
  ``rigid_contact_friction="max"``, as MuJoCo combines them.
- A particle contact's friction holds the particle to its shape's surface
  velocity, ``Contacts.soft_contact_body_vel``, which the collision pipeline
  reports for conveyors and animated meshes; rigid contacts do not read
  ``Contacts.rigid_contact_surface_velocity``. Face and rod-segment contacts
  take barrel cylinders' bulge, read the shapes' ``COLLIDE_PARTICLES`` flags
  at each step, and let particles of no world touch every shape, as Newton's
  particle contacts do.
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
- ``shishi_odoshi_fba``: water of :class:`~newton.solvers.SolverAPICWater`
  coupled to a body that FBA integrates: a spout fills a bamboo tube on a
  hinge until it tips, pours into a basin, swings back onto a stone and fills
  again. Once per step FBA applies the water's impulse on the tube over the
  last water step, and the water's step sees the tube where FBA left it; each
  frame's steps replay from one CUDA graph.
- ``water_wheel_fba``: an overshot water wheel through the same coupling: water
  from a pipe fills the buckets just past the wheel's top, and their weight
  turns the wheel against its axle's damping, its load, until the two torques
  meet.
- ``sand_castle_fba``: a wrecking ball on a rod swings through a castle of damp
  sand, :class:`~newton.solvers.SolverImplicitMPM` sand with friction and a
  small cohesion. The two solvers exchange the sand's impulse on the ball once
  per FBA step, MPM seeing the ball as a collider that turns about the rod's
  pivot with its moment of inertia; the sand takes nearly half of the ball's
  energy.
- ``batter_mixer_fba``: a mixer's paddle, turned by a motor's constant torque,
  stirs viscous batter that sticks to its bowl, through the same coupling. The
  batter's drag sets the paddle's speed, and the paddle marbles vanilla and
  chocolate batter.
- ``meat_cutter_fba``: a lever cutter, pressed down by a hand and lifted back,
  cuts a slab of pork belly in two through the same coupling. The meat's
  layers of lean, fat and skin are elastic MPM solids that yield past their
  own strengths.
- ``convex_pile_fba``: a thousand rigid bodies of eight convex shapes drop
  into an open box and settle in a pile.
- ``convex_hulls_fba``: convex hull colliders: a stack of five cubes, and cubes
  dropped into a U of three hulls on one body and into a concave U decomposed
  into convex hulls by CoACD.
- ``nonconvex_pile_fba``: a thousand rigid bodies of eight non-convex shapes,
  CoACD-decomposed meshes and compounds of boxes, drop into an open box and
  settle in a pile.
- ``chain_net_fba``: a net of 1,080 interlocked steel chain links, after the
  huge chain net of Affine Body Dynamics (Lan et al. 2022, Fig. 13), its rim
  fixed, falls into a bowl and swings back; each link collides as 14 capsules,
  and ``--size`` sets the links per side of its grid.
- ``twisting_bar_fba``, ``squeezing_ball_fba``: a soft bar twisted by its
  pinned ends and a soft ball dragged by spinning cylinders.
- ``softbody_franka_fba``: a Franka arm integrated by
  :class:`~newton.solvers.SolverFeatherstone` picks up a soft ball.
- ``cable_gripper_fba``: soft fingers curled by cables lift a box.
- ``particle_rods_fba``: particle rods: three ropes of rising bending stiffness
  swing down over a bar from a pinned end, a rod clamped in a block sags and
  swings, and a coil of rope drops onto the ground.
- ``hemp_rope_fba``: a 3 m, 12 mm hemp rope lowered by a circling hand coils
  onto the floor, its segments colliding as capsules.
- ``desk_cables_fba``: an HDMI cable, a USB-C cable and a power cord at a
  laptop, with stiff, heavier plugs; a hand carries the HDMI cable across the
  desk over the USB-C cable.
- ``elastic_cords_fba``: a bungee cord and an elastic cord stretch by a fifth
  as their balls swing down from the level, beside a polyester rope that holds
  its length.
- ``rod_gripper_fba``: a rigid gripper picks up a rope by its middle and lays
  it down elsewhere, holding it by the friction of its pads
  (``--grasp friction``) or by particle constraint rows at the particles
  between them (``--grasp pins``).
- ``rope_on_knives_fba``: an 8 mm rope with its particles 12.5 mm apart drops
  onto five steel blades 1 mm thick and comes to rest across their edges by
  its segment contacts (``rod_shape_contacts=True``). With
  ``--no-segment-contacts``, the blades pass between its particles and the
  rope falls through them to the floor.
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
  shoulders, with self-contact, the lite operator accumulating its contact
  forces and light Rayleigh stiffness damping, and a soft bunny by its ears.
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
