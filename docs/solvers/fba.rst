.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

FBA
===

:class:`~newton.solvers.SolverFBA` simulates cloth with projective dynamics.
Each step runs a fixed number of local-global iterations: the local phase
projects every element onto its constraint set, and the global phase solves a
linear system whose matrix is factored once.

.. experimental::

   :class:`~newton.solvers.SolverFBA` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

When to Choose FBA
------------------

FBA suits stiff cloth: each step is implicit, so stiff cloth steps stably at
large time steps.

Choose another solver for soft bodies, for scenes with contacts, for models
with several worlds or with rigid bodies, and for differentiable simulation.
See the :ref:`Supported Features` matrix for a comparison with the other
solvers. The solver builds on projective dynamics and a quadratic bending
energy:

- Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark
  Pauly. 2014. Projective Dynamics: Fusing Constraint Projections for Fast
  Simulation. *ACM Trans. Graph.* 33, 4.
  https://doi.org/10.1145/2601097.2601116
- Miklós Bergou, Max Wardetzky, David Harmon, Denis Zorin, and Eitan
  Grinspun. 2006. A Quadratic Bending Model for Inextensible Surfaces.
  *Symposium on Geometry Processing*.

Construction and Stepping
-------------------------

Step the solver from one state to the next with
:meth:`~newton.solvers.SolverFBA.step`:

.. code-block:: python

   import warp as wp

   import newton

   builder = newton.ModelBuilder()
   builder.add_cloth_grid(
       pos=wp.vec3(0.0, 0.0, 1.0),
       rot=wp.quat_identity(),
       vel=wp.vec3(0.0),
       dim_x=20,
       dim_y=20,
       cell_x=0.05,
       cell_y=0.05,
       mass=0.005,
       tri_ke=1.0e4,
       edge_ke=1.0e-4,
       fix_left=True,
   )
   model = builder.finalize()

   solver = newton.solvers.SolverFBA(model, iterations=10)
   state_0, state_1 = model.state(), model.state()

   for _ in range(120):
       solver.step(state_0, state_1, None, None, 1.0 / 60.0)
       state_0, state_1 = state_1, state_0

The cloth swings down from its left edge, whose particles the builder pins by
giving them zero mass. The first step factors the system matrix on the CPU and
takes longer than the others. The matrix depends on the time step, the
particle masses, the pinned and inactive particles, and the elements and their
materials; the solver factors it again when the time step changes or after
:meth:`~newton.solvers.SolverFBA.notify_model_changed` with
``SHAPE_PROPERTIES``, and other changes to those arrays take effect only then.
A step reads the number of soft contacts in its contacts back to the host, so
it cannot be captured in a CUDA graph; it captures and replays its own
local-global iterations. FBA runs on the CPU and on CUDA devices.

Model Inputs
------------

- **Cloth**: triangles with As-Rigid-As-Possible stretching, which takes
  ``tri_ke`` as the stiffness. Interior edges add the isometric bending
  energy of Bergou et al. with ``edge_ke`` as a bending modulus [N·m] and
  ``edge_rest_angle`` as the rest angle. The same ``edge_ke`` bends an FBA
  cloth more stiffly than a :class:`~newton.solvers.SolverVBD` one, which
  scales it by the edge length.
- **Pins**: particles with zero inverse mass follow their pin targets through
  a stiff spring of ``pin_stiffness`` [N/m]. The targets start at the model
  positions and move with :meth:`~newton.solvers.SolverFBA.set_pin_targets`.
  Particles without the ``ACTIVE`` flag that have a mass hold where the input
  state has them.

FBA does not simulate tetrahedra, rigid bodies or contacts: a model with
tetrahedra or bodies, and contacts that hold soft contacts, raise
``NotImplementedError``. It does not simulate springs either, or the inputs
of :class:`~newton.Control`.

Accuracy
--------

Positions are float32 and velocities come from position differences, so far
from the origin velocity rounding grows: over a 350 cm free fall a cloth
drifts by 0.08 cm. The bending energy is quadratic in the positions, so a
fold with a large rest angle settles a little away from that angle under
load; in Newton's shared cloth tests the folds of a curved cloth settle
0.11 rad from their rest angles.
