Fix `SolverFBA(rigid_iteration_graph=True)` failing with an `AttributeError` on a step without rigid contact rows before the first step with them, such as a body falling onto the ground.
