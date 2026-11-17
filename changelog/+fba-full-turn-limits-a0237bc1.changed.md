Leave out, with a `NewtonWarning`, the limits of `SolverFBA` revolute joints whose range spans a full turn or more, such as the +-2 pi of many arm descriptions, which raised `NotImplementedError`.
