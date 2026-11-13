Speed up `SolverFBA` steps on the sparse NSN storage: the CUDA graph of a step's NSN step replays at later steps while its rows keep their count, instead of being captured again at every step.
