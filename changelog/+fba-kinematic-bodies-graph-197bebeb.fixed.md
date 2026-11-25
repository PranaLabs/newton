Fix recording `SolverFBA` steps into a CUDA graph in models whose bodies are all kinematic, whose steps read the body flags back from the device to find that the solver integrates none of them.
