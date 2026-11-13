Speed up `SolverFBA` springs: the tension solve of the spring chains runs as parallel cyclic reduction over all springs at once instead of one thread per chain.
