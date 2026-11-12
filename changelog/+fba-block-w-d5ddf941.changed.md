Speed up `SolverFBA` steps of many rigid contacts by keeping the Delassus matrix of the rigid rows in 3x3 blocks, which the fused conjugate residual iterations multiply with fewer memory reads.
