Speed up `SolverFBA` steps whose contact rows touch bodies alone by padding the rows to a capacity kept across steps, so that the CUDA graph of the NSN step replays from step to step.
