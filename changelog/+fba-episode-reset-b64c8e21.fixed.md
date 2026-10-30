Preserve gravity on CPU in the Franka cloth FBA examples and support resetting
selected folding episodes, including robot state, grasp constraints and task
phases. Allow per-world self-contact distances in
`SolverFBA.set_particle_self_contact()` so resetting a shirt's landing phase
preserves neighboring episodes.
