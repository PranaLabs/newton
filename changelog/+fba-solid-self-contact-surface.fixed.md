Exclude tetrahedral interior particles from SolverFBA vertex-face self-contact queries so solid bodies collide through their surface. Preserve independent-particle, cloth, and vertex-vertex contact behavior.

Solid vertex-face neighbor filtering now includes tetrahedron edges, preventing opposite sides of the same local volume from being treated as separate colliders. Cloth and vertex-vertex filtering are unchanged.
