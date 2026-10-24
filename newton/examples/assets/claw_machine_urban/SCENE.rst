Urban claw machine FBA scene
===========================

Entry point: ``uv run --no-sync -m newton.examples claw_machine_fba``.

The default is the complete 576-frame (19.2 s at 30 fps) sequence. Optional
``--start settled --num-frames 396`` begins at 6.0 s;
``--start grasp --num-frames 282`` begins at 9.8 s. These restore one saved
physical state, then run dynamics. They do not play a cached animation.

Retained settings
-----------------

* 25 soft toys, 87,982 particles, 305,339 tetrahedra; target is rear green bear 9.
* 240 Hz, 5 local/global iterations and 10 linear iterations, Full NSN,
  Neo-Hookean material, face contacts and Coulomb friction enabled.
* Exact per-element material field in simulation.npz; mass damping 12 / s.
* Self/inter-toy margin 41 mm, gap 3 mm, rest exclusion 45 mm;
  toy 20's intra-toy margin is 20 mm. Particle radii are retained per instance,
  including the target bear's 28 mm rigid-contact radius. The accepted visual
  separation between the fingers and bear remains part of this configuration.
* Four force-limited motor references drive X, Y, Z and the carrier plate;
  hinged links and closure constraints drive the fingers dynamically.
* Initial 0.7 s display hold and 0.75 s gravity ramp. Finger collision shapes
  are added after separated settling at 6 s. Contact/factorization state is
  restarted at 6 s and 9.8 s, matching the original run. Physical positions and
  velocities are carried forward unchanged; every toy stays dynamic.
* Sparse Full NSN storage is explicitly enabled for this example. It retains
  the full operator and nonzero fp64 values; it does not switch to Lite NSN,
  reduce iterations or freeze toys. The dense contact operator is still kept.

The standard viewer uses the supplied visual meshes, material colors and
textures. Photoreal Prana rendering and its camera animation remain separate
from the simulation example. --viewer null runs without visual mesh loading.

Validation during integration
-----------------------------

The entry point was run for one frame from the 9.8 s checkpoint. Soft positions
matched the corresponding original result exactly; the maximum rigid pose
component difference was 2.91e-11. Small sparse/dense parity and per-group
contact-margin tests passed. Visual assets were loaded and rendered without
advancing simulation. A full grasp rerun was cancelled at the user's request;
no new full-run validation is claimed.
