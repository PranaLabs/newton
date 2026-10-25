Urban claw machine FBA scene
===========================

Run the complete 600-frame (20 s at 30 fps) scene::

    uv run --no-sync -m newton.examples claw_machine_fba

Optional ``--start settled --num-frames 420`` begins at 6.0 s;
``--start grasp --num-frames 285`` begins at 10.5 s with the claw open.
These restore positions and velocities once, then run dynamics. They do not
play a cached animation. Scene parameters and camera endpoints are stored in
scene.json; no experiment output directory is required.

Physics and grasp
-----------------

* 25 soft toys, 87,982 particles, 305,339 tetrahedra; target is rear green bear 9.
* 240 Hz, 5 local/global iterations and 10 linear iterations, Full NSN,
  Neo-Hookean material, face contacts and Coulomb friction enabled.
* Original per-element materials and masses in simulation.npz; mass damping 12 / s.
* The 39 finger capsules use a 2 mm particle/face contact thickness. Detection
  retains conservative particle radii; the normal constraint uses the smaller
  thickness. Other shapes and soft contacts retain their original settings.
* Self/inter-toy margin 41 mm, gap 3 mm, rest exclusion 45 mm;
  toy 20's intra-toy margin is 20 mm. Tray/wall particle radii remain unchanged.
* Four force-limited motor references drive X, Y, Z and the carrier plate;
  hinged links drive the fingers. The open claw recenters around the bear,
  closes with a 14 N force limit, then ramps to 20 N and 35 N while lifting.
* Initial 0.7 s display hold and 0.75 s gravity ramp. Finger collision shapes
  are added at 6 s. Contact/factorization caches restart at 6, 10.5 and 13.5 s
  to match the trajectory refinement boundaries. Positions and velocities
  remain continuous; every toy stays dynamic throughout.
* Sparse Full NSN storage retains the full operator and nonzero fp64 values;
  it does not change the solver budget. The dense contact operator is retained.

``--finger-contact-radius`` overrides the default 0.002 m finger thickness.
``--grasp-plan`` optionally loads alternate motor keyframes and force schedules;
custom keyframes may require an appropriate ``--num-frames`` value.

Camera
------

The standard viewer starts with the wide cabinet view, transitions smoothly
between 6 and 8.5 s, then holds a fixed 18-degree left view. The final camera is
25% farther away than the earlier 32-degree view and shifted to keep the bear
clear of the front cabinet pillar. This follows the accepted Prana camera path;
Newton's viewer uses its own lighting. --viewer null skips visual mesh loading.

Validation
----------

Run the full dynamics and grasp checks with::

    uv run --no-sync -m newton.examples claw_machine_fba --viewer null --test

Test mode checks positive tetrahedron volumes at every 240 Hz step, finite
states, unchanged material fields, dynamic particle masses, sustained lift and
pile clearance throughout the final two-second hold, and horizontal carry.

The updated default entry point was checked continuously from 0 through 18 s
(4,320 physical steps), with no inverted tetrahedra; the minimum volume ratio
was 0.090563. The run was stopped at the user's request before the final
suspension check. The reduced-radius regression test and all pre-commit checks
passed. The viewer camera matches the accepted render path at all 601 samples.
