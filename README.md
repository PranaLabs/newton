[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Newton with SolverFBA

This branch of [Newton](https://github.com/newton-physics/newton) adds `SolverFBA`, a solver for cloth and
volumetric soft bodies with robust frictional contact, after the paper
[*Fast But Accurate: A Real-Time Hyperelastic Simulator with Robust Frictional Contact*](https://doi.org/10.1145/3730834)
(ACM Transactions on Graphics 44(4), 2025). It is based on Newton `main` at
[`b09a2214`](https://github.com/newton-physics/newton/commit/b09a22143be8015d18491e1df5d3fc147add791d)
(2026-05-14); everything outside `SolverFBA` and its examples is upstream Newton.

<p align="center">
  <img width="960" src="https://github.com/PranaLabs/newton/releases/download/fba-readme-media/fba_examples.gif" alt="SolverFBA examples">
  <br>
  <sub>Clockwise from top left: <code>cloth_dual_franka_fba</code> (1.65x speed), <code>shirt_fold_fba</code> (3.75x),
  <code>pneunet_gripper_fba</code> (1.1x), <code>cloth_on_knives_fba</code> (0.5x). Rendered with Newton's OpenGL viewer.</sub>
</p>

## SolverFBA

`SolverFBA` steps the elastic bodies with projective dynamics: each step runs a fixed number of local-global
iterations, the local phase projects every element onto its constraint set, and the global phase solves with a
system matrix that is factored once. Contacts are complementarity conditions: each iteration takes one semismooth
Newton step of the contact and constraint multipliers, so contacts do not penetrate and need no contact stiffness.

- **Cloth**: ARAP, corotational, or Neo-Hookean stretching and isometric bending.
- **Soft bodies**: tetrahedra with the same stretching models; tetrahedron activations grow or shrink them, as the
  chambers of a pneumatic actuator lengthen.
- **Frictional contact**: particle and face contacts on shapes, cloth self-contact with vertex-face and edge-edge
  contacts, friction anchors that keep stacked layers from creeping apart, and a contact warm start.
- **Damping**: implicit Rayleigh damping, per particle if needed.
- **Constraints**: pinned particles with moving targets, and rows that hold lengths or linear combinations of
  particle positions, such as cables through a soft body or the cloth a gripper holds.
- **Rigid bodies**: rigid bodies with bilateral joints, or bodies that another Newton solver integrates.
- **Devices**: CUDA and CPU.

`SolverFBA` is experimental: its API and behavior may change. The
[FBA solver guide](https://pranalabs.github.io/newton/fba/solvers/fba.html) covers its inputs, contact options, and limitations, and the
[API reference](https://pranalabs.github.io/newton/fba/api/_generated/newton.solvers.SolverFBA.html) its parameters.

## Requirements

- **Python** 3.10+
- **OS:** Linux (x86-64, aarch64), Windows (x86-64), or macOS (CPU only)
- **GPU:** Warp 1.18, which requires an NVIDIA Turing GPU (compute capability 7.5+) and driver R580+ for CUDA
  acceleration. No local CUDA Toolkit is required; macOS runs on CPU.

## Installation

`SolverFBA` is not in the `newton` package on PyPI. Install this branch from source:

```bash
git clone https://github.com/PranaLabs/newton.git
cd newton
uv run --extra examples -m newton.examples cloth_dual_franka_fba
```

or, without [uv](https://docs.astral.sh/uv/), in a virtual environment:

```bash
pip install -e ".[examples]"
python -m newton.examples cloth_dual_franka_fba
```

## Quickstart

A cloth dropped on the ground:

```python
import warp as wp

import newton

builder = newton.ModelBuilder()
builder.add_ground_plane()
builder.add_cloth_grid(
    pos=wp.vec3(-0.5, -0.5, 0.5),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0),
    dim_x=20,
    dim_y=20,
    cell_x=0.05,
    cell_y=0.05,
    mass=0.005,
    tri_ke=1.0e4,
    edge_ke=1.0e-4,
    particle_radius=0.005,
)
model = builder.finalize()

solver = newton.solvers.SolverFBA(model, iterations=10)
pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
contacts = pipeline.contacts()
state_0, state_1 = model.state(), model.state()

for _ in range(120):
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, None, contacts, 1.0 / 60.0)
    state_0, state_1 = state_1, state_0
```

The first step factors the system matrix on the CPU and takes longer than the others.

## Examples

Each example below runs with `python -m newton.examples <name>`, or with
`uv run --extra examples -m newton.examples <name>` from a uv checkout. The examples that come with upstream Newton
are still in the package; `python -m newton.examples --list` lists them all. The clips are rendered with Newton's
OpenGL viewer; they play at real time or faster, the longer examples sped up to at most 10 seconds per clip.

<table>
  <tr>
    <td colspan="3"><h3>Robot Manipulation</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_dual_franka_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/cloth_dual_franka_fba.webp" alt="Cloth Dual Franka FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_shirt_fold_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/shirt_fold_fba.webp" alt="Shirt Fold FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_softbody_franka_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/softbody_franka_fba.webp" alt="Softbody Franka FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_dual_franka_fba</code><br>
      <sub>two Franka arms fold a towel twice into a square</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples shirt_fold_fba</code><br>
      <sub>two Franka arms fold a T-shirt dropped on a table</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_franka_fba</code><br>
      <sub>a Franka arm picks up a soft ball held by friction alone</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_pneunet_gripper_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/pneunet_gripper_fba.webp" alt="PneuNet Gripper FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_cable_gripper_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/cable_gripper_fba.webp" alt="Cable Gripper FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_claw_machine_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/claw_machine_fba.webp" alt="Claw Machine FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples pneunet_gripper_fba</code><br>
      <sub>a pneumatic three-finger gripper closes around a ball</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_gripper_fba</code><br>
      <sub>soft fingers curled by embedded cables lift a box</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples claw_machine_fba</code><br>
      <sub>a claw picks a toy out of a pile of 25 soft toys</sub>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Cloth</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_stretching_cloth_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/stretching_cloth_fba.webp" alt="Stretching Cloth FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_shirt_pins_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/shirt_pins_fba.webp" alt="Shirt Pins FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_cloth_on_knives_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/cloth_on_knives_fba.webp" alt="Cloth on Knives FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples stretching_cloth_fba</code><br>
      <sub>Neo-Hookean cloth pulled apart by pins on two opposite edges</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples shirt_pins_fba</code><br>
      <sub>a T-shirt moved by two handles at its shoulders, with self-contact</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_on_knives_fba</code><br>
      <sub>cloth dropped on notched blades, held off their tips by face contacts</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_cloth_on_boxes_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/cloth_on_boxes_fba.webp" alt="Cloth on Boxes FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_on_boxes_fba</code><br>
      <sub>cloth on stacked boxes and a hinged plank that tips onto them</sub>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Rigid Bodies</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/contacts/example_convex_pile_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/convex_pile_fba.webp" alt="Convex Pile FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/contacts/example_convex_hulls_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/convex_hulls_fba.webp" alt="Convex Hulls FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/contacts/example_nonconvex_pile_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/nonconvex_pile_fba.webp" alt="Nonconvex Pile FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples convex_pile_fba</code><br>
      <sub>a thousand convex bodies drop into a box and settle in a pile</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples convex_hulls_fba</code><br>
      <sub>a cube stack and cubes dropped into a compound U and a CoACD-decomposed U</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples nonconvex_pile_fba</code><br>
      <sub>a thousand non-convex bodies, as convex parts, settle in a box</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/contacts/example_chain_net_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/chain_net_fba.webp" alt="Chain Net FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples chain_net_fba</code><br>
      <sub>a net of 1,080 interlocked chain links, its rim fixed, falls into a bowl</sub>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Soft Bodies</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_twisting_bar_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/twisting_bar_fba.webp" alt="Twisting Bar FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_squeezing_ball_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/squeezing_ball_fba.webp" alt="Squeezing Ball FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_softbody_cantilever_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/softbody_cantilever_fba.webp" alt="Softbody Cantilever FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples twisting_bar_fba</code><br>
      <sub>a stiff Neo-Hookean bar (E = 1 GPa) twisted by its pinned ends</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples squeezing_ball_fba</code><br>
      <sub>a soft ball dragged through spinning cylinders by friction</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_cantilever_fba</code><br>
      <sub>two soft beams, one with Rayleigh damping</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_bunny_ears_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/bunny_ears_fba.webp" alt="Bunny Ears FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cable/example_particle_rods_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/particle_rods_fba.webp" alt="Particle Rods FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cable/example_hemp_rope_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/hemp_rope_fba.webp" alt="Hemp Rope FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples bunny_ears_fba</code><br>
      <sub>a soft bunny carried and swung by its ear tips</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples particle_rods_fba</code><br>
      <sub>ropes and a rod made of particles swing over a bar, sag and coil</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples hemp_rope_fba</code><br>
      <sub>a hemp rope lowered by a circling hand coils onto the floor</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cable/example_desk_cables_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/desk_cables_fba.webp" alt="Desk Cables FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cable/example_elastic_cords_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/elastic_cords_fba.webp" alt="Elastic Cords FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cable/example_rod_gripper_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/rod_gripper_fba.webp" alt="Rod Gripper FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples desk_cables_fba</code><br>
      <sub>HDMI, USB-C and power cables with plugs; a hand carries one across the desk</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples elastic_cords_fba</code><br>
      <sub>bungee and elastic cords stretch as their balls swing, a rope does not</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples rod_gripper_fba</code><br>
      <sub>a rigid gripper picks up a rope by friction or by pinned particles</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cable/example_rope_on_knives_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/rope_on_knives_fba.webp" alt="Rope on Knives FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples rope_on_knives_fba</code><br>
      <sub>a coarse rope drops onto thin blades and rests on their edges by segment contacts</sub>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Water and MPM Coupling</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_shishi_odoshi_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media-2026-11-24/shishi_odoshi_fba.webp" alt="Shishi-odoshi FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_water_wheel_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media-2026-11-24/water_wheel_fba.webp" alt="Water Wheel FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_sand_castle_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/sand_castle_fba.webp" alt="Sand Castle FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples shishi_odoshi_fba</code><br>
      <sub>Water fills a hinged bamboo tube until it tips, pours into a basin and knocks back on a stone</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples water_wheel_fba</code><br>
      <sub>Water from a bamboo pipe fills the buckets of an overshot wheel and turns it against its load</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples sand_castle_fba</code><br>
      <sub>A wrecking ball on a rod swings through a castle of damp MPM sand</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_batter_mixer_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/batter_mixer_fba.webp" alt="Batter Mixer FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_meat_cutter_fba.py">
        <img width="320" src="https://github.com/PranaLabs/newton/releases/download/fba-gallery-media/meat_cutter_fba.webp" alt="Meat Cutter FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples batter_mixer_fba</code><br>
      <sub>A motor's paddle marbles viscous vanilla and chocolate MPM batter in a bowl</sub>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples meat_cutter_fba</code><br>
      <sub>A lever cutter, pressed by a hand, cuts a layered slab of MPM pork belly in two</sub>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
</table>

### Performance

Mean time per frame of each example over its demo, as long as its clip above, after 10 warm-up frames, without
rendering (`--viewer null`), on one NVIDIA GeForce RTX 5090, with this branch's code as of 2026-11-18. Real time is the simulated time
per wall-clock time; contacts are the mean count per frame.

| Example | Scene | Frame | ms per frame | Real time |
| --- | --- | ---: | ---: | ---: |
| `cloth_dual_franka_fba` | 1,681 particles, 3,200 triangles, 18 bodies | 1/60 s | 13.0 | 1.29x |
| `shirt_fold_fba` | 6,480 particles, 12,748 triangles, 18 bodies | 1/60 s | 16.7 | 1.00x |
| `softbody_franka_fba` | 2,197 particles, 1,728 triangles, 10,368 tetrahedra, 9 bodies | 1/60 s | 20.4 | 0.82x |
| `pneunet_gripper_fba` | 7,363 particles, 9,612 triangles, 27,395 tetrahedra, 1 body | 1/100 s | 39.6 | 0.25x |
| `cable_gripper_fba` | 1,050 particles, 1,408 triangles, 3,200 tetrahedra, 1 body | 1/100 s | 25.6 | 0.39x |
| `claw_machine_fba` | 87,982 particles, 131,566 triangles, 305,339 tetrahedra, 10 bodies | 1/30 s | 411.0 | 0.08x |
| `stretching_cloth_fba` | 20,201 particles, 40,000 triangles | 1/100 s | 3.8 | 2.60x |
| `shirt_pins_fba` | 7,931 particles, 15,650 triangles | 1/60 s | 12.7 | 1.31x |
| `cloth_on_knives_fba` | 5,101 particles, 10,000 triangles | 1/100 s | 28.5 | 0.35x |
| `cloth_on_boxes_fba` | 289 particles, 512 triangles, 3 bodies | 1/60 s | 22.7 | 0.74x |
| `convex_pile_fba` | 1,000 bodies, 6,310 contacts | 1/50 s | 6.8 | 2.93x |
| `convex_hulls_fba` | 9 bodies, 50 contacts | 1/50 s | 3.5 | 5.70x |
| `nonconvex_pile_fba` | 1,000 bodies, 18,360 contacts | 1/50 s | 16.1 | 1.25x |
| `chain_net_fba` | 1,008 bodies, 7,500 contacts | 1/50 s | 14.2 | 1.41x |
| `twisting_bar_fba` | 11,340 particles, 5,780 triangles, 58,956 tetrahedra | 1/100 s | 6.5 | 1.55x |
| `squeezing_ball_fba` | 6,859 particles, 3,888 triangles, 34,992 tetrahedra, 4 bodies | 1/100 s | 23.1 | 0.43x |
| `softbody_cantilever_fba` | 10,584 particles, 6,760 triangles, 43,940 tetrahedra | 1/50 s | 7.7 | 2.59x |
| `bunny_ears_fba` | 4,155 particles, 5,294 triangles, 15,844 tetrahedra | 1/60 s | 3.7 | 4.45x |
| `particle_rods_fba` | 395 particles | 1/50 s | 10.0 | 2.00x |
| `hemp_rope_fba` | 251 particles | 1/50 s | 9.3 | 2.14x |
| `desk_cables_fba` | 533 particles | 1/50 s | 10.1 | 1.97x |
| `elastic_cords_fba` | 251 particles | 1/50 s | 33.1 | 0.60x |
| `rod_gripper_fba` | 102 particles, 3 bodies | 1/50 s | 18.0 | 1.11x |
| `rope_on_knives_fba` | 58 particles | 1/50 s | 9.0 | 2.21x |
| `shishi_odoshi_fba` | up to 59,017 water particles, 1 body | 1/30 s | 21.7 | 1.53x |
| `water_wheel_fba` | up to 20,000 water particles, 1 body | 1/30 s | 9.1 | 3.66x |
| `sand_castle_fba` | 44,125 sand particles (MPM), 1 body | 1/30 s | 142.5 | 0.23x |
| `batter_mixer_fba` | 56,194 batter particles (MPM), 1 body | 1/30 s | 208.4 | 0.16x |
| `meat_cutter_fba` | 55,202 meat particles (MPM), 1 body | 1/30 s | 468.1 | 0.07x |

The water examples (measured on 2026-11-24) couple `SolverAPICWater` on a grid of 5 mm cells with `SolverFBA` once per
step, 10 steps per frame, each frame replayed from one CUDA graph; the water flows in over the demo, up to the capacity of
its pool. The MPM examples couple an MPM grid of 5 mm voxels (7.5 mm for the sand, 4 mm for the meat) with `SolverFBA`
once per step, 16 steps per frame.

### Example Options

All examples share a common set of command-line options. The most commonly used ones are:

| Argument        | Description                                                              | Default                      |
| --------------- | ------------------------------------------------------------------------ | ---------------------------- |
| `--viewer`      | Viewer to use: `gl`, `usd`, `rtx`, `rerun`, `viser`, or `null`.          | `gl`                         |
| `--device`      | Compute device to use, e.g., `cpu`, `cuda:0`.                            | `None` (default Warp device) |
| `--num-frames`  | Total number of frames to simulate.                                      | `100`                        |
| `--output-path` | Path to the output USD file (used by the `usd` viewer).                  | `output.usd`                 |

For the complete, always-current list of options — plus any arguments a specific example adds — run:

```bash
python -m newton.examples <example_name> --help
```

### Example Usage

```bash
# List available examples
python -m newton.examples --list

# Run with the USD viewer and save to my_output.usd
python -m newton.examples cloth_dual_franka_fba --viewer usd --output-path my_output.usd

# Run on a selected device
python -m newton.examples shirt_fold_fba --device cuda:0

# Combine options
python -m newton.examples twisting_bar_fba --viewer gl --num-frames 500 --device cpu
```

## Citation

If you use `SolverFBA` in your research, please cite:

```bibtex
@article{Zeng2025FBA,
  title   = {Fast But Accurate: A Real-Time Hyperelastic Simulator with Robust Frictional Contact},
  author  = {Zeng, Ziqiu and Luo, Siyuan and Shi, Fan and Zhang, Zhongkai},
  journal = {ACM Transactions on Graphics},
  volume  = {44},
  number  = {4},
  pages   = {1--19},
  year    = {2025},
  doi     = {10.1145/3730834},
}
```

## Newton

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp) and
a [Linux Foundation](https://www.linuxfoundation.org/) project; see the
[Newton repository](https://github.com/newton-physics/newton) and its
[documentation](https://newton-physics.github.io/newton/latest/guide/overview.html) for everything else. Code is
licensed under [Apache-2.0](LICENSE.md). Documentation is licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Additional and third-party license texts are available in
[`newton/licenses`](newton/licenses).
