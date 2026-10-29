[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![Downloads](https://api.pepy.tech/badge/newton/month)](https://pepy.tech/projects/newton)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push - AWS GPU](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml)

# Newton

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp), specifically targeting roboticists and simulation researchers.

Newton extends and generalizes Warp's ([deprecated](https://github.com/NVIDIA/warp/discussions/735)) `warp.sim` module, and integrates
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as its primary backend. Newton emphasizes GPU-based computation, [OpenUSD](https://openusd.org/) support, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

Newton is a [Linux Foundation](https://www.linuxfoundation.org/) project that is community-built and maintained. Code is licensed under [Apache-2.0](https://github.com/newton-physics/newton/blob/main/LICENSE.md). Documentation is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Additional and third-party license texts are available in [`newton/licenses`](https://github.com/newton-physics/newton/tree/main/newton/licenses).

Newton was initiated by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Requirements

- **Python** 3.10+
- **OS:** Linux (x86-64, aarch64), Windows (x86-64), or macOS (CPU only)
- **GPU:** Fresh installs currently select Warp 1.18, which requires an NVIDIA Turing GPU (compute capability 7.5+) and driver R580+ for CUDA acceleration.
  On an R545–R579 driver or an older supported GPU, install with `warp-lang==1.17.0` (see the installation guide).
  No local CUDA Toolkit is required; macOS runs on CPU.

For detailed system requirements, see the [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html). For tested configurations and Newton's versioning and deprecation policies, see the [compatibility guide](https://newton-physics.github.io/newton/latest/guide/compatibility.html).

## Quickstart

```bash
pip install "newton[examples]"
python -m newton.examples
```

To install from source with [uv](https://docs.astral.sh/uv/), see the [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html).

## Examples

The examples below run on `SolverFBA`, which this branch adds. The examples that come with
upstream Newton are still in the package; `python -m newton.examples --list` lists them all.

Before running the examples below, install Newton with the examples extra:

```bash
pip install "newton[examples]"
```

If you run the examples from a source checkout with uv, use
`uv run --extra examples -m newton.examples <example_name>` instead of the
`python -m newton.examples <example_name>` commands below.

<table>
  <tr>
    <td colspan="3"><h3>Cloth Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_stretching_cloth_fba.py">
        <img width="320" src="docs/images/examples/example_stretching_cloth_fba.jpg" alt="Stretching Cloth FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_shirt_pins_fba.py">
        <img width="320" src="docs/images/examples/example_shirt_pins_fba.jpg" alt="Shirt Pins FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_cloth_dual_franka_fba.py">
        <img width="320" src="docs/images/examples/example_cloth_dual_franka_fba.jpg" alt="Cloth Dual Franka FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples stretching_cloth_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples shirt_pins_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_dual_franka_fba</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/cloth/example_shirt_fold_fba.py">
        <img width="320" src="docs/images/examples/example_shirt_fold_fba.jpg" alt="Shirt Fold FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples shirt_fold_fba</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Multi-Physics Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_cloth_on_knives_fba.py">
        <img width="320" src="docs/images/examples/example_cloth_on_knives_fba.jpg" alt="Cloth on Knives FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/multiphysics/example_cloth_on_boxes_fba.py">
        <img width="320" src="docs/images/examples/example_cloth_on_boxes_fba.jpg" alt="Cloth on Boxes FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_on_knives_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_on_boxes_fba</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Softbody Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_twisting_bar_fba.py">
        <img width="320" src="docs/images/examples/example_twisting_bar_fba.jpg" alt="Twisting Bar FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_squeezing_ball_fba.py">
        <img width="320" src="docs/images/examples/example_squeezing_ball_fba.jpg" alt="Squeezing Ball FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_softbody_cantilever_fba.py">
        <img width="320" src="docs/images/examples/example_softbody_cantilever_fba.jpg" alt="Softbody Cantilever FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples twisting_bar_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples squeezing_ball_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_cantilever_fba</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_softbody_franka_fba.py">
        <img width="320" src="docs/images/examples/example_softbody_franka_fba.jpg" alt="Softbody Franka FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_cable_gripper_fba.py">
        <img width="320" src="docs/images/examples/example_cable_gripper_fba.jpg" alt="Cable Gripper FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_bunny_ears_fba.py">
        <img width="320" src="docs/images/examples/example_bunny_ears_fba.jpg" alt="Bunny Ears FBA">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_franka_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_gripper_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples bunny_ears_fba</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_pneunet_gripper_fba.py">
        <img width="320" src="docs/images/examples/example_pneunet_gripper_fba.jpg" alt="PneuNet Gripper FBA">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="newton/examples/softbody/example_claw_machine_fba.py">
        <img width="320" src="docs/images/examples/example_claw_machine_fba.jpg" alt="Claw Machine FBA">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples pneunet_gripper_fba</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples claw_machine_fba</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
</table>

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

## Research using Newton

See [PUBLICATIONS.md](https://github.com/newton-physics/newton/blob/main/PUBLICATIONS.md) for research using Newton, inclusion criteria, and instructions for adding your publication.

## Contributing and Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for ways to contribute and the pull-request process. The [governance contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) cover legal requirements, project roles, and approval authority. Use the [development guide](https://newton-physics.github.io/newton/latest/guide/development.html) for environment setup and development workflows. Code and public API changes must follow the [source code guidelines](CODING_GUIDELINES.rst); reviewers may use the suggested [review guidelines](REVIEW_GUIDELINES.rst).

## Support and Community Discussion

For questions, please consult the [Newton documentation](https://newton-physics.github.io/newton/latest/guide/overview.html) first before creating [a discussion in the main repository](https://github.com/newton-physics/newton/discussions).

## Code of Conduct

By participating in this community, you agree to abide by the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

## Project Governance, Legal, and Members

Please see the [newton-governance repository](https://github.com/newton-physics/newton-governance) for more information about project governance.
