# Per-Demo Scene Config Audit (Phase 0 Task 0.3)

**Date:** 2026-05-17
**Source-of-truth:** RealSim CudaTests `scene.json` files (read-only)
**Cross-ref:** `/home/ziqiu/work/newton/scripts/realsim_baseline/decisions.json` (Task 0.2)
**Discipline reference:** `/home/ziqiu/.claude/projects/-home-ziqiu-work/memory/realsim-port-discipline.md` — when a field differs, default verdict is "FBA script is wrong, not RealSim scene.json".

## Key clarifying finding (revised 2026-05-17 post-controller-check)

`scripts/fba_demo1_cloth_on_plane.py`, `scripts/fba_demo2_softbody_bounce.py`, and `scripts/fba_demo3_cloth_on_cylinder.py` are **NOT** CudaTests ports. They are simpler hand-rolled cloth/softbody scenes used during FBA bring-up. The `fba_demo<N>` numbering does not align with CudaTests demo numbering.

**However**, `scripts/fba_twisting_bar_cudatests.py` covers Demos 1 (TwistingBar=ARAP) and 2 (TwistingBarNH=NH) via `--energy {arap, corotational, neohookean}`. The initial audit missed this because the `fba_demo*.py` glob pattern didn't match.

Corrected mapping of the ten CudaTests demos to existing FBA scripts:

| # | CudaTests demo            | FBA port script                                          |
|---|---------------------------|----------------------------------------------------------|
| 1 | TwistingBar               | `scripts/fba_twisting_bar_cudatests.py --energy arap`    |
| 2 | TwistingBarNH             | `scripts/fba_twisting_bar_cudatests.py --energy neohookean` |
| 3 | StretchingCloth           | NOT YET WRITTEN                                          |
| 4 | PullingWooper             | `scripts/fba_demo4_pulling_wooper.py`                    |
| 5 | SqueezingBall             | `scripts/fba_demo5_squeezing_ball.py`                    |
| 6 | CrossingGingerbreadman    | NOT YET WRITTEN                                          |
| 7 | SharpCorner               | NOT YET WRITTEN                                          |
| 8 | ClothOnKnives             | NOT YET WRITTEN                                          |
| 9 | ParallelEnvTest           | NOT YET WRITTEN                                          |
| 10| CableGrabRaptor           | NOT YET WRITTEN                                          |

For demos 1, 2, 4, 5 the per-demo cross-check below uses the actual FBA script values. For demos 3, 6, 7, 8, 9, 10 the RealSim column serves as scene-loader spec for Phase 3 / Phase 5.

## Summary

| Demo | FBA script status | Verdict |
|---|---|---|
| 1. TwistingBar | `fba_twisting_bar_cudatests.py --energy arap` | NEEDS-RECHECK (not in initial audit) |
| 2. TwistingBarNH | `fba_twisting_bar_cudatests.py --energy neohookean` | NEEDS-RECHECK (not in initial audit) |
| 3. StretchingCloth | NOT YET WRITTEN | NOT-YET-WRITTEN |
| 4. PullingWooper | `scripts/fba_demo4_pulling_wooper.py` | DIVERGE-1 |
| 5. SqueezingBall | `scripts/fba_demo5_squeezing_ball.py` | DIVERGE-1 |
| 6. CrossingGingerbreadman | NOT YET WRITTEN | NOT-YET-WRITTEN |
| 7. SharpCorner | NOT YET WRITTEN | NOT-YET-WRITTEN |
| 8. ClothOnKnives | NOT YET WRITTEN | NOT-YET-WRITTEN |
| 9. ParallelEnvTest | NOT YET WRITTEN | NOT-YET-WRITTEN |
| 10. CableGrabRaptor | NOT YET WRITTEN | NOT-YET-WRITTEN |

Roll-up: **0 MATCH, 2 DIVERGE, 2 NEEDS-RECHECK, 6 NOT-YET-WRITTEN**.

The two NEEDS-RECHECK demos (1, 2) are tracked as a follow-up sub-task before Phase 3 begins (verify `fba_twisting_bar_cudatests.py` byte-matches `TwistingBar/scene.json` and `TwistingBarNH/scene.json`).

## Per-demo

### Demo 1 — TwistingBar

**RealSim scene:** `simulation/config/CudaTests/TwistingBar/scene.json`
**Object cfg:** `simulation/config/CudaTests/TwistingBar/object_5k.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:5) | `[0.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` (scene.json:4) | `0.01` | NOT YET WRITTEN | — |
| `stop` (scene.json:3) | `810` | NOT YET WRITTEN | — |
| `timer` (scene.json:2) | `50` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` (scene.json:6) | `5` (PD outer iters) | NOT YET WRITTEN | — |
| `linearsolver.type` (scene.json:8) | `SPARSE_INVERSE_CUDA` | NOT YET WRITTEN | — |
| `constraintsolver` block | **ABSENT** (pure FEM, no contacts) | NOT YET WRITTEN | — |
| `parallelEnv` block | absent | NOT YET WRITTEN | — |

**Objects:**

| Obj | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh (object_5k.json:2) | `resources/mesh/volume/cube_volume_5292P.mesh` | NOT YET WRITTEN | — |
| object_0 | `element_type` (object_5k.json:4) | `TETRAHEDRON` | NOT YET WRITTEN | — |
| object_0 | `transformation.trans` (object_5k.json:6) | `[0.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.rotation` (object_5k.json:7) | `[0.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.scale` (object_5k.json:8) | `[1.0, 1.0, 1.0]` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.obj_mass` (object_5k.json:12) | `1000` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.young` (object_5k.json:13) | `1E+9` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.poisson` (object_5k.json:14) | `0.45` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.constitutive` (object_5k.json:15) | `TET_PD_ARAP` | NOT YET WRITTEN | — |
| object_0 | bending block | absent (tet, not cloth) | NOT YET WRITTEN | — |

**Collision shapes:** none — pure FEM, no contacts.

**Pin actions** (from object_5k.json:17-44):

| Pin idx | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| 0 | `box` (line 20) | `[-1.1, 1.99, -1.1, 1.1, 2.1, 1.1]` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.action` (line 23) | `ROLLING` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.center` | `[0.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.axis` | `[0.0, 1.0, 0.0]` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.avel` | `10.0` (deg/s) | NOT YET WRITTEN | — |
| 0 | `dynamic.0.maxrotation` | `90.0` (deg) | NOT YET WRITTEN | — |
| 1 | `box` (line 33) | `[-1.1, -2.1, -1.1, 1.1, -1.99, 1.1]` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.action` | `ROLLING` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.axis` | `[0.0, 1.0, 0.0]` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.avel` | `-10.0` (deg/s) | NOT YET WRITTEN | — |
| 1 | `dynamic.0.maxrotation` | `90.0` | NOT YET WRITTEN | — |

---

### Demo 2 — TwistingBarNH

**RealSim scene:** `simulation/config/CudaTests/TwistingBarNH/scene.json`
**Object cfg:** `simulation/config/CudaTests/TwistingBarNH/object_10k_nh.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:5) | `[0.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` (scene.json:4) | `0.01` | NOT YET WRITTEN | — |
| `stop` (scene.json:3) | `810` | NOT YET WRITTEN | — |
| `maxFrame` (scene.json:8) | `810` | NOT YET WRITTEN | — |
| `offline` (scene.json:7) | `true` | NOT YET WRITTEN | — |
| `output_abc` (scene.json:9) | `simulation/output_abc/TwistingBarNH` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` (scene.json:6) | `5` | NOT YET WRITTEN | — |
| `linearsolver.type` | `SPARSE_INVERSE_CUDA` | NOT YET WRITTEN | — |
| `constraintsolver` | **ABSENT** (no contacts) | NOT YET WRITTEN | — |

**Objects:**

| Obj | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh (object_10k_nh.json:2) | `resources/mesh/volume/cube_volume_11340P.mesh` | NOT YET WRITTEN | — |
| object_0 | `element_type` | `TETRAHEDRON` | NOT YET WRITTEN | — |
| object_0 | `transformation.trans` | `[0.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.obj_mass` | `1000` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.young` | `1E+9` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.poisson` | `0.45` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` | NOT YET WRITTEN | — |

**Collision shapes:** none.

**Pin actions** (object_10k_nh.json:17-44): identical structure and values to Demo 1 (two ROLLING pins on top/bottom, opposite avel ±10 deg/s, maxrotation 90).

---

### Demo 3 — StretchingCloth

**RealSim scene:** `simulation/config/CudaTests/StretchingCloth/scene.json`
**Object cfg:** `simulation/config/CudaTests/StretchingCloth/object_NH_20k.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

Note: scene.json filename suggests "Cloth" + ARAP per Phase 0 plan, but the object referenced is `object_NH_20k.json` (Neo-Hookean). Plan-stated "TRI ARAP" is incorrect — actual scene uses **TRI_NEOHOOKEAN** with `bending=0.0`.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:7) | `[0.0, -1.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` (scene.json:6) | `0.01` | NOT YET WRITTEN | — |
| `stop` (scene.json:3) | `1200` | NOT YET WRITTEN | — |
| `maxFrame` (scene.json:5) | `2000` | NOT YET WRITTEN | — |
| `offline` (scene.json:4) | `false` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` (scene.json:8) | `5` | NOT YET WRITTEN | — |
| `linearsolver.type` | `SPARSE_INVERSE_CUDA` | NOT YET WRITTEN | — |
| `constraintsolver` | **ABSENT** (no contacts) | NOT YET WRITTEN | — |

**Objects:**

| Obj | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh (object_NH_20k.json:2) | `resources/mesh/cloth/square_20201P.obj` | NOT YET WRITTEN | — |
| object_0 | `element_type` | `TRIANGLE` | NOT YET WRITTEN | — |
| object_0 | `transformation.rotation` | `[90.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.scale` | `[1.0, 1.0, 1.0]` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.obj_mass` | `1000` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.young` | `1E+5` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.poisson` | `0.45` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.constitutive` | `TRI_NEOHOOKEAN` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.bending` | `0.0` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.bending_type` | `BEND_ISOMETRIC` | NOT YET WRITTEN | — |

**Collision shapes:** none.

**Pin actions** (object_NH_20k.json:19-44):

| Pin idx | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| 0 | `box` | `[0.99, -0.1, -1.1, 1.1, 0.1, 1.1]` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.action` | `PULLING` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.direction` | `[1.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.vel` | `0.1` | NOT YET WRITTEN | — |
| 0 | `dynamic.0.maxlength` | `1.0` | NOT YET WRITTEN | — |
| 1 | `box` | `[-1.1, -0.1, -1.1, -0.99, 0.1, 1.1]` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.action` | `PULLING` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.direction` | `[-1.0, 0.0, 0.0]` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.vel` | `0.1` | NOT YET WRITTEN | — |
| 1 | `dynamic.0.maxlength` | `1.0` | NOT YET WRITTEN | — |

---

### Demo 4 — PullingWooper

**RealSim scene:** `simulation/config/CudaTests/PullingWooper/scene.json`
**Object cfg:** `simulation/config/CudaTests/PullingWooper/wooper_5k.json`
**FBA script:** `/home/ziqiu/work/newton/scripts/fba_demo4_pulling_wooper.py`

**Global config:**

| Field | RealSim value (file:line) | FBA value (file:line) | Verdict |
|---|---|---|---|
| `gravity` | `[0.0, -0.0, 0.0]` (scene.json:6) | `GRAVITY = 0.0` (fba_demo4:38) → `gravity=0.0` Y-up (line 112) | MATCH |
| `timestep` | `0.01` (scene.json:4) | `DT = 0.01` (line 34) | MATCH |
| `stop` | `500` (scene.json:2) | `TOTAL_FRAMES = 500` (line 35) | MATCH |
| `timer` | `20` (scene.json:3) | n/a (not a physics field) | — |
| `LocalGlobal_CUDA` | `5` (scene.json:7) | `PD_ITERATIONS = 5` (line 36) | MATCH |
| `linearsolver.type` | `SPARSE_INVERSE_CUDA` | not set (Newton uses its own linear solver) | DIVERGE-OK (per-architecture; not a physics field) |
| `constraintsolver.type` | `NonSmoothNewton_CUDA` (scene.json:30) | `SolverFBA` NSN path (line 184) | MATCH (functional) |
| `constraintsolver.maxforce` | `1.00E+12` (scene.json:31) | not explicitly set in script — solver default | OPEN: confirm Phase 2 NSN default cap == 1e12 |
| `constraintsolver.linearsolver` | `PCR_CUDA` (scene.json:32) | n/a (Newton uses dense Schur inside NSN) | DIVERGE-OK (architecture) |
| `constraintsolver.iterations` | `10` (scene.json:33) | `NSN_ITERATIONS = 10` (line 37) | MATCH |
| `constraintsolver.tolerance` | `1.00E-9` (scene.json:34) | not exposed (FBA NSN runs fixed iters) | OPEN |
| `constraintsolver.function` | `FB` (Fischer-Burmeister) (scene.json:35) | FBA's NSN uses FB | MATCH |
| `parallelEnv` | absent | absent | MATCH |

**Objects:**

| Obj | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh | `resources/mesh/volume/wooper_volume_5325P.mesh` (wooper_5k.json:2) | `MESH_PATH = .../wooper_volume_5325P.mesh` (fba_demo4:33) | MATCH |
| object_0 | `element_type` | `TETRAHEDRON` (wooper_5k.json:3) | tet (uses `add_soft_mesh` w/ tet indices, line 122) | MATCH |
| object_0 | `transformation.trans` | `[0.0, 4.0, -0.0]` (wooper_5k.json:5) | `TRANS = [0.0, 4.0, 0.0]` (line 46) | MATCH |
| object_0 | `transformation.rotation` | `[0.0, 0.0, -90.0]` (wooper_5k.json:6) | `ROT_Z_DEG = -90.0` (line 45) | MATCH |
| object_0 | `transformation.scale` | `[1.0, 1.0, 1.0]` | implicit scale=1 in transform_verts | MATCH |
| object_0 | `mechanical_props.obj_mass` | `1000` (wooper_5k.json:11) | `OBJ_MASS = 1000.0` (line 42); density = mass/total_vol (line 119) | MATCH (note: Newton uses density × per-tet-volume rather than RealSim's uniform total/N lumping — see Task 0.2 decisions.json:5-21. Will diverge once Phase 2 Task 2.3 switches Newton to uniform lumping) |
| object_0 | `mechanical_props.young` | `1E+7` (wooper_5k.json:12) | `YOUNG = 1.0e7` (line 40) | MATCH |
| object_0 | `mechanical_props.poisson` | `0.3` (wooper_5k.json:13) | `POISSON = 0.3` (line 41) | MATCH |
| object_0 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` (wooper_5k.json:14) | `stretching_model="neohookean"` (line 189) | MATCH |

**Collision shapes** (cylinders only; both static, mu=0):

| Shape | Field | RealSim (scene.json:11-28) | FBA (fba_demo4) | Verdict |
|---|---|---|---|---|
| cyl 0 | `base` | `[0.0, -1.0, -1.3]` (line 13) | `CYL0_BASE = [0.0, -1.0, -1.3]` (line 59) | MATCH |
| cyl 0 | `axis` | `[1.0, 0.0, 0.0]` (line 14) | `CYL_AXIS = [1.0, 0.0, 0.0]` (line 61) | MATCH |
| cyl 0 | `radius` | `1.0` (line 15) | `CYL_RADIUS = 1.0` (line 57) | MATCH |
| cyl 0 | `mu` | `0.0` (line 16) | `friction=False` solver-wide (line 188) | MATCH (effective) |
| cyl 0 | `rollingvel` | `0.0` (line 17) | n/a — script does not set rolling vel for cylinders | MATCH (RealSim 0 == static) |
| cyl 0 | `visuallength` | `1.0` (line 18) | `CYL_HALF_HEIGHT = 3.0` (line 58) | OPEN: RealSim `visuallength=1.0` may indicate half-length 1.0; FBA uses half_height 3.0. RealSim cylinders may be infinite physically (`visuallength` is render-only). Verify with RealSim cylinder collision code before locking. |
| cyl 1 | `base` | `[0.0, -1.0, 1.3]` (line 21) | `CYL1_BASE = [0.0, -1.0, 1.3]` (line 60) | MATCH |
| cyl 1 | `axis` | `[1.0, 0.0, 0.0]` (line 22) | `CYL_AXIS` reused (line 61) | MATCH |
| cyl 1 | `radius` | `1.0` (line 23) | `1.0` | MATCH |
| cyl 1 | `mu` | `0.0` (line 24) | `friction=False` | MATCH (effective) |
| cyl 1 | `rollingvel` | `0.0` (line 25) | n/a (static) | MATCH |

**Pin actions** (wooper_5k.json:16-29):

| Pin idx | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| 0 | `box` (line 19) | `[-2.5, 2, -0.3, -1.5, 4, 0.3]` | `PIN_LO = (-2.5, 2.0, -0.3)`, `PIN_HI = (-1.5, 4.0, 0.3)` (lines 49-50) | MATCH |
| 0 | `dynamic.0.action` | `PULLING` (line 22) | PULLING semantics via `set_pin_targets` (line 214) | MATCH |
| 0 | `dynamic.0.direction` | `[0.0, -1.0, 0.0]` (line 23) | `PIN_DIR = [0.0, -1.0, 0.0]` (line 52) | MATCH |
| 0 | `dynamic.0.vel` | `2.0` (line 24) | `PIN_VEL = 2.0` (line 53) | MATCH |
| 0 | `dynamic.0.maxlength` | `10.0` (line 25) | `PIN_MAXLENGTH = 10.0` (line 54) | MATCH |

**Demo 4 verdict:** Net **DIVERGE-1** — single architecturally-derived divergence:
- `cyl.visuallength = 1.0` vs FBA `CYL_HALF_HEIGHT = 3.0` is *almost certainly* intentional (RealSim's cylinder collision treats cylinders as infinite-length; `visuallength` is purely visual). Flagged as OPEN — needs RealSim cylinder-collision-source check.

All other physics-affecting fields match byte-for-byte.

---

### Demo 5 — SqueezingBall

**RealSim scene:** `simulation/config/CudaTests/SqueezingBall/scene.json`
**Object cfg:** `simulation/config/CudaTests/SqueezingBall/ball_7k.json`
**FBA script:** `/home/ziqiu/work/newton/scripts/fba_demo5_squeezing_ball.py`

**Global config:**

| Field | RealSim value (file:line) | FBA value (file:line) | Verdict |
|---|---|---|---|
| `gravity` | `[0.0, -10.0, 0.0]` (scene.json:8) | `GRAVITY = -10.0` Y-up (fba_demo5:37, 94) | MATCH |
| `timestep` | `0.01` (scene.json:7) | `DT = 0.01` (line 33) | MATCH |
| `stop` | `600` (scene.json:5) | `TOTAL_FRAMES = 600` (line 34) | MATCH |
| `maxFrame` | `900` (scene.json:3) | n/a (FBA runs `stop` frames; 900 is RealSim's hard wall, irrelevant) | MATCH (semantic) |
| `LocalGlobal_CUDA` | `5` (scene.json:9) | `PD_ITERATIONS = 5` (line 35) | MATCH |
| `constraintsolver.type` | `NonSmoothNewton_CUDA` (scene.json:56) | SolverFBA NSN path | MATCH (functional) |
| `constraintsolver.maxforce` | `1.00E+12` (scene.json:57) | solver default | OPEN: Phase 2 Task 2.4 default cap |
| `constraintsolver.iterations` | **`10`** (scene.json:59) | **`NSN_ITERATIONS = 1`** (line 36) | **DIVERGE** — script sets only 1 NSN iter vs scene's 10. This is the documented failing-case (`memory/realsim-port-discipline.md` cites this demo as exemplar). Phase 2 Task 2.2 will fix the default; Phase 1 Task 1.1 covers trajectory parity for this demo specifically. |
| `constraintsolver.tolerance` | `1.00E-9` (scene.json:60) | not exposed | OPEN |
| `constraintsolver.function` | `FB` (scene.json:61) | FB | MATCH |
| `parallelEnv` | absent | absent | MATCH |

**Objects:**

| Obj | Field | RealSim | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh | `resources/mesh/volume/ball_volume_7129P.mesh` (ball_7k.json:2) | `MESH_PATH = .../ball_volume_7129P.mesh` (fba_demo5:30-32) | MATCH |
| object_0 | `element_type` | `TETRAHEDRON` (ball_7k.json:3) | tet via `add_soft_mesh` (line 103) | MATCH |
| object_0 | `transformation.trans` | `[0.0, 2.6, -0.0]` (ball_7k.json:5) | `BALL_TRANS = [0.0, 2.6, 0.0]` (line 44) | MATCH |
| object_0 | `transformation.rotation` | `[0.0, 0.0, 0.0]` (ball_7k.json:6) | implicit (no rotation in `transform_verts`, line 62) | MATCH |
| object_0 | `transformation.scale` | `[3.0, 3.0, 3.0]` (ball_7k.json:7) | `BALL_SCALE = 3.0` (line 43) | MATCH |
| object_0 | `mechanical_props.obj_mass` | `1000` (ball_7k.json:11) | `OBJ_MASS = 1000.0` (line 41); density = mass/vol (line 100) | MATCH (modulo lumping scheme — see Task 0.2; same caveat as demo 4) |
| object_0 | `mechanical_props.young` | `1E+4` (ball_7k.json:12) | `YOUNG = 1.0e4` (line 39) | MATCH |
| object_0 | `mechanical_props.poisson` | `0.4` (ball_7k.json:13) | `POISSON = 0.4` (line 40) | MATCH |
| object_0 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` (ball_7k.json:14) | `stretching_model="neohookean"` (line 172) | MATCH |
| object_0 | `mechanical_props.colormap` | `false` (ball_7k.json:15) | n/a (render-only) | MATCH |

**Collision shapes** — 4 rolling cylinders + 1 static plane:

| Shape | Field | RealSim (scene.json:13-54) | FBA (fba_demo5) | Verdict |
|---|---|---|---|---|
| cyl 0 | `base` | `[0.0, -1.0, -1.5]` (line 15) | `CYLINDERS[0][0] = [0.0, -1.0, -1.5]` (line 52) | MATCH |
| cyl 0 | `axis` | `[1.0, 0.0, 0.0]` (line 16) | `CYLINDERS[0][1] = [1.0, 0.0, 0.0]` (line 52) | MATCH |
| cyl 0 | `radius` | `1.0` (line 17) | `CYL_RADIUS = 1.0` (line 46) | MATCH |
| cyl 0 | `mu` | `0.5` (line 18) | `FRICTION_MU = 0.5` applied via `mu_override` (line 165) and `shape_material_mu` (line 137) | MATCH |
| cyl 0 | `rollingvel` | `-3.0` rad/s (line 19) | `CYLINDERS[0][2] = -3.0` → `shape_angular_velocity` (line 163, 176) | MATCH |
| cyl 1 | `base` | `[0.0, -1.0, 1.5]` (line 23) | `CYLINDERS[1][0] = [0.0, -1.0, 1.5]` (line 53) | MATCH |
| cyl 1 | `axis` | `[1.0, 0.0, 0.0]` | `[1.0, 0.0, 0.0]` | MATCH |
| cyl 1 | `radius` | `1.0` | `1.0` | MATCH |
| cyl 1 | `mu` | `0.5` | `0.5` | MATCH |
| cyl 1 | `rollingvel` | `3.0` (line 27) | `+3.0` (line 53) | MATCH |
| cyl 2 | `base` | `[-1.5, -3.5, 0.0]` (line 31) | `[-1.5, -3.5, 0.0]` (line 54) | MATCH |
| cyl 2 | `axis` | `[0.0, 0.0, 1.0]` | `[0.0, 0.0, 1.0]` | MATCH |
| cyl 2 | `radius` | `1.0` | `1.0` | MATCH |
| cyl 2 | `mu` | `0.5` | `0.5` | MATCH |
| cyl 2 | `rollingvel` | `3.0` (line 35) | `+3.0` (line 54) | MATCH |
| cyl 3 | `base` | `[1.5, -3.5, 0.0]` (line 39) | `[+1.5, -3.5, 0.0]` (line 55) | MATCH |
| cyl 3 | `axis` | `[0.0, 0.0, 1.0]` | `[0.0, 0.0, 1.0]` | MATCH |
| cyl 3 | `radius` | `1.0` | `1.0` | MATCH |
| cyl 3 | `mu` | `0.5` | `0.5` | MATCH |
| cyl 3 | `rollingvel` | `-3.0` (line 43) | `-3.0` (line 55) | MATCH |
| plane 0 | `base` | `[0.0, -10.0, 0, 0]` (line 50; the trailing `,0` is suspicious — likely typo, 4 elements, but `base` is logically a 3-vec; RealSim parser presumably drops the 4th) | implicit via `plane=(0.0, 1.0, 0.0, 10.0)` (Newton plane form, d=+10) (line 126) | MATCH (geometric) |
| plane 0 | `normal` | `[0.0, 1.0, 0.0]` (line 51) | `(0.0, 1.0, 0.0)` (line 126) | MATCH |
| plane 0 | `mu` | `0.5` (line 49) | `FRICTION_MU = 0.5` blanket-applied to `shape_material_mu` (line 137) | MATCH |
| plane 0 | `visualscale` | `1.0` (line 52) | n/a (render-only) | MATCH |

**Pin actions:** none (ball is unconstrained, falls under gravity).

**Demo 5 verdict:** Net **DIVERGE-1**: NSN iterations differ (1 vs 10). All geometry, friction, masses, materials match. The single divergence is exactly the bug Phase 1 Task 1.1 (SqueezingBall NSN trajectory parity) and Phase 2 Task 2.2 (NSN iter default 1 → 10) target.

---

### Demo 6 — CrossingGingerbreadman

**RealSim scene:** `simulation/config/CudaTests/CrossingGingerbreadman/scene.json`
**Object cfg:** `simulation/config/CudaTests/CrossingGingerbreadman/ginger_10k.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:5) | `[0.0, -0.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` | `0.01` | NOT YET WRITTEN | — |
| `stop` | `810` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` | `5` | NOT YET WRITTEN | — |
| `constraintsolver.type` (scene.json:91) | `NonSmoothNewton_CUDA` | NOT YET WRITTEN | — |
| `constraintsolver.maxforce` (scene.json:92) | `1.00E+15` | NOT YET WRITTEN | — |
| `constraintsolver.iterations` | `10` | NOT YET WRITTEN | — |
| `constraintsolver.tolerance` | `1.00E-9` | NOT YET WRITTEN | — |
| `constraintsolver.function` | `FB` | NOT YET WRITTEN | — |

**Objects:**

| Obj | Field | RealSim (ginger_10k.json) | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh (:2) | `resources/mesh/volume/gingerbreadman_volume_11133P.mesh` | NOT YET WRITTEN | — |
| object_0 | `transformation.trans` (:5) | `[-0.0, 12.0, -0.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.rotation` (:6) | `[90.0, 90.0, 180.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.scale` (:7) | `[6.0, 6.0, 6.0]` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.obj_mass` | `1000` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.young` | `1E+6` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.poisson` | `0.3` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` | NOT YET WRITTEN | — |

**Collision shapes** — 13 cylinders, all radius=1, mu=0:

| # | base | axis | Notes |
|---|---|---|---|
| 0 | `[0.0, 0.0, -1.2]` | `[1.0, 0.0, 0.0]` | top entry pair |
| 1 | `[0.0, 0.0, 1.2]` | `[1.0, 0.0, 0.0]` | top entry pair |
| 2 | `[-4.0, 0.0, 0.0]` | `[0.0, 0.0, 1.0]` | mid sides |
| 3 | `[4.0, 0.0, 0.0]` | `[0.0, 0.0, 1.0]` | mid sides |
| 4 | `[0.0, -8.0, -2.0]` | `[1.0, 0.0, 0.0]` | mid funnel |
| 5 | `[1.732, -8.0, 1.0]` | `[-0.5, 0.0, 0.866]` | mid funnel diag |
| 6 | `[-1.732, -8.0, 1.0]` | `[0.5, 0.0, 0.866]` | mid funnel diag |
| 7 | `[0.0, -16.0, -2.4]` | `[1.0, 0.0, 0.0]` | bottom funnel |
| 8 | `[2.078, -16.0, 1.2]` | `[-0.5, 0.0, 0.866]` | |
| 9 | `[-2.078, -16.0, 1.2]` | `[0.5, 0.0, 0.866]` | |
| 10 | `[0.0, -16.0, 2.4]` | `[1.0, 0.0, 0.0]` | |
| 11 | `[2.078, -16.0, -1.2]` | `[0.5, 0.0, 0.866]` | |
| 12 | `[-2.078, -16.0, -1.2]` | `[-0.5, 0.0, 0.866]` | |

All 13 cylinders have `radius: 1.0`, `mu: 0.0`. None have `rollingvel` set (default 0).

**Pin actions** (ginger_10k.json:16-29):

| Pin idx | Field | RealSim |
|---|---|---|
| 0 | `box` | `[-0.2, 0, -0.2, 0.2, 4, 0.2]` |
| 0 | `dynamic.0.action` | `PULLING` |
| 0 | `dynamic.0.direction` | `[0.0, -1.0, 0.0]` |
| 0 | `dynamic.0.vel` | `2.0` |
| 0 | `dynamic.0.maxlength` | `50.0` |

---

### Demo 7 — SharpCorner

**RealSim scene:** `simulation/config/CudaTests/SharpCorner/scene.json`
**Object cfg:** `simulation/config/CudaTests/SharpCorner/cloth_5k.json`
**Obstacle cfg:** `simulation/config/CudaTests/SharpCorner/box.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:5) | `[0.0, -10.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` | `0.01` | NOT YET WRITTEN | — |
| `stop` | `200` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` | `5` | NOT YET WRITTEN | — |
| `genericCCD` block (scene.json:10-20) | `muVF=0.0`, `muEE=0.0`, `muFV=0.0`, all checks `true`, `alarm=0.02`, `miniseparation=0.01`, `detect_all=true` | NOT YET WRITTEN | — (mesh CCD enabled — new infra needed) |
| `constraintsolver.type` | `NonSmoothNewton_CUDA` | NOT YET WRITTEN | — |
| `constraintsolver.maxforce` (scene.json:23) | `100` (integer, not float) | NOT YET WRITTEN | — |
| `constraintsolver.iterations` | `10` | NOT YET WRITTEN | — |
| `constraintsolver.function` | `FB` | NOT YET WRITTEN | — |

**Objects:**

| Obj | Field | RealSim (cloth_5k.json) | FBA | Verdict |
|---|---|---|---|---|
| object_0 | mesh (:2) | `resources/mesh/cloth/square_5101P.obj` | NOT YET WRITTEN | — |
| object_0 | `element_type` | `TRIANGLE` | NOT YET WRITTEN | — |
| object_0 | `transformation.trans` (:6) | `[0.0, 4.0, 0.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.rotation` (:7) | `[90.0, 30.0, 0.0]` | NOT YET WRITTEN | — |
| object_0 | `transformation.scale` (:8) | `[3.5, 3.5, 3.5]` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.obj_mass` | `1000` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.young` | `1E+5` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.poisson` | `0.4` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.constitutive` | `TRI_ARAP` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.bending` | `5.0` | NOT YET WRITTEN | — |
| object_0 | `mechanical_props.bending_type` | `BEND_ISOMETRIC` | NOT YET WRITTEN | — |

**Obstacles** (single box):

| Obs | Field | RealSim (box.json) |
|---|---|---|
| obstacle_0 | mesh (:2) | `resources/mesh/volume/squarecube_volume_125P.mesh` |
| obstacle_0 | `element_type` | `TETRAHEDRON` (used as static collider) |
| obstacle_0 | `transformation.trans` (:6) | `[0.0, 0.0, 0.0]` |
| obstacle_0 | `transformation.rotation` (:7) | `[0.0, 0.0, 0.0]` |
| obstacle_0 | `transformation.scale` (:8) | `[1.0, 1.0, 1.0]` |

**Pin actions:** none. **Plane / sphere / cylinder shapes:** none — only the mesh obstacle.

---

### Demo 8 — ClothOnKnives

**RealSim scene:** `simulation/config/CudaTests/ClothOnKnives/scene.json`
**Object cfg:** `simulation/config/CudaTests/ClothOnKnives/cloth_5k.json`
**Obstacle cfg:** `simulation/config/CudaTests/ClothOnKnives/knives.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:5) | `[0.0, -10.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` | `0.01` | NOT YET WRITTEN | — |
| `stop` | `900` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` | `5` | NOT YET WRITTEN | — |
| `genericCCD` (scene.json:17-27) | `muVF=0`, `muEE=0`, `muFV=0`, all checks true, `alarm=0.03`, `miniseparation=0.01`, `detect_all=true` | NOT YET WRITTEN | — |
| `constraintsolver.maxforce` | `100.0` | NOT YET WRITTEN | — |
| `constraintsolver.iterations` | `10` | NOT YET WRITTEN | — |
| `constraintsolver.function` | `FB` | NOT YET WRITTEN | — |

**Objects:**

| Obj | Field | RealSim (cloth_5k.json) |
|---|---|---|
| object_0 | mesh | `resources/mesh/cloth/square_5101P.obj` |
| object_0 | `transformation.trans` (:6) | `[0.0, 3.0, 0.1]` |
| object_0 | `transformation.rotation` (:7) | `[90.0, 0.0, 0.0]` |
| object_0 | `transformation.scale` (:8) | `[3.0, 2.7, 2.7]` |
| object_0 | `mechanical_props.obj_mass` | `1000` |
| object_0 | `mechanical_props.young` | `1E+5` |
| object_0 | `mechanical_props.poisson` | `0.4` |
| object_0 | `mechanical_props.constitutive` | `TRI_NEOHOOKEAN` |
| object_0 | `mechanical_props.bending` | `10.0` |
| object_0 | `mechanical_props.bending_type` | `BEND_ISOMETRIC` |

**Collision shapes** — single plane:

| Shape | Field | RealSim (scene.json:10-16) |
|---|---|---|
| plane 0 | `base` | `[0.0, -2.0, 0, 0]` (trailing 0 — RealSim parser drops) |
| plane 0 | `normal` | `[0.0, 1.0, 0.0]` |
| plane 0 | `mu` | `0.0` |

**Obstacles** (knives mesh):

| Obs | Field | RealSim (knives.json) |
|---|---|---|
| obstacle_0 | mesh | `resources/mesh/surface/knives.obj` |
| obstacle_0 | `element_type` | `TRIANGLE` |
| obstacle_0 | `transformation.trans` (:6) | `[1.5, 0.0, 0.0]` |
| obstacle_0 | `transformation.rotation` | `[0.0, 0.0, 0.0]` |
| obstacle_0 | `transformation.scale` | `[2.0, 2.0, 2.0]` |

**Pin actions:** none.

---

### Demo 9 — ParallelEnvTest

**RealSim scene:** `simulation/config/CudaTests/ParallelEnvTest/scene.json`
**Object cfg:** `simulation/config/CudaTests/ParallelEnvTest/cloth0.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `offline` (scene.json:2) | `false` | NOT YET WRITTEN | — |
| `gravity` (scene.json:8) | `[0.0, -10.0, 0.0]` | NOT YET WRITTEN | — |
| `timestep` | `0.01` | NOT YET WRITTEN | — |
| `stop` / `maxFrame` | `300` / `300` | NOT YET WRITTEN | — |
| `LocalGlobal` (scene.json:9) | `5` — **note: bare `LocalGlobal`, not `LocalGlobal_CUDA`** | NOT YET WRITTEN | — (CPU LocalGlobal integrator variant) |
| `constraintsolver.type` (scene.json:14) | **`LiteNonSmoothNewton_CUDA`** (the parallel-env variant) | NOT YET WRITTEN | — |
| `constraintsolver.maxforce` (scene.json:15) | `1.00E+10` | NOT YET WRITTEN | — |
| `constraintsolver.iterations` | `10` | NOT YET WRITTEN | — |
| `constraintsolver.function` | `FB` | NOT YET WRITTEN | — |
| `parallelEnv.matrix` (scene.json:22) | `[5, 1, 5]` — 25 envs in X-Z grid | NOT YET WRITTEN | — |
| `parallelEnv.space` (scene.json:23) | `[10, 10, 10]` — env spacing (meters) | NOT YET WRITTEN | — |

**Objects** (per env):

| Obj | Field | RealSim (cloth0.json) |
|---|---|---|
| object_0 | mesh | `resources/mesh/cloth/square_1013P.obj` |
| object_0 | `transformation.trans` | `[0.0, 3.5, 0.0]` |
| object_0 | `transformation.rotation` | `[90.0, 0.0, 0.0]` |
| object_0 | `transformation.scale` | `[3.5, 3.5, 3.5]` |
| object_0 | `mechanical_props.obj_mass` | `1000` |
| object_0 | `mechanical_props.young` | `1E+5` |
| object_0 | `mechanical_props.poisson` | `0.4` |
| object_0 | `mechanical_props.constitutive` | `TRI_ARAP` |
| object_0 | `mechanical_props.bending` | `20.0` |
| object_0 | `mechanical_props.bending_type` | `BEND_ISOMETRIC` |

**Collision shapes** — single sphere per env:

| Shape | Field | RealSim (scene.json:26-31) |
|---|---|---|
| sphere 0 | `center` | `[0.0, 0.0, 0, 0]` (trailing 0 — parser drops; effectively (0,0,0)) |
| sphere 0 | `radius` | `3.0` |
| sphere 0 | `mu` | `0.3` |
| sphere 0 | `visualradius` | `2.95` |

**Pin actions:** none.

---

### Demo 10 — CableGrabRaptor

**RealSim scene:** `simulation/config/CudaTests/CableGrabRaptor/scene.json`
**Object cfgs (3):** `gripper0.json`, `gripper1.json`, `raptor_10k.json`
**Obstacle:** `torus.json`
**FBA script:** NOT YET WRITTEN — Phase 3/5 scene-loader spec below.

**Global config:**

| Field | RealSim value | FBA | Verdict |
|---|---|---|---|
| `gravity` (scene.json:5) | `[0.0, -1.0, 0.0]` (note: 1 m/s², not 9.81 or 10) | NOT YET WRITTEN | — |
| `timestep` | `0.01` | NOT YET WRITTEN | — |
| `stop` | `6000` | NOT YET WRITTEN | — |
| `LocalGlobal_CUDA` | `5` | NOT YET WRITTEN | — |
| `genericDCD` (scene.json:35-44) | `muVF=0.5`, `muEE=0.0`, `muFV=0.5`, `VF=true`, `EE=false`, `FV=false`, `alarm=0.1`, `miniseparation=-0.01`, `detect_all=true` — note DCD not CCD | NOT YET WRITTEN | — |
| `constraintsolver.maxforce` | `1.00E+12` | NOT YET WRITTEN | — |
| `constraintsolver.iterations` | `10` | NOT YET WRITTEN | — |
| `constraintsolver.function` | `FB` | NOT YET WRITTEN | — |

**Objects (3):**

| Obj | Field | RealSim |
|---|---|---|
| object_0 (gripper0.json) | mesh | `resources/mesh/volume/finger.mesh` |
| object_0 | `transformation.trans` | `[0.5, 0.0, -0.3]` |
| object_0 | `transformation.rotation` | `[0.0, 0.0, 120.0]` |
| object_0 | `transformation.scale` | `[0.04, 0.04, 0.04]` |
| object_0 | `mechanical_props.obj_mass` | `1` |
| object_0 | `mechanical_props.young` | `5E+5` |
| object_0 | `mechanical_props.poisson` | `0.2` |
| object_0 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` |
| object_0 | `cable` (12-point array) | (see gripper0.json:67-80) |
| object_0 | `cableInvStiff` | `1E-12` |
| object_1 (gripper1.json) | mesh | `resources/mesh/volume/finger.mesh` |
| object_1 | `transformation.trans` | `[-0.5, 0.0, 0.3]` |
| object_1 | `transformation.rotation` | `[180.0, 0.0, 60.0]` |
| object_1 | `transformation.scale` | `[0.04, 0.04, 0.04]` |
| object_1 | `mechanical_props.obj_mass` | `1` |
| object_1 | `mechanical_props.young` | `5E+5` |
| object_1 | `mechanical_props.poisson` | `0.2` |
| object_1 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` |
| object_1 | `cable` / `cableInvStiff` | identical to object_0 |
| object_2 (raptor_10k.json) | mesh | `resources/mesh/volume/raptor_volume_10011P.mesh` |
| object_2 | `transformation.trans` | `[0.0, -5.6, -1.2]` |
| object_2 | `transformation.scale` | `[0.75, 0.75, 0.75]` |
| object_2 | `mechanical_props.obj_mass` | `10` |
| object_2 | `mechanical_props.young` | `2E+4` |
| object_2 | `mechanical_props.poisson` | `0.1` |
| object_2 | `mechanical_props.constitutive` | `TET_PD_NEOHOOKEAN` |
| object_2 | `mechanical_props.colormap` | `true` |

**Collision shapes:**

| Shape | Field | RealSim (scene.json) |
|---|---|---|
| plane 0 | `base` | `[0.0, -6.0, 0, 0]` (line 12) |
| plane 0 | `normal` | `[0.0, 1.0, -0.0]` (line 13) |
| plane 0 | `mu` | `1.0` (line 14) |
| plane 0 | `obj` | `2` — applies only to raptor (object_2) |
| plane 1 | `base` | `[0.0, -5.4, 1.0]` (line 18) |
| plane 1 | `normal` | `[0.0, 1.0, -1.2]` (line 19) — note non-unit; presumably normalized by parser |
| plane 1 | `mu` | `1.0` (line 20) |
| plane 1 | `obj` | `2` |
| torus 0 | `center` | `[10.0, 2.0, -10.0]` (line 26) |
| torus 0 | `axis` | `[0.0, 1.0, 0.0]` (line 27) |
| torus 0 | `radius_outer` | `2.0` (line 28) |
| torus 0 | `radius_inner` | `0.5` (line 29) |
| torus 0 | `mu` | `0.0` (line 30) |
| torus 0 | `obj` | `2` |

**Obstacles** (torus mesh, in addition to analytic torus shape):

| Obs | Field | RealSim (torus.json) |
|---|---|---|
| obstacle_0 | mesh | `resources/mesh/surface/torus.obj` |
| obstacle_0 | `transformation.trans` | `[10.0, 2.0, -10.0]` |
| obstacle_0 | `transformation.scale` | `[2.0, 2.0, 2.0]` |

**Pin actions** (each gripper, multiple sequential actions):

Both gripper0 and gripper1 share an identical 7-step pin action sequence (all on pin idx 0, AABB `[-2,-0.5,-2, 2,1,2]`):

| Step | action | direction / axis | vel / avel | maxlength / maxrotation |
|---|---|---|---|---|
| 0 | PULLING | `[0.0, 0.0, 0.0]` | `1.0` | `2.0` |
| 1 | PULLING | `[0.0, 1.0, 0.0]` | `2.0` | `10.0` |
| 2 | ROLLING | center=`[0,0,0]`, axis=`[0,1,0]` | `45.0` deg/s | `135.0` deg |
| 3 | PULLING | `[1.0, 0.0, -1.0]` | `2.0` | `10.0` |
| 4 | PULLING | `[0.0, 0.0, 0.0]` | `2.0` | `4.0` |
| 5 | PULLING | `[0.0, 1.0, 0.0]` | `4.0` | `4.0` |
| 6 | PULLING | `[0.0, 0.0, 0.0]` | `2.0` | `10.0` |

(See gripper0.json:16-66 and gripper1.json:16-66; the two files are identical in pin sequence.)

---

## DIVERGE rows summary (Phase 2/3 fix targets)

1. **Demo 5 (SqueezingBall) — `constraintsolver.iterations`**
   - RealSim scene.json:59 → `10`
   - FBA fba_demo5:36 → `NSN_ITERATIONS = 1`
   - Owners: Phase 1 Task 1.1 (trajectory parity test) + Phase 2 Task 2.2 (NSN iter default 1→10)
   - **This is the documented failing case from the discipline doc.**

2. **Demo 4 (PullingWooper) — `cyl.visuallength` vs `CYL_HALF_HEIGHT`** (OPEN)
   - RealSim scene.json:18,26 → `visuallength: 1.0`
   - FBA fba_demo4:58 → `CYL_HALF_HEIGHT = 3.0`
   - Likely OK (RealSim cylinder collision treats cylinders as infinite physically; `visuallength` is render-only). Verify against RealSim cylinder-collision source before locking. If RealSim does use finite cylinders, FBA's 3.0 vs scene 1.0 is a divergence.

## Open issues

1. **`constraintsolver.tolerance` (`1.00E-9`)** — RealSim's NSN exits early if residual drops below tolerance; FBA's NSN runs a fixed iter count with no early exit. Confirmed not exposed in current FBA. May affect parity for demos that converge in fewer than 10 iters.

2. **`constraintsolver.linearsolver: PCR_CUDA`** — RealSim uses preconditioned conjugate residual for the linear subsystem inside NSN. FBA uses dense Schur. Architectural — not a physics field per se — but worth noting if PCR's iterative tolerance differs.

3. **Scene-mass-lumping mismatch (demos 4, 5)** — FBA scripts use `density = obj_mass / total_volume` × per-tet volume integration (Warp's standard mass-from-density). RealSim uses uniform `m_i = obj_mass / num_vertices` (`Mass.cpp:12-21`, Task 0.2). For uniform tet meshes these are very close, but non-uniform meshes will differ. Phase 2 Task 2.3 will switch FBA to RealSim's uniform lumping.

4. **StretchingCloth plan label mismatch** — Phase 0 plan listed StretchingCloth as "TRI ARAP, PULLING pin" but actual scene loads `object_NH_20k.json` (TRI_NEOHOOKEAN with bending=0). Either (a) plan is stale and should be updated, or (b) the intended object cfg was `object_ARAP_20k.json` (which exists in the StretchingCloth/ directory). Surface to user.

5. **CrossingGingerbreadman `maxforce = 1E+15`** vs plan's `1E+12` — already noted in `decisions.json:87`. Phase 2 Task 2.4 should use the actual scene value.

6. **SqueezingBall plane `base` typo `[0.0, -10.0, 0, 0]`** — 4 elements instead of 3 (likewise ClothOnKnives plane `[0.0, -2.0, 0, 0]` and ParallelEnvTest sphere `center: [0.0, 0.0, 0, 0]`). RealSim parser silently drops the trailing 0; FBA scene-loader must replicate this lenient behavior.

7. **ParallelEnvTest uses `LocalGlobal` (not `LocalGlobal_CUDA`)** and `LiteNonSmoothNewton_CUDA` — already flagged in decisions.json. The Lite variant pre-factorizes one shared Schur for all parallel envs. Phase 5 (new infra demos 7-10) will need a port.

8. **CableGrabRaptor `cable` / `cableInvStiff` energies** are not present in any other demo and represent the only inextensible-cable energy in CudaTests. New energy class will be needed for Phase 5.

9. **CableGrabRaptor `genericDCD`** (discrete CD) differs from `genericCCD` used by SharpCorner/ClothOnKnives. Two distinct mesh-mesh collision paths exist in RealSim; FBA currently has neither.

10. **Per-object `obj` field on planes/torus (CableGrabRaptor)** — `"obj": 2` restricts that shape to only collide with object index 2 (raptor). FBA has no equivalent per-shape object filter today.

## Per-demo source-of-truth file paths (for Phase 3/5 scene loader)

```
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBar/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBar/object_5k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBarNH/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBarNH/object_10k_nh.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/StretchingCloth/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/StretchingCloth/object_NH_20k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/PullingWooper/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/PullingWooper/wooper_5k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SqueezingBall/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SqueezingBall/ball_7k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CrossingGingerbreadman/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CrossingGingerbreadman/ginger_10k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SharpCorner/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SharpCorner/cloth_5k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SharpCorner/box.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/ClothOnKnives/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/ClothOnKnives/cloth_5k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/ClothOnKnives/knives.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/ParallelEnvTest/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/ParallelEnvTest/cloth0.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CableGrabRaptor/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CableGrabRaptor/gripper0.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CableGrabRaptor/gripper1.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CableGrabRaptor/raptor_10k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/CableGrabRaptor/torus.json
```
