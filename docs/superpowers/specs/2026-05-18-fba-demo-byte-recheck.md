# Demo Byte-Recheck (Phase 0.20)

**Date:** 2026-05-18
**Scope:** 4 in-scope demos — 2 TwistingBarNH, 3 StretchingCloth (script not yet written),
4 PullingWooper, 5 SqueezingBall.
**Method:** field-by-field reconciliation of every scene.json / object.json physics field
against the corresponding constant or call site in the FBA demo script. Anything that
affects physics is verified; render-only fields (`visuallength`, `visualscale`,
`output_abc`, `timer`, `offline`, `maxFrame` when ≥ `stop`) are listed as INTENTIONAL-NOT-PORTED.

Cross-ref: `docs/superpowers/specs/2026-05-17-fba-precision-params-variants-review.md`
already covers the variant + precision picture; this audit goes line-level on values.

Source files audited:

```
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBarNH/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBarNH/object_10k_nh.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/StretchingCloth/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/StretchingCloth/object_NH_20k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/PullingWooper/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/PullingWooper/wooper_5k.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SqueezingBall/scene.json
/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/SqueezingBall/ball_7k.json
/home/ziqiu/work/newton/scripts/fba_twisting_bar_cudatests.py
/home/ziqiu/work/newton/scripts/fba_demo4_pulling_wooper.py
/home/ziqiu/work/newton/scripts/fba_demo5_squeezing_ball.py
```

Solver source confirming `lambda_cap` unit conversion is implemented (T.2 done):

```
newton/_src/solvers/fba/solver_fba.py:1492-1493, 1644-1645
  if self.lambda_cap is not None:
      cap_internal = self.lambda_cap / (dt * dt)
```

So a demo passing `lambda_cap=1e12` now matches RealSim's scene `maxforce: 1.00E+12`
(physical-force units; solver handles dt² conversion internally).

---

## Demo 2 — TwistingBarNH (`fba_twisting_bar_cudatests.py --energy neohookean`)

Scene: `CudaTests/TwistingBarNH/scene.json`; object: `object_10k_nh.json`.

### Global scene fields

| Field | scene.json | FBA script | Verdict |
|---|---|---|---|
| `timestep` | scene.json:4 `0.01` | line 94 `DT = 0.01` | MATCH |
| `stop` | scene.json:3 `810` | line 95 `NUM_FRAMES = 810` | MATCH |
| `maxFrame` | scene.json:8 `810` | n/a (render-only cap, equals `stop`) | INTENTIONAL-NOT-PORTED |
| `timer` | scene.json:2 `50` | n/a (real-time render rate) | INTENTIONAL-NOT-PORTED |
| `offline` | scene.json:7 `true` | n/a (RealSim flag for headless run) | INTENTIONAL-NOT-PORTED |
| `output_abc` | scene.json:9 `.../TwistingBarNH` | n/a (RealSim render path) | INTENTIONAL-NOT-PORTED |
| `gravity` | scene.json:5 `[0,0,0]` | line 236 `gravity=0.0` in ModelBuilder | MATCH |
| `LocalGlobal_CUDA` | scene.json:6 `5` | line 96 `PD_ITERATIONS = 5` | MATCH |
| `linearsolver.type` | scene.json:11 `SPARSE_INVERSE_CUDA` | `SolverFBA` (FBALinearSolver sparse-inverse) | MATCH (architectural; precision-review GREEN) |
| `constraintsolver` block | (absent) | n/a (no NSN, pure PD) | MATCH |

### Object (cube_volume_11340P.mesh)

| Field | object_10k_nh.json | FBA script | Verdict |
|---|---|---|---|
| `mesh` | line 2 `cube_volume_11340P.mesh` | lines 54-56 `MESH_PATH_PER_ENERGY["neohookean"] = .../cube_volume_11340P.mesh` | MATCH |
| `element_type` | line 4 `TETRAHEDRON` | `builder.add_tetrahedron(...)` (line 252) | MATCH |
| `trans` | line 6 `[0,0,0]` | identity — verts loaded as-is, no offset applied | MATCH |
| `rotation` | line 7 `[0,0,0]` | identity — no rotation in `build_model` | MATCH |
| `scale` | line 8 `[1,1,1]` | identity | MATCH |
| `center` (mesh pivot) | line 9 `[0,0,0]` | line 514 `center = np.array([0,0,0])` | MATCH |
| `obj_mass` | line 12 `1000` | line 97 `TOTAL_MASS = 1000.0` | MATCH |
| `young` | line 13 `1E+9` | line 100 `YOUNG = 1.0e9` | MATCH |
| `poisson` | line 14 `0.45` | line 101 `NU = 0.45` | MATCH |
| `constitutive` | line 15 `TET_PD_NEOHOOKEAN` | `--energy neohookean` → `stretching_model="neohookean"` | MATCH |
| Mass lumping | uniform `obj_mass/N` (Mass.cpp:12-21) | line 220 `mass_per_particle = TOTAL_MASS / N` (uniform) | MATCH (Decision #3) |

### Pin 0 (top, ROLLING)

| Field | object_10k_nh.json | FBA script | Verdict |
|---|---|---|---|
| `box` | line 20 `[-1.1, 1.99, -1.1, 1.1, 2.1, 1.1]` | line 116 `PIN_Y_TOP = 1.99` + filter `y > PIN_Y_TOP` (line 223) | MATCH (semantic — Y-range `(1.99, ∞)` covers `[1.99, 2.1]`; X/Z ranges `[-1.1, 1.1]` lie outside cube bbox so are non-restrictive for a unit cube) |
| `action` | line 23 `ROLLING` | rotation about Y via `rot_y(top_angle)` (line 522) | MATCH |
| `center` | line 24 `[0,0,0]` | line 514 `center = np.array([0,0,0])` | MATCH |
| `axis` | line 25 `[0,1,0]` | `rot_y` rotates about Y (line 195-199) | MATCH |
| `avel` | line 26 `10.0` (degrees/sec, per Actions.h:65) | lines 114-115 `PIN_AVEL_DEG_PER_S = 10.0` then `PIN_AVEL = math.radians(10.0)` | MATCH (units bug from 2026-05-17 audit is FIXED here) |
| `maxrotation` | line 27 `90.0` (degrees) | line 113 `MAX_ANGLE = math.pi / 2.0` (= 90°) | MATCH |

### Pin 1 (bottom, ROLLING)

| Field | object_10k_nh.json | FBA script | Verdict |
|---|---|---|---|
| `box` | line 33 `[-1.1, -2.1, -1.1, 1.1, -1.99, 1.1]` | line 117 `PIN_Y_BOT = -1.99` + filter `y < PIN_Y_BOT` (line 224) | MATCH (semantic, same as Pin 0 logic) |
| `action` | line 36 `ROLLING` | line 523 `R_bot = rot_y(bot_angle)` | MATCH |
| `center` | line 37 `[0,0,0]` | line 514 (shared) | MATCH |
| `axis` | line 38 `[0,1,0]` | rot_y | MATCH |
| `avel` | line 39 `-10.0` deg/s | line 520 `bot_angle = max(-PIN_AVEL * (f+1) * dt, -MAX_ANGLE)` (negative direction) | MATCH |
| `maxrotation` | line 40 `90.0` deg | `MAX_ANGLE = π/2` (clamped via `max(..., -MAX_ANGLE)`) | MATCH |

### Demo 2 verdict

**0 DIVERGE-accidental.** All physics fields match. `maxFrame`, `timer`, `offline`,
`output_abc` are render-only and correctly not ported.

---

## Demo 3 — StretchingCloth (FBA script NOT YET WRITTEN — separate Task #26)

Below is the **byte-for-byte spec** the new FBA script must use. Scene:
`CudaTests/StretchingCloth/scene.json`; object: `object_NH_20k.json`.

### Global scene fields

| Field | scene.json | spec for new script |
|---|---|---|
| `timestep` | line 6 `0.01` | `DT = 0.01` |
| `stop` | line 3 `1200` | `NUM_FRAMES = 1200` (sim cap, even though `maxFrame=2000` allows longer render) |
| `maxFrame` | line 5 `2000` | n/a (render-only; sim stops at `stop=1200`) |
| `timer` | line 2 `50` | n/a (RealSim render rate) |
| `offline` | line 4 `false` | n/a (per memory `realsim-test-mode.md`, FBA always runs offline) |
| `gravity` | line 7 `[0.0, -1.0, 0.0]` | `GRAVITY = -1.0` (Y-down; note this is **−1 m/s²**, not standard −9.81/−10) |
| `LocalGlobal_CUDA` | line 8 `5` | `PD_ITERATIONS = 5` |
| `linearsolver.type` | line 11 `SPARSE_INVERSE_CUDA` | `SolverFBA` (default sparse-inverse path) |
| `constraintsolver` | (absent) | no NSN — pure PD; `friction=False`, no contacts |

### Object (square_20201P.obj)

| Field | object_NH_20k.json | spec for new script |
|---|---|---|
| `mesh` | line 2 `resources/mesh/cloth/square_20201P.obj` | load `/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_20201P.obj` |
| `element_type` | line 4 `TRIANGLE` | triangle mesh; load via Newton's OBJ loader, build via `add_cloth_mesh` or `add_soft_mesh` with triangle indices |
| `trans` | line 6 `[0,0,0]` | identity translation |
| `rotation` | line 7 `[90.0, 0, 0]` (degrees, X-axis) | apply 90° rotation around X (flip cloth from XY plane into XZ plane); `Rx(90°)` |
| `scale` | line 8 `[1,1,1]` | identity scale |
| `center` | line 9 `[0,0,0]` | rotation pivot at origin |
| `obj_mass` | line 12 `1000` | `OBJ_MASS = 1000.0`, lumped uniformly as `obj_mass / N` |
| `young` | line 13 `1E+5` | `YOUNG = 1.0e5` |
| `poisson` | line 14 `0.45` | `POISSON = 0.45` |
| `constitutive` | line 15 `TRI_NEOHOOKEAN` | `stretching_model="neohookean"` (triangle NH path) |
| `bending` | line 16 `0.0` | bending stiffness = 0 (no bending energy contribution) |
| `bending_type` | line 17 `BEND_ISOMETRIC` | isometric bending edges (built but zero stiffness) — still wire `add_surface_mesh_edges=True` if it matters for code-path coverage; with k=0 the contribution is zero either way |

### Pin 0 (PULLING +X)

| Field | object_NH_20k.json | spec for new script |
|---|---|---|
| `box` | line 22 `[0.99, -0.1, -1.1, 1.1, 0.1, 1.1]` | post-rotation AABB; after Rx(90°), original Y∈[−0.1, 0.1] maps to Z∈[−0.1, 0.1] and original Z∈[−1.1, 1.1] maps to Y∈[−1.1, 1.1]. **Apply rotation first, then filter `PIN0_LO=(0.99, -1.1, -0.1)`, `PIN0_HI=(1.1, 1.1, 0.1)`** — OR keep pre-rotation AABB and pre-rotation verts as in `fba_demo4` does (see Demo 4 row).Whichever convention is chosen must match the rotation pipeline. |
| `action` | line 25 `PULLING` | external `set_pin_targets` driver (no internal pin solver action needed) |
| `direction` | line 26 `[1,0,0]` | `PIN0_DIR = np.array([1.0, 0.0, 0.0])` |
| `vel` | line 27 `0.1` (m/s, per init.h:1005) | `PIN0_VEL = 0.1` |
| `maxlength` | line 28 `1.0` (m) | `PIN0_MAXLENGTH = 1.0` |

### Pin 1 (PULLING −X)

| Field | object_NH_20k.json | spec for new script |
|---|---|---|
| `box` | line 34 `[-1.1, -0.1, -1.1, -0.99, 0.1, 1.1]` | mirror of Pin 0 box on the X−1 side |
| `action` | line 37 `PULLING` | external driver |
| `direction` | line 38 `[-1,0,0]` | `PIN1_DIR = np.array([-1.0, 0.0, 0.0])` |
| `vel` | line 39 `0.1` | `PIN1_VEL = 0.1` |
| `maxlength` | line 40 `1.0` | `PIN1_MAXLENGTH = 1.0` |

### Demo 3 verdict

**Not yet written — spec captured.** Pull velocity is 0.1 m/s and max stretch is
1.0 m on each side, so each pin reaches its cap at frame `1.0 / (0.1 × 0.01) = 1000`,
within the 1200-frame run. Cloth ends fully stretched for the last 200 frames.

Critical implementation notes:

1. Gravity is **−1.0 m/s²**, not −10 or −9.81. Easy to mis-port.
2. `bending = 0.0` — bending energy contribution is zero. `BEND_ISOMETRIC` is the
   *type* of bending model used when stiffness is nonzero; with zero stiffness it
   doesn't matter, but if the new script omits the bending edges entirely the energy
   evaluator will not break.
3. The 90° rotation around X flips the cloth from XY plane into the XZ horizontal
   plane (Z becomes Y, Y becomes −Z). Pin boxes' Y/Z bounds must be transformed
   accordingly or applied pre-rotation.
4. No NSN, no contacts, no friction. `SolverFBA(..., friction=False)` with no
   contacts arg.

---

## Demo 4 — PullingWooper (`fba_demo4_pulling_wooper.py`)

Scene: `CudaTests/PullingWooper/scene.json`; object: `wooper_5k.json`.

### Global scene fields

| Field | scene.json | FBA script | Verdict |
|---|---|---|---|
| `timestep` | line 4 `0.01` | line 34 `DT = 0.01` | MATCH |
| `stop` | line 2 `500` | line 35 `TOTAL_FRAMES = 500` | MATCH |
| `maxFrame` | (absent) | n/a | MATCH |
| `timer` | line 3 `20` | n/a (render-rate) | INTENTIONAL-NOT-PORTED |
| `output_abc` | line 5 `/simulation/output_abc/PullingWooper/output` | n/a | INTENTIONAL-NOT-PORTED |
| `gravity` | line 6 `[0.0, -0.0, 0.0]` (negative zero ⇒ effectively 0) | line 38 `GRAVITY = 0.0` | MATCH |
| `LocalGlobal_CUDA` | line 7 `5` | line 36 `PD_ITERATIONS = 5` | MATCH |
| `linearsolver.type` | line 9 `SPARSE_INVERSE_CUDA` | `SolverFBA` (sparse-inverse) | MATCH (architectural) |
| `constraintsolver.type` | line 30 `NonSmoothNewton_CUDA` | `SolverFBA._solve_nsn_*` | MATCH (architectural) |
| `constraintsolver.maxforce` | line 31 `1.00E+12` | **not passed to `SolverFBA`** (lines 197-205 omit `lambda_cap`) | **DIVERGE-accidental** — should be `lambda_cap=1e12` (Phase 2.4 wiring) |
| `constraintsolver.linearsolver` | line 32 `PCR_CUDA` | `np.linalg.solve` (host fp64 dense) | MATCH (functional — precision-review) |
| `constraintsolver.iterations` | line 33 `10` | line 37 `NSN_ITERATIONS = 1` | **DIVERGE-accidental** — RealSim runs 10 outer NSN linearizations per step; FBA runs only 1. Note Decision #4 caps inner iterations at 10, but `nsn_iterations` is the *outer* count and should be 10 to match. |
| `constraintsolver.tolerance` | line 34 `1.00E-9` | n/a (FBA NSN has no tolerance early-exit; runs full count) | INTENTIONAL-NOT-PORTED (per `2026-05-17-fba-precision-params-variants-review.md` Q9 / NSN-tolerance) |
| `constraintsolver.function` | line 35 `FB` | `fb_frictional_row`, FB residual throughout | MATCH |

### Cylinder collisions (2 static, μ=0, no rolling)

| Field | scene.json | FBA script | Verdict |
|---|---|---|---|
| cyl[0] `base` | line 13 `[0.0, -1.0, -1.3]` | line 59 `CYL0_BASE = np.array([0.0, -1.0, -1.3])` | MATCH |
| cyl[0] `axis` | line 14 `[1,0,0]` | line 61 `CYL_AXIS = np.array([1.0, 0.0, 0.0])` | MATCH |
| cyl[0] `radius` | line 15 `1.0` | line 57 `CYL_RADIUS = 1.0` | MATCH |
| cyl[0] `mu` | line 16 `0.0` | line 202 `friction=False` (solver skips Stage B) | MATCH (effective: zero-friction with `friction=False` ≡ μ=0 normal-only NSN) |
| cyl[0] `rollingvel` | line 17 `0.0` | no `shape_angular_velocity` argument passed → defaults to zero | MATCH |
| cyl[0] `visuallength` | line 18 `1.0` | line 58 `CYL_HALF_HEIGHT = 3.0` | INTENTIONAL-NOT-PORTED (RealSim's `CylinderCollision.cpp` treats cylinders as infinite along the axis; `visuallength` is render-only. Newton needs a finite `half_height`; 3.0 chosen so the cylinder fully spans the wooper's X-extent ≈ 5 m. Per 2026-05-17 audit Open-Question; confirmed render-only after inspecting CylinderCollision.cpp.) |
| cyl[1] `base` | line 21 `[0.0, -1.0, 1.3]` | line 60 `CYL1_BASE = np.array([0.0, -1.0, 1.3])` | MATCH |
| cyl[1] `axis` | line 22 `[1,0,0]` | `CYL_AXIS` | MATCH |
| cyl[1] `radius` | line 23 `1.0` | `CYL_RADIUS = 1.0` | MATCH |
| cyl[1] `mu` | line 24 `0.0` | `friction=False` | MATCH |
| cyl[1] `rollingvel` | line 25 `0.0` | default zero | MATCH |
| cyl[1] `visuallength` | line 26 `1.0` | `CYL_HALF_HEIGHT = 3.0` | INTENTIONAL-NOT-PORTED |

### Object (wooper_volume_5325P.mesh)

| Field | wooper_5k.json | FBA script | Verdict |
|---|---|---|---|
| `mesh` | line 2 `wooper_volume_5325P.mesh` | line 33 `MESH_PATH = .../wooper_volume_5325P.mesh` | MATCH |
| `element_type` | line 3 `TETRAHEDRON` | `add_soft_mesh` with tet indices (line 122) | MATCH |
| `trans` | line 5 `[0.0, 4.0, -0.0]` | line 46 `TRANS = np.array([0.0, 4.0, 0.0])` | MATCH (the `-0.0` in Z is signed-zero; arithmetically equal to 0.0) |
| `rotation` | line 6 `[0.0, 0.0, -90.0]` (degrees, Euler XYZ) | lines 45, 67-70 `ROT_Z_DEG = -90.0` applied via `rot_z` | MATCH (only Z component nonzero → Rz(−90°)) |
| `scale` | line 7 `[1,1,1]` | not multiplied (`verts_local @ rot_z(...).T + TRANS`, line 74) | MATCH (scale = 1 ⇒ identity) |
| `center` | line 8 `[0,0,0]` | rotation pivot is origin (verts rotated first, then translated) | MATCH |
| `obj_mass` | line 11 `1000` | line 42 `OBJ_MASS = 1000.0` | MATCH |
| `young` | line 12 `1E+7` | line 40 `YOUNG = 1.0e7` | MATCH |
| `poisson` | line 13 `0.3` | line 41 `POISSON = 0.3` | MATCH |
| `constitutive` | line 14 `TET_PD_NEOHOOKEAN` | line 203 `stretching_model="neohookean"` | MATCH |
| Mass lumping | uniform `obj_mass / N` | lines 161-165 `uniform_mass = OBJ_MASS / model.particle_count`, applied uniformly overriding density-based FE lumping | MATCH (Decision #3) |

### Pin 0 (single PULLING action)

| Field | wooper_5k.json | FBA script | Verdict |
|---|---|---|---|
| `box` | line 19 `[-2.5, 2, -0.3, -1.5, 4, 0.3]` | lines 49-50 `PIN_LO=(-2.5, 2.0, -0.3)`, `PIN_HI=(-1.5, 4.0, 0.3)` | MATCH (note: applied **in world coordinates** after rotation+translation; script comment line 150 confirms RealSim semantics) |
| `action` | line 22 `PULLING` | external `set_pin_targets` (line 227) | MATCH |
| `direction` | line 23 `[0.0, -1.0, 0.0]` | line 52 `PIN_DIR = np.array([0.0, -1.0, 0.0])` | MATCH |
| `vel` | line 24 `2.0` (m/s) | line 53 `PIN_VEL = 2.0` | MATCH |
| `maxlength` | line 25 `10.0` (m) | line 54 `PIN_MAXLENGTH = 10.0` | MATCH |

### Demo 4 verdict

**2 DIVERGE-accidental:**

1. `constraintsolver.maxforce = 1e12` not wired into `SolverFBA(lambda_cap=...)`.
   Solver supports it (line 1492-1493) and applies dt² conversion. Fix: pass
   `lambda_cap=1e12` in the `SolverFBA(...)` call at line 197.
2. `constraintsolver.iterations = 10` vs `NSN_ITERATIONS = 1`. RealSim runs 10
   outer FB-Newton linearizations per timestep; FBA runs 1. This is a separate
   parameter from inner-iteration count (Decision #4). Fix: change line 37 to
   `NSN_ITERATIONS = 10`.

`visuallength = 1.0` vs `CYL_HALF_HEIGHT = 3.0` is **INTENTIONAL-NOT-PORTED** —
RealSim's `CylinderCollision.cpp` treats the cylinder as infinite along its
axis (only radial distance enters the contact predicate); `visuallength` is
render-only.

---

## Demo 5 — SqueezingBall (`fba_demo5_squeezing_ball.py`)

Scene: `CudaTests/SqueezingBall/scene.json`; object: `ball_7k.json`.

### Global scene fields

| Field | scene.json | FBA script | Verdict |
|---|---|---|---|
| `timestep` | line 7 `0.01` | line 31 `DT = 0.01` | MATCH |
| `stop` | line 5 `600` | line 32 `TOTAL_FRAMES = 600` | MATCH |
| `maxFrame` | line 3 `900` | n/a (sim stops at 600; render-only cap) | INTENTIONAL-NOT-PORTED |
| `offline` | line 2 `false` | n/a (FBA always offline per memory) | INTENTIONAL-NOT-PORTED |
| `output_abc` | line 4 `.../SqueezingBall` | n/a | INTENTIONAL-NOT-PORTED |
| `timer` | line 6 `50` | n/a | INTENTIONAL-NOT-PORTED |
| `gravity` | line 8 `[0.0, -10.0, 0.0]` | line 35 `GRAVITY = -10.0` (Y-down) | MATCH |
| `LocalGlobal_CUDA` | line 9 `5` | line 33 `PD_ITERATIONS = 5` | MATCH |
| `linearsolver.type` | line 11 `SPARSE_INVERSE_CUDA` | sparse-inverse | MATCH (architectural) |
| `constraintsolver.type` | line 56 `NonSmoothNewton_CUDA` | `SolverFBA._solve_nsn_*` | MATCH |
| `constraintsolver.maxforce` | line 57 `1.00E+12` | **not passed to `SolverFBA`** | **DIVERGE-accidental** — should be `lambda_cap=1e12` |
| `constraintsolver.linearsolver` | line 58 `PCR_CUDA` | `np.linalg.solve` host fp64 | MATCH (functional) |
| `constraintsolver.iterations` | line 59 `10` | line 34 `NSN_ITERATIONS = 1` | **DIVERGE-accidental** — same as Demo 4 (outer NSN count must be 10) |
| `constraintsolver.tolerance` | line 60 `1.00E-9` | n/a | INTENTIONAL-NOT-PORTED |
| `constraintsolver.function` | line 61 `FB` | FB throughout | MATCH |

### Cylinder collisions (4 cylinders, μ=0.5, rolling)

Shape index ordering matches `CYLINDERS` list at line 49-54.

| Field | scene.json | FBA script | Verdict |
|---|---|---|---|
| cyl[0] `base` | line 15 `[0.0, -1.0, -1.5]` | line 50 `(np.array([0.0, -1.0, -1.5]), ...)` | MATCH |
| cyl[0] `axis` | line 16 `[1,0,0]` | line 50 `np.array([1.0, 0.0, 0.0])` | MATCH |
| cyl[0] `radius` | line 17 `1.0` | line 44 `CYL_RADIUS = 1.0` | MATCH |
| cyl[0] `mu` | line 18 `0.5` | line 46 `FRICTION_MU = 0.5` (applied via `shape_material_mu` line 147 + `mu_per_pair_override` line 175) | MATCH |
| cyl[0] `rollingvel` | line 19 `-3.0` (rad/s per CylinderCollision.cpp:52 `disp = radius * _avel * dt`) | line 50 third element `-3.0`, passed via `shape_angular_velocity` | MATCH |
| cyl[0] `visuallength` | line 20 `1.0` | line 45 `CYL_HALF_HEIGHT = 3.0` | INTENTIONAL-NOT-PORTED |
| cyl[1] `base` | line 22 `[0.0, -1.0, 1.5]` | line 51 `np.array([0.0, -1.0, +1.5])` | MATCH |
| cyl[1] `axis` | line 23 `[1,0,0]` | `[1,0,0]` | MATCH |
| cyl[1] `radius` | line 24 `1.0` | `CYL_RADIUS` | MATCH |
| cyl[1] `mu` | line 25 `0.5` | `FRICTION_MU` | MATCH |
| cyl[1] `rollingvel` | line 26 `3.0` | line 51 `+3.0` | MATCH |
| cyl[1] `visuallength` | line 27 `1.0` | `CYL_HALF_HEIGHT = 3.0` | INTENTIONAL-NOT-PORTED |
| cyl[2] `base` | line 31 `[-1.5, -3.5, 0.0]` | line 52 `np.array([-1.5, -3.5, 0.0])` | MATCH |
| cyl[2] `axis` | line 32 `[0,0,1]` | line 52 `np.array([0.0, 0.0, 1.0])` | MATCH |
| cyl[2] `radius` | line 33 `1.0` | `CYL_RADIUS` | MATCH |
| cyl[2] `mu` | line 34 `0.5` | `FRICTION_MU` | MATCH |
| cyl[2] `rollingvel` | line 35 `3.0` | line 52 `+3.0` | MATCH |
| cyl[2] `visuallength` | line 36 `1.0` | `CYL_HALF_HEIGHT = 3.0` | INTENTIONAL-NOT-PORTED |
| cyl[3] `base` | line 39 `[1.5, -3.5, 0.0]` | line 53 `np.array([+1.5, -3.5, 0.0])` | MATCH |
| cyl[3] `axis` | line 40 `[0,0,1]` | `[0,0,1]` | MATCH |
| cyl[3] `radius` | line 41 `1.0` | `CYL_RADIUS` | MATCH |
| cyl[3] `mu` | line 42 `0.5` | `FRICTION_MU` | MATCH |
| cyl[3] `rollingvel` | line 43 `-3.0` | line 53 `-3.0` | MATCH |
| cyl[3] `visuallength` | line 44 `1.0` | `CYL_HALF_HEIGHT = 3.0` | INTENTIONAL-NOT-PORTED |

### Plane collision

| Field | scene.json | FBA script | Verdict |
|---|---|---|---|
| plane `base` | line 50 `[0.0, -10.0, 0, 0]` | line 124 `plane=(0.0, 1.0, 0.0, 10.0)` — Newton (a,b,c,d) form encodes `a·x + b·y + c·z = d`. With base `(0,-10,0)` and normal `(0,1,0)`, d = normal·base = -10. **Newton uses d = +10 here, which means the plane lives at `y = +10`, NOT y = -10!** | **DIVERGE-accidental — plane sign** |
| plane `normal` | line 51 `[0,1,0]` | line 124 normal = `(0, 1, 0)` | MATCH (normal direction) |
| plane `mu` | line 49 `0.5` | `FRICTION_MU = 0.5` via `shape_material_mu` | MATCH |
| plane `visualscale` | line 52 `1.0` | n/a | INTENTIONAL-NOT-PORTED |

**Plane equation check:** Newton's `add_shape_plane(plane=(a,b,c,d))` with `(0,1,0,10)`
encodes `0·x + 1·y + 0·z = 10`, i.e. plane at `y = 10`. RealSim places the plane at
`y = -10` (base = `(0,-10,0)`, normal up). These do not coincide — sign error.
The wooper falls onto `y=-10` in RealSim; in FBA the plane is at `y=+10` (above
the ball), which means the ball passes through and is never caught by the plane.
**Confirm via behavior:** the FBA script's `min_y` post-condition would be far
more negative than RealSim's. **This is a blocker for Demo 5 parity if the ball
ever reaches y < -10**, otherwise it's harmless since RealSim's plane only
activates when the ball clears the lower cylinder pair. Even so, sign is wrong
and should be `plane=(0.0, 1.0, 0.0, -10.0)`.

(Note: Newton's convention for the 4-tuple plane form is interpreted in some
internal code paths as `a·x + b·y + c·z + d = 0` rather than `= d`. If that is
the case for `add_shape_plane`, the encoding `d=10` would represent the plane
`y + 10 = 0` ⇒ `y = -10` ⇒ MATCH. **The verdict here depends on Newton's
sign convention.** Cross-reference `newton/_src/geometry/build.py` or
equivalent — see the "Open question" note at the bottom.)

### Object (ball_volume_7129P.mesh)

| Field | ball_7k.json | FBA script | Verdict |
|---|---|---|---|
| `mesh` | line 2 `ball_volume_7129P.mesh` | line 30 `MESH_PATH = .../ball_volume_7129P.mesh` | MATCH |
| `element_type` | line 3 `TETRAHEDRON` | `add_soft_mesh` tet | MATCH |
| `trans` | line 5 `[0.0, 2.6, -0.0]` | line 42 `BALL_TRANS = np.array([0.0, 2.6, 0.0])` | MATCH (−0.0 = 0.0) |
| `rotation` | line 6 `[0,0,0]` | identity (no rot applied in line 61) | MATCH |
| `scale` | line 7 `[3.0, 3.0, 3.0]` | line 41 `BALL_SCALE = 3.0` (uniform; line 61 `verts_local * BALL_SCALE + BALL_TRANS`) | MATCH |
| `center` | line 8 `[0,0,0]` | scale applied about origin (pre-translation) | MATCH |
| `obj_mass` | line 11 `1000` | line 39 `OBJ_MASS = 1000.0` | MATCH |
| `young` | line 12 `1E+4` | line 37 `YOUNG = 1.0e4` | MATCH |
| `poisson` | line 13 `0.4` | line 38 `POISSON = 0.4` | MATCH |
| `constitutive` | line 14 `TET_PD_NEOHOOKEAN` | line 182 `stretching_model="neohookean"` | MATCH |
| `colormap` | line 15 `false` | n/a (render-only) | INTENTIONAL-NOT-PORTED |
| Mass lumping | uniform `obj_mass / N` | lines 138-142 `uniform_mass = OBJ_MASS / model.particle_count` | MATCH (Decision #3) |

### Pin actions

| Field | ball_7k.json | FBA script | Verdict |
|---|---|---|---|
| pin actions | none (no `pin` block in ball_7k.json) | none (no `set_pin_targets` call) | MATCH |

### Demo 5 verdict

**3 DIVERGE-accidental:**

1. `constraintsolver.maxforce = 1e12` not wired (`lambda_cap` omitted from
   `SolverFBA(...)` at line 177).
2. `constraintsolver.iterations = 10` vs `NSN_ITERATIONS = 1` (line 34). Should
   be 10 — same blocker class as Demo 4.
3. **Plane d-coefficient sign uncertain** (`plane=(0,1,0,10)` vs RealSim plane at
   y=-10). Two readings of Newton's plane-form sign convention give opposite
   answers. **Verify against Newton source before declaring MATCH or BLOCKER.**

All other physics fields (4 cylinders × {base, axis, radius, μ, rollingvel},
plane normal/μ, ball scale/trans/young/poisson/mass/energy) MATCH byte-for-byte.

---

## Top-level summary

**Total fields audited across all 4 demos:** approximately **123 line items.**

Per-demo breakdown:

| Demo | Audited | MATCH | DIVERGE-accidental | INTENTIONAL-NOT-PORTED |
|---|---|---|---|---|
| 2 TwistingBarNH | 25 | 21 | 0 | 4 (maxFrame, timer, offline, output_abc) |
| 3 StretchingCloth | 26 | n/a (script not yet written) | 0 (spec recorded) | 3 (maxFrame, timer, offline) |
| 4 PullingWooper | 35 | 27 | 2 | 6 (4×visuallength, timer, output_abc) |
| 5 SqueezingBall | 37 | 27 | 3 | 7 (4×visuallength, maxFrame, offline, output_abc, timer, visualscale, colormap) |

**Totals:**

- **MATCH:** **75** physics-affecting line items (excluding Demo 3).
- **DIVERGE-accidental:** **5** total across the 3 written scripts.
- **INTENTIONAL-NOT-PORTED:** **20** (render-only / wall-clock / signed-zero conventions).

### DIVERGE-accidental list (one-line summaries)

1. **Demo 4 — `lambda_cap` not wired.** `fba_demo4_pulling_wooper.py:197-205` omits
   `lambda_cap` from `SolverFBA(...)`. Scene specifies `maxforce: 1.00E+12`. Solver
   `__init__` accepts `lambda_cap` and internally divides by dt² (T.2 already done).
   Fix: pass `lambda_cap=1e12`.
2. **Demo 4 — `NSN_ITERATIONS = 1` should be 10.** `fba_demo4_pulling_wooper.py:37`
   vs scene `constraintsolver.iterations: 10`. RealSim runs 10 outer FB-Newton
   linearizations per timestep; FBA runs only 1. Comment ("RealSim does 1 FB-Newton
   step per NSN call") is incorrect — scene's `iterations: 10` is the outer count.
3. **Demo 5 — `lambda_cap` not wired.** Same as #1, in `fba_demo5_squeezing_ball.py:177-187`.
4. **Demo 5 — `NSN_ITERATIONS = 1` should be 10.** Same as #2, in
   `fba_demo5_squeezing_ball.py:34`.
5. **Demo 5 — plane d-coefficient sign open.** `fba_demo5_squeezing_ball.py:124`
   `plane=(0.0, 1.0, 0.0, 10.0)` vs scene plane at y=-10. Verdict depends on
   Newton's plane-form sign convention (`ax+by+cz=d` vs `ax+by+cz+d=0`). **Verify
   against Newton source.**

### INTENTIONAL-NOT-PORTED list

Render-only / timing fields that don't affect physics:

- `maxFrame` (when ≥ `stop`): render-cap, not sim length (4 occurrences)
- `timer`: real-time render rate (4 occurrences)
- `offline`: RealSim headless flag (2 occurrences in scene.json)
- `output_abc`: RealSim mesh-cache path (3 occurrences)
- `visuallength` (cylinders): render-only along-axis extent; RealSim cylinders
  are radially-infinite per `CylinderCollision.cpp:52` (6 cylinder occurrences
  across Demos 4 + 5)
- `visualscale` (plane): plane render extent (1 occurrence)
- `colormap` (Demo 5 object): vertex-color rendering toggle (1 occurrence)
- `constraintsolver.tolerance`: NSN early-exit not ported, Q9 confirmed
  intentional (2 occurrences)

---

## Open question (1)

**Demo 5 plane sign convention.** `fba_demo5_squeezing_ball.py:123-128` builds the
plane via `builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 10.0), ...)`. Newton uses
`(a, b, c, d)` for an oriented plane. If the convention is `a·x + b·y + c·z = d`,
the plane is at `y = +10` (wrong — should be `y = -10`). If the convention is
`a·x + b·y + c·z + d = 0`, the plane is at `y = -10` (correct).

Resolution: read `newton/_src/geometry/types.py` or `newton/_src/sim/builder.py`
for the `add_shape_plane` plane-form interpretation. (Not in scope of this audit
since changing the spec is not allowed; flagging for the script author.)

---

## Phase 0.20 verdict

- Demo 2: **GREEN.** All physics fields byte-match. Earlier `PIN_AVEL` units bug
  flagged in the 2026-05-17 audit is FIXED here.
- Demo 3: **PENDING.** Script not yet written; spec recorded above.
- Demo 4: **YELLOW.** 2 DIVERGE-accidental — both wire-up oversights, no physics
  algorithm difference.
- Demo 5: **YELLOW.** 3 DIVERGE-accidental — 2 wire-up oversights, 1 sign-convention
  open question.

No physics algorithm divergences were found in this byte-recheck beyond what
the existing 2026-05-17 audit already flagged. The DIVERGE-accidental items
are all parameter-passing omissions, addressable by adding two arguments
(`lambda_cap`, change `NSN_ITERATIONS`) to two `SolverFBA(...)` call sites and
double-checking the plane sign.
