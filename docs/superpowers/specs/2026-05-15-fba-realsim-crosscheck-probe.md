# RealSim FBA Cross-Check Probe

**Date:** 2026-05-15  
**Purpose:** Establish minimum viable RealSim reference trajectory for bit-level comparison against Newton SolverFBA port.  
**Next step:** Write Newton-side comparison code (separate task).

---

## 1. Binary Status

### Original binary (Sep 23 2025, 202 MB)

- **Realtime mode** (`"offline": false` or absent): works correctly. Ran 50 steps on `SimpleTrianglePinTest` successfully.
- **Offline mode** (`"offline": true`): segfaults in `PDTriangleStretchingEnergy::accumulateMatrix(triplets, coeff)` during `LocalGlobalSolver::init()`. The `_area` vector is empty at call time. Root cause is not fully confirmed in the old binary (likely a GCC version / optimization difference that emerged between Sep 2025 build and current GCC 13).

### Rebuild (May 15 2026, 240 MB)

Required due to two issues in the build environment:
1. CMake cache referenced Python 3.9 torch libs (`~/.../python3.9/...`) but the `realsim_py` conda env was upgraded to Python 3.12 — `libc10.so` path was stale.
2. Mixed torch installations: cmake pulled `libtorch_cuda.so` from pip's `~/.local/lib/python3.12/site-packages/torch/` but `libc10_cuda.so` from conda's `realsim_py` env. The pip torch is newer and requires `c10::cuda::get_fabric_clique_id` which the conda `libc10_cuda.so` lacks.

Fix applied:
```bash
cd /home/ziqiu/work/RealSim_py/realsim_py/build
cmake /home/ziqiu/work/RealSim_py/realsim_py \
  -Dc10_LIBRARY=~/.local/lib/python3.12/site-packages/torch/lib/libc10.so \
  -DC10_CUDA_LIBRARY=~/.local/lib/python3.12/site-packages/torch/lib/libc10_cuda.so \
  -DTORCH_LIBRARY=~/.local/lib/python3.12/site-packages/torch/lib/libtorch.so \
  -DTorch_DIR=~/.local/lib/python3.12/site-packages/torch/share/cmake/Torch \
  -DCMAKE_CXX_FLAGS="-Wno-array-bounds"
cmake --build . --target RealSim -j$(nproc)
```

Rebuild succeeded (warnings about AVX bounds in NeoHookean LBFGS are benign). Binary: `/home/ziqiu/work/RealSim_py/realsim_py/build/bin/RealSim` (May 15 2026).

**Note:** The new binary crashes with `free(): invalid pointer` in the destructor after writing Alembic output. This is a pre-existing double-ownership bug in `PDTriangleStretchingEnergy::setGeometry` which does `_geometry = static_cast<shared_ptr<STriangleMesh>>(raw_ptr)` creating an independent owning shared_ptr from a raw pointer that is also owned by `Meshes[i]`. The crash happens AFTER the simulation loop and AFTER the Alembic file is fully written — the output is valid.

---

## 2. Output Alembic File

| Property | Value |
|----------|-------|
| Path | `/home/ziqiu/work/RealSim_py/realsim_py/simulation/output_abc/FBACrossCheck/output_obj_0.abc` |
| Size | 75,202 bytes |
| Format | Alembic Ogawa 1.8.6 |
| Object name | `mesh0` |
| Frames | 50 |
| Vertices per frame | 113 |

---

## 3. Test Scene: FBACrossCheck

### Scene config

`/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/Tests/FBACrossCheck/scene.json`:
```json
{
    "offline": true,
    "maxFrame": 50,
    "timestep": 0.01,
    "gravity": [0.0, -10.0, 0.0],
    "LocalGlobal" : 5,
    "linearsolver": { "type": "SPARSE_INVERSE_CUDA" },
    "output_abc": "simulation/output_abc/FBACrossCheck",
    "objects": { "object_0": "simulation/config/Tests/FBACrossCheck/cloth.json" },
    "obstacles": {}
}
```

### Cloth config

`/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/Tests/FBACrossCheck/cloth.json`:
```json
{
    "mesh": "resources/mesh/cloth/square_113P.obj",
    "element_type": "TRIANGLE",
    "transformation": {
        "trans": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "center": [0.0, 0.0, 0.0]
    },
    "mechanical_props": {
        "obj_mass": 1.0,
        "young": 1E+4,
        "poisson": 0.4,
        "constitutive": "TRI_ARAP",
        "bending": 0.1,
        "bending_type": "BEND_ISOMETRIC"
    },
    "pin": {
        "0": { "box": [-1.1, 0.8, -0.1, 1.1, 1.1, 0.1] }
    }
}
```

---

## 4. Mesh Details

**File:** `resources/mesh/cloth/square_113P.obj`  
**Created with:** Blender 4.3.0

| Property | Value |
|----------|-------|
| Vertices | 113 |
| Triangles | 196 |
| Bounding box X | [-1.0, 1.0] |
| Bounding box Y | [-1.0, 1.0] |
| Bounding box Z | [0.0, 0.0] (flat in XY plane) |

---

## 5. Pin Specification

**Box predicate:** `[-1.1, 0.8, -0.1, 1.1, 1.1, 0.1]`  
Format: `[xmin, ymin, zmin, xmax, ymax, zmax]`

Captures all vertices with `y ∈ [0.8, 1.1]` (the full top edge at y=1.0).

**Pinned vertex indices (0-based):**

| Index | x | y | z |
|-------|---|---|---|
| 2 | -1.000000 | 1.000000 | 0.000000 |
| 3 | 1.000000 | 1.000000 | 0.000000 |
| 22 | 0.714286 | 1.000000 | 0.000000 |
| 23 | 0.428571 | 1.000000 | 0.000000 |
| 24 | 0.142857 | 1.000000 | 0.000000 |
| 25 | -0.142857 | 1.000000 | 0.000000 |
| 26 | -0.428571 | 1.000000 | 0.000000 |
| 27 | -0.714286 | 1.000000 | 0.000000 |

8 vertices pinned total (entire top edge).

**Verification from trajectory:** Frame 0 and Frame 49 both show v[2] = (-1.0, 1.0, 0.0) and v[3] = (1.0, 1.0, 0.0) unchanged — top corners confirmed pinned.

---

## 6. Trajectory Spot-Check

Positions of first 5 vertices (vertex indices 0–4):

### Frame 0 (initial)
```
v[0] = (-1.000000, -1.000000,  0.000000)   # bottom-left corner, free
v[1] = ( 1.000000, -1.000000,  0.000000)   # bottom-right corner, free
v[2] = (-1.000000,  1.000000,  0.000000)   # top-left corner, PINNED
v[3] = ( 1.000000,  1.000000,  0.000000)   # top-right corner, PINNED
v[4] = (-1.000000,  0.714286,  0.000000)   # left edge, free
```

### Frame 1
```
v[0] = (-0.999992, -1.000420,  0.000000)
v[1] = ( 0.999992, -1.000420,  0.000000)
v[2] = (-1.000000,  1.000000,  0.000000)   # pinned, unchanged
v[3] = ( 1.000000,  1.000000,  0.000000)   # pinned, unchanged
v[4] = (-0.999992,  0.714209,  0.000000)
```

### Frame 2
```
v[0] = (-0.999975, -1.000724,  0.000000)
v[1] = ( 0.999975, -1.000724,  0.000000)
v[2] = (-1.000000,  1.000000,  0.000000)   # pinned
v[3] = ( 1.000000,  1.000000,  0.000000)   # pinned
v[4] = (-0.999988,  0.714169,  0.000000)
```

### Frame 49 (final)
```
v[0] = (-0.999971, -1.000657,  0.000000)
v[1] = ( 0.999971, -1.000657,  0.000000)
v[2] = (-1.000000,  1.000000,  0.000000)   # pinned
v[3] = ( 1.000000,  1.000000,  0.000000)   # pinned
v[4] = (-0.999989,  0.714179,  0.000000)
```

**Observations:**
- The cloth is flat (Z = 0 everywhere) — expected since the mesh is planar in XY and there's no out-of-plane perturbation.
- Under gravity `[0, -10, 0]`, vertices move in -Y. The displacement is small (~0.001 m) because the cloth is stiff (`young = 1e4 Pa` equivalent weight, with ARAP constitutive).
- The simulation reaches a near-steady state by frame ~20 (displacements plateau).
- Z remains exactly 0.0 throughout all frames — the flat mesh under flat-field gravity stays planar.

---

## 7. Parameter Mapping for Newton Side

When writing Newton comparison code, use:

| Parameter | RealSim value | Newton equivalent |
|-----------|--------------|-------------------|
| dt | 0.01 | `model.dt = 0.01` |
| gravity | [0, -10, 0] | `model.gravity = wp.vec3(0, -10, 0)` |
| LG iterations | 5 | `solver.iterations = 5` |
| constitutive | TRI_ARAP | `tri_model = ARAP` |
| bending | 0.1 (`BEND_ISOMETRIC`) | `edge_ke ≈ 0.1` |
| young | 1e4 | `tri_ke ∝ 2μ` (see stability-diagnosis.md for conversion) |
| poisson | 0.4 | used to compute λ, μ |
| obj_mass | 1.0 (total) | `particle_mass = 1.0 / 113` per vertex |
| linear solver | SPARSE_INVERSE_CUDA | SolverFBA (exact sparse inverse) |
| pin | top edge (8 verts, indices 2,3,22-27) | `particle_mass[i] = 0` for those indices |
| mesh | square_113P.obj (flat, [-1,1]x[-1,1]) | create matching grid or parse .obj |
| stop | 50 frames | `num_frames = 50` |

---

## 8. Notes and Concerns

1. **Z stays 0.0**: The planar mesh under planar gravity will never develop out-of-plane motion. This is mathematically correct but means the cross-check validates only 2D deformation. This is fine for TRI_ARAP which operates on 2D triangles embedded in 3D.

2. **Binary crash after write**: The `free(): invalid pointer` crash is a double-free in the energy destructor. The output file is complete and valid — the crash happens after `writeAlembicMultiFrame` finishes. No action needed for this probe; the Newton comparison task should note that rerunning the simulation will re-overwrite the abc file correctly.

3. **Offline mode constraint**: `"offline": true` + `"maxFrame": N` is required for abc output. `"stop": N` only works in Realtime (GUI) mode which has no abc output path.

4. **obj_mass semantics**: In RealSim, `obj_mass` is the total mass of the object (1.0 kg for all 113 vertices). Newton's `particle_mass` is per-vertex, so `particle_mass = 1.0 / 113 ≈ 0.00885 kg` per vertex.

5. **ARAP weight**: RealSim's ARAP weight is `_weight = area_weighted * young_modulus_term`. The exact formula is in `PDTriangleStretchingEnergy.cpp`. See `2026-05-15-fba-stability-diagnosis.md` for the derivation of `_weight = 2μ` where `μ = E / (2(1+ν))`.

---

## 9. Files Created

- `/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/Tests/FBACrossCheck/scene.json`
- `/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/Tests/FBACrossCheck/cloth.json`
- `/home/ziqiu/work/RealSim_py/realsim_py/simulation/output_abc/FBACrossCheck/output_obj_0.abc` (75,202 bytes, 50 frames)
- `/tmp/read_abc.cpp` (Alembic reader utility, compile with instructions below)

### Recompile read_abc:
```bash
g++ -std=c++17 \
  -I/home/ziqiu/work/RealSim_py/realsim_py/deps/alembic/lib \
  -I/home/ziqiu/work/RealSim_py/realsim_py/build/deps/alembic/lib \
  -I/home/ziqiu/work/RealSim_py/realsim_py/deps/Imath/src/Imath \
  -I/home/ziqiu/work/RealSim_py/realsim_py/build/deps/Imath/config \
  /tmp/read_abc.cpp \
  /home/ziqiu/work/RealSim_py/realsim_py/build/lib/libAlembic.a \
  /home/ziqiu/work/RealSim_py/realsim_py/build/lib/libImath-3_2.a \
  -o /tmp/read_abc
```

### Rerun simulation:
```bash
cd /home/ziqiu/work/RealSim_py/realsim_py
./build/bin/RealSim -m simulation/config/Tests/FBACrossCheck/scene.json
# Expect: crash after write, but abc file at simulation/output_abc/FBACrossCheck/output_obj_0.abc is valid
```
