# Global Review Summary — FBA vs RealSim 全局对齐审计

**Date:** 2026-05-17
**Scope:** 三个并行 review 的合并报告
**In-scope demos (post 2026-05-17 fourth scope cut):** Demo 2 TwistingBarNH, 3 StretchingCloth, 4 PullingWooper, 5 SqueezingBall. (Demo 1, 6, 7, 8, 9, 10 all dropped.)

## 子 review 索引

1. **LBFGS / NH local projection**: `docs/superpowers/specs/2026-05-17-fba-lbfgs-compliance-review.md` — **GREEN** (0 accidental)
2. **NSN inner solver**: `docs/superpowers/specs/2026-05-17-fba-nsn-compliance-review.md` — **GREEN post A1/A2/A3 fix** (3 accidental found, 3 fixed)
3. **LG+NSN integration (with/without contact)**: `docs/superpowers/specs/2026-05-17-fba-lg-nsn-integration-review.md` — **GREEN** (0 accidental)
4. **Collision detection (per shape)**: `docs/superpowers/specs/2026-05-17-fba-collision-detection-review.md` — **YELLOW** (1 architecture blocker AA.2 Demo 8)
5. **Precision + Parameters + RealSim variants**: `docs/superpowers/specs/2026-05-17-fba-precision-params-variants-review.md` — **YELLOW** (3 parameter blockers in scripts)

---

## RealSim 多变体——in-scope 最小子集

RealSim 提供多种 LinearSolver / NSN / Collider 实现。In-scope 7 demo 实际只用以下一组：

| 组件 | 唯一需要的变体 |
|---|---|
| LocalGlobal driver | `CUDALocalGlobalSolver` |
| A 矩阵 LinearSolver | `CUDASparseInverseSolver` (`SPARSE_INVERSE_CUDA`) |
| NSN 约束求解器 | `CUDANonSmoothNewtonSolver` (single-env, **NOT** Lite — ParallelEnv 已 out-of-scope) |
| NSN 内层 Schur LCP | `CUDADenseJacobiPCRSolver` (FBA 用 `np.linalg.solve` 替代——algorithmically equivalent for dense system) |
| 接触检测 | `CylinderCollision`, `PlaneCollision`, `GenericCCD` (mesh) |

**所有其他 RealSim 变体（CG, Cholesky CPU, LiteNSN, SphereCollision 等）超出 in-scope demo 范围——FBA 不需要实现这些，也确认 FBA 当前没有多余的实现路径要清理（除两处历史死代码，下文 Phase 4 cleanup 列出）。**

---

## DIVERGE-accidental 汇总（按严重度排序）

### 🔴 BLOCKERS（需要修才能进 Phase 3 demo verification）

| ID | 文件:行 | 内容 | 状态 |
|---|---|---|---|
| **PIN_AVEL-1** | `scripts/fba_twisting_bar_cudatests.py:98` | `PIN_AVEL=10.0` 当成 rad/s；RealSim `Actions.h:65` 是 deg/s → π/180。FBA 旋转 **57.3× 过快**。影响 Demo 1, 2 (都用这同一脚本) | 待修 |
| **MESH-1** | `scripts/fba_twisting_bar_cudatests.py` | Demo 1 用了 Demo 2 的 `cube_volume_11340P.mesh`；Demo 1 应是 `cube_volume_5292P.mesh` | 待修 |
| **AA.2** | `solver_fba.py` (架构性) | Demo 8 ClothOnKnives: RealSim `GenericCCD` 用 VF/EE + barycentric multi-DOF；Newton `mesh_query_point_sign_normal` 是 single-particle SDF。检测方向也反 | 架构决策待定 |

### 🟡 已 fix 待 GPU 验证

| ID | 文件:行 | 内容 | 状态 |
|---|---|---|---|
| **A1** | `solver_fba.py:~1441` | Stage A 多余 `np.maximum(lam, 0.0)`——`λ_n ≥ 0` clamp 把 FB-Newton 退化回 PGS | ✅ Fixed (uncommitted) |
| **A2** | `solver_fba.py:~1587-1598` | Stage B Coulomb cone 用 unsigned `lam_n`；RealSim 用 signed + 两个独立 if | ✅ Fixed (uncommitted) |
| **A3** | `solver_fba.py:~967` | Stage A correction-apply guard 是 single-sided；Stage B 用 symmetric `abs(.)` | ✅ Fixed (uncommitted) |

### 🟢 已知 + 已排进 Phase 1 sub-task

| ID | 内容 | Tracking |
|---|---|---|
| Y.1 | Sphere/cylinder normal pene0 缺 0.01m 互穿 cushion | Phase 1.3.e (Task #17) |
| ~~Y.2~~ | ~~Plane tangent pene0~~ | **RECLASSIFIED 2026-05-17 → MATCH-by-identity.** Projection-onto-plane identity proves `pene0_n = n·base` (P-independent) and `pene0_t = t·P` with FBA's P = `p_t0` = RealSim's `p_t0`. Detection-set divergence (`p_t0` vs `p_t1`) is out-of-scope. Task #14 closed |
| T.2 | `lambda_cap` 单位 scale (fp64 `1/dt²` 偏差) | Phase 1.3.d (Task #16) |
| AA.1 | μ mixing: FBA `sqrt(particle_mu·shape_mu)`；RealSim 直接用 shape μ | 当前 in-scope demo 通过 `mu_per_pair_override` 或 `friction=False` 掩盖；新加任务跟踪 |
| W | Position-LCP 多 PD 迭代轻微 drift | Phase 1.3.g (Task #19)——UPGRADED 直接 port `_systemlinearsolver->solve` re-derive |
| Demo 4/5 `lambda_cap` 未 wire | 当前 maxforce=1e12 不激活；非 blocker | Phase 2.4 (Task #10) |

### 🔵 Precision DIVERGE-accidental（below 1e-3 tolerance, 不 blocker）

| ID | 文件:行 | 内容 |
|---|---|---|
| P1 | `linear_solver.py:1333` | `lam.astype(np.float32)` 在 J^T gather 之前 |
| P2 | (legacy) | `_A_inv_Jt_d` 存 `wp.vec3` fp32, 仅 `use_isodof=False` |
| P3 | `insert_component_kernel` | 每 PD outer iter 一次 fp64→fp32 downcast |

### 🟣 Implementation-choice DIVERGE-intentional (per "完全对齐 A 级" 2026-05-17)

These are reclassified from earlier ambiguity to explicit intentional architectural/implementation choices. They are NOT algorithmic substitutions and do NOT change observable behavior outside FP-level noise:

| ID | 内容 | Justification |
|---|---|---|
| ARCH.1 | dt²-scaling: FBA stores `A_FBA = A_R/dt²` and `λ_FBA = λ_R/dt²`; RealSim stores `A_R = M + dt²·L`. Observable `x_unc`, `Δx_correction`, and trajectory identical in infinite precision; fp32-level FP-order differs | Architectural choice — reverting would be ~2 weeks rewrite; no algorithmic effect. Documented in `audit-v2.md` |
| ARCH.2 | fp32 primary + fp64 in LBFGS inner only; RealSim is fp64 throughout | Precision choice — full fp64 conversion costs perf, no algorithmic effect |
| ARCH.3 | Warp `wp.svd2/svd3` vs Eigen `JacobiSVD` | Different SVD algorithm — produces same projected `P` up to FP noise but not bit-identical |
| ARCH.4 | Persistent `ω` buffer in FBA (bookkeeping for iter-0 penetration); RealSim recomputes every NSN iter | Bookkeeping artifact only; `ω` re-derived per iter inside FBA NSN loop too |
| ARCH.5 | Newton's BSR sparse Cholesky vs RealSim's cuSparse direct | Different factorization implementation; same factored solve |

---

## 三个 Open Questions 需要你拍

### Q-G1: Demo 8 ClothOnKnives 架构决策（AA.2）

RealSim 的 `GenericCCD` 是 vertex-face / edge-edge 的连续碰撞检测，约束行用 barycentric multi-DOF（一个接触约束牵涉 ≥3 个粒子）。Newton 的 collision pipeline 是 single-particle SDF 接触（cloth 粒子 vs knife SDF），FBA 的接触 buffer schema 也是 single-particle。**两者不可能 byte-equivalent**。

选项：
- **(a)** Port GenericCCD 到 Newton + 扩 FBA contact schema 支持 barycentric multi-DOF——重活，2-3 周
- **(b)** Drop Demo 8 from scope——in-scope 变成 6 个 demo (1,2,3,4,5,6)，全是 rigid collider，scope 清爽
- **(c)** 仍跑 Demo 8，但接受质化"似乎差不多"而非 byte-aligned——违反 plan 的 binary success 标准

### Q-G2: AA.1 μ mixing 是否改

FBA 当前 `μ_eff = sqrt(particle_mu · shape_mu)`，RealSim 直接用 `shape_mu`。In-scope demo 因为 override 或无摩擦掩盖；但保留 sqrt mixing 会让未来任何 demo 默认行为不对。

- **(a)** 移除 sqrt mixing，与 RealSim 对齐——一行改动
- **(b)** 保留 sqrt mixing 作 Newton-side 通用 contact API 兼容——但需要每个 demo 都强制 override

### Q-G3: PIN_AVEL 和 mesh 文件 bug（PIN_AVEL-1, MESH-1）

明显的脚本 bug。要不要顺手在下一个 commit 里修？修了等于 Demo 1 + Demo 2 重新生效。

- **(a)** 是，同一 batch 修（推荐）
- **(b)** 单独 commit
