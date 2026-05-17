# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics cloth solver)."""

import warp as wp

# 3x2 matrix type (wp.mat32 is not available in Warp 1.14; use types.matrix).
mat32 = wp.types.matrix(shape=(3, 2), dtype=wp.float32)

# ---- LBFGS history types (M = 8 history slots, faithful to RealSim
# LBFGS.hpp:36 default ``M = 8`` used by ``NHProjectionProblem2D`` /
# ``NHProjectionProblem3D``). DIM = 2 (Tri) or 3 (Tet).
_LBFGS_M = 8
_lbfgs_mat_2xM = wp.types.matrix(shape=(2, _LBFGS_M), dtype=wp.float64)
_lbfgs_mat_3xM = wp.types.matrix(shape=(3, _LBFGS_M), dtype=wp.float64)
_lbfgs_vec_M = wp.types.vector(length=_LBFGS_M, dtype=wp.float64)

# Locked LBFGS / line-search parameters. Faithful to RealSim defaults
# (Minimizer.hpp:66-70, LBFGS.hpp:47, HyperelasticProblemS.h:11):
#   max_iters = 50, ls_decrease (c_1) = 1e-4, tol = 1e-6, M = 8.
_LBFGS_MAX_ITERS = wp.constant(50)
_LBFGS_LS_MAX_ITERS = wp.constant(64)  # Cap RealSim's 100000 to a Warp-friendly bound.
_LBFGS_C1 = wp.constant(wp.float64(1.0e-4))
_LBFGS_TOL = wp.constant(wp.float64(1.0e-6))
# Sigma feasibility floor (RealSim PDNeohookeanTriangleEnergyParallel.h:68
# / PDNeohookeanTetrahedronEnergyParallel.h:69 all-three-tiny clamp).
_NH_EPS = wp.constant(wp.float64(1.0e-6))
# Huge-but-finite "infinity" returned by ``value()`` when sigma_i < 0 to
# block the line search. Faithful to RealSim HyperelasticProblemS.h:39/97
# which returns ``std::numeric_limits<float>::max() ~= 3.4e38`` rather than
# a true +infinity. Using ``1e30`` (USER-CONFIRMED SUBSTITUTION per Q9): Warp
# ``@wp.func`` cannot throw, and a finite ceiling avoids NaN propagation
# on GPU floating-point ops.
_NH_VALUE_INF = wp.constant(wp.float64(1.0e30))


@wp.kernel
def gather_per_particle_kernel(
    contributions: wp.array2d[wp.vec3],  # (num_elements, n_verts_per_element)
    offsets: wp.array[wp.int32],  # (n_particles + 1,)
    element_idx: wp.array[wp.int32],  # (total_incidence,)
    local_idx: wp.array[wp.int32],  # (total_incidence,)
    rhs: wp.array[wp.vec3],  # (n_particles,) — ACCUMULATED into
):
    """Sum per-element contributions into per-particle rhs.

    For each particle, reads its incident-element entries from the CSR
    adjacency (built by ``build_particle_element_csr``) and sums the
    corresponding ``contributions[element_idx, local_idx]`` vec3 values.
    Adds the sum to ``rhs`` (does NOT overwrite) so other terms (inertial
    prediction, pin energy, contact lambda correction) already in rhs are
    preserved.

    Args:
        contributions: ``(num_elements, n_verts_per_element)`` per-element
            contribution to each of its incident vertices.
        offsets: CSR row offsets (``n_particles + 1`` entries).
        element_idx: CSR entry -> element index.
        local_idx: CSR entry -> local vertex index within element.
        rhs: ``(n_particles,)`` accumulator (in-out).
    """
    p = wp.tid()
    start = offsets[p]
    end = offsets[p + 1]
    s = wp.vec3(0.0, 0.0, 0.0)
    for k in range(start, end):
        s = s + contributions[element_idx[k], local_idx[k]]
    rhs[p] = rhs[p] + s


@wp.kernel
def apply_permutation_vec3_kernel(
    src: wp.array[wp.vec3],
    perm: wp.array[wp.int32],
    dst: wp.array[wp.vec3],
):
    """dst[i] = src[perm[i]] — per-particle gather."""
    tid = wp.tid()
    dst[tid] = src[perm[tid]]


@wp.kernel
def extract_component_kernel(
    src: wp.array[wp.vec3],
    component: int,
    dst: wp.array[wp.float64],
):
    """dst[i] = src[i][component] — extracts one vec3 component into a float64 array."""
    tid = wp.tid()
    dst[tid] = wp.float64(src[tid][component])


@wp.kernel
def insert_component_kernel(
    src: wp.array[wp.float64],
    component: int,
    dst: wp.array[wp.vec3],
):
    """dst[i][component] = src[i] — inserts a float64 scalar back into a vec3 component."""
    tid = wp.tid()
    v = dst[tid]
    v[component] = wp.float32(src[tid])
    dst[tid] = v


@wp.kernel
def scale_by_diag_kernel(
    src: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    dst: wp.array[wp.float64],
):
    """dst[i] = diag[i] * src[i] — used for D-inverse scaling in the linear solve."""
    tid = wp.tid()
    dst[tid] = diag[tid] * src[tid]


@wp.kernel
def apply_permutation_scalar_kernel(
    src: wp.array[wp.float64],
    perm: wp.array[wp.int32],
    dst: wp.array[wp.float64],
):
    """dst[i] = src[perm[i]] — scalar gather variant."""
    tid = wp.tid()
    dst[tid] = src[perm[tid]]


@wp.kernel
def zero_vec3_kernel(arr: wp.array[wp.vec3]):
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_inertial_kernel(
    x_prev: wp.array[wp.vec3],
    v_prev: wp.array[wp.vec3],
    f_ext: wp.array[wp.vec3],
    inv_mass: wp.array[wp.float32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: float,
    x_inertia: wp.array[wp.vec3],
):
    """x_inertia = x_prev + dt*v_prev + dt^2*(f_ext*im + g).

    Gravity is applied unconditionally to all particles, including pinned ones
    (inv_mass == 0). ``x_inertia`` is used as the warm-start ``x_cur`` for the
    PD outer iteration; the local projection kernels (ARAP / bending) read
    ``x_cur`` to compute deformation gradients. Wrong pinned positions bias the
    deformation gradient at triangles touching the pin, compounding to large
    errors over many steps. The pin constraint is enforced via the large
    ``pin_stiffness`` diagonal entry in ``A`` and the matching
    ``pin_stiffness * x_ref`` scatter to the RHS — NOT by gating gravity here.
    For pinned particles (im == 0) ``f_ext * im == 0``, so only gravity
    contributes; the resulting ``x_inertia`` is never fed into the inertia RHS
    term (``add_inertia_to_rhs`` weights by mass, which is 0 for pins).
    """
    tid = wp.tid()
    w_idx = wp.max(particle_world[tid], 0)
    g = gravity[w_idx]
    im = inv_mass[tid]
    a = (
        f_ext[tid] * im + g
    )  # apply gravity unconditionally; pin softness is enforced elsewhere via large diagonal + x_ref RHS scatter
    x_inertia[tid] = x_prev[tid] + v_prev[tid] * dt + a * (dt * dt)


@wp.kernel
def add_inertia_to_rhs_kernel(
    x_inertia: wp.array[wp.vec3],
    mass: wp.array[wp.float32],
    dt: float,
    rhs: wp.array[wp.vec3],
):
    """rhs += (m/dt^2) * x_inertia.  Free-particle term only; mass==0 for pins, contributes 0."""
    tid = wp.tid()
    m = mass[tid]
    w = m / (dt * dt)
    rhs[tid] = rhs[tid] + x_inertia[tid] * w


@wp.kernel
def write_velocity_kernel(
    x_prev: wp.array[wp.vec3],
    x_new: wp.array[wp.vec3],
    inv_mass: wp.array[wp.float32],
    dt: float,
    v_new: wp.array[wp.vec3],
):
    """v_new = (x_new - x_prev) / dt for free particles; zero for pins."""
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        v_new[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v_new[tid] = (x_new[tid] - x_prev[tid]) / dt


@wp.func
def svd_3x2(F: mat32) -> mat32:
    """ARAP projection: SVD-clamp 3x2 deformation gradient to nearest rotation.

    Reduces 3x2 SVD to a 2x2 symmetric eigenproblem on F^T F and uses Warp's
    built-in ``wp.svd2``. ``svd2(FtF)`` returns the singular values of FtF,
    which equal the squared singular values of F (since FtF is symmetric PSD).
    ARAP clamps singular values to 1, yielding the nearest rigid map
    ``P = U V^T`` (3x2).

    Note: ``wp.mat32`` is absent in Warp 1.14; ``mat32`` is defined at module
    level as ``wp.types.matrix(shape=(3, 2), dtype=wp.float32)``.
    """
    FtF = wp.transpose(F) * F  # 2x2 symmetric PSD

    # wp.svd2 of symmetric PSD: U == V (up to sign); use V as right singular vectors of F
    _U2, sigma_sq, V2 = wp.svd2(FtF)

    s0 = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1 = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    # U columns of the 3x2 SVD: u_i = F v_i / s_i
    v0 = wp.vec2(V2[0, 0], V2[1, 0])
    v1 = wp.vec2(V2[0, 1], V2[1, 1])
    u0 = (F * v0) / s0  # 3-vector
    u1 = (F * v1) / s1  # 3-vector

    # ARAP projection: P = U V^T (3x2) with singular values clamped to 1.
    p_col0 = u0 * V2[0, 0] + u1 * V2[0, 1]
    p_col1 = u0 * V2[1, 0] + u1 * V2[1, 1]
    return mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )


@wp.func
def project_arap_3x3(F: wp.mat33) -> wp.mat33:
    """ARAP 3D projection: F -> nearest proper rotation R.

    Uses Warp's ``wp.svd3`` then handles the reflection case:
    if ``det(U) * det(V) < 0``, flip the last column of U so that
    ``R = U' * V^T`` has ``det(R) = +1``.

    Args:
        F: 3x3 deformation gradient.

    Returns:
        R: 3x3 proper rotation (nearest rotation to F).
    """
    U, _sigma, V = wp.svd3(F)

    # Detect reflection: det(U) * det(V) < 0 means R = U*V^T would have det = -1.
    # Fix by flipping the column of U corresponding to the smallest singular value.
    # wp.svd3 returns sigma in descending order, so index 2 is the smallest.
    detUV = wp.determinant(U) * wp.determinant(V)
    if detUV < 0.0:
        # Flip column 2 of U. wp.mat33 is row-major: U[row, col].
        # To flip col 2: negate U[0,2], U[1,2], U[2,2].
        U = wp.mat33(
            U[0, 0],
            U[0, 1],
            -U[0, 2],
            U[1, 0],
            U[1, 1],
            -U[1, 2],
            U[2, 0],
            U[2, 1],
            -U[2, 2],
        )
    return U * wp.transpose(V)


@wp.kernel
def project_stretching_arap_tet_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-tet ARAP local projection scatter for PD softbody.

    Computes F = Ds * Dm_inv (3x3), projects to nearest rotation R via SVD
    with reflection handling, and scatters ``w * Dm_inv * R^T`` into the four
    stencil vertices via atomic_add.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        rhs: Output RHS accumulator (atomic-add target), shape ``[particle_count]``.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    # Ds = [p1-p0 | p2-p0 | p3-p0]  (column-stack 3 edge vectors -> 3x3)
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    R = project_arap_3x3(F)

    # proj = w * Dm_inv * R^T  (3x3)
    w = tet_weight[t]
    RT = wp.transpose(R)
    proj = w * (Dm_inv * RT)

    # Scatter stencil (RealSim PDTetrahedronEnergy.cpp:191-194):
    #   rhs[t[0]] += -proj.row(0) - proj.row(1) - proj.row(2)
    #   rhs[t[1]] += proj.row(0)
    #   rhs[t[2]] += proj.row(1)
    #   rhs[t[3]] += proj.row(2)
    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    wp.atomic_add(rhs, i0, -(row0 + row1 + row2))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)
    wp.atomic_add(rhs, i3, row2)


@wp.kernel
def project_stretching_arap_tet_compute_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Compute per-tet ARAP contribution to each of its 4 local vertices.

    Same math as :func:`project_stretching_arap_tet_kernel` but writes to a
    pre-allocated ``(T, 4)`` scratch buffer instead of atomic-adding into rhs.
    Pair with :func:`gather_per_particle_kernel` for the deterministic
    reduction.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        contributions: Output ``(tet_count, 4)`` per-local-vertex contribution.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    R = project_arap_3x3(F)

    w = tet_weight[t]
    RT = wp.transpose(R)
    proj = w * (Dm_inv * RT)

    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    contributions[t, 0] = -(row0 + row1 + row2)
    contributions[t, 1] = row0
    contributions[t, 2] = row1
    contributions[t, 3] = row2


@wp.func
def project_corotational_sigma3d(sigma: wp.vec3, mu: float, lam: float) -> wp.vec3:
    """Closed-form 3D corotational local projection on SVD singular values.

    Minimises E(s) = mu*||s-I||^2 + (lam/2)*tr(s-I)^2 + (k/2)*||s-s0||^2
    with k = 2*mu (PD penalty weight), I = (1,1,1) identity singular values.

    Setting grad_E = 0 yields the 3x3 linear system (4*mu*I + lam*11^T)*s = b,
    solved in closed form via Sherman-Morrison:

        alpha = 4*mu,  beta = lam,  n = 3
        inv(alpha*I + beta*11^T) = (1/alpha)*I - beta/(alpha*(alpha + n*beta))*11^T
        b_i = 2*mu + 3*lam + 2*mu*s0_i
        sum_b = 6*mu + 9*lam + 2*mu*(s0[0]+s0[1]+s0[2])
        s_proj[i] = b_i/alpha - beta*sum_b / (alpha*(alpha + 3*beta))

    Note: uses the standard symmetric trace formula (NOT mimicking RealSim's
    Eigen .trace() bug on Vector2d which only reads index 0). The 2D cloth
    Corot in this codebase uses the same standard formulation; this 3D tet
    Corot is the natural 3D extension of the same energy.

    See ``docs/superpowers/specs/2026-05-15-fba-corot-realsim-discrepancy.md``
    for details on the RealSim Eigen .trace() discrepancy (applies to both 2D
    cloth and 3D tet corotational).

    Args:
        sigma: Singular values from ``wp.svd3(F)``, shape (3,), in descending order.
        mu: First Lame parameter (shear modulus) [Pa].
        lam: Second Lame parameter [Pa].

    Returns:
        Projected singular-value triple ``(s_proj_0, s_proj_1, s_proj_2)``.
    """
    k = 2.0 * mu  # PD penalty weight
    alpha = 4.0 * mu  # diagonal of system matrix (2*mu + lam + k = 4*mu + lam) minus lam*11^T rank-1 part
    beta = lam
    n = 3.0

    b0 = 2.0 * mu + 3.0 * lam + k * sigma[0]
    b1 = 2.0 * mu + 3.0 * lam + k * sigma[1]
    b2 = 2.0 * mu + 3.0 * lam + k * sigma[2]
    sum_b = b0 + b1 + b2

    scale = beta * sum_b / (alpha * (alpha + n * beta))
    proj0 = b0 / alpha - scale
    proj1 = b1 / alpha - scale
    proj2 = b2 / alpha - scale
    return wp.vec3(proj0, proj1, proj2)


@wp.kernel
def project_stretching_corotational_tet_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-tet corotational local projection scatter for PD softbody.

    Computes F = Ds * Dm_inv (3x3), extracts singular values via ``wp.svd3``,
    applies the closed-form 3D corotational projection to get ``sigma_proj``,
    reconstructs ``P = U * diag(sigma_proj) * Vt``, and scatters
    ``w * Dm_inv * P^T`` into the four stencil vertices via atomic_add.

    No reflection handling is needed: ``wp.svd3`` returns non-negative singular
    values and the corotational projection preserves positivity; the
    reconstructed P naturally has the correct orientation.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        mu: First Lamé parameter [Pa].
        lam: Second Lamé parameter [Pa].
        rhs: Output RHS accumulator (atomic-add target), shape ``[particle_count]``.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    # Ds = [p1-p0 | p2-p0 | p3-p0]  (column-stack 3 edge vectors -> 3x3)
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    # Corotational projection: closed-form 3D solve on singular values.
    sigma_proj = project_corotational_sigma3d(wp.vec3(sigma[0], sigma[1], sigma[2]), mu, lam)

    # Reconstruct P = U * diag(sigma_proj) * V^T  (3x3).
    # P = (U * diag(s)) * V^T
    s0 = sigma_proj[0]
    s1 = sigma_proj[1]
    s2 = sigma_proj[2]
    US = wp.mat33(
        U[0, 0] * s0,
        U[0, 1] * s1,
        U[0, 2] * s2,
        U[1, 0] * s0,
        U[1, 1] * s1,
        U[1, 2] * s2,
        U[2, 0] * s0,
        U[2, 1] * s1,
        U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    # proj = w * Dm_inv * P^T  (3x3)
    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    # Scatter stencil (RealSim PDTetrahedronEnergy.cpp:191-194):
    #   rhs[t[0]] += -proj.row(0) - proj.row(1) - proj.row(2)
    #   rhs[t[1]] += proj.row(0)
    #   rhs[t[2]] += proj.row(1)
    #   rhs[t[3]] += proj.row(2)
    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    wp.atomic_add(rhs, i0, -(row0 + row1 + row2))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)
    wp.atomic_add(rhs, i3, row2)


@wp.kernel
def project_stretching_corotational_tet_compute_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Compute per-tet corotational contribution to each of its 4 local vertices.

    Same math as :func:`project_stretching_corotational_tet_kernel` but writes
    to a pre-allocated ``(T, 4)`` scratch buffer instead of atomic-adding into
    rhs. Pair with :func:`gather_per_particle_kernel` for the deterministic
    reduction.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        mu: First Lamé parameter [Pa].
        lam: Second Lamé parameter [Pa].
        contributions: Output ``(tet_count, 4)`` per-local-vertex contribution.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    sigma_proj = project_corotational_sigma3d(wp.vec3(sigma[0], sigma[1], sigma[2]), mu, lam)

    s0 = sigma_proj[0]
    s1 = sigma_proj[1]
    s2 = sigma_proj[2]
    US = wp.mat33(
        U[0, 0] * s0,
        U[0, 1] * s1,
        U[0, 2] * s2,
        U[1, 0] * s0,
        U[1, 1] * s1,
        U[1, 2] * s2,
        U[2, 0] * s0,
        U[2, 1] * s1,
        U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    contributions[t, 0] = -(row0 + row1 + row2)
    contributions[t, 1] = row0
    contributions[t, 2] = row1
    contributions[t, 3] = row2


# ---------------------------------------------------------------------------
# RealSim-faithful LBFGS port for Tri/Tet Neo-Hookean local projection.
#
# Mirrors RealSim's ``mcl::optlib::LBFGS<double, DIM>`` (M = 8, ``c_1 = 1e-4``,
# tol = 1e-6, max_iters = 50, BacktrackingCubic line search) used at
# ``PDNeohookeanTriangleEnergyParallel.h:74`` and
# ``PDNeohookeanTetrahedronEnergyParallel.h:80``. The Tri (2D) and Tet (3D)
# variants are kept separate (faithful to Q12) for clarity. All arithmetic
# is in ``wp.float64`` (Q13). USER-CONFIRMED SUBSTITUTIONS (Q9):
#   - ``nh_value_2d/3d`` returns ``1e30`` instead of ``+infinity`` for sigma_i < 0
#     (Warp ``@wp.func`` cannot represent semantically clean +infinity without NaN
#     risk; RealSim itself returns ``std::numeric_limits<float>::max()`` ~=
#     3.4e38, not literal infinity).
#   - ``nh_gradient_2d/3d`` clamps sigma_i >= 1e-6 internally instead of
#     RealSim's ``std::runtime_error`` (Warp cannot throw).
# ---------------------------------------------------------------------------


@wp.func
def nh_value_2d(
    sigma: wp.vec2d,
    sigma_init: wp.vec2d,
    mu: wp.float64,
    lam: wp.float64,
    k: wp.float64,
) -> wp.float64:
    """Tri NH objective value Psi(sigma) + (k/2)||sigma-sigma_init||^2.

    Mirrors ``NHProjectionProblem2D::value`` (HyperelasticProblemS.h:93-102)
    and ``energy_density`` (HyperelasticProblemS.h:81-91). For sigma_i < 0 returns
    a huge finite barrier ``1e30`` (substitution per Q9). For sigma_i >= 0 returns
    ``(mu/2)(sigma_0^2 + sigma_1^2 - 2*log(J) - 3) + (lam/2)*log^2(J) + (k/2)||sigma-sigma_init||^2``
    with ``J = sigma_0*sigma_1``.
    """
    if sigma[0] < wp.float64(0.0) or sigma[1] < wp.float64(0.0):
        return _NH_VALUE_INF
    s0 = sigma[0]
    s1 = sigma[1]
    j = s0 * s1
    # Guard log domain in case the line search lands on sigma_i > 0 but j extremely small.
    j = wp.max(j, _NH_EPS * _NH_EPS)
    log_j = wp.log(j)
    log_i3 = wp.float64(2.0) * log_j  # log(I_3) = log(J^2) = 2*log(J)
    i1 = s0 * s0 + s1 * s1
    psi = wp.float64(0.5) * mu * (i1 - log_i3 - wp.float64(3.0)) + wp.float64(0.125) * lam * log_i3 * log_i3
    d0 = s0 - sigma_init[0]
    d1 = s1 - sigma_init[1]
    return psi + wp.float64(0.5) * k * (d0 * d0 + d1 * d1)


@wp.func
def nh_gradient_2d(
    sigma: wp.vec2d,
    sigma_init: wp.vec2d,
    mu: wp.float64,
    lam: wp.float64,
    k: wp.float64,
) -> wp.vec2d:
    """Tri NH gradient grad Psi + k(sigma - sigma_init).

    Mirrors ``NHProjectionProblem2D::gradient`` (HyperelasticProblemS.h:104-116).
    sigma_i is clamped to ``1e-6`` internally before computing ``log(J)`` and
    ``1/sigma_i`` (Q9 substitution: RealSim throws ``runtime_error`` when J <= 0;
    Warp cannot throw, and the LBFGS line search will reject any step where
    ``value()`` returned the +infinity barrier so a gracefully-clamped gradient is
    never actually consumed for an accepted step).
    """
    s0 = wp.max(sigma[0], _NH_EPS)
    s1 = wp.max(sigma[1], _NH_EPS)
    j = s0 * s1
    log_j = wp.log(j)
    inv0 = wp.float64(1.0) / s0
    inv1 = wp.float64(1.0) / s1
    g0 = mu * (s0 - inv0) + lam * log_j * inv0 + k * (sigma[0] - sigma_init[0])
    g1 = mu * (s1 - inv1) + lam * log_j * inv1 + k * (sigma[1] - sigma_init[1])
    return wp.vec2d(g0, g1)


@wp.func
def nh_value_3d(
    sigma: wp.vec3d,
    sigma_init: wp.vec3d,
    mu: wp.float64,
    lam: wp.float64,
    k: wp.float64,
) -> wp.float64:
    """Tet NH objective value. Mirrors ``NHProjectionProblem3D::value``
    (HyperelasticProblemS.h:35-44, energy at :23-33). sigma_i < 0 returns 1e30."""
    if sigma[0] < wp.float64(0.0) or sigma[1] < wp.float64(0.0) or sigma[2] < wp.float64(0.0):
        return _NH_VALUE_INF
    s0 = sigma[0]
    s1 = sigma[1]
    s2 = sigma[2]
    j = s0 * s1 * s2
    j = wp.max(j, _NH_EPS * _NH_EPS * _NH_EPS)
    log_j = wp.log(j)
    log_i3 = wp.float64(2.0) * log_j
    i1 = s0 * s0 + s1 * s1 + s2 * s2
    psi = wp.float64(0.5) * mu * (i1 - log_i3 - wp.float64(3.0)) + wp.float64(0.125) * lam * log_i3 * log_i3
    d0 = s0 - sigma_init[0]
    d1 = s1 - sigma_init[1]
    d2 = s2 - sigma_init[2]
    return psi + wp.float64(0.5) * k * (d0 * d0 + d1 * d1 + d2 * d2)


@wp.func
def nh_gradient_3d(
    sigma: wp.vec3d,
    sigma_init: wp.vec3d,
    mu: wp.float64,
    lam: wp.float64,
    k: wp.float64,
) -> wp.vec3d:
    """Tet NH gradient. Mirrors ``NHProjectionProblem3D::gradient``
    (HyperelasticProblemS.h:46-59). sigma_i clamped to 1e-6 inside (Q9)."""
    s0 = wp.max(sigma[0], _NH_EPS)
    s1 = wp.max(sigma[1], _NH_EPS)
    s2 = wp.max(sigma[2], _NH_EPS)
    j = s0 * s1 * s2
    log_j = wp.log(j)
    inv0 = wp.float64(1.0) / s0
    inv1 = wp.float64(1.0) / s1
    inv2 = wp.float64(1.0) / s2
    g0 = mu * (s0 - inv0) + lam * log_j * inv0 + k * (sigma[0] - sigma_init[0])
    g1 = mu * (s1 - inv1) + lam * log_j * inv1 + k * (sigma[1] - sigma_init[1])
    g2 = mu * (s2 - inv2) + lam * log_j * inv2 + k * (sigma[2] - sigma_init[2])
    return wp.vec3d(g0, g1, g2)


@wp.func
def _cubic_step(
    fx0: wp.float64,
    gtp: wp.float64,
    fxa: wp.float64,
    alpha: wp.float64,
    fxp: wp.float64,
    alphap: wp.float64,
) -> wp.float64:
    """Cubic-interpolation step length per Backtracking.hpp:129-143.

    Solves the 2x2 system for the cubic coefficients (r[0], r[1]) fitted to
    f(alpha), f'(alpha), f(alpha_p), f(alpha). Returns the larger root of the derived
    quadratic; falls back to the secant form if the cubic degenerates to a
    quadratic (``r[0] ~= 0``).
    """
    mult = wp.float64(1.0) / (alpha * alpha * alphap * alphap * (alpha - alphap))
    a00 = alphap * alphap
    a01 = -alpha * alpha
    a10 = -alphap * alphap * alphap
    a11 = alpha * alpha * alpha
    b0 = fxa - fx0 - alpha * gtp
    b1 = fxp - fx0 - alphap * gtp
    r0 = mult * (a00 * b0 + a01 * b1)
    r1 = mult * (a10 * b0 + a11 * b1)
    if wp.abs(r0) <= wp.float64(0.0):
        # Cubic degenerated to quadratic: alpha_new = -gtp / (2*r1).
        return -gtp / (wp.float64(2.0) * r1)
    disc = r1 * r1 - wp.float64(3.0) * r0 * gtp
    d = wp.sqrt(wp.max(disc, wp.float64(0.0)))
    return (-r1 + d) / (wp.float64(3.0) * r0)


@wp.func
def _clamp_range(alpha: wp.float64, low: wp.float64, high: wp.float64) -> wp.float64:
    """Backtracking.hpp:116-120 ``range`` clamp."""
    if alpha < low:
        return low
    if alpha > high:
        return high
    return alpha


@wp.func
def _backtracking_cubic_2d(
    x: wp.vec2d,
    x0: wp.vec2d,
    p: wp.vec2d,
    fx0: wp.float64,
    gtp: wp.float64,
    alpha0: wp.float64,
    mu: wp.float64,
    lam: wp.float64,
    k: wp.float64,
) -> wp.float64:
    """BacktrackingCubic Armijo line search for Tri NH.

    Mirrors ``BacktrackingCubic::search`` (Backtracking.hpp:79-113). ``fx0``
    and ``gtp = grad f(x)*p`` are passed in (already computed by the caller) --
    RealSim recomputes them inside ``search`` but doing so wastes a gradient
    eval per LBFGS iter (section 3.4 of the spec). Returns ``-1`` on failure (Armijo
    not satisfied within the inner iteration cap).
    """
    p_norm_sq = p[0] * p[0] + p[1] * p[1]
    if p_norm_sq <= wp.float64(0.0):
        return _LBFGS_C1  # Matches Backtracking.hpp:43 (return decrease).

    alpha = alpha0
    fxp = fx0
    alphap = alpha

    for _i in range(_LBFGS_LS_MAX_ITERS):
        x_trial = wp.vec2d(x[0] + alpha * p[0], x[1] + alpha * p[1])
        fxa = nh_value_2d(x_trial, x0, mu, lam, k)
        armijo_rhs = fx0 + alpha * _LBFGS_C1 * gtp
        if fxa <= armijo_rhs:
            return alpha

        if _i == 0:
            # First-failure: quadratic interpolation (Backtracking.hpp:99-100).
            alpha_tmp = gtp / (wp.float64(2.0) * (fx0 + gtp - fxa))
        else:
            alpha_tmp = _cubic_step(fx0, gtp, fxa, alpha, fxp, alphap)

        fxp = fxa
        alphap = alpha
        alpha = _clamp_range(alpha_tmp, wp.float64(0.1) * alpha, wp.float64(0.5) * alpha)

    return wp.float64(-1.0)


@wp.func
def _backtracking_cubic_3d(
    x: wp.vec3d,
    x0: wp.vec3d,
    p: wp.vec3d,
    fx0: wp.float64,
    gtp: wp.float64,
    alpha0: wp.float64,
    mu: wp.float64,
    lam: wp.float64,
    k: wp.float64,
) -> wp.float64:
    """Tet variant of the BacktrackingCubic Armijo line search."""
    p_norm_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
    if p_norm_sq <= wp.float64(0.0):
        return _LBFGS_C1

    alpha = alpha0
    fxp = fx0
    alphap = alpha

    for _i in range(_LBFGS_LS_MAX_ITERS):
        x_trial = wp.vec3d(x[0] + alpha * p[0], x[1] + alpha * p[1], x[2] + alpha * p[2])
        fxa = nh_value_3d(x_trial, x0, mu, lam, k)
        armijo_rhs = fx0 + alpha * _LBFGS_C1 * gtp
        if fxa <= armijo_rhs:
            return alpha

        if _i == 0:
            alpha_tmp = gtp / (wp.float64(2.0) * (fx0 + gtp - fxa))
        else:
            alpha_tmp = _cubic_step(fx0, gtp, fxa, alpha, fxp, alphap)

        fxp = fxa
        alphap = alpha
        alpha = _clamp_range(alpha_tmp, wp.float64(0.1) * alpha, wp.float64(0.5) * alpha)

    return wp.float64(-1.0)


@wp.func
def project_neohookean_sigma2d_lbfgs(
    sigma_init: wp.vec2d,
    mu: wp.float64,
    lam: wp.float64,
) -> wp.vec2d:
    """Tri NH local projection via RealSim's LBFGS (M=8, BacktrackingCubic).

    Faithful port of ``mcl::optlib::LBFGS<double, 2>::minimize`` (LBFGS.hpp:52-151)
    applied to ``NHProjectionProblem2D`` (HyperelasticProblemS.h:73-120) at
    PDNeohookeanTriangleEnergyParallel.h:74. History is freshly zeroed per
    call (LBFGS.hpp:53-67 -- stack-local), matching the per-element /
    per-PD-outer-iter reset semantics. Convergence: ``||grad|| < 1e-6`` OR
    ``||x_prev - x|| < 1e-6`` (HyperelasticProblemS.h:67-70). Returns ``sigma_init``
    on line-search failure (RealSim's FAILURE escape).
    """
    k = wp.float64(2.0) * mu

    # All-tiny clamp (PDNeohookeanTriangleEnergyParallel.h:68-72). Note the
    # ``sigma_2 < 0`` flip in the Tet wrapper is *omitted* in the Tri wrapper
    # because ``Eigen::JacobiSVD<Mat3x2>`` returns non-negative singular
    # values natively. Our Tri caller uses ``wp.svd2(F^T F)`` which yields
    # eigenvalues >= 0, then we take ``sqrt`` -- also non-negative.
    s0 = sigma_init[0]
    s1 = sigma_init[1]
    if wp.abs(s0) < _NH_EPS and wp.abs(s1) < _NH_EPS:
        s0 = _NH_EPS
        s1 = _NH_EPS
    x = wp.vec2d(s0, s1)
    x0 = x  # Quadratic-penalty anchor (RealSim's ``sigma0``, LBFGS.hpp:52 ``x0``).

    # History storage: ``s`` and ``y`` columns hold history pairs (oldest in
    # col 0). ``alpha``/``rho`` are two-loop scratch. All zeroed.
    s_hist = _lbfgs_mat_2xM()
    y_hist = _lbfgs_mat_2xM()
    alpha_vec = _lbfgs_vec_M()
    rho_vec = _lbfgs_vec_M()

    grad = nh_gradient_2d(x, x0, mu, lam, k)
    gamma_k = wp.float64(1.0)
    alpha_init = wp.float64(1.0)

    # ``history_count`` plays the role of ``min(M, k)`` in RealSim's loop. We
    # use a budget counter and a fixed Python ``range`` (Warp can't
    # mutate the loop var). Restart sets ``history_count = 0`` and decrements
    # ``iters_remaining`` (LBFGS.hpp:105-106 ``max_iters -= k; k = 0``).
    history_count = int(0)
    iters_remaining = _LBFGS_MAX_ITERS

    for _it in range(_LBFGS_MAX_ITERS):
        if iters_remaining <= 0:
            break

        x_old = x
        grad_old = grad
        q = grad

        # L-BFGS first-loop recursion (LBFGS.hpp:86-92), newest -> oldest.
        # Note: RealSim's iter = min(M, k) where ``k`` is the iteration index;
        # since we evict on overflow, ``history_count`` is exactly that value.
        iter_count = history_count
        if iter_count > _LBFGS_M:
            iter_count = _LBFGS_M
        for i_rev in range(_LBFGS_M):
            i = iter_count - 1 - i_rev
            if i < 0:
                break
            sy = s_hist[0, i] * y_hist[0, i] + s_hist[1, i] * y_hist[1, i]
            rho_i = wp.float64(1.0) / sy
            rho_vec[i] = rho_i
            sq = s_hist[0, i] * q[0] + s_hist[1, i] * q[1]
            a_i = rho_i * sq
            alpha_vec[i] = a_i
            q = wp.vec2d(q[0] - a_i * y_hist[0, i], q[1] - a_i * y_hist[1, i])

        # Initial Hessian scaling H_0 = gamma_k * I (Nocedal-Wright "method 1").
        q = wp.vec2d(gamma_k * q[0], gamma_k * q[1])

        # L-BFGS second-loop recursion (LBFGS.hpp:94-99), oldest -> newest.
        for i in range(_LBFGS_M):
            if i >= iter_count:
                break
            qy = q[0] * y_hist[0, i] + q[1] * y_hist[1, i]
            beta = rho_vec[i] * qy
            coeff = alpha_vec[i] - beta
            q = wp.vec2d(q[0] + coeff * s_hist[0, i], q[1] + coeff * s_hist[1, i])

        # Non-descent restart (LBFGS.hpp:101-108).
        dir_dot = q[0] * grad[0] + q[1] * grad[1]
        restarted = False
        if dir_dot <= wp.float64(0.0):
            q = grad
            history_count = 0
            inf_norm = wp.max(wp.abs(grad[0]), wp.abs(grad[1]))
            if inf_norm > wp.float64(0.0):
                alpha_init = wp.min(wp.float64(1.0), wp.float64(1.0) / inf_norm)
            else:
                alpha_init = wp.float64(1.0)
            restarted = True

        # Line search on direction p = -q.
        p_search = wp.vec2d(-q[0], -q[1])
        # fx0 = f(x); gtp = grad f(x)*p = -q*grad. RealSim recomputes the
        # gradient inside ``BacktrackingCubic::search``; we pass it through.
        fx0 = nh_value_2d(x, x0, mu, lam, k)
        gtp = grad[0] * p_search[0] + grad[1] * p_search[1]
        rate = _backtracking_cubic_2d(x, x0, p_search, fx0, gtp, alpha_init, mu, lam, k)
        if rate <= wp.float64(0.0):
            # Linesearch failure -> RealSim returns FAILURE (LBFGS.hpp:113-114).
            return x

        x_last = x
        x = wp.vec2d(x[0] - rate * q[0], x[1] - rate * q[1])
        iters_remaining -= 1

        # Convergence on grad (at x_old, pre-recompute -- faithful) OR step.
        gn = wp.sqrt(grad[0] * grad[0] + grad[1] * grad[1])
        step = wp.vec2d(x_last[0] - x[0], x_last[1] - x[1])
        sn = wp.sqrt(step[0] * step[0] + step[1] * step[1])
        if gn < _LBFGS_TOL or sn < _LBFGS_TOL:
            break

        grad = nh_gradient_2d(x, x0, mu, lam, k)
        s_temp = wp.vec2d(x[0] - x_old[0], x[1] - x_old[1])
        y_temp = wp.vec2d(grad[0] - grad_old[0], grad[1] - grad_old[1])

        if history_count < _LBFGS_M:
            slot = history_count
            s_hist[0, slot] = s_temp[0]
            s_hist[1, slot] = s_temp[1]
            y_hist[0, slot] = y_temp[0]
            y_hist[1, slot] = y_temp[1]
            history_count += 1
        else:
            # Evict oldest, shift left, append newest at col M-1
            # (LBFGS.hpp:131-134 non-circular buffer; semantically equivalent
            # to a circular index with the same first/second-loop ordering).
            for j in range(_LBFGS_M - 1):
                s_hist[0, j] = s_hist[0, j + 1]
                s_hist[1, j] = s_hist[1, j + 1]
                y_hist[0, j] = y_hist[0, j + 1]
                y_hist[1, j] = y_hist[1, j + 1]
            s_hist[0, _LBFGS_M - 1] = s_temp[0]
            s_hist[1, _LBFGS_M - 1] = s_temp[1]
            y_hist[0, _LBFGS_M - 1] = y_temp[0]
            y_hist[1, _LBFGS_M - 1] = y_temp[1]

        denom = y_temp[0] * y_temp[0] + y_temp[1] * y_temp[1]
        if wp.abs(denom) <= wp.float64(0.0):
            break
        sy = s_temp[0] * y_temp[0] + s_temp[1] * y_temp[1]
        gamma_k = sy / denom

        if restarted:
            # On a restart the ``alpha_init`` was set to min(1, 1/||g||inf); the
            # *next* iter resets to 1.0 per RealSim LBFGS.hpp:145.
            alpha_init = wp.float64(1.0)
        else:
            alpha_init = wp.float64(1.0)

    return x


@wp.func
def project_neohookean_sigma3d_lbfgs(
    sigma_init: wp.vec3d,
    mu: wp.float64,
    lam: wp.float64,
) -> wp.vec3d:
    """Tet NH local projection via RealSim's LBFGS (M=8, BacktrackingCubic).

    Faithful port of ``mcl::optlib::LBFGS<double, 3>::minimize`` (LBFGS.hpp:52-151)
    applied to ``NHProjectionProblem3D`` (HyperelasticProblemS.h:15-63) at
    PDNeohookeanTetrahedronEnergyParallel.h:80. Pre-LBFGS sigma fix-up
    mirrors PDNeohookeanTetrahedronEnergyParallel.h:69-78: all-three-tiny
    clamp + sigma_2 sign flip. The sigma_2 flip is a no-op in practice because
    ``wp.svd3`` returns non-negative singular values (Q10 -- surfaced).
    """
    k = wp.float64(2.0) * mu

    s0 = sigma_init[0]
    s1 = sigma_init[1]
    s2 = sigma_init[2]
    if wp.abs(s0) < _NH_EPS and wp.abs(s1) < _NH_EPS and wp.abs(s2) < _NH_EPS:
        s0 = _NH_EPS
        s1 = _NH_EPS
        s2 = _NH_EPS
    if s2 < wp.float64(0.0):
        s2 = -s2  # No-op for wp.svd3 outputs (>= 0), kept for parity.
    x = wp.vec3d(s0, s1, s2)
    x0 = x

    s_hist = _lbfgs_mat_3xM()
    y_hist = _lbfgs_mat_3xM()
    alpha_vec = _lbfgs_vec_M()
    rho_vec = _lbfgs_vec_M()

    grad = nh_gradient_3d(x, x0, mu, lam, k)
    gamma_k = wp.float64(1.0)
    alpha_init = wp.float64(1.0)

    history_count = int(0)
    iters_remaining = _LBFGS_MAX_ITERS

    for _it in range(_LBFGS_MAX_ITERS):
        if iters_remaining <= 0:
            break

        x_old = x
        grad_old = grad
        q = grad

        iter_count = history_count
        if iter_count > _LBFGS_M:
            iter_count = _LBFGS_M
        for i_rev in range(_LBFGS_M):
            i = iter_count - 1 - i_rev
            if i < 0:
                break
            sy = s_hist[0, i] * y_hist[0, i] + s_hist[1, i] * y_hist[1, i] + s_hist[2, i] * y_hist[2, i]
            rho_i = wp.float64(1.0) / sy
            rho_vec[i] = rho_i
            sq = s_hist[0, i] * q[0] + s_hist[1, i] * q[1] + s_hist[2, i] * q[2]
            a_i = rho_i * sq
            alpha_vec[i] = a_i
            q = wp.vec3d(
                q[0] - a_i * y_hist[0, i],
                q[1] - a_i * y_hist[1, i],
                q[2] - a_i * y_hist[2, i],
            )

        q = wp.vec3d(gamma_k * q[0], gamma_k * q[1], gamma_k * q[2])

        for i in range(_LBFGS_M):
            if i >= iter_count:
                break
            qy = q[0] * y_hist[0, i] + q[1] * y_hist[1, i] + q[2] * y_hist[2, i]
            beta = rho_vec[i] * qy
            coeff = alpha_vec[i] - beta
            q = wp.vec3d(
                q[0] + coeff * s_hist[0, i],
                q[1] + coeff * s_hist[1, i],
                q[2] + coeff * s_hist[2, i],
            )

        dir_dot = q[0] * grad[0] + q[1] * grad[1] + q[2] * grad[2]
        restarted = False
        if dir_dot <= wp.float64(0.0):
            q = grad
            history_count = 0
            inf_norm = wp.max(wp.max(wp.abs(grad[0]), wp.abs(grad[1])), wp.abs(grad[2]))
            if inf_norm > wp.float64(0.0):
                alpha_init = wp.min(wp.float64(1.0), wp.float64(1.0) / inf_norm)
            else:
                alpha_init = wp.float64(1.0)
            restarted = True

        p_search = wp.vec3d(-q[0], -q[1], -q[2])
        fx0 = nh_value_3d(x, x0, mu, lam, k)
        gtp = grad[0] * p_search[0] + grad[1] * p_search[1] + grad[2] * p_search[2]
        rate = _backtracking_cubic_3d(x, x0, p_search, fx0, gtp, alpha_init, mu, lam, k)
        if rate <= wp.float64(0.0):
            return x

        x_last = x
        x = wp.vec3d(x[0] - rate * q[0], x[1] - rate * q[1], x[2] - rate * q[2])
        iters_remaining -= 1

        gn = wp.sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2])
        step = wp.vec3d(x_last[0] - x[0], x_last[1] - x[1], x_last[2] - x[2])
        sn = wp.sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2])
        if gn < _LBFGS_TOL or sn < _LBFGS_TOL:
            break

        grad = nh_gradient_3d(x, x0, mu, lam, k)
        s_temp = wp.vec3d(x[0] - x_old[0], x[1] - x_old[1], x[2] - x_old[2])
        y_temp = wp.vec3d(grad[0] - grad_old[0], grad[1] - grad_old[1], grad[2] - grad_old[2])

        if history_count < _LBFGS_M:
            slot = history_count
            s_hist[0, slot] = s_temp[0]
            s_hist[1, slot] = s_temp[1]
            s_hist[2, slot] = s_temp[2]
            y_hist[0, slot] = y_temp[0]
            y_hist[1, slot] = y_temp[1]
            y_hist[2, slot] = y_temp[2]
            history_count += 1
        else:
            for j in range(_LBFGS_M - 1):
                s_hist[0, j] = s_hist[0, j + 1]
                s_hist[1, j] = s_hist[1, j + 1]
                s_hist[2, j] = s_hist[2, j + 1]
                y_hist[0, j] = y_hist[0, j + 1]
                y_hist[1, j] = y_hist[1, j + 1]
                y_hist[2, j] = y_hist[2, j + 1]
            s_hist[0, _LBFGS_M - 1] = s_temp[0]
            s_hist[1, _LBFGS_M - 1] = s_temp[1]
            s_hist[2, _LBFGS_M - 1] = s_temp[2]
            y_hist[0, _LBFGS_M - 1] = y_temp[0]
            y_hist[1, _LBFGS_M - 1] = y_temp[1]
            y_hist[2, _LBFGS_M - 1] = y_temp[2]

        denom = y_temp[0] * y_temp[0] + y_temp[1] * y_temp[1] + y_temp[2] * y_temp[2]
        if wp.abs(denom) <= wp.float64(0.0):
            break
        sy = s_temp[0] * y_temp[0] + s_temp[1] * y_temp[1] + s_temp[2] * y_temp[2]
        gamma_k = sy / denom

        if restarted:
            alpha_init = wp.float64(1.0)
        else:
            alpha_init = wp.float64(1.0)

    return x


@wp.func
def project_neohookean_sigma3d(sigma: wp.vec3, mu: float, lam: float) -> wp.vec3:
    """Project SVD singular values to 3D Neo-Hookean PD equilibrium via 5 Newton iterations.

    Solves: min_{s}  E_NH(s) + (k/2)*||s - s0||^2
    where k = 2*mu (PD penalty weight) and
    E_NH(s) = (mu/2)*(I1 - 2*log(J) - 3) + (lam/2)*log(J)^2
    with I1 = s0^2 + s1^2 + s2^2, J = s0*s1*s2.

    Matches RealSim NHProjectionProblem3D::energy_density (using log_I3 = 2*log(J),
    so 0.125*lam*log_I3^2 = (lam/2)*log^2(J)). Sign verified against RealSim reference.

    Gradient:
        grad_i = mu*(s_i - 1/s_i) + lam*log(J)/s_i + k*(s_i - s0_i)

    Hessian (symmetric 3x3):
        diag_factor = mu + lam - lam*log(J)
        H_ii = mu + diag_factor/s_i^2 + k
        H_ij = lam/(s_i*s_j)  for i != j

    Solved via closed-form 3x3 inverse (cofactor matrix / determinant).
    5 fixed Newton iterations. Clamp s > 1e-6 each iteration.

    Args:
        sigma: Singular values from ``wp.svd3(F)``, shape (3,), in descending order.
        mu: First Lame parameter (shear modulus) [Pa].
        lam: Second Lame parameter [Pa].

    Returns:
        Projected sigma triple suitable for reconstructing P = U diag(sigma) V^T.
    """
    eps = 1.0e-6
    k = 2.0 * mu

    # Initialize Newton iterate from SVD singular values; store initial sigma_0 for penalty.
    s0 = wp.max(sigma[0], eps)
    s1 = wp.max(sigma[1], eps)
    s2 = wp.max(sigma[2], eps)
    sigma0_0 = s0
    sigma0_1 = s1
    sigma0_2 = s2

    for _i in range(5):
        # Clamp before computing log/inv.
        s0 = wp.max(s0, eps)
        s1 = wp.max(s1, eps)
        s2 = wp.max(s2, eps)

        J = s0 * s1 * s2
        log_J = wp.log(J)
        inv0 = 1.0 / s0
        inv1 = 1.0 / s1
        inv2 = 1.0 / s2

        # Gradient.
        g0 = mu * (s0 - inv0) + lam * log_J * inv0 + k * (s0 - sigma0_0)
        g1 = mu * (s1 - inv1) + lam * log_J * inv1 + k * (s1 - sigma0_1)
        g2 = mu * (s2 - inv2) + lam * log_J * inv2 + k * (s2 - sigma0_2)

        # Hessian diagonal factor.
        diag_factor = mu + lam - lam * log_J
        H00 = mu + diag_factor * inv0 * inv0 + k
        H11 = mu + diag_factor * inv1 * inv1 + k
        H22 = mu + diag_factor * inv2 * inv2 + k
        H01 = lam * inv0 * inv1
        H02 = lam * inv0 * inv2
        H12 = lam * inv1 * inv2

        # Closed-form 3x3 inverse via cofactors.
        C00 = H11 * H22 - H12 * H12
        C11 = H00 * H22 - H02 * H02
        C22 = H00 * H11 - H01 * H01
        C01 = -(H01 * H22 - H12 * H02)
        C02 = H01 * H12 - H11 * H02
        C12 = -(H00 * H12 - H01 * H02)

        det_H = H00 * C00 + H01 * C01 + H02 * C02

        dx0 = (C00 * g0 + C01 * g1 + C02 * g2) / det_H
        dx1 = (C01 * g0 + C11 * g1 + C12 * g2) / det_H
        dx2 = (C02 * g0 + C12 * g1 + C22 * g2) / det_H

        s0 = s0 - dx0
        s1 = s1 - dx1
        s2 = s2 - dx2

    s0 = wp.max(s0, eps)
    s1 = wp.max(s1, eps)
    s2 = wp.max(s2, eps)
    return wp.vec3(s0, s1, s2)


@wp.kernel
def project_stretching_neohookean_tet_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-tet Neo-Hookean local projection scatter for PD softbody.

    Computes F = Ds * Dm_inv (3x3), extracts singular values via ``wp.svd3``,
    applies 5-iteration Newton solve of the 3D Neo-Hookean PD projection to get
    ``sigma_proj``, reconstructs ``P = U * diag(sigma_proj) * Vt``, and scatters
    ``w * Dm_inv * P^T`` into the four stencil vertices via atomic_add.

    No reflection handling is needed: ``wp.svd3`` returns non-negative singular
    values and the NH projection preserves positivity; the reconstructed P
    naturally has the correct orientation.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        mu: First Lamé parameter [Pa].
        lam: Second Lamé parameter [Pa].
        rhs: Output RHS accumulator (atomic-add target), shape ``[particle_count]``.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    # Ds = [p1-p0 | p2-p0 | p3-p0]  (column-stack 3 edge vectors -> 3x3)
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    # Neo-Hookean projection: 5-iter Newton solve on singular values.
    sigma_proj = project_neohookean_sigma3d(wp.vec3(sigma[0], sigma[1], sigma[2]), mu, lam)

    # Reconstruct P = U * diag(sigma_proj) * V^T  (3x3).
    s0 = sigma_proj[0]
    s1 = sigma_proj[1]
    s2 = sigma_proj[2]
    US = wp.mat33(
        U[0, 0] * s0,
        U[0, 1] * s1,
        U[0, 2] * s2,
        U[1, 0] * s0,
        U[1, 1] * s1,
        U[1, 2] * s2,
        U[2, 0] * s0,
        U[2, 1] * s1,
        U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    # proj = w * Dm_inv * P^T  (3x3)
    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    # Scatter stencil.
    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    wp.atomic_add(rhs, i0, -(row0 + row1 + row2))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)
    wp.atomic_add(rhs, i3, row2)


@wp.kernel
def project_stretching_neohookean_tet_compute_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Compute per-tet Neo-Hookean contribution to each of its 4 local vertices.

    Same math as :func:`project_stretching_neohookean_tet_kernel` but writes
    to a pre-allocated ``(T, 4)`` scratch buffer instead of atomic-adding into
    rhs. Pair with :func:`gather_per_particle_kernel` for the deterministic
    reduction.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        mu: First Lamé parameter [Pa].
        lam: Second Lamé parameter [Pa].
        contributions: Output ``(tet_count, 4)`` per-local-vertex contribution.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    sigma_proj = project_neohookean_sigma3d(wp.vec3(sigma[0], sigma[1], sigma[2]), mu, lam)

    s0 = sigma_proj[0]
    s1 = sigma_proj[1]
    s2 = sigma_proj[2]
    US = wp.mat33(
        U[0, 0] * s0,
        U[0, 1] * s1,
        U[0, 2] * s2,
        U[1, 0] * s0,
        U[1, 1] * s1,
        U[1, 2] * s2,
        U[2, 0] * s0,
        U[2, 1] * s1,
        U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    contributions[t, 0] = -(row0 + row1 + row2)
    contributions[t, 1] = row0
    contributions[t, 2] = row1
    contributions[t, 3] = row2


@wp.kernel
def project_stretching_neohookean_tet_compute_kernel_lbfgs(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Per-tet Neo-Hookean projection (LBFGS variant of
    :func:`project_stretching_neohookean_tet_compute_kernel`).

    Same scatter math as the Newton variant. The local sigma projection is the
    RealSim-faithful LBFGS solver (:func:`project_neohookean_sigma3d_lbfgs`)
    instead of FBA's 5-iter Newton. SVD inputs are cast fp32 -> fp64 before
    LBFGS; the projected sigma is cast back to fp32 for the P reconstruction
    and scatter, matching the existing kernel's mixed-precision contract.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    # LBFGS in fp64.
    sigma_init = wp.vec3d(wp.float64(sigma[0]), wp.float64(sigma[1]), wp.float64(sigma[2]))
    mu_d = wp.float64(mu)
    lam_d = wp.float64(lam)
    sigma_proj_d = project_neohookean_sigma3d_lbfgs(sigma_init, mu_d, lam_d)
    s0 = wp.float32(sigma_proj_d[0])
    s1 = wp.float32(sigma_proj_d[1])
    s2 = wp.float32(sigma_proj_d[2])

    US = wp.mat33(
        U[0, 0] * s0,
        U[0, 1] * s1,
        U[0, 2] * s2,
        U[1, 0] * s0,
        U[1, 1] * s1,
        U[1, 2] * s2,
        U[2, 0] * s0,
        U[2, 1] * s1,
        U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    contributions[t, 0] = -(row0 + row1 + row2)
    contributions[t, 1] = row0
    contributions[t, 2] = row1
    contributions[t, 3] = row2


@wp.kernel
def project_stretching_arap_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-triangle ARAP local projection scatter for PD cloth.

    Computes the deformation gradient `F = Ds * Dm_inv` (3x2), projects to the
    nearest rotation `P` via SVD-clamp, and scatters the per-particle
    contribution `w * Dm_inv * P^T` into the RHS vector via atomic_add.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (`ke * area`).
        rhs: Output RHS accumulator; receives atomic-add contributions.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    # Deformation gradient F = [p1-p0 | p2-p0] . Dm_inv  (3x2)
    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    P = svd_3x2(F)

    # Local contribution to RHS: proj = w * Dm_inv * P^T  (2x3)
    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]  # column 0 of P as a row of P^T
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]  # column 1 of P as a row of P^T

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    # Scatter: rhs[i0] += -row0 - row1; rhs[i1] += row0; rhs[i2] += row1
    wp.atomic_add(rhs, i0, -(row0 + row1))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)


@wp.kernel
def project_stretching_arap_compute_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Compute per-tri ARAP contribution to each of its 3 local vertices.

    Same math as :func:`project_stretching_arap_kernel` but writes to a
    pre-allocated ``(T, 3)`` scratch buffer instead of atomic-adding into rhs.
    Pair with :func:`gather_per_particle_kernel` for the deterministic
    reduction.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``ke * area``).
        contributions: Output ``(tri_count, 3)`` per-local-vertex contribution.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    P = svd_3x2(F)

    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    contributions[t, 0] = -(row0 + row1)
    contributions[t, 1] = row0
    contributions[t, 2] = row1


@wp.func
def project_corotational_sigma(sigma_sq: wp.vec2, mu: float, lam: float) -> wp.vec2:
    """Closed-form 2D corotational local projection on singular values.

    Minimises E(s) = mu*||s-I||^2 + (lam/2)*tr(s-I)^2  subject to PD's
    quadratic penalty (k/2)*||s-s0||^2 with k = 2*mu.  The resulting 2x2
    linear system has the closed-form solution below.  Both singular
    values share the same energy-form term (symmetric trace), which is
    the standard corotational formulation in elasticity references
    (Sifakis 2012, Bouaziz 2014, etc.).

    Note - mismatch with RealSim reference:
        RealSim's ``CorotProjectionProblem2D::energy_density`` evaluates
        ``(x - I).trace()`` on an ``Eigen::Vector2d``; Eigen's ``trace()``
        on a non-square matrix returns the sum over indices [0, min(rows,
        cols)), i.e. ``x[0]`` only for a 2x1 vector - not ``x[0] + x[1]``
        (verified empirically). RealSim therefore implements an
        anisotropic energy that only penalises ``(sigma_0 - 1)`` in the
        trace term. Newton's implementation here is the standard symmetric
        formulation. Trajectory-level cross-check vs RealSim Corotational
        is therefore expected to differ at the order of mm on stretched
        cloth (see docs/superpowers/specs cross-check write-ups). ARAP
        and Neo-Hookean cross-checks against RealSim agree to <10 microns.

    Args:
        sigma_sq: Squared singular values from ``wp.svd2(FtF)``.
        mu: First Lame parameter (shear modulus) [Pa].
        lam: Second Lame parameter [Pa].

    Returns:
        Projected singular-value pair ``(sigma_proj_0, sigma_proj_1)``.
    """
    s0 = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1 = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    k = 2.0 * mu  # PD penalty weight
    # 2x2 system: diagonal M, off-diagonal lam.
    M = 2.0 * mu + lam + k  # = 4*mu + lam
    det = M * M - lam * lam

    b0 = 2.0 * (mu + lam) + k * s0
    b1 = 2.0 * (mu + lam) + k * s1

    proj0 = (M * b0 - lam * b1) / det
    proj1 = (M * b1 - lam * b0) / det
    return wp.vec2(proj0, proj1)


@wp.kernel
def project_stretching_corotational_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-triangle corotational local projection scatter for PD cloth.

    Computes the deformation gradient ``F = Ds * Dm_inv`` (3x2), extracts
    singular values via ``wp.svd2(FtF)``, applies the closed-form corotational
    projection to get ``sigma_proj``, reconstructs ``P = U * diag(sigma_proj) * Vt``,
    and scatters ``w * Dm_inv * Pt`` into the RHS vector via atomic_add.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``ke * area``).
        mu: First Lame parameter [Pa].
        lam: Second Lame parameter [Pa].
        rhs: Output RHS accumulator; receives atomic-add contributions.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    # Deformation gradient F = [p1-p0 | p2-p0] . Dm_inv  (3x2)
    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    # 3x2 SVD via 2x2 symmetric eigenproblem on FtF.
    FtF = wp.transpose(F) * F
    _U2, sigma_sq, V2 = wp.svd2(FtF)

    # Real singular values of F (sqrt of eigenvalues of FᵀF).
    s0_real = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1_real = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    # Left singular vectors of F: u_i = F v_i / s_i.
    v0 = wp.vec2(V2[0, 0], V2[1, 0])
    v1 = wp.vec2(V2[0, 1], V2[1, 1])
    u0 = (F * v0) / s0_real  # 3-vector
    u1 = (F * v1) / s1_real  # 3-vector

    # Corotational projection: sigma_proj via closed-form 2x2 solve.
    sigma_proj = project_corotational_sigma(sigma_sq, mu, lam)

    # Reconstruct P = U * diag(sigma_proj) * Vt  (3x2).
    # P[:, j] = sum_i u_i * sigma_proj[i] * V2[j, i]
    p_col0 = u0 * (sigma_proj[0] * V2[0, 0]) + u1 * (sigma_proj[1] * V2[0, 1])
    p_col1 = u0 * (sigma_proj[0] * V2[1, 0]) + u1 * (sigma_proj[1] * V2[1, 1])

    P = mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )

    # Local contribution to RHS: proj = w * Dm_inv * Pt  (2x3)
    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    # Scatter: rhs[i0] += -row0 - row1; rhs[i1] += row0; rhs[i2] += row1
    wp.atomic_add(rhs, i0, -(row0 + row1))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)


@wp.kernel
def project_stretching_corotational_compute_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Compute per-tri corotational contribution to each of its 3 local vertices.

    Same math as :func:`project_stretching_corotational_kernel` but writes to a
    pre-allocated ``(T, 3)`` scratch buffer instead of atomic-adding into rhs.
    Pair with :func:`gather_per_particle_kernel` for the deterministic
    reduction.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``ke * area``).
        mu: First Lame parameter [Pa].
        lam: Second Lame parameter [Pa].
        contributions: Output ``(tri_count, 3)`` per-local-vertex contribution.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    FtF = wp.transpose(F) * F
    _U2, sigma_sq, V2 = wp.svd2(FtF)

    s0_real = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1_real = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    v0 = wp.vec2(V2[0, 0], V2[1, 0])
    v1 = wp.vec2(V2[0, 1], V2[1, 1])
    u0 = (F * v0) / s0_real
    u1 = (F * v1) / s1_real

    sigma_proj = project_corotational_sigma(sigma_sq, mu, lam)

    p_col0 = u0 * (sigma_proj[0] * V2[0, 0]) + u1 * (sigma_proj[1] * V2[0, 1])
    p_col1 = u0 * (sigma_proj[0] * V2[1, 0]) + u1 * (sigma_proj[1] * V2[1, 1])

    P = mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )

    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    contributions[t, 0] = -(row0 + row1)
    contributions[t, 1] = row0
    contributions[t, 2] = row1


@wp.func
def project_neohookean_sigma(sigma_sq: wp.vec2, mu: float, lam: float) -> wp.vec2:
    """Project SVD singular values to Neo-Hookean PD equilibrium via 5 Newton iterations.

    Solves: min_sigma  E_NH(sigma) + (k/2)*||sigma - sigma_0||^2
    where k = 2*mu (bulk modulus quadratic penalty).

    Args:
        sigma_sq: Squared singular values from wp.svd2(F^T F).
        mu: First Lame parameter (shear modulus).
        lam: Second Lame parameter.

    Returns:
        Projected sigma (2D vector) suitable for reconstructing P = U diag(sigma) V^T.
    """
    eps = 1.0e-6
    sigma0_0 = wp.sqrt(wp.max(sigma_sq[0], eps * eps))
    sigma0_1 = wp.sqrt(wp.max(sigma_sq[1], eps * eps))

    k = 2.0 * mu
    sigma_0 = sigma0_0
    sigma_1 = sigma0_1

    for _i in range(5):
        # Clamp before computing log/inv
        sigma_0 = wp.max(sigma_0, eps)
        sigma_1 = wp.max(sigma_1, eps)
        J = sigma_0 * sigma_1
        log_J = wp.log(J)
        inv_0 = 1.0 / sigma_0
        inv_1 = 1.0 / sigma_1

        # Gradient
        grad_0 = mu * (sigma_0 - inv_0) + lam * log_J * inv_0 + k * (sigma_0 - sigma0_0)
        grad_1 = mu * (sigma_1 - inv_1) + lam * log_J * inv_1 + k * (sigma_1 - sigma0_1)

        # Hessian diagonal factor: mu + lam - lam*log_J
        h_diag_factor = mu + lam - lam * log_J
        H_00 = mu + h_diag_factor * inv_0 * inv_0 + k
        H_11 = mu + h_diag_factor * inv_1 * inv_1 + k
        H_01 = lam / J

        # 2x2 inverse times gradient (closed form)
        det = H_00 * H_11 - H_01 * H_01
        dx_0 = (H_11 * grad_0 - H_01 * grad_1) / det
        dx_1 = (H_00 * grad_1 - H_01 * grad_0) / det

        sigma_0 = sigma_0 - dx_0
        sigma_1 = sigma_1 - dx_1

    sigma_0 = wp.max(sigma_0, eps)
    sigma_1 = wp.max(sigma_1, eps)
    return wp.vec2(sigma_0, sigma_1)


@wp.kernel
def project_stretching_neohookean_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-triangle Neo-Hookean local projection scatter for PD cloth.

    Computes the deformation gradient ``F = Ds * Dm_inv`` (3x2), extracts
    singular values via ``wp.svd2(FtF)``, applies 5 Newton iterations of the
    Neo-Hookean PD projection to get ``sigma_proj``, reconstructs
    ``P = U * diag(sigma_proj) * Vt``, and scatters ``w * Dm_inv * Pt`` into
    the RHS vector via atomic_add.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``ke * area``).
        mu: First Lame parameter [Pa].
        lam: Second Lame parameter [Pa].
        rhs: Output RHS accumulator; receives atomic-add contributions.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    # Deformation gradient F = [p1-p0 | p2-p0] . Dm_inv  (3x2)
    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    # 3x2 SVD via 2x2 symmetric eigenproblem on FtF.
    FtF = wp.transpose(F) * F
    _U2, sigma_sq, V2 = wp.svd2(FtF)

    # Real singular values of F (sqrt of eigenvalues of FᵀF).
    s0_real = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1_real = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    # Left singular vectors of F: u_i = F v_i / s_i.
    v0 = wp.vec2(V2[0, 0], V2[1, 0])
    v1 = wp.vec2(V2[0, 1], V2[1, 1])
    u0 = (F * v0) / s0_real  # 3-vector
    u1 = (F * v1) / s1_real  # 3-vector

    # Neo-Hookean projection: sigma_proj via 5-iteration Newton solve.
    sigma_proj = project_neohookean_sigma(sigma_sq, mu, lam)

    # Reconstruct P = U * diag(sigma_proj) * Vt  (3x2).
    # P[:, j] = sum_i u_i * sigma_proj[i] * V2[j, i]
    p_col0 = u0 * (sigma_proj[0] * V2[0, 0]) + u1 * (sigma_proj[1] * V2[0, 1])
    p_col1 = u0 * (sigma_proj[0] * V2[1, 0]) + u1 * (sigma_proj[1] * V2[1, 1])

    P = mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )

    # Local contribution to RHS: proj = w * Dm_inv * Pt  (2x3)
    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    # Scatter: rhs[i0] += -row0 - row1; rhs[i1] += row0; rhs[i2] += row1
    wp.atomic_add(rhs, i0, -(row0 + row1))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)


@wp.kernel
def project_stretching_neohookean_compute_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Compute per-tri Neo-Hookean contribution to each of its 3 local vertices.

    Same math as :func:`project_stretching_neohookean_kernel` but writes to a
    pre-allocated ``(T, 3)`` scratch buffer instead of atomic-adding into rhs.
    Pair with :func:`gather_per_particle_kernel` for the deterministic
    reduction.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``ke * area``).
        mu: First Lame parameter [Pa].
        lam: Second Lame parameter [Pa].
        contributions: Output ``(tri_count, 3)`` per-local-vertex contribution.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    FtF = wp.transpose(F) * F
    _U2, sigma_sq, V2 = wp.svd2(FtF)

    s0_real = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1_real = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    v0 = wp.vec2(V2[0, 0], V2[1, 0])
    v1 = wp.vec2(V2[0, 1], V2[1, 1])
    u0 = (F * v0) / s0_real
    u1 = (F * v1) / s1_real

    sigma_proj = project_neohookean_sigma(sigma_sq, mu, lam)

    p_col0 = u0 * (sigma_proj[0] * V2[0, 0]) + u1 * (sigma_proj[1] * V2[0, 1])
    p_col1 = u0 * (sigma_proj[0] * V2[1, 0]) + u1 * (sigma_proj[1] * V2[1, 1])

    P = mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )

    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    contributions[t, 0] = -(row0 + row1)
    contributions[t, 1] = row0
    contributions[t, 2] = row1


@wp.kernel
def project_stretching_neohookean_compute_kernel_lbfgs(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Per-tri Neo-Hookean projection (LBFGS variant of
    :func:`project_stretching_neohookean_compute_kernel`).

    Same scatter math as the Newton variant. The local sigma projection is the
    RealSim-faithful LBFGS solver (:func:`project_neohookean_sigma2d_lbfgs`)
    instead of FBA's 5-iter Newton. SVD inputs (singular values of F derived
    from ``wp.svd2(F^T F)``) are cast fp32 -> fp64 before LBFGS; the projected
    sigma is cast back to fp32 for the P reconstruction and scatter.
    """
    t = wp.tid()
    i0 = tri_indices[3 * t + 0]
    i1 = tri_indices[3 * t + 1]
    i2 = tri_indices[3 * t + 2]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]

    Ds_col0 = p1 - p0
    Ds_col1 = p2 - p0
    Ds = mat32(
        Ds_col0[0],
        Ds_col1[0],
        Ds_col0[1],
        Ds_col1[1],
        Ds_col0[2],
        Ds_col1[2],
    )
    Dm_inv = tri_rest_inv[t]
    F = Ds * Dm_inv

    FtF = wp.transpose(F) * F
    _U2, sigma_sq, V2 = wp.svd2(FtF)

    s0_real = wp.sqrt(wp.max(sigma_sq[0], 1.0e-20))
    s1_real = wp.sqrt(wp.max(sigma_sq[1], 1.0e-20))

    v0 = wp.vec2(V2[0, 0], V2[1, 0])
    v1 = wp.vec2(V2[0, 1], V2[1, 1])
    u0 = (F * v0) / s0_real
    u1 = (F * v1) / s1_real

    sigma_init = wp.vec2d(wp.float64(s0_real), wp.float64(s1_real))
    mu_d = wp.float64(mu)
    lam_d = wp.float64(lam)
    sigma_proj_d = project_neohookean_sigma2d_lbfgs(sigma_init, mu_d, lam_d)
    sp0 = wp.float32(sigma_proj_d[0])
    sp1 = wp.float32(sigma_proj_d[1])

    p_col0 = u0 * (sp0 * V2[0, 0]) + u1 * (sp1 * V2[0, 1])
    p_col1 = u0 * (sp0 * V2[1, 0]) + u1 * (sp1 * V2[1, 1])

    P = mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )

    w = tri_weight[t]
    PT00 = P[0, 0]
    PT01 = P[1, 0]
    PT02 = P[2, 0]
    PT10 = P[0, 1]
    PT11 = P[1, 1]
    PT12 = P[2, 1]

    row0 = wp.vec3(
        w * (Dm_inv[0, 0] * PT00 + Dm_inv[0, 1] * PT10),
        w * (Dm_inv[0, 0] * PT01 + Dm_inv[0, 1] * PT11),
        w * (Dm_inv[0, 0] * PT02 + Dm_inv[0, 1] * PT12),
    )
    row1 = wp.vec3(
        w * (Dm_inv[1, 0] * PT00 + Dm_inv[1, 1] * PT10),
        w * (Dm_inv[1, 0] * PT01 + Dm_inv[1, 1] * PT11),
        w * (Dm_inv[1, 0] * PT02 + Dm_inv[1, 1] * PT12),
    )

    contributions[t, 0] = -(row0 + row1)
    contributions[t, 1] = row0
    contributions[t, 2] = row1


@wp.kernel
def project_bending_kernel(
    positions: wp.array[wp.vec3],  # x_cur — current iterate
    edge_indices: wp.array2d[wp.int32],  # shape (E, 4)
    edge_quad_q: wp.array[wp.vec4],  # length-4 vector q per edge; Q = q*q^T
    edge_weight: wp.array[wp.float32],
    edge_norm: wp.array[wp.float32],  # ||q*x_rest|| — rest curvature magnitude
    rhs: wp.array[wp.vec3],
):
    """Per-edge isometric bending scatter for PD cloth.

    Matches RealSim ``PDIsometricBendingEnergy::localProjection``: for each
    interior edge, the local step projects the current curvature vector
    ``q*x_cur`` onto a sphere of radius ``_norm[i] = ||q*x_rest||``, i.e. pulls
    the curvature *magnitude* back to its rest value while letting the
    direction follow ``x_cur``. This is correct on both flat and curved rest
    cloth; the previous ``q*x_ref`` form only agreed on flat rest.

    The local-projection target is ``ê * _norm[i]`` with
    ``ê = (q*x_cur) / ||q*x_cur||``. We scatter
    ``w * q[a] * ê * _norm[i]`` into each of the 4 stencil vertices via
    ``atomic_add``. When ``||q*x_cur||`` is below ``1e-12`` the contribution is
    skipped (numerator and target both shrink to zero); when
    ``_norm[i] == 0`` (flat rest) the contribution is also zero.

    The Hessian's matching ``w * q*qᵀ`` term is absorbed into the prefactored
    ``A`` in :func:`build_pd_system` and does not depend on ``x_cur``.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        edge_indices: 4-vertex bending stencil, shape ``[edge_count, 4]``.
        edge_quad_q: Per-edge length-4 cotangent vector ``q``.
        edge_weight: Per-edge bending stiffness ``w`` (already scaled by
            ``3 / (A0 + A1)``).
        edge_norm: Per-edge rest curvature magnitude ``||q*x_rest||``.
        rhs: Output RHS accumulator; receives atomic-add contributions.
    """
    e = wp.tid()
    w = edge_weight[e]
    if w == 0.0:
        return
    norm_rest = edge_norm[e]
    if norm_rest == 0.0:
        return

    q = edge_quad_q[e]
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    i2 = edge_indices[e, 2]
    i3 = edge_indices[e, 3]

    # Compute q^T * x_cur (a vec3 because positions are vec3).
    qTxcur = positions[i0] * q[0] + positions[i1] * q[1] + positions[i2] * q[2] + positions[i3] * q[3]
    norm_cur = wp.length(qTxcur)
    if norm_cur < 1.0e-12:
        return

    # Target curvature: unit direction of q*x_cur, scaled to rest magnitude.
    target = qTxcur * (norm_rest / norm_cur)

    # Scatter w * q[a] * target to each row.
    wp.atomic_add(rhs, i0, w * q[0] * target)
    wp.atomic_add(rhs, i1, w * q[1] * target)
    wp.atomic_add(rhs, i2, w * q[2] * target)
    wp.atomic_add(rhs, i3, w * q[3] * target)


@wp.kernel
def project_bending_compute_kernel(
    positions: wp.array[wp.vec3],  # x_cur — current iterate
    edge_indices: wp.array2d[wp.int32],  # shape (E, 4)
    edge_quad_q: wp.array[wp.vec4],  # length-4 vector q per edge; Q = q*q^T
    edge_weight: wp.array[wp.float32],
    edge_norm: wp.array[wp.float32],  # ||q*x_rest|| — rest curvature magnitude
    # output (per-edge, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (E, 4)
):
    """Compute per-edge bending contribution to each of its 4 stencil vertices.

    Same math as :func:`project_bending_kernel` but writes to a pre-allocated
    ``(E, 4)`` scratch buffer instead of atomic-adding into rhs. Pair with
    :func:`gather_per_particle_kernel` for the deterministic reduction.

    The contribution at local vertex ``a`` is ``w * q[a] * target`` where
    ``target = (q*x_cur).normalized() * ||q*x_rest||`` matches the H'-fixed
    normalize-then-scale-by-rest-norm form. When weight, rest curvature, or
    current curvature is too small the contribution is zeroed.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        edge_indices: 4-vertex bending stencil, shape ``[edge_count, 4]``.
        edge_quad_q: Per-edge length-4 cotangent vector ``q``.
        edge_weight: Per-edge bending stiffness ``w`` (already scaled by
            ``3 / (A0 + A1)``).
        edge_norm: Per-edge rest curvature magnitude ``||q*x_rest||``.
        contributions: Output ``(edge_count, 4)`` per-local-vertex contribution.
    """
    e = wp.tid()
    zero = wp.vec3(0.0, 0.0, 0.0)
    w = edge_weight[e]
    if w == 0.0:
        contributions[e, 0] = zero
        contributions[e, 1] = zero
        contributions[e, 2] = zero
        contributions[e, 3] = zero
        return
    norm_rest = edge_norm[e]
    if norm_rest == 0.0:
        contributions[e, 0] = zero
        contributions[e, 1] = zero
        contributions[e, 2] = zero
        contributions[e, 3] = zero
        return

    q = edge_quad_q[e]
    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    i2 = edge_indices[e, 2]
    i3 = edge_indices[e, 3]

    # Compute q^T * x_cur (a vec3 because positions are vec3).
    qTxcur = positions[i0] * q[0] + positions[i1] * q[1] + positions[i2] * q[2] + positions[i3] * q[3]
    norm_cur = wp.length(qTxcur)
    if norm_cur < 1.0e-12:
        contributions[e, 0] = zero
        contributions[e, 1] = zero
        contributions[e, 2] = zero
        contributions[e, 3] = zero
        return

    # Target curvature: unit direction of q*x_cur, scaled to rest magnitude.
    target = qTxcur * (norm_rest / norm_cur)

    # Per-local-vertex contribution: w * q[a] * target.
    contributions[e, 0] = w * q[0] * target
    contributions[e, 1] = w * q[1] * target
    contributions[e, 2] = w * q[2] * target
    contributions[e, 3] = w * q[3] * target


@wp.kernel
def project_pin_kernel(
    pin_indices: wp.array[wp.int32],
    x_ref: wp.array[wp.vec3],  # full vec3 array indexed by particle index
    pin_stiffness: float,
    rhs: wp.array[wp.vec3],
):
    """Per-pin PD soft-constraint scatter.

    For each pinned particle, contributes ``w_pin * x_ref[i]`` to the RHS.
    The matching `w_pin` diagonal term is already in the prefactored ``A``
    (added by ``build_pd_system`` at the pinned particle's diagonal).

    Args:
        pin_indices: Packed indices of pinned particles, shape ``[pin_count]``.
        x_ref: Reference positions [m] for each particle, indexed by particle index.
        pin_stiffness: PD soft-pin weight ``w_pin``.
        rhs: Output RHS accumulator; receives atomic-add contributions.
    """
    k = wp.tid()
    i = pin_indices[k]
    wp.atomic_add(rhs, i, pin_stiffness * x_ref[i])


# ---------------------------------------------------------------------------
# Phase 4 Stage A — contact Jacobian kernels
# ---------------------------------------------------------------------------


@wp.kernel
def zero_scalar_kernel(arr: wp.array[wp.float64]):
    """arr[i] = 0.0 — zeroes a float64 scalar array."""
    tid = wp.tid()
    arr[tid] = wp.float64(0.0)


@wp.kernel
def build_contact_jacobian_vec3_kernel(
    particle_count: int,
    contact_idx: int,
    j_indices: wp.array[wp.int32],
    j_normals: wp.array[wp.vec3],
    j_alpha: wp.array[wp.float32],
    out: wp.array[wp.vec3],
):
    """Set out[j_indices[contact_idx]] = j_alpha[contact_idx] * j_normals[contact_idx].

    Called with ``dim=1`` (single thread).  All other slots are left as-is
    (caller is responsible for zeroing ``out`` before launch).

    Args:
        particle_count: Total particle count (unused at runtime, guards bounds).
        contact_idx: Index of the contact to materialise.
        j_indices: Particle index per contact row, shape ``[M]``.
        j_normals: World-frame contact normal per contact row, shape ``[M]``.
        j_alpha: Jacobian coefficient per contact row (1.0 for particle-shape).
        out: Output vec3 array of length ``particle_count``; only the slot at
            ``j_indices[contact_idx]`` is written.
    """
    _tid = wp.tid()
    idx = j_indices[contact_idx]
    if idx >= 0 and idx < particle_count:
        out[idx] = wp.float32(j_alpha[contact_idx]) * j_normals[contact_idx]


@wp.kernel
def build_contact_jacobian_dir_kernel(
    particle_count: int,
    contact_idx: int,
    j_indices: wp.array[wp.int32],
    j_alpha: wp.array[wp.float32],
    direction: wp.vec3,
    out: wp.array[wp.vec3],
):
    """Set out[j_indices[contact_idx]] = j_alpha[contact_idx] * direction.

    General-direction variant of :func:`build_contact_jacobian_vec3_kernel`.
    Used by Stage B friction to set tangent-direction Jacobian columns (t1, t2)
    without storing them in a per-contact array.

    Called with ``dim=1`` (single thread).  Caller must zero ``out`` before launch.

    Args:
        particle_count: Total particle count (bounds guard).
        contact_idx: Contact row index (0-based).
        j_indices: Particle index per contact, shape ``[M]``.
        j_alpha: Jacobian coefficient per contact (1.0 for particle-shape), shape ``[M]``.
        direction: The unit direction vector (normal, t1, or t2).
        out: Output vec3 array of length ``particle_count``.
    """
    _tid = wp.tid()
    idx = j_indices[contact_idx]
    if idx >= 0 and idx < particle_count:
        out[idx] = wp.float32(j_alpha[contact_idx]) * direction


@wp.kernel
def set_lambda_jacobian_vec3_kernel(
    contact_idx: int,
    j_indices: wp.array[wp.int32],
    j_normals: wp.array[wp.vec3],
    j_alpha: wp.array[wp.float32],
    lam_c: float,
    particle_count: int,
    out: wp.array[wp.vec3],
):
    """Scatter ``lam_c * j_alpha[c] * j_normals[c]`` to ``out[j_indices[c]]``.

    Called with ``dim=1``.  Caller must zero ``out`` before launch if
    accumulation from previous contacts is not desired.

    Args:
        contact_idx: Contact row index ``c``.
        j_indices: Particle index per contact row, shape ``[M]``.
        j_normals: Contact normal per row, shape ``[M]``.
        j_alpha: Coefficient per row (1.0 for particle-shape contacts).
        lam_c: Scalar multiplier (lambda for contact ``c``).
        particle_count: Guard for bounds checking.
        out: Output vec3 array of length ``particle_count``.
    """
    _tid = wp.tid()
    idx = j_indices[contact_idx]
    if idx >= 0 and idx < particle_count:
        out[idx] = lam_c * wp.float32(j_alpha[contact_idx]) * j_normals[contact_idx]


@wp.kernel
def accumulate_vec3_kernel(
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
):
    """dst[i] += src[i] — accumulate one vec3 array into another."""
    tid = wp.tid()
    dst[tid] = dst[tid] + src[tid]


@wp.kernel
def accumulate_schur_W_kernel(
    contact_particle: wp.array[wp.int32],
    contact_dir: wp.array[wp.vec3],
    contact_alpha: wp.array[wp.float32],
    A_inv_Jt: wp.array2d[wp.vec3],
    W: wp.array2d[wp.float64],
):
    """Compute W[c_prime, c] = alpha[c_prime] * dot(dir[c_prime], (A^{-1} J^T)[c, particle[c_prime]]).

    Each thread handles one (c_prime, c) pair in the M_rows x M_rows output W.

    Args:
        contact_particle: Particle index per row, shape ``[M_rows]``.
        contact_dir: Contact direction (normal or tangent) per row, shape ``[M_rows]``.
        contact_alpha: Jacobian coefficient per row, shape ``[M_rows]``.
        A_inv_Jt: Result of A^{-1} applied to each J^T column, shape ``[M_rows, N]`` (2D,
            indexed as A_inv_Jt[row_c, particle]).
        W: Output Schur complement matrix, shape ``[M_rows, M_rows]``.
    """
    c_prime, c = wp.tid()
    p = contact_particle[c_prime]
    y_at_p = A_inv_Jt[c, p]
    alpha = contact_alpha[c_prime]
    n = contact_dir[c_prime]
    W[c_prime, c] = wp.float64(alpha) * wp.float64(wp.dot(n, y_at_p))


@wp.kernel
def subtract_vec3_kernel(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    """out[i] = a[i] - b[i]."""
    tid = wp.tid()
    out[tid] = a[tid] - b[tid]


# ---------------------------------------------------------------------------
# Approach B — multi-RHS batched primitives for Schur complement build
# ---------------------------------------------------------------------------


@wp.kernel
def bsr_mv_multi_rhs_scalar_kernel(
    A_offsets: wp.array[wp.int32],
    A_columns: wp.array[wp.int32],
    A_values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    """y[rhs, row] = sum_k A[row, col_k] * x[rhs, col_k].

    One thread per (rhs_idx, matrix_row). ``A`` is a scalar 1x1 BSR matrix
    (i.e. A_values is a flat 1D array of length ``nnz``; block = scalar).

    Args:
        A_offsets: CSR row offsets, length ``nrow + 1``, int32.
        A_columns: Column indices, length ``nnz``, int32.
        A_values: Scalar values, length ``nnz``, float64.
        x: Input, shape ``(R, N)`` float64.
        y: Output, shape ``(R, N)`` float64; written in-place.
    """
    rhs_idx, row = wp.tid()
    v = wp.float64(0.0)
    beg = A_offsets[row]
    end = A_offsets[row + 1]
    for block in range(beg, end):
        col = A_columns[block]
        v = v + A_values[block] * x[rhs_idx, col]
    y[rhs_idx, row] = v


@wp.kernel
def apply_permutation_multi_rhs_kernel(
    src: wp.array2d[wp.float64],
    perm: wp.array[wp.int32],
    dst: wp.array2d[wp.float64],
):
    """dst[r, i] = src[r, perm[i]] — multi-RHS gather.

    One thread per (rhs_idx, element_idx).

    Args:
        src: Source array, shape ``(R, N)`` float64.
        perm: Permutation, length ``N``, int32.
        dst: Destination array, shape ``(R, N)`` float64.
    """
    rhs_idx, i = wp.tid()
    dst[rhs_idx, i] = src[rhs_idx, perm[i]]


@wp.kernel
def scale_by_diag_multi_rhs_kernel(
    src: wp.array2d[wp.float64],
    diag: wp.array[wp.float64],
    dst: wp.array2d[wp.float64],
):
    """dst[r, i] = diag[i] * src[r, i] — multi-RHS diagonal scaling.

    One thread per (rhs_idx, element_idx).

    Args:
        src: Source array, shape ``(R, N)`` float64.
        diag: Diagonal coefficients, length ``N``, float64.
        dst: Destination array, shape ``(R, N)`` float64.
    """
    rhs_idx, i = wp.tid()
    dst[rhs_idx, i] = diag[i] * src[rhs_idx, i]


@wp.kernel
def pack_jacobian_axis_kernel(
    axis: int,
    contact_particle: wp.array[wp.int32],
    contact_dir: wp.array[wp.vec3],
    contact_alpha: wp.array[wp.float32],
    b_multi: wp.array2d[wp.float64],
):
    """Pack one axis of J^T into b_multi for multi-RHS solve.

    For RHS row ``r`` (i.e. contact row ``r``):
        b_multi[r, :] = 0.0 for all particles except particle[r],
        b_multi[r, particle[r]] = alpha[r] * dir[r][axis].

    Called with ``dim = total_rows``. Caller must zero ``b_multi`` before launch.

    Args:
        axis: Spatial axis to extract (0=x, 1=y, 2=z).
        contact_particle: Particle index per row, shape ``[total_rows]``, int32.
        contact_dir: Contact direction per row, shape ``[total_rows]``, vec3.
        contact_alpha: Jacobian coefficient per row, shape ``[total_rows]``, float32.
        b_multi: Output RHS buffer, shape ``(total_rows, N)`` float64.
    """
    r = wp.tid()
    p = contact_particle[r]
    d = contact_dir[r]
    alpha = wp.float64(contact_alpha[r])
    b_multi[r, p] = alpha * wp.float64(d[axis])


@wp.kernel
def unpack_to_A_inv_Jt_axis_kernel(
    axis: int,
    y_multi: wp.array2d[wp.float64],
    A_inv_Jt: wp.array2d[wp.vec3],
):
    """Write one axis of y_multi back into A_inv_Jt.

    A_inv_Jt[r, i][axis] = float32(y_multi[r, i]).

    Called with ``dim = (total_rows, N)``.

    Args:
        axis: Spatial axis to write (0=x, 1=y, 2=z).
        y_multi: Solved result, shape ``(total_rows, N)`` float64.
        A_inv_Jt: Output buffer, shape ``(total_rows, N)`` vec3; only axis is written.
    """
    r, i = wp.tid()
    v = A_inv_Jt[r, i]
    if axis == 0:
        v[0] = wp.float32(y_multi[r, i])
    elif axis == 1:
        v[1] = wp.float32(y_multi[r, i])
    else:
        v[2] = wp.float32(y_multi[r, i])
    A_inv_Jt[r, i] = v


@wp.kernel
def pack_jacobian_3axis_kernel(
    total_rows: int,
    contact_particle: wp.array[wp.int32],
    contact_dir: wp.array[wp.vec3],
    contact_alpha: wp.array[wp.float32],
    b_multi: wp.array2d[wp.float64],
):
    """Pack all 3 spatial axes of J^T into a single contiguous b_multi for batched solve.

    Rows of ``b_multi`` are laid out by axis-major blocks:
        rows [0, total_rows)               -> axis 0 (x)
        rows [total_rows, 2*total_rows)    -> axis 1 (y)
        rows [2*total_rows, 3*total_rows)  -> axis 2 (z)

    For RHS row ``axis*total_rows + r``:
        b_multi[axis*total_rows + r, particle[r]] = alpha[r] * dir[r][axis].

    Called with ``dim = 3 * total_rows``. Caller must zero ``b_multi`` before launch.

    Args:
        total_rows: Number of contact rows (length of ``contact_particle``).
        contact_particle: Particle index per row, shape ``[total_rows]``, int32.
        contact_dir: Contact direction per row, shape ``[total_rows]``, vec3.
        contact_alpha: Jacobian coefficient per row, shape ``[total_rows]``, float32.
        b_multi: Output RHS buffer, shape ``(3*total_rows, N)`` float64.
    """
    tid = wp.tid()
    axis = tid / total_rows
    r = tid - axis * total_rows
    p = contact_particle[r]
    d = contact_dir[r]
    alpha = wp.float64(contact_alpha[r])
    b_multi[tid, p] = alpha * wp.float64(d[axis])


@wp.kernel
def unpack_to_A_inv_Jt_3axis_kernel(
    total_rows: int,
    y_multi: wp.array2d[wp.float64],
    A_inv_Jt: wp.array2d[wp.vec3],
):
    """Write all 3 axes of y_multi back into A_inv_Jt in a single launch.

    For row ``r`` and node ``i``:
        A_inv_Jt[r, i] = (float32(y_multi[r, i]),
                          float32(y_multi[total_rows + r, i]),
                          float32(y_multi[2*total_rows + r, i])).

    Called with ``dim = (total_rows, N)``.

    Args:
        total_rows: Number of contact rows.
        y_multi: Solved result, shape ``(3*total_rows, N)`` float64.
        A_inv_Jt: Output buffer, shape ``(total_rows, N)`` vec3; fully written.
    """
    r, i = wp.tid()
    A_inv_Jt[r, i] = wp.vec3(
        wp.float32(y_multi[r, i]),
        wp.float32(y_multi[total_rows + r, i]),
        wp.float32(y_multi[2 * total_rows + r, i]),
    )


@wp.kernel
def accumulate_lambda_correction_kernel(
    lam: wp.array[wp.float32],
    A_inv_Jt: wp.array2d[wp.vec3],
    n_rows: int,
    out: wp.array[wp.vec3],
):
    """Compute ``out[i] = Σ_r lam[r] * A_inv_Jt[r, i]`` for one particle per thread.

    Replaces the prior host-side Python loop in
    :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._apply_lambda_correction`
    and :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._apply_lambda_correction_friction`.

    Each thread handles a single particle ``i`` and accumulates the weighted
    sum across rows. Each output cell is written by exactly one thread, so no
    atomics are required.

    Called with ``dim = N`` (number of particles).

    Args:
        lam: Contact impulse vector, shape ``[n_rows]``, float32. Stage A uses
            ``n_rows == M``; Stage B uses ``n_rows == 3*M``.
        A_inv_Jt: Cached ``A⁻¹ Jᵀ`` device buffer, shape ``[n_rows, N]`` vec3.
        n_rows: Number of active contact rows (M or 3M).
        out: Output correction array, shape ``[N]`` vec3; fully written.
    """
    i = wp.tid()
    c = wp.vec3(0.0, 0.0, 0.0)
    for r in range(n_rows):
        c = c + lam[r] * A_inv_Jt[r, i]
    out[i] = c


# ---------------------------------------------------------------------------
# Task P — isodof-restricted Schur build primitives
# ---------------------------------------------------------------------------


@wp.kernel
def compute_wi_kernel(
    ST_offsets: wp.array[wp.int32],
    ST_columns: wp.array[wp.int32],
    ST_values: wp.array[wp.float64],
    D_inv: wp.array[wp.float64],
    isodof_perm: wp.array[wp.int32],
    Wi: wp.array2d[wp.float64],
):
    """Compute ``Wi[a, b] = A^{-1}[isodofs[a], isodofs[b]]`` for selected DOFs.

    Uses ``A^{-1} = P^T S^T D^{-1} S P`` so

        Wi[a, b] = sum_k S[k, ip[a]] * D_inv[k] * S[k, ip[b]]

    where ``ip[a] = invperm[isodofs[a]]``. The sparsity pattern of S is the
    elimination tree, so the sum is over the intersection of column ``ip[a]``
    and column ``ip[b]`` of S.

    ``S`` is provided through its transpose ``S^T`` in CSR form (i.e.,
    ``ST_offsets`` row-indexes ``S^T``, equivalent to column-indexing ``S``).
    For row ``c`` of ``S^T``, the entries ``(ST_columns[k], ST_values[k])`` are
    ``(row_in_S, S[row_in_S, c])`` with ``row_in_S`` sorted ascending — so the
    intersection of two columns of S reduces to a two-pointer merge over the
    sorted row-index lists.

    One thread per ``(a, b)`` pair in the ``(k, k)`` output ``Wi``. Writes both
    ``Wi[a, b]`` and ``Wi[b, a]`` for ``a <= b``; threads with ``a > b`` exit
    early.

    Args:
        ST_offsets: CSR row offsets of ``S^T``, length ``N + 1``, int32.
        ST_columns: CSR column indices of ``S^T``, length ``nnz(S)``, int32 —
            these are the row indices of ``S`` in column order.
        ST_values: CSR values of ``S^T``, length ``nnz(S)``, float64 —
            same numerical values as ``S`` since transposition only reorders.
        D_inv: ``D^{-1}`` diagonal, length ``N``, float64.
        isodof_perm: ``invperm[isodofs[a]]`` for ``a = 0..k-1``, length ``k``,
            int32.
        Wi: Output symmetric matrix, shape ``(k, k)`` float64.
    """
    a, b = wp.tid()
    if b < a:
        return
    ip_a = isodof_perm[a]
    ip_b = isodof_perm[b]
    beg_i = ST_offsets[ip_a]
    end_i = ST_offsets[ip_a + 1]
    beg_j = ST_offsets[ip_b]
    end_j = ST_offsets[ip_b + 1]
    pi = beg_i
    pj = beg_j
    acc = wp.float64(0.0)
    while pi < end_i and pj < end_j:
        ki = ST_columns[pi]
        kj = ST_columns[pj]
        if ki == kj:
            acc = acc + ST_values[pi] * D_inv[ki] * ST_values[pj]
            pi = pi + 1
            pj = pj + 1
        elif ki < kj:
            pi = pi + 1
        else:
            pj = pj + 1
    Wi[a, b] = acc
    if b != a:
        Wi[b, a] = acc


@wp.kernel
def compose_W_from_wi_kernel(
    contact_particle: wp.array[wp.int32],
    contact_dir: wp.array[wp.vec3],
    contact_alpha: wp.array[wp.float32],
    isodof_rank: wp.array[wp.int32],
    Wi: wp.array2d[wp.float64],
    W: wp.array2d[wp.float64],
):
    """Assemble ``W[c', c] = alpha[c'] * alpha[c] * (dir[c'] . dir[c]) * Wi[r', r]``.

    For particle-vs-static-shape contact rows, the Jacobian for row ``c`` has a
    single nonzero column at ``particle[c]`` with value
    ``alpha[c] * dir[c]`` (vec3). The scalar ``A^{-1}`` acts component-wise on
    each spatial axis, so

        (J A^{-1} J^T)[c', c] = alpha[c'] * alpha[c] * (dir[c'] . dir[c])
                                 * A^{-1}[particle[c'], particle[c]]
                              = alpha[c'] * alpha[c] * (dir[c'] . dir[c])
                                 * Wi[rank(particle[c']), rank(particle[c])].

    One thread per ``(c', c)`` pair in the ``(total_rows, total_rows)`` W.

    Args:
        contact_particle: Particle index per contact row, shape ``[total_rows]``.
        contact_dir: Contact direction per row, shape ``[total_rows]``.
        contact_alpha: Jacobian coefficient per row, shape ``[total_rows]``.
        isodof_rank: Per-particle rank within ``isodofs`` (``isodof_rank[p] = a``
            iff ``isodofs[a] == p``), shape ``[N]``. Entries for non-isodof
            particles are ``-1`` but are never indexed because every contact
            row's particle is by construction an isodof.
        Wi: ``(k, k)`` matrix of selected ``A^{-1}`` entries, float64.
        W: Output ``(total_rows, total_rows)`` Schur complement, float64.
    """
    cp, c = wp.tid()
    p_cp = contact_particle[cp]
    p_c = contact_particle[c]
    r_cp = isodof_rank[p_cp]
    r_c = isodof_rank[p_c]
    alpha = wp.float64(contact_alpha[cp]) * wp.float64(contact_alpha[c])
    dot = wp.float64(wp.dot(contact_dir[cp], contact_dir[c]))
    W[cp, c] = alpha * dot * Wi[r_cp, r_c]


@wp.kernel
def gather_jt_lambda_kernel(
    row_offsets: wp.array[wp.int32],  # (N + 1,) CSR offsets into row_indices
    row_indices: wp.array[wp.int32],  # (total_rows,) contact rows incident to each particle
    contact_dir: wp.array[wp.vec3],  # (total_rows,) per-row direction
    contact_alpha: wp.array[wp.float32],  # (total_rows,) per-row Jacobian coefficient
    lam: wp.array[wp.float32],  # (total_rows,) per-row impulse
    out: wp.array[wp.vec3],  # (N,) per-particle J^T * lambda — OVERWRITTEN
):
    """Compute ``out[p] = sum_{r: particle[r]=p} alpha[r] * lam[r] * dir[r]``.

    Particle-centered (deterministic) gather replacing the atomic-scatter
    :func:`build_jt_lambda_vec3_kernel`. One thread per particle reads its
    incident contact rows from the inverse-mapping CSR (built once per
    contact-set update by
    :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.build_schur_complement`)
    and sums ``alpha[r] * lam[r] * dir[r]`` over them.

    For Stage A (unilateral): one row per contact, ``total_rows == M``.
    For Stage B (Coulomb): three rows per contact (n, t1, t2 — all sharing
    the same particle), ``total_rows == 3*M``. Both cases use the same CSR
    layout because rows are indexed by their position in the per-row
    direction/alpha arrays.

    Overwrites ``out`` (does not accumulate).

    Args:
        row_offsets: CSR row offsets (``N + 1`` entries), shape ``[N + 1]``.
        row_indices: CSR entry -> contact row index, shape ``[total_rows]``.
        contact_dir: Contact direction per row, shape ``[total_rows]``.
        contact_alpha: Jacobian coefficient per row, shape ``[total_rows]``.
        lam: Contact impulse per row, shape ``[total_rows]``, float32.
        out: Per-particle accumulator, shape ``[N]``, vec3. Overwritten.
    """
    p = wp.tid()
    start = row_offsets[p]
    end = row_offsets[p + 1]
    s = wp.vec3(0.0, 0.0, 0.0)
    for k in range(start, end):
        r = row_indices[k]
        s = s + (contact_alpha[r] * lam[r]) * contact_dir[r]
    out[p] = s
