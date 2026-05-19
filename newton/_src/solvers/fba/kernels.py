# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics cloth solver)."""

import warp as wp

from newton._src.geometry.kernels import triangle_closest_point_barycentric

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


@wp.func
def project_arap_3x3_d(F: wp.mat33d) -> wp.mat33d:
    """fp64 variant of :func:`project_arap_3x3`.

    Lifts the SVD and reflection-handling rotation projection to ``wp.float64``
    so that ill-conditioned ``F`` (large condition number, e.g. extreme
    stretching of a wooper tail tip) does not lose accuracy through the fp32
    SVD core. Mirrors RealSim's ``signedEigenSVD`` (``SVD.cpp:7-29``) which
    runs an Eigen JacobiSVD in ``double``; the reflection branch here flips
    column 2 of U to keep R a proper rotation.
    """
    U, _sigma, V = wp.svd3(F)
    detUV = wp.determinant(U) * wp.determinant(V)
    if detUV < wp.float64(0.0):
        U = wp.mat33d(
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

    # fp64 SVD-based rotation projection. RealSim's signedEigenSVD
    # (SVD.cpp:7-29) runs Eigen JacobiSVD in double; we mirror that here to
    # avoid fp32 SVD accuracy loss on ill-conditioned F (e.g. extreme tail
    # stretching). proj is then demoted to fp32 for the rhs scatter.
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    R_d = project_arap_3x3_d(F_d)
    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(R_d))

    # Scatter stencil (RealSim PDTetrahedronEnergy.cpp:191-194):
    #   rhs[t[0]] += -proj.row(0) - proj.row(1) - proj.row(2)
    #   rhs[t[1]] += proj.row(0)
    #   rhs[t[2]] += proj.row(1)
    #   rhs[t[3]] += proj.row(2)
    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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

    # fp64 SVD-based rotation projection (see scatter variant for rationale).
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    R_d = project_arap_3x3_d(F_d)
    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(R_d))

    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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


@wp.func
def project_corotational_sigma3d_d(sigma: wp.vec3d, mu: wp.float64, lam: wp.float64) -> wp.vec3d:
    """fp64 variant of :func:`project_corotational_sigma3d`.

    Same closed-form Sherman-Morrison solve, lifted to ``wp.float64`` for the
    Tet Corot local projection so that ill-conditioned singular values feed a
    numerically stable solve. Pair with an fp64 SVD of F.
    """
    k = wp.float64(2.0) * mu
    alpha = wp.float64(4.0) * mu
    beta = lam
    n = wp.float64(3.0)

    b0 = wp.float64(2.0) * mu + wp.float64(3.0) * lam + k * sigma[0]
    b1 = wp.float64(2.0) * mu + wp.float64(3.0) * lam + k * sigma[1]
    b2 = wp.float64(2.0) * mu + wp.float64(3.0) * lam + k * sigma[2]
    sum_b = b0 + b1 + b2

    scale = beta * sum_b / (alpha * (alpha + n * beta))
    proj0 = b0 / alpha - scale
    proj1 = b1 / alpha - scale
    proj2 = b2 / alpha - scale
    return wp.vec3d(proj0, proj1, proj2)


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

    # fp64 SVD + sigma projection + P reconstruction. Mirrors RealSim
    # signedEigenSVD (SVD.cpp:7-29) which runs Eigen JacobiSVD in double.
    # Demoted to fp32 at the rhs scatter boundary only.
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    U_d, sigma_d, V_d = wp.svd3(F_d)
    sigma_proj_d = project_corotational_sigma3d_d(
        wp.vec3d(sigma_d[0], sigma_d[1], sigma_d[2]), wp.float64(mu), wp.float64(lam)
    )

    sd0 = sigma_proj_d[0]
    sd1 = sigma_proj_d[1]
    sd2 = sigma_proj_d[2]
    US_d = wp.mat33d(
        U_d[0, 0] * sd0,
        U_d[0, 1] * sd1,
        U_d[0, 2] * sd2,
        U_d[1, 0] * sd0,
        U_d[1, 1] * sd1,
        U_d[1, 2] * sd2,
        U_d[2, 0] * sd0,
        U_d[2, 1] * sd1,
        U_d[2, 2] * sd2,
    )
    P_d = US_d * wp.transpose(V_d)

    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(P_d))

    # Scatter stencil (RealSim PDTetrahedronEnergy.cpp:191-194):
    #   rhs[t[0]] += -proj.row(0) - proj.row(1) - proj.row(2)
    #   rhs[t[1]] += proj.row(0)
    #   rhs[t[2]] += proj.row(1)
    #   rhs[t[3]] += proj.row(2)
    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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

    # fp64 SVD + sigma projection + P reconstruction (see scatter variant).
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    U_d, sigma_d, V_d = wp.svd3(F_d)
    sigma_proj_d = project_corotational_sigma3d_d(
        wp.vec3d(sigma_d[0], sigma_d[1], sigma_d[2]), wp.float64(mu), wp.float64(lam)
    )

    sd0 = sigma_proj_d[0]
    sd1 = sigma_proj_d[1]
    sd2 = sigma_proj_d[2]
    US_d = wp.mat33d(
        U_d[0, 0] * sd0,
        U_d[0, 1] * sd1,
        U_d[0, 2] * sd2,
        U_d[1, 0] * sd0,
        U_d[1, 1] * sd1,
        U_d[1, 2] * sd2,
        U_d[2, 0] * sd0,
        U_d[2, 1] * sd1,
        U_d[2, 2] * sd2,
    )
    P_d = US_d * wp.transpose(V_d)

    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(P_d))

    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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
def project_neohookean_sigma3d_d(sigma: wp.vec3d, mu: wp.float64, lam: wp.float64) -> wp.vec3d:
    """fp64 variant of :func:`project_neohookean_sigma3d` (5-iter Newton).

    Lifts the per-iteration gradient/Hessian solve to ``wp.float64`` for the
    Tet NH ``newton5`` projection path. Paired with an fp64 SVD of F so that
    high-condition-number ``F`` (e.g. extreme tail-tip stretching) does not
    lose accuracy through fp32 division by tiny singular values.
    """
    eps = wp.float64(1.0e-6)
    k = wp.float64(2.0) * mu

    s0 = wp.max(sigma[0], eps)
    s1 = wp.max(sigma[1], eps)
    s2 = wp.max(sigma[2], eps)
    sigma0_0 = s0
    sigma0_1 = s1
    sigma0_2 = s2

    for _i in range(5):
        s0 = wp.max(s0, eps)
        s1 = wp.max(s1, eps)
        s2 = wp.max(s2, eps)

        J = s0 * s1 * s2
        log_J = wp.log(J)
        inv0 = wp.float64(1.0) / s0
        inv1 = wp.float64(1.0) / s1
        inv2 = wp.float64(1.0) / s2

        g0 = mu * (s0 - inv0) + lam * log_J * inv0 + k * (s0 - sigma0_0)
        g1 = mu * (s1 - inv1) + lam * log_J * inv1 + k * (s1 - sigma0_1)
        g2 = mu * (s2 - inv2) + lam * log_J * inv2 + k * (s2 - sigma0_2)

        diag_factor = mu + lam - lam * log_J
        H00 = mu + diag_factor * inv0 * inv0 + k
        H11 = mu + diag_factor * inv1 * inv1 + k
        H22 = mu + diag_factor * inv2 * inv2 + k
        H01 = lam * inv0 * inv1
        H02 = lam * inv0 * inv2
        H12 = lam * inv1 * inv2

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
    return wp.vec3d(s0, s1, s2)


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

    # fp64 SVD + 5-iter Newton + P reconstruction. Mirrors RealSim
    # signedEigenSVD (SVD.cpp:7-29). Demoted to fp32 at the rhs scatter.
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    U_d, sigma_d, V_d = wp.svd3(F_d)
    sigma_proj_d = project_neohookean_sigma3d_d(
        wp.vec3d(sigma_d[0], sigma_d[1], sigma_d[2]), wp.float64(mu), wp.float64(lam)
    )

    sd0 = sigma_proj_d[0]
    sd1 = sigma_proj_d[1]
    sd2 = sigma_proj_d[2]
    US_d = wp.mat33d(
        U_d[0, 0] * sd0,
        U_d[0, 1] * sd1,
        U_d[0, 2] * sd2,
        U_d[1, 0] * sd0,
        U_d[1, 1] * sd1,
        U_d[1, 2] * sd2,
        U_d[2, 0] * sd0,
        U_d[2, 1] * sd1,
        U_d[2, 2] * sd2,
    )
    P_d = US_d * wp.transpose(V_d)

    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(P_d))

    # Scatter stencil.
    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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

    # fp64 SVD + 5-iter Newton + P reconstruction (see scatter variant).
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    U_d, sigma_d, V_d = wp.svd3(F_d)
    sigma_proj_d = project_neohookean_sigma3d_d(
        wp.vec3d(sigma_d[0], sigma_d[1], sigma_d[2]), wp.float64(mu), wp.float64(lam)
    )

    sd0 = sigma_proj_d[0]
    sd1 = sigma_proj_d[1]
    sd2 = sigma_proj_d[2]
    US_d = wp.mat33d(
        U_d[0, 0] * sd0,
        U_d[0, 1] * sd1,
        U_d[0, 2] * sd2,
        U_d[1, 0] * sd0,
        U_d[1, 1] * sd1,
        U_d[1, 2] * sd2,
        U_d[2, 0] * sd0,
        U_d[2, 1] * sd1,
        U_d[2, 2] * sd2,
    )
    P_d = US_d * wp.transpose(V_d)

    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(P_d))

    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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
    instead of FBA's 5-iter Newton. SVD, LBFGS, sigma projection and P
    reconstruction all run in ``wp.float64`` -- mirroring RealSim's
    ``signedEigenSVD`` (``SVD.cpp:7-29``) + ``mcl::optlib::LBFGS<double,3>``
    contract. Final ``proj`` is demoted to fp32 only at the contributions
    write boundary (``_rhs`` itself is fp32).
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

    # fp64 SVD. wp.svd3 accepts mat33d and returns (mat33d, vec3d, mat33d).
    F_d = wp.mat33d(
        wp.float64(F[0, 0]),
        wp.float64(F[0, 1]),
        wp.float64(F[0, 2]),
        wp.float64(F[1, 0]),
        wp.float64(F[1, 1]),
        wp.float64(F[1, 2]),
        wp.float64(F[2, 0]),
        wp.float64(F[2, 1]),
        wp.float64(F[2, 2]),
    )
    U_d, sigma_d, V_d = wp.svd3(F_d)

    # LBFGS in fp64 on fp64 SVD output.
    mu_d = wp.float64(mu)
    lam_d = wp.float64(lam)
    sigma_proj_d = project_neohookean_sigma3d_lbfgs(wp.vec3d(sigma_d[0], sigma_d[1], sigma_d[2]), mu_d, lam_d)

    # Reconstruct P = U diag(sigma_proj) V^T in fp64.
    sd0 = sigma_proj_d[0]
    sd1 = sigma_proj_d[1]
    sd2 = sigma_proj_d[2]
    US_d = wp.mat33d(
        U_d[0, 0] * sd0,
        U_d[0, 1] * sd1,
        U_d[0, 2] * sd2,
        U_d[1, 0] * sd0,
        U_d[1, 1] * sd1,
        U_d[1, 2] * sd2,
        U_d[2, 0] * sd0,
        U_d[2, 1] * sd1,
        U_d[2, 2] * sd2,
    )
    P_d = US_d * wp.transpose(V_d)

    Dm_inv_d = wp.mat33d(
        wp.float64(Dm_inv[0, 0]),
        wp.float64(Dm_inv[0, 1]),
        wp.float64(Dm_inv[0, 2]),
        wp.float64(Dm_inv[1, 0]),
        wp.float64(Dm_inv[1, 1]),
        wp.float64(Dm_inv[1, 2]),
        wp.float64(Dm_inv[2, 0]),
        wp.float64(Dm_inv[2, 1]),
        wp.float64(Dm_inv[2, 2]),
    )
    w_d = wp.float64(tet_weight[t])
    proj_d = w_d * (Dm_inv_d * wp.transpose(P_d))

    row0 = wp.vec3(wp.float32(proj_d[0, 0]), wp.float32(proj_d[0, 1]), wp.float32(proj_d[0, 2]))
    row1 = wp.vec3(wp.float32(proj_d[1, 0]), wp.float32(proj_d[1, 1]), wp.float32(proj_d[1, 2]))
    row2 = wp.vec3(wp.float32(proj_d[2, 0]), wp.float32(proj_d[2, 1]), wp.float32(proj_d[2, 2]))
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


# Tile sizes for the dense GEMM used by ``compute_wi_tiled_kernel``.
# ``32 x 32 x 32`` with ``128`` threads per block was empirically fastest on
# Demo 5 (k ~ 1200, N ~ 7000) on an RTX 5090 fp64 pipeline — see
# ``scripts/profile_schur_w.py`` and the perf summary in
# ``docs/superpowers/specs/2026-05-18-fba-realsim-perf-comparison.md``.
_WI_TILE_M = wp.constant(32)
_WI_TILE_N = wp.constant(32)
_WI_TILE_K = wp.constant(32)
_WI_TILE_THREADS = 128


@wp.kernel
def densify_S_iso_scaled_kernel(
    ST_offsets: wp.array[wp.int32],
    ST_columns: wp.array[wp.int32],
    ST_values: wp.array[wp.float64],
    D_inv: wp.array[wp.float64],
    isodof_perm: wp.array[wp.int32],
    S_iso_scaled: wp.array2d[wp.float64],
):
    """Densify selected columns of ``S`` (scaled by ``sqrt(D_inv)``) into ``(N_pad, k_pad)``.

    Writes ``S_iso_scaled[i, a] = sqrt(D_inv[i]) * S[i, isodof_perm[a]]`` for
    ``i`` in the nonzero support of column ``isodof_perm[a]`` of ``S``.

    Companion to :func:`compute_wi_tiled_kernel`.  The output buffer must be
    pre-zeroed (the kernel only writes to nonzero positions of S).  Padding
    rows ``i >= N`` and padding columns ``a >= k`` are left as zero, so the
    subsequent tile matmul reduces over the padded zeros without polluting
    the result.

    One thread per ``a`` in ``[0, k)``; the inner loop walks the CSR column
    sequentially (``ST_offsets[ip..ip+1]``) — this is the densification path
    that replaces the prior ``compute_wi_kernel`` two-pointer intersection.

    Args:
        ST_offsets: CSR row offsets of ``S^T`` (column offsets of ``S``),
            length ``N + 1``, int32.
        ST_columns: CSR column indices of ``S^T`` (row indices of ``S``),
            length ``nnz(S)``, int32, sorted within each column.
        ST_values: CSR values of ``S^T``, length ``nnz(S)``, float64.
        D_inv: ``D^{-1}`` diagonal of the ``LDL^T`` factorization, length
            ``N``, float64. ``A^{-1} = P · S^T · D^{-1} · S · P^T`` so the
            sandwich form ``S^T · diag(D^{-1}) · S`` matches the per-particle
            Schur scaling.
        isodof_perm: ``perm[isodofs[a]]`` for ``a = 0..k-1``, length ``k``,
            int32 — selects which columns of ``S`` to densify.
        S_iso_scaled: Output dense buffer of shape ``(N_pad, k_pad)``, float64.
            Must be zero-initialized before launch; only entries at nonzero
            positions of the selected ``S`` columns are written.
    """
    a = wp.tid()
    ip = isodof_perm[a]
    beg = ST_offsets[ip]
    end = ST_offsets[ip + 1]
    for p in range(beg, end):
        i = ST_columns[p]
        S_iso_scaled[i, a] = wp.sqrt(D_inv[i]) * ST_values[p]


@wp.kernel
def compute_wi_tiled_kernel(
    S_iso_scaled: wp.array2d[wp.float64],
    Wi: wp.array2d[wp.float64],
    N_pad: int,
):
    """Compute ``Wi = X^T · X`` via blocked dense GEMM where ``X = sqrt(D_inv) · S_iso``.

    Replaces the O(k² · nnz_col) two-pointer ``compute_wi_kernel`` with a dense
    blocked matrix-matrix multiply.  For Demo 5 (k≈1200, N≈7000, mean S column
    nnz ≈ 1700 — i.e. ``S`` columns are ~24% dense) the densified GEMM is
    >3x faster than the sparse intersection because the inner loop becomes a
    streamed tile-mma instead of a per-(a,b) gather over ~1700 indirected
    fp64 reads.

    Computes the *full* ``(k x k)`` ``Wi`` matrix (not just the upper triangle),
    so ``compose_W_from_wi_kernel`` can index it symmetrically without a
    follow-up mirror pass.

    Launched with ``wp.launch_tiled`` over a ``(k_pad/TILE_M, k_pad/TILE_N)``
    grid; one CUDA block per output tile, ``_WI_TILE_THREADS`` threads per
    block.

    Args:
        S_iso_scaled: Dense densified S-columns buffer of shape
            ``(N_pad, k_pad)``, float64. ``S_iso_scaled[i, a] =
            sqrt(D_inv[i]) * S[i, isodof_perm[a]]``. Padding (``i >= N``,
            ``a >= k``) must be zero.
        Wi: Output dense Schur factor block of shape ``(k_pad, k_pad)``,
            float64.  Both upper and lower triangles are written.
        N_pad: Padded row count of ``S_iso_scaled``; must be a multiple of
            ``_WI_TILE_K``.
    """
    tile_i, tile_j = wp.tid()
    acc = wp.tile_zeros(shape=(_WI_TILE_M, _WI_TILE_N), dtype=wp.float64)
    for k_tile in range(0, N_pad, _WI_TILE_K):
        a = wp.tile_load(S_iso_scaled, shape=(_WI_TILE_K, _WI_TILE_M), offset=(k_tile, tile_i * _WI_TILE_M))
        b = wp.tile_load(S_iso_scaled, shape=(_WI_TILE_K, _WI_TILE_N), offset=(k_tile, tile_j * _WI_TILE_N))
        a_T = wp.tile_transpose(a)
        wp.tile_matmul(a_T, b, acc)
    wp.tile_store(Wi, acc, offset=(tile_i * _WI_TILE_M, tile_j * _WI_TILE_N))


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
def compose_W_lite_kernel(
    contact_particle: wp.array[wp.int32],
    contact_dir: wp.array[wp.vec3],
    contact_alpha: wp.array[wp.float32],
    inv_mass: wp.array[wp.float64],
    dt2: wp.float64,
    W: wp.array2d[wp.float64],
):
    """Assemble the LiteNSN dense Schur ``W_lite = H · (dt²·M⁻¹) · Hᵀ``.

    LiteNSN replaces the full ``A⁻¹`` in :func:`compose_W_from_wi_kernel`'s
    :math:`(J A^{-1} J^T)[c', c]` formula with the diagonal mass-inverse
    approximation :math:`dt² · inv_m[p] · I_3`, so off-particle entries
    collapse to zero:

        W_lite[c', c] = alpha[c'] · alpha[c] · (dir[c'] · dir[c])
                         · dt² · inv_m[particle[c]]      if particle[c'] == particle[c]
                       = 0                               otherwise

    This is the dense projection of the sparse Schur the BSR pipeline produces
    (:meth:`FBALinearSolver.build_schur_lite`); using it from the NSN inner
    keeps the existing dense-PCR consumer path intact while still exercising
    the LiteNSN mathematical approximation.

    Pinned particles (``inv_mass == 0``) zero their entire row+column block —
    matching the full-NSN behavior where ``A`` has an infinite mass on those
    rows.

    One thread per ``(c', c)`` pair.  ``W`` may be a larger pre-allocated buffer
    sliced to the active ``(total_rows, total_rows)`` block; entries outside
    the launch domain are not touched.
    """
    cp, c = wp.tid()
    p_cp = contact_particle[cp]
    p_c = contact_particle[c]
    if p_cp != p_c:
        W[cp, c] = wp.float64(0.0)
        return
    alpha = wp.float64(contact_alpha[cp]) * wp.float64(contact_alpha[c])
    dot = wp.float64(wp.dot(contact_dir[cp], contact_dir[c]))
    W[cp, c] = alpha * dot * dt2 * inv_mass[p_c]


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


# ---------------------------------------------------------------------------
# NSN inner LCP — Fischer-Burmeister rows + Schur build (Option C, fp64).
#
# GPU port of RealSim's ``NonSmoothNewton::buildSchurFB`` per-row scalar
# evaluators and the ``A_schur = ωωᵀ ⊙ W + diag(c)`` assembly. The inner
# Schur solve runs in fp64 for parity with the CPU reference; see audit V
# in ``docs/superpowers/plans/2026-05-17-fba-nsn-gpu-port.md``.
# ---------------------------------------------------------------------------


@wp.func
def fb_unilateral_row_wp(
    penetration: wp.float64,
    lam: wp.float64,
    precond: wp.float64,
    dt: wp.float64,
    pene0: wp.float64,
) -> wp.vec3d:
    """Fischer-Burmeister evaluation for a unilateral (normal) contact row.

    GPU port of RealSim ``NonSmoothNewton.cpp:332-341`` and the CPU
    reference :func:`~newton._src.solvers.fba.solver_fba.fb_unilateral_row`.
    Returns a :class:`warp.vec3d` packing ``(omega, compliance, h)`` —
    Warp ``@wp.func`` cannot return Python tuples, so we use the fp64 vec3
    type for the three scalars.

    Args:
        penetration: ``J·q - pene0`` at current iterate (m).
        lam: Current normal lambda (N).
        precond: ``1 / W_ii`` (1/N).
        dt: Timestep (s).
        pene0: ``n·anchor`` (m).

    Returns:
        ``vec3d(omega, compliance, h)`` matching the CPU reference's tuple
        return semantics. The degenerate-root branch returns
        ``(0, precond/dt², pene0)`` to suppress this row's contribution.
    """
    pene = penetration
    plam = precond * lam
    root = wp.sqrt(pene * pene + plam * plam)
    if root < wp.float64(1.0e-30):
        return wp.vec3d(wp.float64(0.0), precond / (dt * dt), pene0)
    omega = wp.float64(1.0) - pene / root
    compliance = (wp.float64(1.0) - plam / root) * (precond / (dt * dt))
    h = -(pene + plam - root) + omega * (penetration + pene0)
    return wp.vec3d(omega, compliance, h)


@wp.func
def fb_frictional_row_wp(
    penetration: wp.float64,
    lam_t: wp.float64,
    lam_n: wp.float64,
    mu: wp.float64,
    precond: wp.float64,
    dt: wp.float64,
    pene0: wp.float64,
) -> wp.vec3d:
    """Fischer-Burmeister evaluation for a frictional (tangent) contact row.

    GPU port of RealSim ``NonSmoothNewton.cpp:343-378`` and the CPU
    reference :func:`~newton._src.solvers.fba.solver_fba.fb_frictional_row`.
    Returns ``(omega, compliance, h)`` packed as :class:`warp.vec3d`.

    Inactive contact (``lam_n ≤ 0``): ``omega = 0``, ``compliance = 1/dt``,
    ``h = -dt · lam_t``.

    Active contact (``lam_n > 0``): smooth complementarity between slip
    speed ``|penetration|/dt`` and cone slack ``μ·lam_n - |lam_t|``.
    ``omega = 1``; compliance varies between near-zero (stick) and ``~1/dt``
    (slip).

    Args:
        penetration: Tangential ``J·q - pene0`` row value (m).
        lam_t: Current tangent lambda (N).
        lam_n: Companion normal lambda (N).
        mu: Coulomb friction coefficient.
        precond: ``1 / W_ii`` for this tangent row (1/N).
        dt: Timestep (s).
        pene0: ``t·anchor`` (m).

    Returns:
        ``vec3d(omega, compliance, h)`` matching the CPU reference.
    """
    if lam_n <= wp.float64(0.0):
        return wp.vec3d(
            wp.float64(0.0),
            wp.float64(1.0) / dt,
            -dt * lam_t,
        )

    abspenevel = wp.abs(penetration / dt)
    tmp = precond * (mu * lam_n - wp.abs(lam_t))
    root = wp.sqrt(abspenevel * abspenevel + tmp * tmp)
    denom = abspenevel + mu * precond * lam_n - root
    if wp.abs(denom) < wp.float64(1.0e-30):
        compliance = wp.float64(1.0) / dt
    else:
        compliance = ((root - tmp) / denom) * (precond / dt)
    h = -(dt * dt) * compliance * lam_t + pene0
    return wp.vec3d(wp.float64(1.0), compliance, h)


@wp.kernel
def build_a_schur_kernel(
    W: wp.array2d[wp.float64],  # (n_rows, n_rows) Schur complement
    omega: wp.array[wp.float64],  # (n_rows,) per-row weighting
    compliance: wp.array[wp.float64],  # (n_rows,) per-row diagonal compliance
    a_schur: wp.array2d[wp.float64],  # (n_rows, n_rows) OUTPUT
):
    """Assemble ``A_schur = ωωᵀ ⊙ W + diag(compliance)`` on GPU.

    GPU port of the CPU expression in
    :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._solve_nsn_unilateral`
    (``A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)``),
    which mirrors RealSim ``NonSmoothNewton.cpp:332-341`` (unilateral) and
    ``:343-378`` (frictional) row contributions assembled into the inner
    Schur LHS.

    Launch with ``dim=(n_rows, n_rows)``.

    Args:
        W: Dense Schur complement, shape ``[n_rows, n_rows]``.
        omega: Per-row Fischer-Burmeister weighting, shape ``[n_rows]``.
        compliance: Per-row diagonal compliance, shape ``[n_rows]``.
        a_schur: Output LHS matrix, shape ``[n_rows, n_rows]``. Overwritten.
    """
    i, j = wp.tid()
    val = omega[i] * omega[j] * W[i, j]
    if i == j:
        val = val + compliance[i]
    a_schur[i, j] = val


# ---------------------------------------------------------------------------
# NSN inner driver — residual, penetration, rhs, FB row launchers, box clamp.
#
# Steps 6-10 of the NSN GPU port (see
# ``docs/superpowers/plans/2026-05-17-fba-nsn-gpu-port.md``). Each kernel is
# a direct port of a numpy expression in
# :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._solve_nsn_unilateral`
# / ``_solve_nsn_coulomb``; cross-reference RealSim
# ``NonSmoothNewton.cpp:102-171`` for the full FB-Newton update sequence.
# ---------------------------------------------------------------------------


@wp.kernel
def compute_contact_residual_unilateral_kernel(
    contact_offset: wp.array[wp.float64],  # (M,) device, n·anchor
    contact_alpha: wp.array[wp.float32],  # (M,)
    contact_normal: wp.array[wp.vec3],  # (M,) world frame
    contact_particle: wp.array[wp.int32],  # (M,)
    x: wp.array[wp.vec3],  # (N,) particle positions (float32)
    r: wp.array[wp.float64],  # (M,) output residual
):
    """GPU port of :meth:`SolverFBA._compute_contact_residual` (Stage A).

    Computes ``r[c] = offset[c] - alpha[c] * dot(normal[c], x[particle[c]])``
    one contact per thread. Mirrors the CPU loop body exactly, including
    the float32→float64 promotion on ``dot`` (Warp ``wp.dot`` on a ``vec3``
    returns ``float32``; we widen to fp64 before the multiply to match the
    CPU ``np.dot`` working in fp64 after the explicit cast).

    Args:
        contact_offset: ``n·anchor`` per contact (fp64), shape ``[M]``.
        contact_alpha: Per-contact Jacobian scale (fp32), shape ``[M]``.
        contact_normal: World-frame unit normal, shape ``[M]``.
        contact_particle: Per-contact particle index, shape ``[M]``.
        x: Particle positions (fp32), shape ``[N]``.
        r: Output residual (fp64), shape ``[M]``.
    """
    c = wp.tid()
    ip = contact_particle[c]
    alpha = wp.float64(contact_alpha[c])
    n_dot_x = wp.float64(wp.dot(contact_normal[c], x[ip]))
    r[c] = contact_offset[c] - alpha * n_dot_x


@wp.kernel
def compute_contact_residual_coulomb_kernel(
    contact_offset_n: wp.array[wp.float64],  # (M,) n·anchor
    contact_offset_t1: wp.array[wp.float64],  # (M,) t1·anchor (with kinematic shift)
    contact_offset_t2: wp.array[wp.float64],  # (M,) t2·anchor (with kinematic shift)
    contact_alpha: wp.array[wp.float32],  # (M,)
    contact_normal: wp.array[wp.vec3],  # (M,)
    contact_t1: wp.array[wp.vec3],  # (M,)
    contact_t2: wp.array[wp.vec3],  # (M,)
    contact_particle: wp.array[wp.int32],  # (M,)
    x: wp.array[wp.vec3],  # (N,) particle positions (float32)
    r: wp.array[wp.float64],  # (3M,) output residual
):
    """GPU port of :meth:`SolverFBA._compute_contact_residual_friction` (Stage B).

    Emits ``r[3c+0/1/2] = offset_{n/t1/t2}[c] - alpha[c] * dot(axis, x[p[c]])``
    where ``axis`` is normal / t1 / t2 respectively. One contact per thread
    writes three consecutive rows in interleaved ``[r_n, r_t1, r_t2]``
    layout, matching the CPU reference.

    Args:
        contact_offset_n: ``n·anchor`` (fp64), shape ``[M]``.
        contact_offset_t1: ``t1·anchor`` (fp64), shape ``[M]``.
        contact_offset_t2: ``t2·anchor`` (fp64), shape ``[M]``.
        contact_alpha: Per-contact Jacobian scale (fp32), shape ``[M]``.
        contact_normal: World-frame unit normal, shape ``[M]``.
        contact_t1: First tangent vectors, shape ``[M]``.
        contact_t2: Second tangent vectors, shape ``[M]``.
        contact_particle: Per-contact particle index, shape ``[M]``.
        x: Particle positions (fp32), shape ``[N]``.
        r: Output residual (fp64), shape ``[3M]``.
    """
    c = wp.tid()
    ip = contact_particle[c]
    alpha = wp.float64(contact_alpha[c])
    xp = x[ip]
    r[3 * c + 0] = contact_offset_n[c] - alpha * wp.float64(wp.dot(contact_normal[c], xp))
    r[3 * c + 1] = contact_offset_t1[c] - alpha * wp.float64(wp.dot(contact_t1[c], xp))
    r[3 * c + 2] = contact_offset_t2[c] - alpha * wp.float64(wp.dot(contact_t2[c], xp))


@wp.kernel
def compute_precond_unilateral_kernel(
    W: wp.array2d[wp.float64],  # (M, M) Schur W
    dt: wp.float64,
    precond: wp.array[wp.float64],  # (M,) output
):
    """Compute ``precond[i] = dt² · max(|W_ii|, 1e-12)`` (Stage A).

    GPU port of the CPU expression in
    :meth:`SolverFBA._solve_nsn_unilateral`:
    ``precond = (dt*dt) * np.where(|diag_W|>1e-12, max(diag_W, 1e-12), 1.0)``.
    Note the ``np.where`` falls through to ``1.0`` only when ``|diag_W| <=
    1e-12`` (degenerate row); otherwise we take ``max(diag_W, 1e-12)`` which
    differs from ``|diag_W|`` for negative diagonals. We replicate this
    branch exactly so the GPU output bit-matches the CPU reference.

    Args:
        W: Dense Schur complement, shape ``[M, M]``.
        dt: Timestep (fp64).
        precond: Output preconditioner, shape ``[M]``.
    """
    i = wp.tid()
    d = W[i, i]
    dt2 = dt * dt
    if wp.abs(d) > wp.float64(1.0e-12):
        if d > wp.float64(1.0e-12):
            precond[i] = dt2 * d
        else:
            precond[i] = dt2 * wp.float64(1.0e-12)
    else:
        precond[i] = dt2 * wp.float64(1.0)


@wp.kernel
def compute_precond_coulomb_kernel(
    W: wp.array2d[wp.float64],  # (3M, 3M) Schur W
    dt: wp.float64,
    precond: wp.array[wp.float64],  # (3M,) output
):
    """Compute Stage B preconditioner: ``dt²·W_ii`` for normal rows and
    ``dt·W_ii`` for tangent rows, with ``|W_ii|`` floor of ``1e-12``.

    GPU port of the per-contact loop in
    :meth:`SolverFBA._solve_nsn_coulomb` (``w_n/w_t1/w_t2 =
    max(|diag_W[3c..]|, 1e-12)``; ``precond[3c] = dt²·w_n``;
    ``precond[3c+1/2] = dt·w_t1/t2``).

    Args:
        W: Dense Schur complement, shape ``[3M, 3M]``.
        dt: Timestep (fp64).
        precond: Output preconditioner, shape ``[3M]``.
    """
    i = wp.tid()  # 0..3M
    w = wp.abs(W[i, i])
    if w < wp.float64(1.0e-12):
        w = wp.float64(1.0e-12)
    if (i % 3) == 0:
        precond[i] = dt * dt * w
    else:
        precond[i] = dt * w


# Tile width for the NSN matvec kernels below.  ``128`` matches the PCR
# matvec tile size (see ``nsn_pcr_solver.py``) — the NSN n_rows is the same
# Schur dimension and an RTX 5090 fp64 sweep on 2026-05-18 picked the same
# sweet spot at n_rows ~ 600..1500.
_NSN_MATVEC_TILE_K = wp.constant(128)
_NSN_MATVEC_BLOCK_DIM = 128


@wp.kernel
def compute_penetration_kernel(
    r: wp.array[wp.float64],
    W: wp.array2d[wp.float64],
    omega: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    n_rows: wp.int32,
    n_rows_pad: wp.int32,
    dt: wp.float64,
    penetration: wp.array[wp.float64],
):
    """GPU port of ``penetration = -r + dt² · W · (ω · λ)``.

    Mirrors the CPU expression inside the FB-Newton loop:
    ``penetration = -r + (dt*dt) * (W @ (omega * lam))``.

    Block-per-row tiled reduction: each block computes one ``penetration[i]``
    cooperatively by accumulating ``TILE_K``-wide strips of
    ``W[i, j] * omega[j] * lam[j]`` and reducing with ``wp.tile_sum``.
    Launched via :func:`wp.launch_tiled` with ``dim=n_rows`` and
    ``block_dim=_NSN_MATVEC_BLOCK_DIM``.  Replaces the prior single-thread
    inner loop (~4-5x faster at n_rows ~ 600..1500 on RTX 5090 fp64).

    Args:
        r: Contact residual (fp64), shape ``[n_rows]``.
        W: Schur complement, shape ``[n_rows, n_rows]``.
        omega: FB row weights, shape ``[n_rows]``.
        lam: Current lambda, shape ``[n_rows]``.
        n_rows: Active row count (``M`` for Stage A, ``3M`` for Stage B).
        n_rows_pad: ``ceil(n_rows / TILE_K) * TILE_K`` — the K loop runs up
            to this so the partial last tile reduces through the same
            ``tile_sum`` path; bounds-checked ``tile_load`` zero-pads OOB.
        dt: Timestep.
        penetration: Output penetration, shape ``[n_rows]``.
    """
    i = wp.tid()
    wol = wp.float64(0.0)
    for k_tile in range(0, n_rows_pad, _NSN_MATVEC_TILE_K):
        w_row = wp.tile_load(W[i], shape=(_NSN_MATVEC_TILE_K,), offset=(k_tile,))
        o_tile = wp.tile_load(omega, shape=(_NSN_MATVEC_TILE_K,), offset=(k_tile,))
        l_tile = wp.tile_load(lam, shape=(_NSN_MATVEC_TILE_K,), offset=(k_tile,))
        ol = wp.tile_map(wp.mul, o_tile, l_tile)
        prod = wp.tile_map(wp.mul, w_row, ol)
        s = wp.tile_sum(prod)
        # ``tile_extract`` broadcasts the (1,) reduction to every thread in
        # the block; folding into a scalar accumulator keeps the cross-iter
        # state out of shared memory (a shared-tile re-assign currently
        # breaks the Warp 1.14 register-vs-shared promotion).
        wol = wol + wp.tile_extract(s, 0)
    # All threads in the block write the same value — CUDA coalesces
    # identical writes (equivalent to a single thread emit).
    penetration[i] = -r[i] + dt * dt * wol


@wp.kernel
def compute_nsn_rhs_kernel(
    h: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    pene0: wp.array[wp.float64],
    r: wp.array[wp.float64],
    W: wp.array2d[wp.float64],
    lam: wp.array[wp.float64],
    n_rows: wp.int32,
    n_rows_pad: wp.int32,
    dt: wp.float64,
    rhs: wp.array[wp.float64],
):
    """GPU port of ``rhs = (1/dt²) · (h - ω · J·x_corrected)`` (NSN inner RHS).

    Mirrors the CPU expressions:

        J_x = (pene0 - r) + (dt*dt) * (W @ (omega * lam))
        rhs = (1.0 / (dt*dt)) * (h - omega * J_x)

    Block-per-row tiled reduction (see :func:`compute_penetration_kernel`)
    for the ``W @ (omega · lam)`` matvec — significantly faster than the
    prior single-thread inner loop at the M sizes targeted by NSN
    (M ~ 100..3000).  The two expressions stay fused in a single kernel so
    we avoid materialising ``J_x`` to a scratch buffer.

    Args:
        h: Per-row Schur RHS contribution, shape ``[n_rows]``.
        omega: Per-row FB weighting, shape ``[n_rows]``.
        pene0: Per-row anchor projection, shape ``[n_rows]``.
        r: Contact residual, shape ``[n_rows]``.
        W: Schur complement, shape ``[n_rows, n_rows]``.
        lam: Current lambda, shape ``[n_rows]``.
        n_rows: Active row count.
        n_rows_pad: ``ceil(n_rows / TILE_K) * TILE_K`` — see
            :func:`compute_penetration_kernel`.
        dt: Timestep.
        rhs: Output NSN RHS, shape ``[n_rows]``.
    """
    i = wp.tid()
    wol = wp.float64(0.0)
    for k_tile in range(0, n_rows_pad, _NSN_MATVEC_TILE_K):
        w_row = wp.tile_load(W[i], shape=(_NSN_MATVEC_TILE_K,), offset=(k_tile,))
        o_tile = wp.tile_load(omega, shape=(_NSN_MATVEC_TILE_K,), offset=(k_tile,))
        l_tile = wp.tile_load(lam, shape=(_NSN_MATVEC_TILE_K,), offset=(k_tile,))
        ol = wp.tile_map(wp.mul, o_tile, l_tile)
        prod = wp.tile_map(wp.mul, w_row, ol)
        s = wp.tile_sum(prod)
        wol = wol + wp.tile_extract(s, 0)
    j_x = (pene0[i] - r[i]) + dt * dt * wol
    rhs[i] = (wp.float64(1.0) / (dt * dt)) * (h[i] - omega[i] * j_x)


@wp.kernel
def compute_unilateral_fb_kernel(
    penetration: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    precond: wp.array[wp.float64],
    pene0: wp.array[wp.float64],
    dt: wp.float64,
    omega: wp.array[wp.float64],  # output
    compliance: wp.array[wp.float64],  # output
    h: wp.array[wp.float64],  # output
):
    """Evaluate :func:`fb_unilateral_row_wp` for every row.

    Per-row launcher that calls the existing ``@wp.func`` FB evaluator;
    one thread populates ``(omega[i], compliance[i], h[i])``. Matches the
    CPU per-row ``fb_unilateral_row(...)`` loop in
    :meth:`SolverFBA._solve_nsn_unilateral`.

    Args:
        penetration: Per-row penetration ``J·q - pene0``, shape ``[M]``.
        lam: Per-row lambda, shape ``[M]``.
        precond: Per-row preconditioner, shape ``[M]``.
        pene0: Per-row anchor projection, shape ``[M]``.
        dt: Timestep.
        omega: Output FB weighting, shape ``[M]``.
        compliance: Output diagonal compliance, shape ``[M]``.
        h: Output Schur RHS contribution, shape ``[M]``.
    """
    i = wp.tid()
    out = fb_unilateral_row_wp(penetration[i], lam[i], precond[i], dt, pene0[i])
    omega[i] = out[0]
    compliance[i] = out[1]
    h[i] = out[2]


@wp.kernel
def compute_frictional_fb_kernel(
    penetration: wp.array[wp.float64],  # (3M,)
    lam: wp.array[wp.float64],  # (3M,)
    mu: wp.array[wp.float64],  # (M,) per-contact
    precond: wp.array[wp.float64],  # (3M,)
    pene0: wp.array[wp.float64],  # (3M,)
    dt: wp.float64,
    omega: wp.array[wp.float64],  # (3M,) output
    compliance: wp.array[wp.float64],  # (3M,) output
    h: wp.array[wp.float64],  # (3M,) output
):
    """Evaluate the 3-row FB block (normal + 2 tangents) per contact.

    One thread per contact emits the unilateral normal row plus two
    frictional tangent rows, matching the CPU loop body in
    :meth:`SolverFBA._solve_nsn_coulomb`. The tangent rows take the
    *current iterate* of ``lam_n`` as their companion normal lambda
    (RealSim ``NonSmoothNewton.cpp:343-378``); this is identical to the
    CPU reference which reads ``lam[idx_n]`` before computing the tangent
    rows in the same iteration.

    Args:
        penetration: Per-row penetration, shape ``[3M]``.
        lam: Per-row lambda, shape ``[3M]``.
        mu: Per-contact friction coefficient, shape ``[M]``.
        precond: Per-row preconditioner, shape ``[3M]``.
        pene0: Per-row anchor projection, shape ``[3M]``.
        dt: Timestep.
        omega: Output FB weighting, shape ``[3M]``.
        compliance: Output diagonal compliance, shape ``[3M]``.
        h: Output Schur RHS contribution, shape ``[3M]``.
    """
    c = wp.tid()  # 0..M
    idx_n = 3 * c
    idx_t1 = 3 * c + 1
    idx_t2 = 3 * c + 2
    lam_n = lam[idx_n]
    mu_c = mu[c]
    out_n = fb_unilateral_row_wp(penetration[idx_n], lam_n, precond[idx_n], dt, pene0[idx_n])
    omega[idx_n] = out_n[0]
    compliance[idx_n] = out_n[1]
    h[idx_n] = out_n[2]
    out_t1 = fb_frictional_row_wp(penetration[idx_t1], lam[idx_t1], lam_n, mu_c, precond[idx_t1], dt, pene0[idx_t1])
    omega[idx_t1] = out_t1[0]
    compliance[idx_t1] = out_t1[1]
    h[idx_t1] = out_t1[2]
    out_t2 = fb_frictional_row_wp(penetration[idx_t2], lam[idx_t2], lam_n, mu_c, precond[idx_t2], dt, pene0[idx_t2])
    omega[idx_t2] = out_t2[0]
    compliance[idx_t2] = out_t2[1]
    h[idx_t2] = out_t2[2]


@wp.kernel
def fb_newton_lite_coulomb_kernel(
    # ---- contact metadata (M entries each) ----
    contact_particle: wp.array[wp.int32],   # (M,)
    contact_normal: wp.array[wp.vec3],      # (M,)
    contact_tangent1: wp.array[wp.vec3],    # (M,)
    contact_tangent2: wp.array[wp.vec3],    # (M,)
    contact_alpha: wp.array[wp.float32],    # (M,)
    contact_mu: wp.array[wp.float64],       # (M,)
    # ---- per-row inputs (3M entries each) ----
    r: wp.array[wp.float64],                # (3M,)
    pene0: wp.array[wp.float64],            # (3M,)
    # ---- global state ----
    inv_mass: wp.array[wp.float64],         # (N,)
    dt: wp.float64,
    cap_internal: wp.float64,               # lambda_cap / dt² (or sentinel; see caller)
    use_cap: wp.int32,                      # 1 = clip, 0 = no-op
    n_fb_iters: wp.int32,                   # outer FB-Newton iter count
    # ---- inout / outputs ----
    lam: wp.array[wp.float64],              # (3M,) inout
    omega: wp.array[wp.float64],            # (3M,) inout — warm-started on entry, written on exit
    lam_apply: wp.array[wp.float64],        # (3M,) output: dt² · ω · λ_final
):
    """Fused per-contact FB-Newton step for LiteNSN, Stage B (Coulomb).

    Under the lite approximation ``W = H · (dt²·M⁻¹) · Hᵀ``, contacts that touch
    distinct particles decouple completely.  For cloth-on-sphere (and all
    cloth-on-rigid scenes where each cloth particle touches at most one rigid
    obstacle), every particle is touched by exactly one contact, so the
    ``3M × 3M`` Schur system block-diagonalises into ``M`` independent ``3×3``
    blocks.  Each thread of this kernel processes one contact end-to-end:

    1. Build the local ``3×3`` ``W`` block from ``α``, ``dir``, and ``dt² · inv_m``.
    2. Compute the local penetration ``-r + dt² · W · (ω·λ)``.
    3. Evaluate the FB rows (normal + 2 tangents) to refresh ``(ω, c, h)``.
    4. Build ``A_schur = ωωᵀ ⊙ W + diag(c)`` and ``rhs = (1/dt²)·(h - ω · J_x)``.
    5. Invert the ``3×3`` ``A_schur`` and apply ``dλ``.
    6. Coulomb clamp + optional symmetric ``lambda_cap``.
    7. Write ``lam_apply = dt² · ω · λ`` for the host-side correction step.

    Memory cost: ``O(M)`` (no Schur matrix materialised).  Scales linearly
    with contact count and trivially across worlds — the per-particle
    decoupling means there is no inter-world coupling for the constraint solve.

    Args:
        contact_particle, contact_normal, contact_tangent1, contact_tangent2,
        contact_alpha, contact_mu: Per-contact arrays of length ``M``.
        r, pene0: Per-row arrays of length ``3M`` (rows interleaved ``[n, t1, t2]``).
        inv_mass: Per-particle inverse mass, length ``N``.
        dt: Timestep.
        cap_internal: ``lambda_cap / dt²`` (in solver-internal units); only
            applied when ``use_cap == 1``.
        use_cap: ``1`` to enable the symmetric ``[-cap, +cap]`` clip,
            ``0`` to disable.
        n_fb_iters: Number of FB-Newton iterations (locked at ``1`` in
            current SolverFBA; supported up to higher counts for stress tests).
        lam: Per-row lambda buffer, length ``3M``.  Warm-started on entry,
            updated on exit.
        omega: Per-row FB weight buffer, length ``3M``.  Warm-started on entry,
            overwritten with the FB-loop's final value on exit.
        lam_apply: Output per-row impulse to apply to ``x_cur``, length ``3M``.
    """
    c = wp.tid()
    p = contact_particle[c]
    inv_m = inv_mass[p]
    dt2 = dt * dt

    # Per-row indices.
    i_n = 3 * c
    i_t1 = 3 * c + 1
    i_t2 = 3 * c + 2

    # Direction vectors (fp64-promoted dots for accumulation).
    n_d = contact_normal[c]
    t1_d = contact_tangent1[c]
    t2_d = contact_tangent2[c]

    # Local W_3x3 = α² · dot(dir_a, dir_b) · dt² · inv_m.
    # ``α`` is the same for all three rows of a contact (set by the host build).
    a_n = wp.float64(contact_alpha[c])
    coef = a_n * a_n * dt2 * inv_m
    W_nn = coef * wp.float64(wp.dot(n_d, n_d))
    W_nt1 = coef * wp.float64(wp.dot(n_d, t1_d))
    W_nt2 = coef * wp.float64(wp.dot(n_d, t2_d))
    W_t1t1 = coef * wp.float64(wp.dot(t1_d, t1_d))
    W_t1t2 = coef * wp.float64(wp.dot(t1_d, t2_d))
    W_t2t2 = coef * wp.float64(wp.dot(t2_d, t2_d))

    # Precond (matches compute_precond_coulomb_kernel: dt² for normal, dt for tangent).
    eps = wp.float64(1.0e-12)
    w_n_a = wp.abs(W_nn)
    w_t1_a = wp.abs(W_t1t1)
    w_t2_a = wp.abs(W_t2t2)
    if w_n_a < eps:
        w_n_a = eps
    if w_t1_a < eps:
        w_t1_a = eps
    if w_t2_a < eps:
        w_t2_a = eps
    precond_n = dt2 * w_n_a
    precond_t1 = dt * w_t1_a
    precond_t2 = dt * w_t2_a

    # Per-row state at entry.
    lam_n = lam[i_n]
    lam_t1 = lam[i_t1]
    lam_t2 = lam[i_t2]
    om_n = omega[i_n]
    om_t1 = omega[i_t1]
    om_t2 = omega[i_t2]
    r_n = r[i_n]
    r_t1 = r[i_t1]
    r_t2 = r[i_t2]
    p0_n = pene0[i_n]
    p0_t1 = pene0[i_t1]
    p0_t2 = pene0[i_t2]
    mu_c = contact_mu[c]

    for _it in range(n_fb_iters):
        # penetration = -r + dt² · W · (ω · λ).  Pure 3×3 matvec.
        ol_n = om_n * lam_n
        ol_t1 = om_t1 * lam_t1
        ol_t2 = om_t2 * lam_t2
        Wol_n = W_nn * ol_n + W_nt1 * ol_t1 + W_nt2 * ol_t2
        Wol_t1 = W_nt1 * ol_n + W_t1t1 * ol_t1 + W_t1t2 * ol_t2
        Wol_t2 = W_nt2 * ol_n + W_t1t2 * ol_t1 + W_t2t2 * ol_t2
        pen_n = -r_n + dt2 * Wol_n
        pen_t1 = -r_t1 + dt2 * Wol_t1
        pen_t2 = -r_t2 + dt2 * Wol_t2

        # FB evaluation refreshes (ω, c, h) using the current λ.  The tangent
        # rows read ``lam_n`` (companion normal) as RealSim does.
        out_n = fb_unilateral_row_wp(pen_n, lam_n, precond_n, dt, p0_n)
        om_n = out_n[0]
        c_n = out_n[1]
        h_n = out_n[2]
        out_t1 = fb_frictional_row_wp(pen_t1, lam_t1, lam_n, mu_c, precond_t1, dt, p0_t1)
        om_t1 = out_t1[0]
        c_t1 = out_t1[1]
        h_t1 = out_t1[2]
        out_t2 = fb_frictional_row_wp(pen_t2, lam_t2, lam_n, mu_c, precond_t2, dt, p0_t2)
        om_t2 = out_t2[0]
        c_t2 = out_t2[1]
        h_t2 = out_t2[2]

        # A_schur_3x3 = ωωᵀ ⊙ W + diag(c).
        A00 = om_n * om_n * W_nn + c_n
        A01 = om_n * om_t1 * W_nt1
        A02 = om_n * om_t2 * W_nt2
        A11 = om_t1 * om_t1 * W_t1t1 + c_t1
        A12 = om_t1 * om_t2 * W_t1t2
        A22 = om_t2 * om_t2 * W_t2t2 + c_t2

        # rhs = (1/dt²) · (h - ω · J_x) where J_x = (pene0 - r) + dt²·W·(ω·λ)
        # = pene0 + pen (since pen = -r + dt²·W·(ω·λ)).
        Jx_n = p0_n + pen_n
        Jx_t1 = p0_t1 + pen_t1
        Jx_t2 = p0_t2 + pen_t2
        inv_dt2 = wp.float64(1.0) / dt2
        rhs_n = inv_dt2 * (h_n - om_n * Jx_n)
        rhs_t1 = inv_dt2 * (h_t1 - om_t1 * Jx_t1)
        rhs_t2 = inv_dt2 * (h_t2 - om_t2 * Jx_t2)

        # 3×3 symmetric solve via cofactor expansion (no Cholesky needed at
        # this size; the matrix is SPD by construction modulo the ωωᵀ Hadamard
        # which preserves SPD when c > 0 and W is SPD).
        # Determinant of A (symmetric).
        m11 = A11 * A22 - A12 * A12
        m12 = A01 * A22 - A02 * A12
        m13 = A01 * A12 - A02 * A11
        det = A00 * m11 - A01 * m12 + A02 * m13
        if wp.abs(det) < wp.float64(1.0e-30):
            # Singular — skip the update (warm-start λ unchanged).
            continue

        inv_det = wp.float64(1.0) / det
        # Inverse of symmetric 3×3 via cofactor matrix.
        i00 = (A11 * A22 - A12 * A12) * inv_det
        i01 = -(A01 * A22 - A02 * A12) * inv_det
        i02 = (A01 * A12 - A02 * A11) * inv_det
        i11 = (A00 * A22 - A02 * A02) * inv_det
        i12 = -(A00 * A12 - A02 * A01) * inv_det
        i22 = (A00 * A11 - A01 * A01) * inv_det

        dlam_n = i00 * rhs_n + i01 * rhs_t1 + i02 * rhs_t2
        dlam_t1 = i01 * rhs_n + i11 * rhs_t1 + i12 * rhs_t2
        dlam_t2 = i02 * rhs_n + i12 * rhs_t1 + i22 * rhs_t2

        lam_n = lam_n + dlam_n
        lam_t1 = lam_t1 + dlam_t1
        lam_t2 = lam_t2 + dlam_t2

    # Optional symmetric lambda cap (post-loop, matches dense path's
    # lambda_cap_clip after the FB-Newton loop).
    if use_cap == wp.int32(1):
        if lam_n > cap_internal:
            lam_n = cap_internal
        if lam_n < -cap_internal:
            lam_n = -cap_internal
        if lam_t1 > cap_internal:
            lam_t1 = cap_internal
        if lam_t1 < -cap_internal:
            lam_t1 = -cap_internal
        if lam_t2 > cap_internal:
            lam_t2 = cap_internal
        if lam_t2 < -cap_internal:
            lam_t2 = -cap_internal

    # Write back λ, ω, and lam_apply = dt² · ω · λ_final.
    lam[i_n] = lam_n
    lam[i_t1] = lam_t1
    lam[i_t2] = lam_t2
    omega[i_n] = om_n
    omega[i_t1] = om_t1
    omega[i_t2] = om_t2
    lam_apply[i_n] = dt2 * om_n * lam_n
    lam_apply[i_t1] = dt2 * om_t1 * lam_t1
    lam_apply[i_t2] = dt2 * om_t2 * lam_t2


# ----- BSR + sparse-PCR helpers for LiteNSN with self-contact (Phase 3) -----

# Module-level type alias for the 1×1 fp64 BSR block — warp can only resolve
# matrix types when they're statically named at module scope.
_BSR_BLOCK_1X1_F64 = wp.types.matrix((1, 1), wp.float64)


@wp.kernel
def precond_from_bsr_diag_coulomb_kernel(
    diag: wp.array[wp.float64],
    dt: wp.float64,
    precond: wp.array[wp.float64],
):
    """Compute Stage B precond (dt²·|W_ii| for normals, dt·|W_ii| for tangents)
    directly from a BSR diagonal.  Companion to the dense
    :func:`compute_precond_coulomb_kernel`.

    ``warp.sparse.bsr_mm`` collapses a (1×1) block_type product into a
    scalar fp64 BsrMatrix, so ``bsr_get_diag`` returns an
    ``array[wp.float64]`` rather than ``array[mat1x1]``.
    """
    i = wp.tid()
    w = wp.abs(diag[i])
    if w < wp.float64(1.0e-12):
        w = wp.float64(1.0e-12)
    if (i % 3) == 0:
        precond[i] = dt * dt * w
    else:
        precond[i] = dt * w


@wp.kernel
def compute_penetration_sparse_kernel(
    r: wp.array[wp.float64],
    w_omega_lam: wp.array[wp.float64],
    dt: wp.float64,
    penetration: wp.array[wp.float64],
):
    """Per-row ``penetration = -r + dt² · W·(ω·λ)`` for the sparse path.

    Sparse equivalent of the matvec embedded in
    :func:`compute_penetration_kernel`: caller supplies the precomputed
    ``W·(ω·λ)`` (from ``wps.bsr_mv``) so this kernel is a trivial 1-line
    fused multiply-add.
    """
    i = wp.tid()
    penetration[i] = -r[i] + dt * dt * w_omega_lam[i]


@wp.kernel
def build_a_schur_lite_bsr_kernel(
    w_offsets: wp.array[wp.int32],
    w_columns: wp.array[wp.int32],
    w_values: wp.array[wp.float64],          # collapsed scalar values for 1×1 BSR
    omega: wp.array[wp.float64],
    compliance: wp.array[wp.float64],
    a_values: wp.array[wp.float64],          # output, same shape as w_values
):
    """Build A_schur = ωωᵀ ⊙ W + diag(c) in BSR with W's sparsity (scalar
    values since ``warp.sparse`` collapses 1×1 block BSR to scalar storage).
    """
    i = wp.tid()
    start = w_offsets[i]
    end = w_offsets[i + 1]
    om_i = omega[i]
    c_i = compliance[i]
    for k in range(start, end):
        j = w_columns[k]
        om_j = omega[j]
        a_ij = om_i * om_j * w_values[k]
        if i == j:
            a_ij = a_ij + c_i
        a_values[k] = a_ij


@wp.kernel
def compute_nsn_rhs_lite_bsr_kernel(
    h: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    pene0: wp.array[wp.float64],
    r: wp.array[wp.float64],
    w_omega_lam: wp.array[wp.float64],      # W · (ω · λ) precomputed via bsr_mv
    dt: wp.float64,
    rhs: wp.array[wp.float64],
):
    """Per-row RHS for the sparse LiteNSN PCR.

    Mirrors the dense ``compute_nsn_rhs_kernel`` but takes a pre-computed
    ``W · (ω · λ)`` vector (assembled by the caller via
    ``wp.sparse.bsr_mv(W_bsr, omega_times_lam, w_omega_lam)``) instead of
    inlining the matvec — that path is what makes the rhs build sparse
    instead of dense.

        J_x = (pene0 - r) + dt² · w_omega_lam
        rhs = (1/dt²) · (h - omega · J_x)
    """
    i = wp.tid()
    dt2 = dt * dt
    Jx = (pene0[i] - r[i]) + dt2 * w_omega_lam[i]
    rhs[i] = (wp.float64(1.0) / dt2) * (h[i] - omega[i] * Jx)


@wp.kernel
def compute_omega_times_lam_kernel(
    omega: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    """Elementwise ``out[i] = omega[i] · lam[i]`` so we can feed ``bsr_mv``."""
    i = wp.tid()
    out[i] = omega[i] * lam[i]


@wp.kernel
def compute_particle_contact_residual_kernel(
    # Per-row inputs:
    row_particle_a: wp.array[wp.int32],     # (3M_self,)
    row_particle_b: wp.array[wp.int32],     # (3M_self,) all valid (≥0)
    row_dir: wp.array[wp.vec3],             # (3M_self,)
    row_alpha: wp.array[wp.float32],        # (3M_self,)
    row_offset: wp.array[wp.float64],       # (3M_self,) = pene0[i]; gap threshold for n rows, 0 for tangents
    x: wp.array[wp.vec3],                   # (N,) particle positions
    # Output:
    r: wp.array[wp.float64],                # (3M_self,)
):
    """Residual for self-contact rows: ``r[i] = offset[i] - α[i] · dot(dir[i], x[a] - x[b])``.

    Same FB convention as :func:`compute_contact_residual_coulomb_kernel`:
    positive r means "penetrating" (gap_threshold > current_signed_separation),
    so the FB unilateral function drives ``λ > 0`` to push the pair apart.
    """
    i = wp.tid()
    ia = row_particle_a[i]
    ib = row_particle_b[i]
    alpha = wp.float64(row_alpha[i])
    diff = x[ia] - x[ib]
    r[i] = row_offset[i] - alpha * wp.float64(wp.dot(row_dir[i], diff))


@wp.kernel
def apply_dt2_inv_mass_correction_kernel(
    jt_lambda: wp.array[wp.vec3],
    inv_mass: wp.array[wp.float64],
    dt: wp.float64,
    x_cur: wp.array[wp.vec3],
):
    """LiteNSN-consistent position correction ``x += dt² · inv_m · Jᵀ·λ``.

    The full-NSN correction uses ``A⁻¹·Jᵀ·λ`` where ``A`` is the PD
    prefactor (mass + stretching + bending stiffness).  Under the lite
    approximation, the Schur was built with ``A_lite = M/dt²``, so for
    consistency the correction must also use the same diagonal mass-inverse
    rather than the full PD ``A``.  Using full ``A⁻¹`` here spreads the
    self-contact impulse across the cloth via elastic coupling — for
    1013-particle cloth this can shrink the per-step displacement by 20-50×,
    which is why visual self-contact appeared to fail even though the
    FB-Newton inner produced healthy λ values (~12k).
    """
    p = wp.tid()
    dt2 = dt * dt
    scale = dt2 * inv_mass[p]
    j = jt_lambda[p]
    x_cur[p] = x_cur[p] + wp.vec3(
        wp.float32(scale * wp.float64(j[0])),
        wp.float32(scale * wp.float64(j[1])),
        wp.float32(scale * wp.float64(j[2])),
    )


@wp.kernel
def scatter_jt_lambda_two_particle_kernel(
    n_rows: wp.int32,
    row_particle_a: wp.array[wp.int32],
    row_particle_b: wp.array[wp.int32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    lam: wp.array[wp.float32],
    out: wp.array[wp.vec3],            # atomic-add target, length N
):
    """Per-row atomic scatter of ``Jᵀ·λ`` for 2-particle rows.

    For each row r:
        out[particle_a] += +α[r] · λ[r] · dir[r]
        out[particle_b] += -α[r] · λ[r] · dir[r]   (only if particle_b ≥ 0)

    Atomic adds make this non-deterministic across runs, but for the small
    n_self contributions this is much simpler than building a per-step CSR.
    Caller is responsible for zeroing ``out`` (or seeding it with the rigid
    contribution) before launch.
    """
    i = wp.tid()
    if i >= n_rows:
        return
    pa = row_particle_a[i]
    contrib = row_alpha[i] * lam[i] * row_dir[i]
    wp.atomic_add(out, pa, contrib)
    pb = row_particle_b[i]
    if pb >= wp.int32(0):
        wp.atomic_add(out, pb, -contrib)


@wp.kernel
def gather_jt_lambda_two_particle_kernel(
    # Per-particle CSR: row_offsets[p+1] - row_offsets[p] = number of
    # signed entries for particle p.
    row_offsets: wp.array[wp.int32],       # (N+1,)
    row_indices: wp.array[wp.int32],       # (E,) — row idx
    row_signs: wp.array[wp.float32],       # (E,) — +1.0 or -1.0
    # Per-row direction and alpha (length = total self-contact rows).
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    lam: wp.array[wp.float32],
    out: wp.array[wp.vec3],
):
    """Particle-centred ``Jᵀ·λ`` gather over rows that touch 2 particles.

    For each particle ``p``, sums ``sign · α[r] · λ[r] · dir[r]`` over the
    rows in its CSR slot.  ``sign`` is ``+1`` for rows where ``p ==
    row_particle_a[r]`` and ``-1`` for rows where ``p == row_particle_b[r]``,
    mirroring the ``∂/∂q`` of a relative-position constraint.

    Writes (NOT accumulates) into ``out`` — caller zeroes ``out`` only on
    particles that have entries.  Particles with empty rows leave ``out[p]``
    unchanged from its prior value (so this kernel can be composed with the
    rigid-side gather without zeroing rigid-only particles).
    """
    p = wp.tid()
    start = row_offsets[p]
    end = row_offsets[p + 1]
    if start == end:
        return
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(start, end):
        ri = row_indices[k]
        s = row_signs[k]
        acc = acc + s * row_alpha[ri] * lam[ri] * row_dir[ri]
    out[p] = out[p] + acc


@wp.kernel
def emit_particle_contact_rows_kernel(
    n_pairs: wp.int32,
    pair_a: wp.array[wp.int32],             # (n_pairs,)
    pair_b: wp.array[wp.int32],             # (n_pairs,)
    pair_normal: wp.array[wp.vec3],         # (n_pairs,) unit from b → a
    gap_threshold: wp.float64,              # 2 · particle_radius
    row_offset_base: wp.int32,              # where in the output arrays self rows start
    row_particle_a: wp.array[wp.int32],
    row_particle_b: wp.array[wp.int32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    row_offset: wp.array[wp.float64],
    contact_mu: wp.array[wp.float64],       # (M_total,) per-contact μ — slot pair_to_contact_idx + p
):
    """Convert one broadphase pair into a Stage B contact (3 rows).

    Tangent basis: build a stable orthonormal frame from the normal.  Pick
    the world axis least aligned with ``n`` and Gram-Schmidt — avoids the
    degeneracy of crossing ``n`` with itself when ``n`` happens to align
    with a coordinate axis.

    Friction μ is written by the host (single ``mu_d.fill_(self_friction_mu)``
    call) — keeping it out of the kernel lets the caller pre-compose μ
    arrays without re-launching this emit.
    """
    p = wp.tid()
    if p >= n_pairs:
        return
    a = pair_a[p]
    b = pair_b[p]
    n = pair_normal[p]

    ax = wp.float32(wp.abs(n[0]))
    ay = wp.float32(wp.abs(n[1]))
    az = wp.float32(wp.abs(n[2]))
    if ax <= ay and ax <= az:
        ref = wp.vec3(1.0, 0.0, 0.0)
    elif ay <= az:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(0.0, 0.0, 1.0)
    t1 = wp.cross(n, ref)
    t1 = t1 / wp.max(wp.length(t1), wp.float32(1.0e-9))
    t2 = wp.cross(n, t1)

    base = row_offset_base + 3 * p
    row_particle_a[base + 0] = a
    row_particle_b[base + 0] = b
    row_dir[base + 0] = n
    row_alpha[base + 0] = wp.float32(1.0)
    row_offset[base + 0] = gap_threshold
    row_particle_a[base + 1] = a
    row_particle_b[base + 1] = b
    row_dir[base + 1] = t1
    row_alpha[base + 1] = wp.float32(1.0)
    row_offset[base + 1] = wp.float64(0.0)
    row_particle_a[base + 2] = a
    row_particle_b[base + 2] = b
    row_dir[base + 2] = t2
    row_alpha[base + 2] = wp.float32(1.0)
    row_offset[base + 2] = wp.float64(0.0)


# ===========================================================================
# 4-slot (v-t/e-e) particle-contact kernels — Phase 1 + 2 of the v-t/e-e plan.
#
# LOCKED-B (uniform 4-slot row layout): all particle-contact rows carry
# ``(pa, pb, pc, pd)`` int32 + ``(wa, wb, wc, wd)`` float32.  Sentinel
# ``p* = -1`` (with matching ``w* = 0``) skips that slot in every consumer.
#
# Residual / Jacobian semantics:
#   r = offset - sum_k wa..wd * dot(row_dir, x[p_k])    (for k where p_k >= 0)
# Scatter:
#   for each row r and each slot k with p_k >= 0:
#     out[p_k] += w_k * alpha * dir * lambda
# This generalises the existing 2-particle v-v formulation
# (``pa=v1, pb=v2, wa=+1, wb=-1, pc=pd=-1, wc=wd=0``) directly.
# ===========================================================================


@wp.kernel
def emit_pp_rows_uniform_kernel(
    # 2-particle v-v broadphase pairs (existing ParticleContactBroadphase path).
    n_pairs: wp.int32,
    pair_a: wp.array[wp.int32],
    pair_b: wp.array[wp.int32],
    pair_normal: wp.array[wp.vec3],
    gap_threshold: wp.float64,
    row_offset_base: wp.int32,
    # 4-slot output arrays.
    row_pa: wp.array[wp.int32],
    row_pb: wp.array[wp.int32],
    row_pc: wp.array[wp.int32],
    row_pd: wp.array[wp.int32],
    row_wa: wp.array[wp.float32],
    row_wb: wp.array[wp.float32],
    row_wc: wp.array[wp.float32],
    row_wd: wp.array[wp.float32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    row_offset: wp.array[wp.float64],
    contact_mu: wp.array[wp.float64],
):
    """Emit a 2-particle v-v contact (3 rows) in the uniform 4-slot layout.

    For each pair (a, b) with unit normal ``n`` (b → a), writes:
      slot a:  pa = a, wa = +1
      slot b:  pb = b, wb = -1
      slots c/d: -1, 0 (unused)
    """
    p = wp.tid()
    if p >= n_pairs:
        return
    a = pair_a[p]
    b = pair_b[p]
    n = pair_normal[p]

    ax = wp.float32(wp.abs(n[0]))
    ay = wp.float32(wp.abs(n[1]))
    az = wp.float32(wp.abs(n[2]))
    if ax <= ay and ax <= az:
        ref = wp.vec3(1.0, 0.0, 0.0)
    elif ay <= az:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(0.0, 0.0, 1.0)
    t1 = wp.cross(n, ref)
    t1 = t1 / wp.max(wp.length(t1), wp.float32(1.0e-9))
    t2 = wp.cross(n, t1)

    base = row_offset_base + 3 * p

    for k in range(3):
        idx = base + k
        row_pa[idx] = a
        row_pb[idx] = b
        row_pc[idx] = wp.int32(-1)
        row_pd[idx] = wp.int32(-1)
        row_wa[idx] = wp.float32(1.0)
        row_wb[idx] = wp.float32(-1.0)
        row_wc[idx] = wp.float32(0.0)
        row_wd[idx] = wp.float32(0.0)
        row_alpha[idx] = wp.float32(1.0)

    row_dir[base + 0] = n
    row_offset[base + 0] = gap_threshold
    row_dir[base + 1] = t1
    row_offset[base + 1] = wp.float64(0.0)
    row_dir[base + 2] = t2
    row_offset[base + 2] = wp.float64(0.0)


@wp.kernel
def vt_count_hits_kernel(
    # Vertex-of-cloth count (= particle count for the cloth meshes the detector covers).
    n_query_particles: wp.int32,
    vertex_colliding_triangles_count: wp.array[wp.int32],
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32],
    # Output (length n_query_particles): clamped count per vertex.
    clamped_count: wp.array[wp.int32],
):
    v = wp.tid()
    if v >= n_query_particles:
        return
    raw = vertex_colliding_triangles_count[v]
    cap = vertex_colliding_triangles_buffer_sizes[v]
    clamped_count[v] = wp.min(raw, cap)


@wp.kernel
def emit_vt_rows_kernel(
    # Flat job array: one thread per (vertex, hit) pair.
    n_hits: wp.int32,
    # CSR over per-vertex hit job indices: for thread tid, find vertex v such
    # that vt_hit_offsets[v] <= tid < vt_hit_offsets[v+1], then hit i = tid - offsets[v].
    vt_hit_offsets: wp.array[wp.int32],     # (n_query_particles + 1,)
    n_query_particles: wp.int32,
    # Detector output (CSR per vertex; 2 ints per hit: (query_vertex, triangle_id)).
    vertex_colliding_triangles: wp.array[wp.int32],
    vertex_colliding_triangles_offsets: wp.array[wp.int32],
    # Triangle topology + positions.
    tri_indices: wp.array2d[wp.int32],      # (T, 3)
    particle_q: wp.array[wp.vec3],          # (N,)
    # Geometry.
    threshold: wp.float32,
    gap_threshold: wp.float64,
    # Output emission counter (atomic).
    contact_count: wp.array[wp.int32],      # (1,) — number of emitted contacts
    max_contacts: wp.int32,                 # capacity check (max_contacts = cap // 3)
    row_offset_base: wp.int32,
    # 4-slot output arrays.
    row_pa: wp.array[wp.int32],
    row_pb: wp.array[wp.int32],
    row_pc: wp.array[wp.int32],
    row_pd: wp.array[wp.int32],
    row_wa: wp.array[wp.float32],
    row_wb: wp.array[wp.float32],
    row_wc: wp.array[wp.float32],
    row_wd: wp.array[wp.float32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    row_offset: wp.array[wp.float64],
    contact_mu: wp.array[wp.float64],
    friction_mu: wp.float64,
):
    """One thread per (vertex, hit). Emits 3 rows in the 4-slot layout.

    Outputs a v-v special-case row when ``max(b1,b2,b3) > 1 - 1e-6`` (closest
    point at a triangle vertex), else a 4-particle v-t row.  Skips when the
    detector returned a sentinel (``triangle_id == -1``) or when the post-
    refit distance exceeds ``threshold``.
    """
    tid = wp.tid()
    if tid >= n_hits:
        return

    # ---- Resolve (vertex, local_hit_index) via binary search over CSR offsets. ----
    lo = wp.int32(0)
    hi = n_query_particles
    while lo < hi:
        mid = (lo + hi) // 2
        if vt_hit_offsets[mid + 1] <= tid:
            lo = mid + 1
        else:
            hi = mid
    v = lo
    local = tid - vt_hit_offsets[v]

    csr_off = vertex_colliding_triangles_offsets[v]
    raw_v = vertex_colliding_triangles[2 * (csr_off + local) + 0]
    tri_id = vertex_colliding_triangles[2 * (csr_off + local) + 1]
    if tri_id < wp.int32(0):
        return
    # The detector stores (query_vertex, triangle_id); query_vertex == v.
    _ = raw_v  # unused

    # ---- Closest point on triangle. ----
    t1i = tri_indices[tri_id, 0]
    t2i = tri_indices[tri_id, 1]
    t3i = tri_indices[tri_id, 2]
    a = particle_q[t1i]
    b = particle_q[t2i]
    c = particle_q[t3i]
    p = particle_q[v]
    bary = triangle_closest_point_barycentric(a, b, c, p)
    b1 = bary[0]
    b2 = bary[1]
    b3 = bary[2]
    closest = b1 * a + b2 * b + b3 * c
    diff = p - closest
    dist = wp.length(diff)
    if dist < wp.float32(1.0e-9):
        return
    if dist > threshold:
        return
    n = diff / dist

    # ---- Allocate a contact slot. ----
    cid = wp.atomic_add(contact_count, 0, 1)
    if cid >= max_contacts:
        return  # over capacity — silently drop (caller checks contact_count)

    base = row_offset_base + 3 * cid

    # ---- V-v fast path: closest at one triangle vertex. ----
    max_bary = wp.max(wp.max(b1, b2), b3)
    if max_bary > wp.float32(1.0) - wp.float32(1.0e-6):
        # Pick which triangle vertex.
        if b1 >= b2 and b1 >= b3:
            pb_id = t1i
        elif b2 >= b3:
            pb_id = t2i
        else:
            pb_id = t3i
        # Build tangent basis from n.
        ax_ = wp.float32(wp.abs(n[0]))
        ay_ = wp.float32(wp.abs(n[1]))
        az_ = wp.float32(wp.abs(n[2]))
        if ax_ <= ay_ and ax_ <= az_:
            ref = wp.vec3(1.0, 0.0, 0.0)
        elif ay_ <= az_:
            ref = wp.vec3(0.0, 1.0, 0.0)
        else:
            ref = wp.vec3(0.0, 0.0, 1.0)
        tan1 = wp.cross(n, ref)
        tan1 = tan1 / wp.max(wp.length(tan1), wp.float32(1.0e-9))
        tan2 = wp.cross(n, tan1)
        for k in range(3):
            idx = base + k
            row_pa[idx] = v
            row_pb[idx] = pb_id
            row_pc[idx] = wp.int32(-1)
            row_pd[idx] = wp.int32(-1)
            row_wa[idx] = wp.float32(1.0)
            row_wb[idx] = wp.float32(-1.0)
            row_wc[idx] = wp.float32(0.0)
            row_wd[idx] = wp.float32(0.0)
            row_alpha[idx] = wp.float32(1.0)
        row_dir[base + 0] = n
        row_offset[base + 0] = gap_threshold
        row_dir[base + 1] = tan1
        row_offset[base + 1] = wp.float64(0.0)
        row_dir[base + 2] = tan2
        row_offset[base + 2] = wp.float64(0.0)
        contact_mu[cid] = friction_mu
        return

    # ---- V-t row (4 slots). ----
    # Build tangent basis from n.
    ax_ = wp.float32(wp.abs(n[0]))
    ay_ = wp.float32(wp.abs(n[1]))
    az_ = wp.float32(wp.abs(n[2]))
    if ax_ <= ay_ and ax_ <= az_:
        ref = wp.vec3(1.0, 0.0, 0.0)
    elif ay_ <= az_:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(0.0, 0.0, 1.0)
    tan1 = wp.cross(n, ref)
    tan1 = tan1 / wp.max(wp.length(tan1), wp.float32(1.0e-9))
    tan2 = wp.cross(n, tan1)

    wa = wp.float32(1.0)
    wb = -b1
    wc = -b2
    wd = -b3
    # Slot whose bary == 0 → drop with -1 sentinel (matches "3 valid slots" case).
    pb_id = t1i
    pc_id = t2i
    pd_id = t3i
    if b1 < wp.float32(1.0e-7):
        pb_id = wp.int32(-1)
        wb = wp.float32(0.0)
    if b2 < wp.float32(1.0e-7):
        pc_id = wp.int32(-1)
        wc = wp.float32(0.0)
    if b3 < wp.float32(1.0e-7):
        pd_id = wp.int32(-1)
        wd = wp.float32(0.0)

    for k in range(3):
        idx = base + k
        row_pa[idx] = v
        row_pb[idx] = pb_id
        row_pc[idx] = pc_id
        row_pd[idx] = pd_id
        row_wa[idx] = wa
        row_wb[idx] = wb
        row_wc[idx] = wc
        row_wd[idx] = wd
        row_alpha[idx] = wp.float32(1.0)
    row_dir[base + 0] = n
    row_offset[base + 0] = gap_threshold
    row_dir[base + 1] = tan1
    row_offset[base + 1] = wp.float64(0.0)
    row_dir[base + 2] = tan2
    row_offset[base + 2] = wp.float64(0.0)
    contact_mu[cid] = friction_mu


@wp.kernel
def ee_count_hits_kernel(
    # Per-edge clamped collision count = min(raw, buffer_sizes).
    n_edges: wp.int32,
    edge_colliding_edges_count: wp.array[wp.int32],
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32],
    # Output (length n_edges): clamped count per edge.
    clamped_count: wp.array[wp.int32],
):
    e = wp.tid()
    if e >= n_edges:
        return
    raw = edge_colliding_edges_count[e]
    cap = edge_colliding_edges_buffer_sizes[e]
    clamped_count[e] = wp.min(raw, cap)


@wp.kernel
def emit_ee_rows_kernel(
    # Flat job array: one thread per (edge, hit) pair.
    n_hits: wp.int32,
    # CSR over per-edge hit-job indices: thread tid -> edge e such that
    # ee_hit_offsets[e] <= tid < ee_hit_offsets[e+1], local k = tid - offsets[e].
    ee_hit_offsets: wp.array[wp.int32],     # (n_edges + 1,)
    n_edges: wp.int32,
    # Detector output (CSR per edge; 2 ints per hit: (query_edge_id, target_edge_id)).
    edge_colliding_edges: wp.array[wp.int32],
    edge_colliding_edges_offsets: wp.array[wp.int32],
    # Edge topology + positions.
    # Newton's bending-stencil edge_indices layout: columns (2, 3) are the
    # two segment endpoints.
    edge_indices: wp.array2d[wp.int32],     # (E, 4)
    particle_q: wp.array[wp.vec3],          # (N,)
    # Geometry.
    threshold: wp.float32,
    gap_threshold: wp.float64,
    parallel_eps: wp.float32,
    # Output emission counter (atomic).
    contact_count: wp.array[wp.int32],      # (1,) — number of emitted contacts (vt+ee combined)
    max_contacts: wp.int32,                 # capacity check (max_contacts = cap // 3)
    row_offset_base: wp.int32,
    # 4-slot output arrays.
    row_pa: wp.array[wp.int32],
    row_pb: wp.array[wp.int32],
    row_pc: wp.array[wp.int32],
    row_pd: wp.array[wp.int32],
    row_wa: wp.array[wp.float32],
    row_wb: wp.array[wp.float32],
    row_wc: wp.array[wp.float32],
    row_wd: wp.array[wp.float32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    row_offset: wp.array[wp.float64],
    contact_mu: wp.array[wp.float64],
    friction_mu: wp.float64,
):
    """One thread per (edge, hit). Emits 3 rows (n + t1 + t2) per e-e pair.

    Per LOCKED-B 4-slot layout:
        pa = a1, pb = a2, pc = b1, pd = b2
        wa = +(1 - alpha),  wb = +alpha
        wc = -(1 - beta),   wd = -beta
    where (alpha, beta) are parametric positions on edges A and B such that
    closest point on A is ``pa1 + alpha · (pa2 - pa1)`` and on B is
    ``pb1 + beta · (pb2 - pb1)``.

    Degenerate-parallel guard: if |cross(da, db)|² < parallel_eps² the pair
    is skipped (the v-t path catches the contact via a vertex). Matches the
    detector's own `edge_edge_parallel_epsilon` semantics — we re-test here
    because the detector still records parallel pairs.
    """
    tid = wp.tid()
    if tid >= n_hits:
        return

    # ---- Resolve (edge, local_hit_index) via binary search over CSR offsets. ----
    lo = wp.int32(0)
    hi = n_edges
    while lo < hi:
        mid = (lo + hi) // 2
        if ee_hit_offsets[mid + 1] <= tid:
            lo = mid + 1
        else:
            hi = mid
    e0 = lo
    local = tid - ee_hit_offsets[e0]

    csr_off = edge_colliding_edges_offsets[e0]
    raw_e0 = edge_colliding_edges[2 * (csr_off + local) + 0]
    e1 = edge_colliding_edges[2 * (csr_off + local) + 1]
    if e1 < wp.int32(0):
        return
    _ = raw_e0  # unused (== e0 by construction)

    # ---- Endpoints. ----
    a1 = edge_indices[e0, 2]
    a2 = edge_indices[e0, 3]
    b1 = edge_indices[e1, 2]
    b2 = edge_indices[e1, 3]
    pa1 = particle_q[a1]
    pa2 = particle_q[a2]
    pb1 = particle_q[b1]
    pb2 = particle_q[b2]
    da = pa2 - pa1
    db = pb2 - pb1

    # ---- Degenerate parallel guard. ----
    cross_v = wp.cross(da, db)
    cross_sq = wp.dot(cross_v, cross_v)
    pe = wp.float32(parallel_eps)
    if cross_sq < pe * pe:
        return

    # ---- Closest points on the two segments. ----
    st = wp.closest_point_edge_edge(pa1, pa2, pb1, pb2, parallel_eps)
    alpha = st[0]
    beta = st[1]
    dist = st[2]
    if dist < wp.float32(1.0e-9):
        return
    if dist > threshold:
        return

    cA = pa1 + alpha * da
    cB = pb1 + beta * db
    dvec = cA - cB
    n = dvec / dist

    # ---- Allocate a contact slot (shared atomic w/ v-t). ----
    cid = wp.atomic_add(contact_count, 0, 1)
    if cid >= max_contacts:
        return
    base = row_offset_base + 3 * cid

    # ---- Tangent basis (stable in-plane, matches v-t emit). ----
    ax_ = wp.float32(wp.abs(n[0]))
    ay_ = wp.float32(wp.abs(n[1]))
    az_ = wp.float32(wp.abs(n[2]))
    if ax_ <= ay_ and ax_ <= az_:
        ref = wp.vec3(1.0, 0.0, 0.0)
    elif ay_ <= az_:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(0.0, 0.0, 1.0)
    tan1 = wp.cross(n, ref)
    tan1 = tan1 / wp.max(wp.length(tan1), wp.float32(1.0e-9))
    tan2 = wp.cross(n, tan1)

    wa = wp.float32(1.0) - alpha
    wb = alpha
    wc = -(wp.float32(1.0) - beta)
    wd = -beta

    for k in range(3):
        idx = base + k
        row_pa[idx] = a1
        row_pb[idx] = a2
        row_pc[idx] = b1
        row_pd[idx] = b2
        row_wa[idx] = wa
        row_wb[idx] = wb
        row_wc[idx] = wc
        row_wd[idx] = wd
        row_alpha[idx] = wp.float32(1.0)
    row_dir[base + 0] = n
    row_offset[base + 0] = gap_threshold
    row_dir[base + 1] = tan1
    row_offset[base + 1] = wp.float64(0.0)
    row_dir[base + 2] = tan2
    row_offset[base + 2] = wp.float64(0.0)
    contact_mu[cid] = friction_mu


@wp.kernel
def compute_particle_contact_residual_4slot_kernel(
    row_pa: wp.array[wp.int32],
    row_pb: wp.array[wp.int32],
    row_pc: wp.array[wp.int32],
    row_pd: wp.array[wp.int32],
    row_wa: wp.array[wp.float32],
    row_wb: wp.array[wp.float32],
    row_wc: wp.array[wp.float32],
    row_wd: wp.array[wp.float32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    row_offset: wp.array[wp.float64],
    x: wp.array[wp.vec3],
    r: wp.array[wp.float64],
):
    """4-slot residual: ``r[i] = offset[i] - α[i] · dot(dir[i], Σ_k w_k · x[p_k])``.

    Sentinel ``p_k = -1`` (with matching ``w_k = 0``) skips that slot.  For
    v-v rows (pa, pb valid, wa=+1, wb=-1, pc=pd=-1, wc=wd=0) this reduces
    exactly to the original ``offset - α · dir · (x[a] - x[b])`` formula.
    """
    i = wp.tid()
    alpha = wp.float64(row_alpha[i])
    d = row_dir[i]

    acc = wp.vec3(0.0, 0.0, 0.0)
    pa = row_pa[i]
    if pa >= wp.int32(0):
        acc = acc + row_wa[i] * x[pa]
    pb = row_pb[i]
    if pb >= wp.int32(0):
        acc = acc + row_wb[i] * x[pb]
    pc = row_pc[i]
    if pc >= wp.int32(0):
        acc = acc + row_wc[i] * x[pc]
    pd = row_pd[i]
    if pd >= wp.int32(0):
        acc = acc + row_wd[i] * x[pd]

    r[i] = row_offset[i] - alpha * wp.float64(wp.dot(d, acc))


@wp.kernel
def scatter_jt_lambda_n_particle_kernel(
    n_rows: wp.int32,
    row_pa: wp.array[wp.int32],
    row_pb: wp.array[wp.int32],
    row_pc: wp.array[wp.int32],
    row_pd: wp.array[wp.int32],
    row_wa: wp.array[wp.float32],
    row_wb: wp.array[wp.float32],
    row_wc: wp.array[wp.float32],
    row_wd: wp.array[wp.float32],
    row_dir: wp.array[wp.vec3],
    row_alpha: wp.array[wp.float32],
    lam: wp.array[wp.float32],
    out: wp.array[wp.vec3],
):
    """Per-row atomic scatter of ``Jᵀ·λ`` for 4-particle rows.

    Per row r and each slot k with ``p_k >= 0 and w_k != 0``:
        out[p_k] += w_k · α[r] · λ[r] · dir[r]

    Caller is responsible for zeroing ``out`` before launch.
    """
    i = wp.tid()
    if i >= n_rows:
        return
    contrib = row_alpha[i] * lam[i] * row_dir[i]

    pa = row_pa[i]
    if pa >= wp.int32(0):
        wp.atomic_add(out, pa, row_wa[i] * contrib)
    pb = row_pb[i]
    if pb >= wp.int32(0):
        wp.atomic_add(out, pb, row_wb[i] * contrib)
    pc = row_pc[i]
    if pc >= wp.int32(0):
        wp.atomic_add(out, pc, row_wc[i] * contrib)
    pd = row_pd[i]
    if pd >= wp.int32(0):
        wp.atomic_add(out, pd, row_wd[i] * contrib)


# ----- Per-particle fused FB-Newton kernel for LiteNSN (Phase 1 of Franka plan) -----
#
# Generalises ``fb_newton_lite_coulomb_kernel`` to handle ``k_p ∈ {1..4}``
# contacts per particle.  Each thread processes one *unique-contacted* particle
# end-to-end; the local 3·k_p × 3·k_p Schur block is constructed in registers
# (fp64 ``mat12``) and solved via in-place Cholesky.  Sparsity stays
# block-diagonal-by-particle — no BSR matrix, no SpGEMM.
#
# Particles with ``k_p > _MAX_KP_PER_PARTICLE`` are detected on the host side
# and routed to the BSR + sparse-PCR path (the same one Phase 3 will use for
# self-contact).  For the Franka cloth scene, the worst case is one cloth
# vertex touching both finger pads + the table = 3, so k_max=4 leaves
# headroom.  ``mat12`` is the largest type we materialise per thread.
_MAX_KP_PER_PARTICLE = wp.constant(4)
_MAX_ROWS_PER_PARTICLE = wp.constant(12)  # 3 · _MAX_KP_PER_PARTICLE
_MAT12_F64 = wp.types.matrix((12, 12), wp.float64)
_VEC12_F64 = wp.types.vector(12, wp.float64)


@wp.kernel
def fb_newton_lite_coulomb_per_particle_kernel(
    # ---- per-isodof iteration metadata ----
    isodof_unique: wp.array[wp.int32],     # (K,) unique contacted particle indices
    isodof_row_offsets: wp.array[wp.int32], # (N+1,) CSR over ALL particles
    isodof_row_indices: wp.array[wp.int32], # (total_rows,) row idx sorted by particle
    # ---- per-row data (3M entries, interleaved [n, t1, t2] per contact) ----
    row_particle: wp.array[wp.int32],       # (3M,) — particle index per row
    row_dir: wp.array[wp.vec3],             # (3M,) — direction per row (fp32)
    row_alpha: wp.array[wp.float32],        # (3M,) — Jacobian coef per row
    # ---- per-contact data (M entries) ----
    contact_mu: wp.array[wp.float64],       # (M,) — friction coef per contact
    # ---- per-row force inputs (3M entries) ----
    r: wp.array[wp.float64],                # (3M,)
    pene0: wp.array[wp.float64],            # (3M,)
    # ---- global state ----
    inv_mass: wp.array[wp.float64],         # (N,)
    dt: wp.float64,
    cap_internal: wp.float64,               # lambda_cap / dt²
    use_cap: wp.int32,
    n_fb_iters: wp.int32,
    # ---- inout / outputs ----
    lam: wp.array[wp.float64],              # (3M,) inout
    omega: wp.array[wp.float64],            # (3M,) inout
    lam_apply: wp.array[wp.float64],        # (3M,) output
):
    """Per-particle fused FB-Newton for LiteNSN, Stage B (Coulomb), k_p ∈ {1..4}.

    Each thread handles one unique-contacted particle ``p``.  Its ``k_p`` contact
    rows live consecutively in ``isodof_row_indices[start:end]``; each row carries
    its own ``(particle, dir, alpha)`` triple and its own (n, t1, t2) sub-triplet
    in the global row arrays.  Under the lite approximation, ``W_lite`` is
    block-diagonal-by-particle, so this thread is fully independent.

    Local state lives in registers / L1 (warp ``mat12``).  Cholesky on a
    ``3·k_p × 3·k_p`` PSD matrix runs in-place over the same buffer.

    For ``k_p > _MAX_KP_PER_PARTICLE`` the kernel emits a sentinel (skips the
    update; warm-start λ unchanged); host code re-runs that step through the BSR
    path.  This branch is never taken in the cloth-on-rigid scenes covered by
    the Franka plan.
    """
    iso = wp.tid()
    p = isodof_unique[iso]
    row_start = isodof_row_offsets[p]
    row_end = isodof_row_offsets[p + 1]
    k_total = row_end - row_start  # = 3 · k_p
    if k_total <= 0:
        return
    if k_total > _MAX_ROWS_PER_PARTICLE:
        # Defer to BSR path (host re-dispatches).  Don't touch λ / ω here.
        return

    inv_m = inv_mass[p]
    dt2 = dt * dt

    # Load up to 12 row indices and their per-row data into local arrays.
    row_idx = _VEC12_F64(wp.float64(0.0))  # store as float for warp matrix-compat indexing
    # Actually we want int row indices — use 12 separate locals (warp doesn't
    # have small int vectors).  Materialise as a stack of fixed-size scalars.
    # We can fall back to loading from the CSR each time we need the row index.

    # Build local W (3k×3k) directly from (alpha, dir) triples.
    # W[a, b] = α_a · α_b · dot(dir_a, dir_b) · dt² · inv_m
    W = _MAT12_F64(wp.float64(0.0))
    for ai in range(k_total):
        ra = isodof_row_indices[row_start + ai]
        alpha_a = wp.float64(row_alpha[ra])
        dir_a = row_dir[ra]
        for bi in range(k_total):
            rb = isodof_row_indices[row_start + bi]
            alpha_b = wp.float64(row_alpha[rb])
            dir_b = row_dir[rb]
            W[ai, bi] = alpha_a * alpha_b * wp.float64(wp.dot(dir_a, dir_b)) * dt2 * inv_m

    # Load per-row state into vec12 locals.
    lam_l = _VEC12_F64(wp.float64(0.0))
    om_l = _VEC12_F64(wp.float64(0.0))
    r_l = _VEC12_F64(wp.float64(0.0))
    pene0_l = _VEC12_F64(wp.float64(0.0))
    for ai in range(k_total):
        ra = isodof_row_indices[row_start + ai]
        lam_l[ai] = lam[ra]
        om_l[ai] = omega[ra]
        r_l[ai] = r[ra]
        pene0_l[ai] = pene0[ra]

    # Precond per row: dt²·|W_ii| for normals, dt·|W_ii| for tangents.
    # Rows interleave (n, t1, t2) per contact => row_mod3 == 0 is normal.
    eps = wp.float64(1.0e-12)
    precond_l = _VEC12_F64(wp.float64(0.0))
    for ai in range(k_total):
        ra = isodof_row_indices[row_start + ai]
        w_abs = wp.abs(W[ai, ai])
        if w_abs < eps:
            w_abs = eps
        if (ra % 3) == 0:
            precond_l[ai] = dt2 * w_abs
        else:
            precond_l[ai] = dt * w_abs

    for _it in range(n_fb_iters):
        # penetration = -r + dt²·W·(ω·λ).  3k_p dot products.
        pen_l = _VEC12_F64(wp.float64(0.0))
        for ai in range(k_total):
            acc = wp.float64(0.0)
            for bi in range(k_total):
                acc = acc + W[ai, bi] * om_l[bi] * lam_l[bi]
            pen_l[ai] = -r_l[ai] + dt2 * acc

        # FB evaluation (rows interleave [n, t1, t2] within each contact; the
        # tangent rows read lam_n from the same contact's normal).
        c_l = _VEC12_F64(wp.float64(0.0))
        h_l = _VEC12_F64(wp.float64(0.0))
        for ai in range(k_total):
            ra = isodof_row_indices[row_start + ai]
            rmod = ra % 3
            if rmod == 0:
                # Normal row.
                out_n = fb_unilateral_row_wp(pen_l[ai], lam_l[ai], precond_l[ai], dt, pene0_l[ai])
                om_l[ai] = out_n[0]
                c_l[ai] = out_n[1]
                h_l[ai] = out_n[2]
            else:
                # Tangent row.  Find the companion normal row in this particle's
                # local block — same contact c = ra // 3, normal row = 3·c.
                # The normal might not be at local index ai-rmod because the
                # local order = sorted-by-row order which preserves
                # n,t1,t2 grouping (rows from one contact are consecutive,
                # since they share a particle and are inserted contiguously).
                # Find the local index of the companion normal row.
                companion_global = 3 * (ra // 3)
                # Linear scan over local rows (k_total ≤ 12) to find it.
                lam_n_local = wp.float64(0.0)
                for ci in range(k_total):
                    if isodof_row_indices[row_start + ci] == companion_global:
                        lam_n_local = lam_l[ci]
                contact_idx = ra // 3
                mu_c = contact_mu[contact_idx]
                out_t = fb_frictional_row_wp(pen_l[ai], lam_l[ai], lam_n_local, mu_c, precond_l[ai], dt, pene0_l[ai])
                om_l[ai] = out_t[0]
                c_l[ai] = out_t[1]
                h_l[ai] = out_t[2]

        # A_schur = ωωᵀ ⊙ W + diag(c) — block-PSD by construction.
        A = _MAT12_F64(wp.float64(0.0))
        for ai in range(k_total):
            for bi in range(k_total):
                a_ij = om_l[ai] * om_l[bi] * W[ai, bi]
                if ai == bi:
                    a_ij = a_ij + c_l[ai]
                A[ai, bi] = a_ij

        # rhs = (1/dt²) · (h - ω · J_x) where J_x = pene0 + pen.
        inv_dt2 = wp.float64(1.0) / dt2
        rhs_l = _VEC12_F64(wp.float64(0.0))
        for ai in range(k_total):
            Jx = pene0_l[ai] + pen_l[ai]
            rhs_l[ai] = inv_dt2 * (h_l[ai] - om_l[ai] * Jx)

        # In-place Cholesky: A = L·Lᵀ.  L overwrites the lower triangle of A.
        # PSD by construction; bail with sentinel diagonal (1) if a diagonal
        # underflows so the subsequent solve produces zero update.
        ok = wp.int32(1)
        for j in range(k_total):
            s = A[j, j]
            for kk in range(j):
                s = s - A[j, kk] * A[j, kk]
            if s < eps:
                ok = wp.int32(0)
                # break early by setting all remaining diags to 1; rhs solve
                # will then produce ~zero update.  Don't actually break — warp
                # doesn't always promote early-return cleanly in nested loops.
                A[j, j] = wp.float64(1.0)
                continue
            ljj = wp.sqrt(s)
            A[j, j] = ljj
            inv_ljj = wp.float64(1.0) / ljj
            for ii in range(j + 1, k_total):
                t = A[ii, j]
                for kk in range(j):
                    t = t - A[ii, kk] * A[j, kk]
                A[ii, j] = t * inv_ljj

        # Forward substitute L·y = rhs.
        y_l = _VEC12_F64(wp.float64(0.0))
        for ii in range(k_total):
            t = rhs_l[ii]
            for kk in range(ii):
                t = t - A[ii, kk] * y_l[kk]
            y_l[ii] = t / A[ii, ii]

        # Back substitute Lᵀ·dlam = y.
        dlam_l = _VEC12_F64(wp.float64(0.0))
        for ii_rev in range(k_total):
            ii = k_total - 1 - ii_rev
            t = y_l[ii]
            for kk in range(ii + 1, k_total):
                t = t - A[kk, ii] * dlam_l[kk]
            dlam_l[ii] = t / A[ii, ii]

        # λ += dlam.  Skip the update for singular A (ok==0).
        if ok == wp.int32(1):
            for ai in range(k_total):
                lam_l[ai] = lam_l[ai] + dlam_l[ai]

    # Coulomb cone clamp per contact (each contact = 3 consecutive rows).
    # Identify contact-c normal local index, clamp its t1 and t2.
    for ai in range(k_total):
        ra = isodof_row_indices[row_start + ai]
        if (ra % 3) != 0:
            continue
        # Find t1, t2 local indices (ra+1, ra+2 global).
        t1_g = ra + 1
        t2_g = ra + 2
        t1_local = wp.int32(-1)
        t2_local = wp.int32(-1)
        for ci in range(k_total):
            rc = isodof_row_indices[row_start + ci]
            if rc == t1_g:
                t1_local = ci
            elif rc == t2_g:
                t2_local = ci
        if t1_local >= wp.int32(0) and t2_local >= wp.int32(0):
            lam_n_val = lam_l[ai]
            contact_idx = ra // 3
            mu_c = contact_mu[contact_idx]
            upper = mu_c * lam_n_val
            lower = -mu_c * lam_n_val
            if lam_l[t1_local] > upper:
                lam_l[t1_local] = upper
            if lam_l[t1_local] < lower:
                lam_l[t1_local] = lower
            if lam_l[t2_local] > upper:
                lam_l[t2_local] = upper
            if lam_l[t2_local] < lower:
                lam_l[t2_local] = lower

    # Optional symmetric lambda cap.
    if use_cap == wp.int32(1):
        for ai in range(k_total):
            v = lam_l[ai]
            if v > cap_internal:
                v = cap_internal
            if v < -cap_internal:
                v = -cap_internal
            lam_l[ai] = v

    # Write back λ, ω, lam_apply = dt² · ω · λ.
    for ai in range(k_total):
        ra = isodof_row_indices[row_start + ai]
        lam[ra] = lam_l[ai]
        omega[ra] = om_l[ai]
        lam_apply[ra] = dt2 * om_l[ai] * lam_l[ai]


@wp.kernel
def axpy_lambda_kernel(
    dlam: wp.array[wp.float64],
    lam: wp.array[wp.float64],  # inout
):
    """``lam += dlam`` (NSN Newton update step)."""
    i = wp.tid()
    lam[i] = lam[i] + dlam[i]


@wp.kernel
def coulomb_box_clamp_kernel(
    mu: wp.array[wp.float64],  # (M,)
    lam: wp.array[wp.float64],  # (3M,) inout
):
    """Per-contact Coulomb signed-cone clamp on tangent lambdas.

    Mirrors RealSim ``NonSmoothNewton.cpp:395-403`` (friction branch) and
    the CPU reference loop in :meth:`SolverFBA._solve_nsn_coulomb`. ``lam_n``
    is read **signed** (per RealSim — the ``if(_lambda[cid] < 0.0)`` line
    is explicitly commented out) so ``upper = mu·lam_n`` and
    ``lower = -mu·lam_n`` can swap when ``lam_n < 0``. The two ``if``
    statements per tangent must run in the exact CPU order (``> upper``
    first, then ``< lower``) — when ``lam_n < 0`` the first clamp pulls
    ``lam_t`` down to a negative ``upper``; the second then clamps it back
    up to ``lower`` (positive), yielding RealSim's saturated behavior
    rather than the degenerate ``np.clip(low > high)`` artifact.

    Args:
        mu: Per-contact friction coefficient (fp64), shape ``[M]``.
        lam: Per-row lambda (fp64), shape ``[3M]``. Modified in place.
    """
    c = wp.tid()
    lam_n = lam[3 * c]
    upper = mu[c] * lam_n
    lower = -mu[c] * lam_n
    if lam[3 * c + 1] > upper:
        lam[3 * c + 1] = upper
    if lam[3 * c + 1] < lower:
        lam[3 * c + 1] = lower
    if lam[3 * c + 2] > upper:
        lam[3 * c + 2] = upper
    if lam[3 * c + 2] < lower:
        lam[3 * c + 2] = lower


@wp.kernel
def lambda_cap_clip_kernel(
    cap_internal: wp.float64,
    lam: wp.array[wp.float64],  # inout
):
    """Symmetric clip: ``lam[i] = clip(lam[i], -cap, +cap)``.

    Implements the CPU post-loop ``np.clip(lam, -cap_internal,
    cap_internal, out=lam)`` where ``cap_internal = lambda_cap / dt²``.
    The conversion factor is applied at the caller; this kernel only
    knows the internal-units cap.

    Args:
        cap_internal: Symmetric cap in internal lambda units (fp64).
        lam: Lambda buffer (fp64), shape ``[n_rows]``. Modified in place.
    """
    i = wp.tid()
    if lam[i] > cap_internal:
        lam[i] = cap_internal
    if lam[i] < -cap_internal:
        lam[i] = -cap_internal


@wp.kernel
def compute_lam_apply_kernel(
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    dt: wp.float64,
    lam_apply: wp.array[wp.float64],
):
    """Compute ``lam_apply = dt² · ω · lam`` (position-LCP correction units).

    See :meth:`SolverFBA._solve_nsn_unilateral` docstring for the
    derivation: RealSim's correction is ``Δq = dt² · A⁻¹ · Jᵀ · (ω · λ)``
    whereas FBA's ``apply_lambda_correction_combined`` adds
    ``A⁻¹·Jᵀ·λ_apply`` directly, so we pre-scale by ``dt²·ω`` here.

    Args:
        lam: Force-units lambda (fp64), shape ``[n_rows]``.
        omega: Per-row FB weighting (fp64), shape ``[n_rows]``.
        dt: Timestep (fp64).
        lam_apply: Output correction-units lambda, shape ``[n_rows]``.
    """
    i = wp.tid()
    lam_apply[i] = dt * dt * omega[i] * lam[i]


@wp.kernel
def build_coulomb_pene0_kernel(
    offset_n: wp.array[wp.float64],
    offset_t1: wp.array[wp.float64],
    offset_t2: wp.array[wp.float64],
    pene0: wp.array[wp.float64],
):
    """Interleave per-contact normal/tangent offsets into the ``(3M,)`` pene0.

    One thread per contact writes ``pene0[3c:3c+3] = [offset_n, offset_t1,
    offset_t2]`` matching the row layout consumed by
    :func:`compute_frictional_fb_kernel` and :func:`compute_nsn_rhs_kernel`.
    Replaces the host-side ``pene0_b[0::3] = ...`` slicing previously run
    each PD outer iter.

    Args:
        offset_n: ``n·anchor`` (fp64), shape ``[M]``.
        offset_t1: Kinematic-shifted ``t1·anchor`` (fp64), shape ``[M]``.
        offset_t2: Kinematic-shifted ``t2·anchor`` (fp64), shape ``[M]``.
        pene0: Output interleaved pene0 (fp64), shape ``[>=3M]``.
    """
    c = wp.tid()
    pene0[3 * c + 0] = offset_n[c]
    pene0[3 * c + 1] = offset_t1[c]
    pene0[3 * c + 2] = offset_t2[c]


@wp.kernel
def apply_tangent_kinematic_shift_kernel(
    t1: wp.array[wp.vec3],
    t2: wp.array[wp.vec3],
    v_anchor: wp.array[wp.vec3d],
    t1_offset_base: wp.array[wp.float64],
    t2_offset_base: wp.array[wp.float64],
    dt: wp.float64,
    t1_offset_out: wp.array[wp.float64],
    t2_offset_out: wp.array[wp.float64],
):
    """Apply per-step kinematic anchor shift to tangent offsets.

    Mirrors the host expression
    ``offset_t = offset_t_base + dt · dot(t, v_anchor)`` previously
    computed in :meth:`SolverFBA.step` from host-side mirrors. Running on
    device eliminates the per-step ``tangent1_d.numpy() / tangent2_d.numpy()
    / v_anchor_d.numpy()`` download chain.

    Args:
        t1: Per-contact first tangent (fp32 vec3), shape ``[M]``.
        t2: Per-contact second tangent (fp32 vec3), shape ``[M]``.
        v_anchor: Per-contact kinematic anchor velocity (fp64 vec3),
            shape ``[M]``.
        t1_offset_base: Base ``t1·anchor`` (fp64), shape ``[M]``.
        t2_offset_base: Base ``t2·anchor`` (fp64), shape ``[M]``.
        dt: Timestep (fp64).
        t1_offset_out: Output shifted offset ``t1·anchor + dt·t1·v_anchor``
            (fp64), shape ``[M]``.
        t2_offset_out: Output shifted offset ``t2·anchor + dt·t2·v_anchor``
            (fp64), shape ``[M]``.
    """
    c = wp.tid()
    v = v_anchor[c]
    t1c = t1[c]
    t2c = t2[c]
    dot1 = wp.float64(t1c[0]) * v[0] + wp.float64(t1c[1]) * v[1] + wp.float64(t1c[2]) * v[2]
    dot2 = wp.float64(t2c[0]) * v[0] + wp.float64(t2c[1]) * v[1] + wp.float64(t2c[2]) * v[2]
    t1_offset_out[c] = t1_offset_base[c] + dt * dot1
    t2_offset_out[c] = t2_offset_base[c] + dt * dot2


@wp.kernel
def cast_fp64_to_fp32_kernel(
    src: wp.array[wp.float64],
    dst: wp.array[wp.float32],
):
    """Narrow ``fp64 -> fp32`` element-wise copy.

    Used to feed the device-resident fp64 ``lam_apply`` produced by the
    NSN inner driver into :func:`gather_jt_lambda_kernel`, which reads
    fp32 per-row impulses (matching the rest of the linear-solver
    arithmetic). Both arrays must have the same length; only the first
    ``dim`` entries (set via the kernel launch dim) are touched.

    Args:
        src: Source fp64 array, shape ``[n_rows]``.
        dst: Destination fp32 array, shape ``[n_rows]``.
    """
    i = wp.tid()
    dst[i] = wp.float32(src[i])


# ---------------------------------------------------------------------------
# Step 5.5 — update_contacts per-contact bookkeeping ported to GPU.
#
# CPU implementation pulled six per-contact buffers to host every step and
# ran a Python ``for`` loop to evaluate the body transform, sphere/cylinder
# cushion, and (Stage B) the tangent basis / kinematic anchor velocity.
# These kernels move that loop on-device; only the lexsort (intentional S.1
# for determinism) and the singleton ``soft_contact_count`` read still hit
# the host. See :meth:`newton._src.solvers.fba.solver_fba.SolverFBA.update_contacts`.
# ---------------------------------------------------------------------------


@wp.kernel
def compute_world_anchor_kernel(
    body_pos: wp.array[wp.vec3],  # (M,) contact anchor in body-local or world frame
    shape_idx: wp.array[wp.int32],  # (M,) shape index per contact
    shape_body: wp.array[wp.int32],  # (n_shapes,) body index per shape (-1 = static)
    body_q: wp.array[wp.transform],  # (n_bodies,) world-frame body transform
    has_body_q: wp.int32,  # 1 if body_q is valid (n_bodies > 0)
    has_shape_body: wp.int32,  # 1 if shape_body is valid
    world_anchor: wp.array[wp.vec3],  # (M,) OUTPUT contact anchor in world frame
):
    """Promote per-contact ``body_pos`` to world frame.

    Matches the CPU reference in
    :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA.update_contacts`:
    if the contact's shape is attached to a dynamic body, transform
    ``body_pos`` through the body quaternion; otherwise pass through
    (static shapes already store world-frame anchors per the collision
    pipeline convention).

    Args:
        body_pos: Per-contact anchor, shape ``[M]``.
        shape_idx: Per-contact shape index, shape ``[M]``; ``< 0`` skips body lookup.
        shape_body: Body index per shape (``-1`` for static), shape ``[n_shapes]``.
        body_q: World-frame body transforms, shape ``[n_bodies]``.
        has_body_q: ``1`` when ``body_q`` is populated; ``0`` triggers pass-through.
        has_shape_body: ``1`` when ``shape_body`` is populated; ``0`` triggers pass-through.
        world_anchor: Output world-frame anchors, shape ``[M]``.
    """
    c = wp.tid()
    bpos = body_pos[c]
    s = shape_idx[c]
    if s < 0 or has_shape_body == 0 or has_body_q == 0:
        world_anchor[c] = bpos
        return
    b_idx = shape_body[s]
    if b_idx < 0:
        world_anchor[c] = bpos
        return
    world_anchor[c] = wp.transform_point(body_q[b_idx], bpos)


@wp.kernel
def compute_normal_offset_kernel(
    normal: wp.array[wp.vec3],  # (M,) world-frame unit normal per contact
    world_anchor: wp.array[wp.vec3],  # (M,) world-frame contact anchor
    shape_idx: wp.array[wp.int32],  # (M,)
    shape_type: wp.array[wp.int32],  # (n_shapes,)
    has_shape_type: wp.int32,
    sphere_geo_type: wp.int32,  # GeoType.SPHERE value
    cylinder_geo_type: wp.int32,  # GeoType.CYLINDER value
    offset: wp.array[wp.float64],  # (M,) OUTPUT
):
    """Compute ``offset[c] = n·world_anchor`` plus sphere/cylinder cushion.

    Matches the CPU loop in
    :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA.update_contacts`:
    apply the 0.01 m RealSim interpenetration cushion to sphere and
    cylinder normal rows (``SphereCollision.cpp:121``,
    ``CylinderCollision.cpp:84``); plane and mesh contacts have no cushion.

    Args:
        normal: Per-contact unit normal, shape ``[M]``.
        world_anchor: Per-contact world-frame anchor, shape ``[M]``.
        shape_idx: Per-contact shape index, shape ``[M]``.
        shape_type: Geometry type per shape, shape ``[n_shapes]``.
        has_shape_type: ``1`` when ``shape_type`` is populated.
        sphere_geo_type: Integer value of :class:`~newton.geometry.GeoType.SPHERE`.
        cylinder_geo_type: Integer value of :class:`~newton.geometry.GeoType.CYLINDER`.
        offset: Output ``n·anchor`` (with cushion if applicable), shape ``[M]``.
    """
    c = wp.tid()
    val = wp.float64(wp.dot(normal[c], world_anchor[c]))
    s = shape_idx[c]
    if has_shape_type == 1 and s >= 0:
        st = shape_type[s]
        if st == sphere_geo_type or st == cylinder_geo_type:
            val = val - wp.float64(0.01)
    offset[c] = val


@wp.kernel
def compute_tangent_basis_kernel(
    normal: wp.array[wp.vec3],  # (M,) world-frame unit normal per contact
    shape_idx: wp.array[wp.int32],  # (M,)
    shape_omega: wp.array[wp.float64],  # (n_shapes,) angular vel about local +Z
    shape_transform: wp.array[wp.transform],  # (n_shapes,) shape→world transform
    has_shape_omega: wp.int32,
    has_shape_transform: wp.int32,
    t1_out: wp.array[wp.vec3],  # (M,) OUTPUT first tangent
    t2_out: wp.array[wp.vec3],  # (M,) OUTPUT second tangent
    is_spinning_out: wp.array[wp.int32],  # (M,) OUTPUT flag for v_anchor
):
    """Compute per-contact orthonormal tangent basis ``(t1, t2)``.

    Mirrors :func:`~newton._src.solvers.fba.solver_fba.compute_tangent_basis`
    for the default case (pick a reference direction, cross with ``n``).
    When the contact's shape has nonzero ``shape_omega`` and a valid
    transform, switch to the cylinder-rolling-aligned basis used in
    ``update_contacts``: ``t1 = normalize(cross(n, axis_world))``,
    ``t2 = normalize(cross(n, t1))``. ``is_spinning_out[c]`` is set to ``1``
    only when the override succeeded - the v_anchor kernel uses that flag
    to decide whether to emit a kinematic anchor velocity.

    Args:
        normal: Per-contact unit normal, shape ``[M]``.
        shape_idx: Per-contact shape index, shape ``[M]``.
        shape_omega: Per-shape angular velocity about local +Z, shape ``[n_shapes]``.
        shape_transform: Per-shape transform, shape ``[n_shapes]``.
        has_shape_omega: ``1`` when ``shape_omega`` is populated.
        has_shape_transform: ``1`` when ``shape_transform`` is populated.
        t1_out: Output first tangent vectors, shape ``[M]``.
        t2_out: Output second tangent vectors, shape ``[M]``.
        is_spinning_out: Output spinning-shape flags, shape ``[M]``.
    """
    c = wp.tid()
    n = normal[c]
    # Default arbitrary basis: pick world Y when n is nearly aligned to world X,
    # otherwise world X — matches ``compute_tangent_basis`` exactly.
    if wp.abs(n[0]) > 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(1.0, 0.0, 0.0)
    t1 = wp.cross(n, ref)
    t1_len = wp.length(t1)
    t1 = t1 / (t1_len + 1.0e-30)
    t2 = wp.cross(n, t1)
    t2_len = wp.length(t2)
    t2 = t2 / (t2_len + 1.0e-30)
    is_spin = wp.int32(0)

    s = shape_idx[c]
    if has_shape_omega == 1 and has_shape_transform == 1 and s >= 0 and s < shape_omega.shape[0]:
        if shape_omega[s] != wp.float64(0.0):
            xf = shape_transform[s]
            axis_world = wp.transform_vector(xf, wp.vec3(0.0, 0.0, 1.0))
            t1_vec = wp.cross(n, axis_world)
            n1 = wp.length(t1_vec)
            if n1 >= 1.0e-9:
                t1 = t1_vec / n1
                t2_raw = wp.cross(n, t1)
                n2 = wp.length(t2_raw)
                t2 = t2_raw / wp.max(n2, 1.0e-30)
                is_spin = wp.int32(1)
    t1_out[c] = t1
    t2_out[c] = t2
    is_spinning_out[c] = is_spin


@wp.kernel
def compute_tangent_offsets_kernel(
    t1: wp.array[wp.vec3],
    t2: wp.array[wp.vec3],
    world_anchor: wp.array[wp.vec3],
    tangent1_offset: wp.array[wp.float64],  # (M,) OUTPUT
    tangent2_offset: wp.array[wp.float64],  # (M,) OUTPUT
):
    """Compute per-contact tangent offsets ``t·world_anchor``.

    Provides the friction residual reference position so
    ``r_t = t·anchor - alpha·t·x_unc`` measures tangential displacement
    from the contact point rather than from the world origin.

    Args:
        t1: First tangent vectors, shape ``[M]``.
        t2: Second tangent vectors, shape ``[M]``.
        world_anchor: World-frame contact anchors, shape ``[M]``.
        tangent1_offset: Output ``t1·anchor`` (fp64), shape ``[M]``.
        tangent2_offset: Output ``t2·anchor`` (fp64), shape ``[M]``.
    """
    c = wp.tid()
    a = world_anchor[c]
    tangent1_offset[c] = wp.float64(wp.dot(t1[c], a))
    tangent2_offset[c] = wp.float64(wp.dot(t2[c], a))


@wp.kernel
def compute_v_anchor_kernel(
    is_spinning: wp.array[wp.int32],  # (M,) flag from compute_tangent_basis_kernel
    shape_idx: wp.array[wp.int32],  # (M,)
    shape_omega: wp.array[wp.float64],  # (n_shapes,)
    shape_transform: wp.array[wp.transform],  # (n_shapes,)
    world_anchor: wp.array[wp.vec3],  # (M,) world-frame anchor
    v_anchor: wp.array[wp.vec3d],  # (M,) OUTPUT fp64 kinematic velocity
):
    """Kinematic anchor velocity for rolling cylinder contacts.

    For non-spinning contacts ``v_anchor = 0``. For spinning cylinder
    shapes ``v_anchor = -omega · cross(axis_world, r_local)`` where
    ``r_local = world_anchor - shape_translation`` and ``axis_world`` is
    the shape's local +Z direction in world frame. The leading minus
    matches RealSim's sign convention; see comment in CPU reference
    :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA.update_contacts`.

    Args:
        is_spinning: Per-contact spinning-shape flag from
            :func:`compute_tangent_basis_kernel`, shape ``[M]``.
        shape_idx: Per-contact shape index, shape ``[M]``.
        shape_omega: Per-shape angular velocity, shape ``[n_shapes]``.
        shape_transform: Per-shape transform, shape ``[n_shapes]``.
        world_anchor: Per-contact world-frame anchor, shape ``[M]``.
        v_anchor: Output kinematic anchor velocity (fp64 vec3), shape ``[M]``.
    """
    c = wp.tid()
    if is_spinning[c] == 0:
        v_anchor[c] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        return
    s = shape_idx[c]
    xf = shape_transform[s]
    axis_world = wp.transform_vector(xf, wp.vec3(0.0, 0.0, 1.0))
    shape_p = wp.transform_get_translation(xf)
    a = world_anchor[c]
    r_local = wp.vec3(a[0] - shape_p[0], a[1] - shape_p[1], a[2] - shape_p[2])
    cross_ar = wp.cross(axis_world, r_local)
    omega = shape_omega[s]
    v_anchor[c] = wp.vec3d(
        -omega * wp.float64(cross_ar[0]),
        -omega * wp.float64(cross_ar[1]),
        -omega * wp.float64(cross_ar[2]),
    )


@wp.kernel
def compute_v_anchor_from_body_kernel(
    shape_idx: wp.array[wp.int32],          # (M,) shape index per contact
    shape_body: wp.array[wp.int32],         # (n_shapes,) body index per shape (-1 = static)
    body_q: wp.array[wp.transform],         # (n_bodies,) body world transforms
    body_qd: wp.array[wp.spatial_vector],   # (n_bodies,) body spatial velocities (ω_world, v_world)
    world_anchor: wp.array[wp.vec3],        # (M,) world-frame anchor
    v_anchor: wp.array[wp.vec3d],           # (M,) inout — accumulate body-driven motion
):
    """Add body-driven kinematic anchor velocity for contacts whose rigid side
    is parented to a dynamic body.

    For each contact ``c``:
      parent = shape_body[shape_idx[c]]
      if parent < 0: skip (static collider — leave v_anchor as-is so the
                              spinning-cylinder path from
                              :func:`compute_v_anchor_kernel` survives).
      else:
        ω_body = wp.spatial_top(body_qd[parent])
        v_body = wp.spatial_bottom(body_qd[parent])
        r      = world_anchor[c] - body_q[parent].translation
        v_anchor[c] = v_body + cross(ω_body, r)

    Newton's :class:`wp.spatial_vector` packs the body spatial velocity as
    ``(ω, v)`` per ``newton.utils.transform`` conventions:
    ``spatial_top`` → angular, ``spatial_bottom`` → linear.

    Designed to run **after** :func:`compute_v_anchor_kernel`: the spinning
    path writes a value, this path overwrites it for body-parented shapes
    (because per-shape ω no longer makes sense when the shape moves with a
    body whose own ω drives the surface velocity).
    """
    c = wp.tid()
    s = shape_idx[c]
    parent = shape_body[s]
    if parent < 0:
        return
    qd = body_qd[parent]
    omega_body = wp.spatial_top(qd)
    v_body = wp.spatial_bottom(qd)
    body_xf = body_q[parent]
    body_p = wp.transform_get_translation(body_xf)
    a = world_anchor[c]
    r = wp.vec3(a[0] - body_p[0], a[1] - body_p[1], a[2] - body_p[2])
    cross_or = wp.cross(omega_body, r)
    v_anchor[c] = wp.vec3d(
        wp.float64(v_body[0] + cross_or[0]),
        wp.float64(v_body[1] + cross_or[1]),
        wp.float64(v_body[2] + cross_or[2]),
    )
