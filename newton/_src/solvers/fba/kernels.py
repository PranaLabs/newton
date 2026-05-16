# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics cloth solver)."""

import warp as wp

# 3x2 matrix type (wp.mat32 is not available in Warp 1.14; use types.matrix).
mat32 = wp.types.matrix(shape=(3, 2), dtype=wp.float32)


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
    """x_inertia = x_prev + dt·v_prev + dt²·(f_ext·im + g).

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
    """rhs += (m/dt²) · x_inertia.  Free-particle term only; mass==0 for pins, contributes 0."""
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
def project_stretching_arap_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-triangle ARAP local projection scatter for PD cloth.

    Computes the deformation gradient `F = Ds · Dm_inv` (3x2), projects to the
    nearest rotation `P` via SVD-clamp, and scatters the per-particle
    contribution `w · Dm_inv · P^T` into the RHS vector via atomic_add.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (`ke · area`).
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
def project_bending_kernel(
    positions: wp.array[wp.vec3],  # x_cur (unused in MVP scatter; reserved)
    x_ref: wp.array[wp.vec3],  # rest positions (defines rest curvature)
    edge_indices: wp.array2d[wp.int32],  # shape (E, 4)
    edge_quad_q: wp.array[wp.vec4],  # length-4 vector q per edge; Q = q*q^T
    edge_weight: wp.array[wp.float32],
    rhs: wp.array[wp.vec3],
):
    """Per-edge isometric bending scatter for PD cloth.

    RealSim `PDIsometricBendingEnergy::localProjection` scatters
    `w * q[a] * (q^T * x_ref)` into each of the 4 stencil vertices via atomic_add.
    For flat rest (where the cotangent q satisfies q^T * x_ref = 0), the scatter
    is zero. The Hessian's matching `-w * q*q^T * x_cur` term is absorbed into
    the prefactored A in build_pd_system.

    Note: ``positions`` (x_cur) is reserved for future contact / non-flat-rest
    variants and is not read in the MVP implementation.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        x_ref: Reference particle positions [m] (defines rest curvature).
        edge_indices: 4-vertex bending stencil, shape ``[edge_count, 4]``.
        edge_quad_q: Per-edge length-4 cotangent vector ``q``.
        edge_weight: Per-edge bending stiffness ``w``.
        rhs: Output RHS accumulator; receives atomic-add contributions.
    """
    e = wp.tid()
    q = edge_quad_q[e]
    w = edge_weight[e]
    if w == 0.0:
        return

    i0 = edge_indices[e, 0]
    i1 = edge_indices[e, 1]
    i2 = edge_indices[e, 2]
    i3 = edge_indices[e, 3]

    # Compute q^T * x_ref (a vec3 because positions are vec3).
    qTxref = x_ref[i0] * q[0] + x_ref[i1] * q[1] + x_ref[i2] * q[2] + x_ref[i3] * q[3]

    # Scatter w * q[a] * (q^T * x_ref) to each row.
    wp.atomic_add(rhs, i0, w * q[0] * qTxref)
    wp.atomic_add(rhs, i1, w * q[1] * qTxref)
    wp.atomic_add(rhs, i2, w * q[2] * qTxref)
    wp.atomic_add(rhs, i3, w * q[3] * qTxref)


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
