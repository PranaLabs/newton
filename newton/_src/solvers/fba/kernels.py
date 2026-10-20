# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics solver for cloth and tetrahedral soft bodies)."""

import warp as wp

# 3x2 matrix type: Warp has no ``wp.mat32``.
mat32 = wp.types.matrix(shape=(3, 2), dtype=wp.float32)

# ---- LBFGS history types: M = 8 history slots for the Neo-Hookean singular
# value problems, DIM = 2 (Tri) or 3 (Tet).
_LBFGS_M = 8
_lbfgs_mat_2xM = wp.types.matrix(shape=(2, _LBFGS_M), dtype=wp.float64)
_lbfgs_mat_3xM = wp.types.matrix(shape=(3, _LBFGS_M), dtype=wp.float64)
_lbfgs_vec_M = wp.types.vector(length=_LBFGS_M, dtype=wp.float64)

# Fixed LBFGS / line-search parameters:
#   max_iters = 50, ls_decrease (c_1) = 1e-4, tol = 1e-6, M = 8.
_LBFGS_MAX_ITERS = wp.constant(50)
_LBFGS_LS_MAX_ITERS = wp.constant(64)  # Bound on the trial steps of one line search.
_LBFGS_C1 = wp.constant(wp.float64(1.0e-4))
_LBFGS_TOL = wp.constant(wp.float64(1.0e-6))
# Sigma feasibility floor, also the value of the clamp applied when all
# singular values are tiny.
_NH_EPS = wp.constant(wp.float64(1.0e-6))
# Huge-but-finite "infinity" returned by ``value()`` when sigma_i < 0 to
# block the line search. Warp ``@wp.func`` cannot throw, and a finite
# ceiling avoids NaN propagation on GPU floating-point ops.
_NH_VALUE_INF = wp.constant(wp.float64(1.0e30))


@wp.kernel
def gather_per_particle_kernel(
    contributions: wp.array2d[wp.vec3],  # (num_elements, n_verts_per_element)
    offsets: wp.array[wp.int32],  # (n_particles + 1,)
    element_idx: wp.array[wp.int32],  # (total_incidence,)
    local_idx: wp.array[wp.int32],  # (total_incidence,)
    rhs: wp.array[wp.vec3],  # (n_particles,) ACCUMULATED into
):
    """Sum per-element contributions into per-particle rhs.

    For each particle, reads its incident-element entries from the CSR
    adjacency (built by ``build_particle_element_csr``) and sums the
    corresponding ``contributions[element_idx, local_idx]`` vec3 values.
    Adds the sum to ``rhs`` (does NOT overwrite) so the terms already in it,
    the inertia, the pins and the other energies, are preserved.

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
def extract_component_kernel(
    src: wp.array[wp.vec3],
    component: int,
    dst: wp.array[wp.float64],
):
    """dst[i] = src[i][component]: extracts one vec3 component into a float64 array."""
    tid = wp.tid()
    dst[tid] = wp.float64(src[tid][component])


@wp.kernel
def insert_component_kernel(
    src: wp.array[wp.float64],
    component: int,
    dst: wp.array[wp.vec3],
):
    """dst[i][component] = src[i]: inserts a float64 scalar back into a vec3 component."""
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
    """dst[i] = diag[i] * src[i], used for D-inverse scaling in the linear solve."""
    tid = wp.tid()
    dst[tid] = diag[tid] * src[tid]


@wp.kernel
def apply_permutation_scalar_kernel(
    src: wp.array[wp.float64],
    perm: wp.array[wp.int32],
    dst: wp.array[wp.float64],
):
    """dst[i] = src[perm[i]]: scalar gather variant."""
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
    ``pin_stiffness * x_ref`` scatter to the RHS, NOT by gating gravity here.
    For pinned particles (im == 0) ``f_ext * im == 0``, so only gravity
    contributes; the resulting ``x_inertia`` is never fed into the inertia RHS
    term (``add_inertia_to_rhs_kernel`` weights by mass, which is 0 for pins).
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
def copy_particles_kernel(
    particles: wp.array[wp.int32],
    source: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
):
    """``target[p] = source[p]`` for each listed particle ``p``."""
    p = particles[wp.tid()]
    target[p] = source[p]


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
    """ARAP projection: the 3x2 matrix with orthonormal columns nearest to the deformation gradient ``F``.

    The right singular vectors ``v_i`` of ``F`` come from the 2x2 eigenproblem of ``F^T F`` (``wp.svd2``),
    and ``P = u_a v_a^T + u_b v_b^T``. ``u_a = F v_a / |F v_a|`` takes the larger of the two images, and
    ``u_b`` is ``F v_b`` made orthogonal to ``u_a`` and normalized rather than divided by its singular
    value: when a triangle is crushed, that singular value vanishes and float32 ``F^T F`` resolves it no
    better than ``1e-4`` of the larger one, so dividing would amplify rounding noise into a huge
    projection. Below ``1e-5`` of the larger image, ``u_b`` is any unit vector orthogonal to ``u_a``,
    equally near, and ``F = 0`` takes an arbitrary rigid placement.

    Note: Warp has no ``wp.mat32``; ``mat32`` is defined at module
    level as ``wp.types.matrix(shape=(3, 2), dtype=wp.float32)``.
    """
    FtF = wp.transpose(F) * F  # 2x2 symmetric PSD
    _U2, _sigma_sq, V2 = wp.svd2(FtF)
    va = wp.vec2(V2[0, 0], V2[1, 0])
    vb = wp.vec2(V2[0, 1], V2[1, 1])
    a = F * va
    b = F * vb
    if wp.length(b) > wp.length(a):
        va, vb = vb, va
        a, b = b, a
    na = wp.length(a)
    ua = wp.vec3(1.0, 0.0, 0.0)
    if na > 1.0e-12:
        ua = a / na
    b = b - wp.dot(b, ua) * ua
    nb = wp.length(b)
    ub = wp.vec3(0.0)
    if nb > 1.0e-5 * na and nb > 1.0e-12:
        ub = b / nb
    else:
        axis = wp.vec3(0.0, 1.0, 0.0)
        if wp.abs(ua[1]) > 0.9:
            axis = wp.vec3(0.0, 0.0, 1.0)
        ub = wp.normalize(wp.cross(axis, ua))
    p_col0 = ua * va[0] + ub * vb[0]
    p_col1 = ua * va[1] + ub * vb[1]
    return mat32(
        p_col0[0],
        p_col1[0],
        p_col0[1],
        p_col1[1],
        p_col0[2],
        p_col1[2],
    )


@wp.func
def project_arap_3x3_d(F: wp.mat33d) -> wp.mat33d:
    """ARAP 3D projection: F -> nearest proper rotation R, in ``wp.float64``.

    Uses ``wp.svd3`` in fp64, so that ill-conditioned ``F`` (large condition
    number, e.g. a thin tip stretched far along one axis) does not lose
    accuracy through an fp32 SVD, then handles the reflection case: if
    ``det(U) * det(V) < 0``, it flips the last column of U, the one of the
    smallest singular value, so that ``R = U' * V^T`` has ``det(R) = +1``.

    Args:
        F: 3x3 deformation gradient.

    Returns:
        R: 3x3 proper rotation (nearest rotation to F).
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
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Compute per-tet ARAP contribution to each of its 4 local vertices.

    Computes ``F = Ds * Dm_inv`` (3x3), projects it to the nearest rotation ``R``
    by an fp64 SVD with reflection handling, and writes ``w * Dm_inv * R^T`` to
    the four stencil vertices.
    Writes into a pre-allocated ``(T, 4)`` scratch buffer; :func:`gather_per_particle_kernel`
    sums the contributions per particle in a fixed order, so the right-hand side is deterministic.

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

    # fp64 SVD-based rotation projection, to avoid fp32 SVD accuracy loss on
    # ill-conditioned F (e.g. extreme tail stretching). The contributions are
    # demoted to fp32.
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
def project_corotational_sigma3d_d(sigma: wp.vec3d, mu: wp.float64, lam: wp.float64) -> wp.vec3d:
    """Closed-form 3D corotational local projection on SVD singular values, in ``wp.float64``.

    Minimises E(s) = mu*||s-I||^2 + (lam/2)*tr(s-I)^2 + (k/2)*||s-s0||^2
    with k = 2*mu (PD penalty weight), I = (1,1,1) identity singular values.

    Setting grad_E = 0 yields the 3x3 linear system (4*mu*I + lam*11^T)*s = b,
    solved in closed form via Sherman-Morrison:

        alpha = 4*mu,  beta = lam,  n = 3
        inv(alpha*I + beta*11^T) = (1/alpha)*I - beta/(alpha*(alpha + n*beta))*11^T
        b_i = 2*mu + 3*lam + 2*mu*s0_i
        sum_b = 6*mu + 9*lam + 2*mu*(s0[0]+s0[1]+s0[2])
        s_proj[i] = b_i/alpha - beta*sum_b / (alpha*(alpha + 3*beta))

    The trace term sums all three singular values,
    ``tr(s-I) = (s_0-1) + (s_1-1) + (s_2-1)``, as the 2D cloth Corot does.
    The Tet Corot local projection runs it in fp64 so that ill-conditioned
    singular values feed a numerically stable solve; pair it with an fp64 SVD
    of F.

    Args:
        sigma: Singular values from ``wp.svd3(F)``, in descending order.
        mu: First Lame parameter (shear modulus) [Pa].
        lam: Second Lame parameter [Pa].

    Returns:
        Projected singular-value triple ``(s_proj_0, s_proj_1, s_proj_2)``.
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
    lame: wp.array[wp.vec2],
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Compute per-tet corotational contribution to each of its 4 local vertices.

    Computes ``F = Ds * Dm_inv`` (3x3), takes its singular values with
    ``wp.svd3``, applies the closed-form 3D corotational projection to get
    ``sigma_proj``, rebuilds ``P = U * diag(sigma_proj) * Vt`` and writes
    ``w * Dm_inv * P^T`` to the four stencil vertices. ``wp.svd3`` returns
    non-negative singular values and the projection keeps them positive, so
    ``P`` needs no reflection handling.
    Writes into a pre-allocated ``(T, 4)`` scratch buffer; :func:`gather_per_particle_kernel`
    sums the contributions per particle in a fixed order, so the right-hand side is deterministic.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        lame: First and second Lamé parameters of each element [Pa].
        contributions: Output ``(tet_count, 4)`` per-local-vertex contribution.
    """
    t = wp.tid()
    mu, lam = lame[t][0], lame[t][1]
    if tet_weight[t] == 0.0:
        # No stiffness, such as a soft body's surface triangles: nothing to project.
        for i in range(4):
            contributions[t, i] = wp.vec3(0.0)
        return
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

    # fp64 SVD + sigma projection + P reconstruction, demoted to fp32 only for
    # the contributions.
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
# LBFGS for the Tri/Tet Neo-Hookean local projection.
#
# L-BFGS (M = 8, ``c_1 = 1e-4``, tol = 1e-6, max_iters = 50) with a cubic
# backtracking line search minimizes the Neo-Hookean energy of the singular
# values plus the PD penalty. The Tri (2D) and Tet (3D) variants are kept
# separate for clarity. All arithmetic is in ``wp.float64``. Warp functions
# cannot throw, so:
#   - ``nh_value_2d/3d`` returns ``1e30`` instead of ``+infinity`` for sigma_i < 0
#     (a semantically clean +infinity risks NaN propagation);
#   - ``nh_gradient_2d/3d`` clamps sigma_i >= 1e-6 internally instead of
#     raising.
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

    For sigma_i < 0 returns the huge finite barrier ``1e30``. For sigma_i >= 0 returns
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

    sigma_i is clamped to ``1e-6`` internally before computing ``log(J)`` and
    ``1/sigma_i``: Warp cannot throw, and the LBFGS line search rejects any step
    where ``value()`` returned the +infinity barrier, so a clamped gradient is
    never consumed for an accepted step.
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
    """Tet NH objective value Psi(sigma) + (k/2)||sigma-sigma_init||^2, with
    ``J = sigma_0*sigma_1*sigma_2``. sigma_i < 0 returns 1e30."""
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
    """Tet NH gradient grad Psi + k(sigma - sigma_init). sigma_i is clamped to
    1e-6 inside, as in :func:`nh_gradient_2d`."""
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
    """Cubic-interpolation step length of the backtracking line search.

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
    """Clamp the trial step length to ``[low, high]``."""
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
    """Cubic backtracking Armijo line search for Tri NH.

    ``fx0`` and ``gtp = grad f(x)*p`` are passed in, already computed by the
    caller, which saves a gradient evaluation per LBFGS iteration. Returns
    ``-1`` on failure (Armijo not satisfied within the inner iteration cap).
    """
    p_norm_sq = p[0] * p[0] + p[1] * p[1]
    if p_norm_sq <= wp.float64(0.0):
        return _LBFGS_C1  # A zero direction returns the decrease factor.

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
            # First failure: quadratic interpolation.
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
    """Tet variant of the cubic backtracking Armijo line search."""
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
    """Tri NH local projection by LBFGS (M=8) with a cubic backtracking line search.

    Minimizes the Neo-Hookean energy of the singular values plus the PD penalty
    ``(k/2)||sigma - sigma_init||^2``, ``k = 2 mu``. The history is stack-local
    and starts empty in every call, so every element and PD outer iteration
    starts afresh. Convergence: ``||grad|| < 1e-6`` OR ``||x_prev - x|| < 1e-6``.
    Returns the current iterate on line-search failure.
    """
    k = wp.float64(2.0) * mu

    # All-tiny clamp. The Tet wrapper's ``sigma_2 < 0`` flip is omitted here:
    # the Tri caller uses ``wp.svd2(F^T F)``, which yields eigenvalues >= 0, and
    # takes their ``sqrt``, also non-negative.
    s0 = sigma_init[0]
    s1 = sigma_init[1]
    if wp.abs(s0) < _NH_EPS and wp.abs(s1) < _NH_EPS:
        s0 = _NH_EPS
        s1 = _NH_EPS
    x = wp.vec2d(s0, s1)
    x0 = x  # Quadratic-penalty anchor.

    # History storage: ``s`` and ``y`` columns hold history pairs (oldest in
    # col 0). ``alpha``/``rho`` are two-loop scratch. All zeroed.
    s_hist = _lbfgs_mat_2xM()
    y_hist = _lbfgs_mat_2xM()
    alpha_vec = _lbfgs_vec_M()
    rho_vec = _lbfgs_vec_M()

    grad = nh_gradient_2d(x, x0, mu, lam, k)
    gamma_k = wp.float64(1.0)
    alpha_init = wp.float64(1.0)

    # ``history_count`` is ``min(M, k)`` for ``k`` iterations since the last
    # restart. A budget counter and a fixed Python ``range`` bound the
    # iterations across restarts (Warp can't mutate the loop var). A restart
    # sets ``history_count = 0``; ``iters_remaining`` keeps counting down.
    history_count = int(0)
    iters_remaining = _LBFGS_MAX_ITERS

    for _it in range(_LBFGS_MAX_ITERS):
        if iters_remaining <= 0:
            break

        x_old = x
        grad_old = grad
        q = grad

        # L-BFGS first-loop recursion, newest -> oldest, over the
        # ``min(M, k)`` stored pairs: eviction on overflow keeps
        # ``history_count`` at that value.
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

        # L-BFGS second-loop recursion, oldest -> newest.
        for i in range(_LBFGS_M):
            if i >= iter_count:
                break
            qy = q[0] * y_hist[0, i] + q[1] * y_hist[1, i]
            beta = rho_vec[i] * qy
            coeff = alpha_vec[i] - beta
            q = wp.vec2d(q[0] + coeff * s_hist[0, i], q[1] + coeff * s_hist[1, i])

        # Non-descent restart.
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
        # fx0 = f(x); gtp = grad f(x)*p = -q*grad, passed to the line search.
        fx0 = nh_value_2d(x, x0, mu, lam, k)
        gtp = grad[0] * p_search[0] + grad[1] * p_search[1]
        rate = _backtracking_cubic_2d(x, x0, p_search, fx0, gtp, alpha_init, mu, lam, k)
        if rate <= wp.float64(0.0):
            # Line-search failure: return the current iterate.
            return x

        x_last = x
        x = wp.vec2d(x[0] - rate * q[0], x[1] - rate * q[1])
        iters_remaining -= 1

        # Convergence on grad (at x_old, before it is recomputed) OR step.
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
            # Evict oldest, shift left, append newest at col M-1 (equivalent
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
            # On a restart ``alpha_init`` was set to min(1, 1/||g||inf); the
            # *next* iter resets it to 1.0.
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
    """Tet NH local projection by LBFGS (M=8) with a cubic backtracking line search.

    As :func:`project_neohookean_sigma2d_lbfgs` with three singular values. The
    sigma fix-up before the minimization clamps all-three-tiny singular values
    and flips the sign of a negative sigma_2. The flip is a no-op in practice
    because ``wp.svd3`` returns non-negative singular values.
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
        s2 = -s2  # No-op for wp.svd3 outputs (>= 0), kept as a guard.
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


@wp.kernel
def project_stretching_neohookean_tet_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    lame: wp.array[wp.vec2],
    # output (per-tet, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 4)
):
    """Per-tet Neo-Hookean projection by L-BFGS.

    Projects the singular values of ``F`` with
    :func:`project_neohookean_sigma3d_lbfgs` and writes each tet's
    contributions to its four particles for the per-particle gather. SVD,
    L-BFGS, sigma projection and P reconstruction all run in ``wp.float64``.
    Final ``proj`` is demoted to fp32 only at the contributions write boundary
    (``_rhs`` itself is fp32).
    """
    t = wp.tid()
    mu, lam = lame[t][0], lame[t][1]
    if tet_weight[t] == 0.0:
        # No stiffness, such as a soft body's surface triangles: nothing to project.
        for i in range(4):
            contributions[t, i] = wp.vec3(0.0)
        return
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
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Compute per-tri ARAP contribution to each of its 3 local vertices.

    Computes the deformation gradient ``F = Ds * Dm_inv`` (3x2), projects it to
    the nearest rotation ``P`` by clamping its singular values to 1, and writes
    ``w * Dm_inv * P^T`` to the three vertices.
    Writes into a pre-allocated ``(T, 3)`` scratch buffer; :func:`gather_per_particle_kernel`
    sums the contributions per particle in a fixed order, so the right-hand side is deterministic.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``2 * k_mu * area``).
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
    (Sifakis 2012, Bouaziz 2014, etc.): the trace term sums both singular
    values, ``tr(s-I) = (s_0-1) + (s_1-1)``.

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
    lame: wp.array[wp.vec2],
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Compute per-tri corotational contribution to each of its 3 local vertices.

    Computes the deformation gradient ``F = Ds * Dm_inv`` (3x2), takes its
    singular values from ``wp.svd2(FtF)``, applies the closed-form corotational
    projection to get ``sigma_proj``, rebuilds ``P = U * diag(sigma_proj) * Vt``
    and writes ``w * Dm_inv * P^T`` to the three vertices.
    Writes into a pre-allocated ``(T, 3)`` scratch buffer; :func:`gather_per_particle_kernel`
    sums the contributions per particle in a fixed order, so the right-hand side is deterministic.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2x2 rest-pose inverse (``Dm_inv``).
        tri_weight: Per-triangle stretching weight (``2 * k_mu * area``).
        lame: First and second Lamé parameters of each element [Pa].
        contributions: Output ``(tri_count, 3)`` per-local-vertex contribution.
    """
    t = wp.tid()
    mu, lam = lame[t][0], lame[t][1]
    if tri_weight[t] == 0.0:
        # No stiffness, such as a soft body's surface triangles: nothing to project.
        for i in range(3):
            contributions[t, i] = wp.vec3(0.0)
        return
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


@wp.kernel
def project_stretching_neohookean_kernel(
    positions: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],  # flat shape (3*T,)
    tri_rest_inv: wp.array[wp.mat22],
    tri_weight: wp.array[wp.float32],
    lame: wp.array[wp.vec2],
    # output (per-tri, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (T, 3)
):
    """Per-tri Neo-Hookean projection by L-BFGS.

    Projects the singular values of ``F`` with
    :func:`project_neohookean_sigma2d_lbfgs` and writes each triangle's
    contributions to its three particles for the per-particle gather. SVD
    inputs (singular values of F derived from ``wp.svd2(F^T F)``) are cast
    fp32 -> fp64 before L-BFGS; the projected sigma is cast back to fp32 for
    the P reconstruction.
    """
    t = wp.tid()
    mu, lam = lame[t][0], lame[t][1]
    if tri_weight[t] == 0.0:
        # No stiffness, such as a soft body's surface triangles: nothing to project.
        for i in range(3):
            contributions[t, i] = wp.vec3(0.0)
        return
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
    positions: wp.array[wp.vec3],  # x_cur (current iterate)
    edge_indices: wp.array2d[wp.int32],  # shape (E, 4)
    edge_quad_q: wp.array[wp.vec4],  # length-4 vector q per edge; Q = q*q^T
    edge_weight: wp.array[wp.float32],
    edge_norm: wp.array[wp.float32],  # ||q*x_rest|| (rest curvature magnitude)
    # output (per-edge, per-local-vertex contributions)
    contributions: wp.array2d[wp.vec3],  # (E, 4)
):
    """Compute per-edge bending contribution to each of its 4 stencil vertices.

    The local step projects the current curvature vector ``q*x_cur`` onto the
    sphere of radius ``||q*x_rest||``: it restores the curvature *magnitude* of
    the rest shape and lets the direction follow ``x_cur``, which holds on flat
    and curved rest cloth alike. The contribution at local vertex ``a`` is
    ``w * q[a] * target`` with ``target = (q*x_cur).normalized() * ||q*x_rest||``.
    When the weight, the rest curvature or the current curvature (below
    ``1e-12``) vanishes, the contributions are zero. The matching Hessian term
    ``w * q*q^T`` is part of the prefactored system matrix and does not depend
    on ``x_cur``.
    Writes into a pre-allocated ``(E, 4)`` scratch buffer; :func:`gather_per_particle_kernel`
    sums the contributions per particle in a fixed order, so the right-hand side is deterministic.

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
# Contact Jacobian kernels
# ---------------------------------------------------------------------------


@wp.kernel
def accumulate_vec3_kernel(
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
):
    """dst[i] += src[i]: accumulate one vec3 array into another."""
    tid = wp.tid()
    dst[tid] = dst[tid] + src[tid]


# ---------------------------------------------------------------------------
# Schur build primitives restricted to the contacted DOFs (isodofs)
# ---------------------------------------------------------------------------


# Tile sizes for the dense GEMM used by ``compute_wi_tiled_kernel``.
# ``32 x 32 x 32`` with ``128`` threads per block was empirically fastest for
# k ~ 1200 contacted DOFs among N ~ 7000 particles on an RTX 5090 in fp64.
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
    sequentially (``ST_offsets[ip..ip+1]``).

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
            int32. Selects which columns of ``S`` to densify.
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

    A dense blocked matrix-matrix multiply instead of an O(k² · nnz_col) sparse
    column intersection. For k≈1200, N≈7000 and a mean S column
    nnz ≈ 1700 (``S`` columns ~24% dense), the densified GEMM is
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

    The NSN inner solve consumes it through the dense-PCR path.

    Pinned particles (``inv_mass == 0``) zero their entire row+column block,
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
    out: wp.array[wp.vec3],  # (N,) per-particle J^T * lambda, OVERWRITTEN
):
    """Compute ``out[p] = sum_{r: particle[r]=p} alpha[r] * lam[r] * dir[r]``.

    Particle-centered gather, deterministic because it needs no atomics: one
    thread per particle reads its incident contact rows from the
    inverse-mapping CSR (built once per contact set by the Schur build, see
    :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.build_schur_complement`)
    and sums ``alpha[r] * lam[r] * dir[r]`` over them.

    Frictionless (unilateral) contacts: one row per contact, ``total_rows == M``.
    Frictional (Coulomb) contacts: three rows per contact (n, t1, t2, all sharing
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
# NSN inner LCP: Fischer-Burmeister rows and Schur build, in fp64.
#
# Per-row scalar evaluators of the Fischer-Burmeister functions and the
# ``A_schur = ωωᵀ ⊙ W + diag(c)`` assembly. The inner Schur solve runs in
# fp64, as the host reference in the tests does.
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

    GPU version of the host reference ``fb_unilateral_row``
    (``newton/tests/fba/_reference.py``). Returns a :class:`warp.vec3d`
    packing ``(omega, compliance, h)``: Warp ``@wp.func`` cannot return Python
    tuples, so the fp64 vec3 type carries the three scalars.

    Args:
        penetration: ``J·q - pene0`` at current iterate (m).
        lam: Current normal lambda (N).
        precond: ``1 / W_ii`` (1/N).
        dt: Timestep (s).
        pene0: ``n·anchor`` (m).

    Returns:
        ``vec3d(omega, compliance, h)`` matching the CPU reference's tuple
        return semantics. The degenerate-root branch returns
        ``(0, precond/dt², 0)``: the residual vanishes there, so the row
        takes no Newton step.
    """
    pene = penetration
    plam = precond * lam
    root = wp.sqrt(pene * pene + plam * plam)
    if root < wp.float64(1.0e-30):
        return wp.vec3d(wp.float64(0.0), precond / (dt * dt), wp.float64(0.0))
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

    GPU version of the host reference ``fb_frictional_row``
    (``newton/tests/fba/_reference.py``).
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

    GPU version of the host expression in
    ``solve_nsn_unilateral`` (``newton/tests/fba/_reference.py``),
    ``A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)``,
    which assembles the unilateral and frictional row contributions into the
    inner Schur LHS.

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
# NSN inner driver: residual, penetration, rhs, FB row launchers, box clamp.
#
# Each kernel computes a NumPy expression of ``solve_nsn_unilateral`` or
# ``solve_nsn_coulomb`` (``newton/tests/fba/_reference.py``) on the device.
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
    """Frictionless (unilateral) contact residual.

    Computes ``r[c] = offset[c] - alpha[c] * dot(normal[c], x[particle[c]])``
    one contact per thread. ``wp.dot`` on a ``vec3`` returns ``float32``; it is
    widened to fp64 before the multiply.

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
    """Frictional (Coulomb) contact residual.

    Emits ``r[3c+0/1/2] = offset_{n/t1/t2}[c] - alpha[c] * dot(axis, x[p[c]])``
    where ``axis`` is normal / t1 / t2 respectively. One contact per thread
    writes three consecutive rows in interleaved ``[r_n, r_t1, r_t2]``
    layout.

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
    """Compute ``precond[i] = dt² · max(W_ii, 1e-12)``, or ``dt²`` where ``|W_ii| <= 1e-12`` (frictionless rows).

    The expression of the host reference ``solve_nsn_unilateral``
    (``newton/tests/fba/_reference.py``):
    ``precond = (dt*dt) * np.where(|diag_W|>1e-12, max(diag_W, 1e-12), 1.0)``.
    The ``np.where`` falls through to ``1.0`` only when ``|diag_W| <=
    1e-12`` (degenerate row); otherwise it takes ``max(diag_W, 1e-12)``, which
    differs from ``|diag_W|`` for negative diagonals. The kernel keeps both
    branches, so its output equals the host reference bit for bit.

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
    """Compute the frictional preconditioner: ``dt²·W_ii`` for normal rows and
    ``dt·W_ii`` for tangent rows, with ``|W_ii|`` floor of ``1e-12``.

    Same as the per-contact loop of the host reference
    ``solve_nsn_coulomb`` (``newton/tests/fba/_reference.py``) (``w_n/w_t1/w_t2 =
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
# matvec tile size (see ``nsn_pcr_solver.py``): the NSN n_rows is the same
# Schur dimension, and an fp64 sweep on an RTX 5090 found the same sweet
# spot at n_rows ~ 600..1500.
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
    """Compute ``penetration = -r + dt² · W · (ω · λ)``.

    The expression of the host reference's FB-Newton loop:
    ``penetration = -r + (dt*dt) * (W @ (omega * lam))``.

    Block-per-row tiled reduction: each block computes one ``penetration[i]``
    cooperatively by accumulating ``TILE_K``-wide strips of
    ``W[i, j] * omega[j] * lam[j]`` and reducing with ``wp.tile_sum``.
    Launched via :func:`wp.launch_tiled` with ``dim=n_rows`` and
    ``block_dim=_NSN_MATVEC_BLOCK_DIM``, about 4-5x faster than a thread per
    row at n_rows ~ 600..1500 on an RTX 5090 in fp64.

    Args:
        r: Contact residual (fp64), shape ``[n_rows]``.
        W: Schur complement, shape ``[n_rows, n_rows]``.
        omega: FB row weights, shape ``[n_rows]``.
        lam: Current lambda, shape ``[n_rows]``.
        n_rows: Active row count (``M`` frictionless, ``3M`` frictional).
        n_rows_pad: ``ceil(n_rows / TILE_K) * TILE_K``. The K loop runs up
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
        # state out of shared memory (re-assigning a shared tile broke the
        # register-vs-shared promotion of Warp 1.14).
        wol = wol + wp.tile_extract(s, 0)
    # All threads in the block write the same value. CUDA coalesces
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
    """Compute the NSN inner right-hand side ``rhs = (1/dt²) · (h - ω · J·x_corrected)``.

    The expressions of the host reference:

        J_x = (pene0 - r) + (dt*dt) * (W @ (omega * lam))
        rhs = (1.0 / (dt*dt)) * (h - omega * J_x)

    Block-per-row tiled reduction (see :func:`compute_penetration_kernel`)
    for the ``W @ (omega · lam)`` matvec.  The two expressions stay fused in
    a single kernel so we avoid materialising ``J_x`` to a scratch buffer.

    Args:
        h: Per-row Schur RHS contribution, shape ``[n_rows]``.
        omega: Per-row FB weighting, shape ``[n_rows]``.
        pene0: Per-row anchor projection, shape ``[n_rows]``.
        r: Contact residual, shape ``[n_rows]``.
        W: Schur complement, shape ``[n_rows, n_rows]``.
        lam: Current lambda, shape ``[n_rows]``.
        n_rows: Active row count.
        n_rows_pad: ``ceil(n_rows / TILE_K) * TILE_K``. See
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
    ``solve_nsn_unilateral`` (``newton/tests/fba/_reference.py``).

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
    ``solve_nsn_coulomb`` (``newton/tests/fba/_reference.py``). The tangent rows take the
    *current iterate* of ``lam_n`` as their companion normal lambda, as the
    host reference does, which reads ``lam[idx_n]`` before computing the
    tangent rows in the same iteration.

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
def contact_mu_kernel(
    contact_shape: wp.array[wp.int32],  # (M,) shape of each contact, -1 for contacts without a shape
    shape_material_mu: wp.array[wp.float32],
    particle_mu: wp.float64,
    # output
    mu: wp.array[wp.float64],
):
    """Friction coefficient of each contact: its shape's, or the particles' for contacts without a shape."""
    c = wp.tid()
    s = contact_shape[c]
    if s >= 0 and s < shape_material_mu.shape[0]:
        mu[c] = wp.float64(shape_material_mu[s])
    else:
        mu[c] = particle_mu


@wp.kernel
def lite_row_dirs_kernel(
    normal: wp.array[wp.vec3],  # (M,)
    tangent1: wp.array[wp.vec3],  # (M,), unused with one row per contact
    tangent2: wp.array[wp.vec3],
    rows_per_contact: int,
    # output
    row_dir: wp.array[wp.vec3],  # (rows_per_contact M,)
):
    """Directions of the lite rows: the normal and tangents of each contact."""
    r = wp.tid()
    if r % rows_per_contact == 0:
        row_dir[r] = normal[r // rows_per_contact]
    elif r % rows_per_contact == 1:
        row_dir[r] = tangent1[r // rows_per_contact]
    else:
        row_dir[r] = tangent2[r // rows_per_contact]


@wp.kernel
def fb_newton_lite_coulomb_kernel(
    # ---- contact metadata (M entries each) ----
    contact_particle: wp.array[wp.int32],  # (M,)
    contact_normal: wp.array[wp.vec3],  # (M,)
    contact_tangent1: wp.array[wp.vec3],  # (M,)
    contact_tangent2: wp.array[wp.vec3],  # (M,)
    contact_alpha: wp.array[wp.float32],  # (M,)
    contact_mu: wp.array[wp.float64],  # (M,)
    # ---- per-row inputs (3M entries each) ----
    r: wp.array[wp.float64],  # (3M,)
    pene0: wp.array[wp.float64],  # (3M,)
    # ---- global state ----
    inv_mass: wp.array[wp.float64],  # (N,)
    dt: wp.float64,
    cap_internal: wp.float64,  # lambda_cap / dt² (or sentinel; see caller)
    use_cap: wp.int32,  # 1 = clip, 0 = no-op
    n_fb_iters: wp.int32,  # outer FB-Newton iter count
    # ---- inout / outputs ----
    lam: wp.array[wp.float64],  # (3M,) inout
    omega: wp.array[wp.float64],  # (3M,) inout: warm-started on entry, written on exit
    lam_apply: wp.array[wp.float64],  # (3M,) output: dt² · ω · λ_final
):
    """Fused per-contact FB-Newton step for LiteNSN with Coulomb friction.

    Under the lite approximation ``W = H · (dt²·M⁻¹) · Hᵀ``, contacts that touch
    distinct particles decouple completely.  For cloth on a sphere (and all
    scenes where each cloth particle touches at most one shape), every
    particle is touched by exactly one contact, so the
    ``3M x 3M`` Schur system block-diagonalises into ``M`` independent ``3x3``
    blocks.  Each thread of this kernel processes one contact end-to-end:

    1. Build the local ``3x3`` ``W`` block from ``alpha``, ``dir``, and ``dt² · inv_m``.
    2. Compute the local penetration ``-r + dt² · W · (ω·λ)``.
    3. Evaluate the FB rows (normal + 2 tangents) to refresh ``(ω, c, h)``.
    4. Build ``A_schur = ωωᵀ ⊙ W + diag(c)`` and ``rhs = (1/dt²)·(h - ω · J_x)``.
    5. Invert the ``3x3`` ``A_schur`` and apply ``dλ``.
    6. Coulomb clamp + optional symmetric ``lambda_cap``.
    7. Write ``lam_apply = dt² · ω · λ`` for the correction step.

    Memory cost: ``O(M)`` (no Schur matrix materialised).  Scales linearly
    with contact count.

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
        n_fb_iters: Number of FB-Newton iterations; SolverFBA takes one per
            local-global iteration.
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
    # ``alpha`` is the same for all three rows of a contact (set by the host build).
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
        # penetration = -r + dt² · W · (ω · λ).  Pure 3x3 matvec.
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
        # rows read ``lam_n``, their companion normal multiplier.
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

        # rhs = (1/dt²) · (h - ω · J_x) with J_x = (pene0 - r) + dt²·W·(ω·λ) under the updated weights, as the
        # dense driver takes it; the penetration above holds the weights of the previous iteration.
        wl_n = om_n * lam_n
        wl_t1 = om_t1 * lam_t1
        wl_t2 = om_t2 * lam_t2
        Jx_n = p0_n - r_n + dt2 * (W_nn * wl_n + W_nt1 * wl_t1 + W_nt2 * wl_t2)
        Jx_t1 = p0_t1 - r_t1 + dt2 * (W_nt1 * wl_n + W_t1t1 * wl_t1 + W_t1t2 * wl_t2)
        Jx_t2 = p0_t2 - r_t2 + dt2 * (W_nt2 * wl_n + W_t1t2 * wl_t1 + W_t2t2 * wl_t2)
        inv_dt2 = wp.float64(1.0) / dt2
        rhs_n = inv_dt2 * (h_n - om_n * Jx_n)
        rhs_t1 = inv_dt2 * (h_t1 - om_t1 * Jx_t1)
        rhs_t2 = inv_dt2 * (h_t2 - om_t2 * Jx_t2)

        # 3x3 symmetric solve via cofactor expansion (no Cholesky needed at
        # this size; the matrix is SPD by construction modulo the ωωᵀ Hadamard
        # which preserves SPD when c > 0 and W is SPD).
        # Determinant of A (symmetric).
        m11 = A11 * A22 - A12 * A12
        m12 = A01 * A22 - A02 * A12
        m13 = A01 * A12 - A02 * A11
        det = A00 * m11 - A01 * m12 + A02 * m13
        if wp.abs(det) < wp.float64(1.0e-30):
            # Singular: skip the update (warm-start λ unchanged).
            continue

        inv_det = wp.float64(1.0) / det
        # Inverse of symmetric 3x3 via cofactor matrix.
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

    # Coulomb box clamp of the tangent multipliers, as the dense NSN and the per-particle lite kernel apply it:
    # no friction without a positive normal multiplier, else within [-mu lam_n, mu lam_n].
    if lam_n <= wp.float64(0.0):
        lam_t1 = wp.float64(0.0)
        lam_t2 = wp.float64(0.0)
    else:
        bound = mu_c * lam_n
        lam_t1 = wp.clamp(lam_t1, -bound, bound)
        lam_t2 = wp.clamp(lam_t2, -bound, bound)

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


# ----- Per-particle fused FB-Newton kernel for LiteNSN -----
#
# Generalises ``fb_newton_lite_coulomb_kernel`` to handle ``k_p ∈ {1..4}``
# contacts per particle.  Each thread processes one *unique-contacted* particle
# end-to-end; the local 3·k_p x 3·k_p Schur block is constructed in registers
# (fp64 ``mat12``) and solved via in-place Cholesky.  Sparsity stays
# block-diagonal by particle.
#
# A particle with more than ``_MAX_KP_PER_PARTICLE`` contacts keeps its
# multipliers: the kernel applies them unchanged. A cloth vertex pinched
# between two pads on a table has 3 contacts, so k_max=4 leaves headroom.
# ``mat12`` is the largest type we materialise per thread.
_MAX_KP_PER_PARTICLE = wp.constant(4)
_MAX_ROWS_PER_PARTICLE = wp.constant(12)  # 3 · _MAX_KP_PER_PARTICLE
_MAT12_F64 = wp.types.matrix((12, 12), wp.float64)
_VEC12_F64 = wp.types.vector(12, wp.float64)


@wp.kernel
def fb_newton_lite_coulomb_per_particle_kernel(
    # ---- per-isodof iteration metadata ----
    isodof_unique: wp.array[wp.int32],  # (K,) unique contacted particle indices
    isodof_row_offsets: wp.array[wp.int32],  # (N+1,) CSR over ALL particles
    isodof_row_indices: wp.array[wp.int32],  # (total_rows,) row idx sorted by particle
    # ---- per-row data (3M entries, interleaved [n, t1, t2] per contact) ----
    row_particle: wp.array[wp.int32],  # (3M,) particle index per row
    row_dir: wp.array[wp.vec3],  # (3M,) direction per row (fp32)
    row_alpha: wp.array[wp.float32],  # (3M,) Jacobian coef per row
    # ---- per-contact data (M entries) ----
    contact_mu: wp.array[wp.float64],  # (M,) friction coef per contact
    # ---- per-row force inputs (3M entries) ----
    r: wp.array[wp.float64],  # (3M,)
    pene0: wp.array[wp.float64],  # (3M,)
    # ---- global state ----
    inv_mass: wp.array[wp.float64],  # (N,)
    dt: wp.float64,
    cap_internal: wp.float64,  # lambda_cap / dt²
    use_cap: wp.int32,
    n_fb_iters: wp.int32,
    # ---- inout / outputs ----
    lam: wp.array[wp.float64],  # (3M,) inout
    omega: wp.array[wp.float64],  # (3M,) inout
    lam_apply: wp.array[wp.float64],  # (3M,) output
):
    """Per-particle fused FB-Newton for LiteNSN with Coulomb friction, k_p ∈ {1..4}.

    Each thread handles one unique-contacted particle ``p``.  Its ``k_p`` contact
    rows live consecutively in ``isodof_row_indices[start:end]``; each row carries
    its own ``(particle, dir, alpha)`` triple and its own (n, t1, t2) sub-triplet
    in the global row arrays.  Under the lite approximation, ``W_lite`` is
    block-diagonal-by-particle, so this thread is fully independent.

    Local state lives in registers / L1 (warp ``mat12``).  Cholesky on a
    ``3·k_p x 3·k_p`` PSD matrix runs in-place over the same buffer.

    The rows of a particle with more than ``_MAX_ROWS_PER_PARTICLE`` keep their
    multipliers.
    """
    iso = wp.tid()
    p = isodof_unique[iso]
    row_start = isodof_row_offsets[p]
    row_end = isodof_row_offsets[p + 1]
    k_total = row_end - row_start  # = 3 · k_p
    if k_total <= 0:
        return
    inv_m = inv_mass[p]
    dt2 = dt * dt
    if k_total > _MAX_ROWS_PER_PARTICLE:
        for ai in range(k_total):
            ra = isodof_row_indices[row_start + ai]
            lam_apply[ra] = dt2 * omega[ra] * lam[ra]
        return

    # Build local W (3kx3k) directly from (alpha, dir) triples.
    # W[a, b] = alpha_a · alpha_b · dot(dir_a, dir_b) · dt² · inv_m
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
                # local block: same contact c = ra // 3, normal row = 3·c.
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

        # A_schur = ωωᵀ ⊙ W + diag(c), block-PSD by construction.
        A = _MAT12_F64(wp.float64(0.0))
        for ai in range(k_total):
            for bi in range(k_total):
                a_ij = om_l[ai] * om_l[bi] * W[ai, bi]
                if ai == bi:
                    a_ij = a_ij + c_l[ai]
                A[ai, bi] = a_ij

        # rhs = (1/dt²) · (h - ω · J_x) with J_x = (pene0 - r) + dt²·W·(ω·λ) under the updated weights, as the
        # dense driver takes it; the penetration above holds the weights of the previous iteration.
        inv_dt2 = wp.float64(1.0) / dt2
        rhs_l = _VEC12_F64(wp.float64(0.0))
        for ai in range(k_total):
            acc = wp.float64(0.0)
            for bi in range(k_total):
                acc = acc + W[ai, bi] * om_l[bi] * lam_l[bi]
            Jx = pene0_l[ai] - r_l[ai] + dt2 * acc
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
                # will then produce ~zero update.  Don't actually break: warp
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

    # Coulomb box clamp per contact (each contact = 3 consecutive rows): no friction without a positive normal
    # multiplier, else within [-mu lam_n, mu lam_n]. Identify contact-c normal local index, clamp its t1 and t2.
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
            bound = wp.float64(0.0)
            if lam_n_val > wp.float64(0.0):
                bound = contact_mu[contact_idx] * lam_n_val
            lam_l[t1_local] = wp.clamp(lam_l[t1_local], -bound, bound)
            lam_l[t2_local] = wp.clamp(lam_l[t2_local], -bound, bound)

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
    """Per-contact Coulomb box clamp on tangent lambdas.

    As ``coulomb_box_clamp`` of the host reference (``newton/tests/fba/_reference.py``)
    does: a contact whose normal multiplier ``lam_n`` is not positive carries
    no friction, so its tangent multipliers are zeroed; the others are clamped
    into ``[-mu·lam_n, mu·lam_n]``. ``lam_n`` keeps the value of the Newton
    step. Bounds read from a negative ``lam_n`` would swap and leave a friction
    force of ``mu·|lam_n|`` in an arbitrary direction.

    Args:
        mu: Per-contact friction coefficient (fp64), shape ``[M]``.
        lam: Per-row lambda (fp64), shape ``[3M]``. Modified in place.
    """
    c = wp.tid()
    lam_n = lam[3 * c]
    if lam_n <= wp.float64(0.0):
        lam[3 * c + 1] = wp.float64(0.0)
        lam[3 * c + 2] = wp.float64(0.0)
        return
    bound = mu[c] * lam_n
    lam[3 * c + 1] = wp.clamp(lam[3 * c + 1], -bound, bound)
    lam[3 * c + 2] = wp.clamp(lam[3 * c + 2], -bound, bound)


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

    See ``solve_nsn_unilateral`` (``newton/tests/fba/_reference.py``) docstring for the
    derivation: the correction is ``Δq = dt² · A⁻¹ · Jᵀ · (ω · λ)``, and
    ``apply_lambda_correction_combined`` adds ``A⁻¹·Jᵀ·λ_apply`` directly, so
    the multipliers are pre-scaled by ``dt²·ω`` here.

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

    Computes ``offset_t = offset_t_base + dt · dot(t, v_anchor)`` on the
    device, so the step downloads no tangents or anchor velocities.

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
# Per-contact bookkeeping of ``SolverFBA._ingest_contacts`` on the device.
#
# These kernels evaluate the normal offset, and for frictional contacts the
# tangent basis and kinematic anchor velocity of each contact, after the
# contacts are sorted into a deterministic order.
# ---------------------------------------------------------------------------


@wp.kernel
def compute_normal_offset_kernel(
    normal: wp.array[wp.vec3],  # (M,) world-frame unit normal per contact
    world_anchor: wp.array[wp.vec3],  # (M,) world-frame contact anchor
    offset: wp.array[wp.float64],  # (M,) OUTPUT
):
    """Compute ``offset[c] = n·world_anchor``: the contact plane through the closest shape point.

    Args:
        normal: Per-contact unit normal, shape ``[M]``.
        world_anchor: Per-contact world-frame anchor, shape ``[M]``.
        offset: Output ``n·anchor``, shape ``[M]``.
    """
    c = wp.tid()
    offset[c] = wp.float64(wp.dot(normal[c], world_anchor[c]))


@wp.kernel
def soft_contact_force_kernel(
    record: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
    rows_per_contact: int,
    lam_apply: wp.array[wp.float64],
    anchor: wp.array[wp.vec3],
    rigid_contact_max: int,
    contact_f: wp.array[wp.spatial_vector],
):
    """Wrench on the shape of each particle contact ``record[c]`` of ``Contacts``, from the forces [N] the step's
    last solve applied along the rows ``n, t1, t2`` (or ``n`` alone) of the contact: the force by the particle and its
    torque about the world origin, in world frame."""
    c = wp.tid()
    r = rows_per_contact * c
    f = normal[c] * float(lam_apply[r])
    if rows_per_contact == 3:
        f = f + tangent1[c] * float(lam_apply[r + 1]) + tangent2[c] * float(lam_apply[r + 2])
    contact_f[rigid_contact_max + record[c]] = wp.spatial_vector(-f, -wp.cross(anchor[c], f))


@wp.kernel
def add_contact_thickness_kernel(
    particle: wp.array[wp.int32],
    shape: wp.array[wp.int32],
    particle_radius: wp.array[wp.float32],
    shape_margin: wp.array[wp.float32],
    offset: wp.array[wp.float64],  # (M,) inout
):
    """Move each shape contact's plane out by the particle radius plus the shape margin.

    A contact without a shape (``shape < 0``) has no margin.
    """
    c = wp.tid()
    radius = particle_radius[particle[c]]
    if shape[c] >= 0:
        radius += shape_margin[shape[c]]
    offset[c] = offset[c] + wp.float64(radius)


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
    ``_ingest_contacts``: ``t1 = normalize(cross(n, axis_world))``,
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
    # otherwise world X (matches ``compute_tangent_basis`` exactly).
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
    the shape's local +Z direction in world frame. The leading minus makes
    a positive ``omega`` move the anchor along ``cross(normal, axis_world)``.

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
