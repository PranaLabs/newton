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
def svd_3x2_frame(F: mat32):
    """The singular vectors and values of the 3x2 deformation gradient ``F = s_a u_a v_a^T + s_b u_b v_b^T``.

    The right singular vectors ``v_i`` come from the 2x2 eigenproblem of ``F^T F`` (``wp.svd2``).
    ``u_a = F v_a / |F v_a|`` takes the larger of the two images, and ``u_b`` is ``F v_b`` made orthogonal to
    ``u_a`` and normalized rather than divided by its singular value: when a triangle is crushed, that singular
    value vanishes and float32 ``F^T F`` resolves it no better than ``1e-4`` of the larger one, so dividing would
    amplify rounding noise into a huge frame. Below ``1e-5`` of the larger image, ``u_b`` is any unit vector
    orthogonal to ``u_a``, and ``F = 0`` takes an arbitrary frame. The singular values ``s_a >= s_b >= 0`` are the
    lengths of the two images.

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
    return ua, ub, va, vb, na, nb


@wp.func
def svd_3x2(F: mat32) -> mat32:
    """ARAP projection: the 3x2 matrix ``P = u_a v_a^T + u_b v_b^T`` with orthonormal columns nearest to the
    deformation gradient ``F``, from :func:`svd_3x2_frame`."""
    ua, ub, va, vb, _sa, _sb = svd_3x2_frame(F)
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
    accuracy through an fp32 SVD. ``wp.svd3`` returns rotations ``U`` and
    ``V``, with the smallest singular value negative where ``det(F) < 0``, so
    ``R = U * V^T`` is the nearest proper rotation also to an inverted ``F``.

    Args:
        F: 3x3 deformation gradient.

    Returns:
        R: 3x3 proper rotation (nearest rotation to F).
    """
    U, _sigma, V = wp.svd3(F)
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
    rotations ``U`` and ``V`` and signed singular values, the smallest
    negative for an inverted tet, which the projection moves toward 1.
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
        # No stiffness: nothing to project.
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
        if dir_dot <= wp.float64(0.0):
            q = grad
            history_count = 0
            inf_norm = wp.max(wp.abs(grad[0]), wp.abs(grad[1]))
            if inf_norm > wp.float64(0.0):
                alpha_init = wp.min(wp.float64(1.0), wp.float64(1.0) / inf_norm)
            else:
                alpha_init = wp.float64(1.0)

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
    and flips the sign of a negative sigma_2, which ``wp.svd3`` returns for an
    inverted tet, so that the projection restores its orientation.
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
        s2 = -s2
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
        if dir_dot <= wp.float64(0.0):
            q = grad
            history_count = 0
            inf_norm = wp.max(wp.max(wp.abs(grad[0]), wp.abs(grad[1])), wp.abs(grad[2]))
            if inf_norm > wp.float64(0.0):
                alpha_init = wp.min(wp.float64(1.0), wp.float64(1.0) / inf_norm)
            else:
                alpha_init = wp.float64(1.0)

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
        # No stiffness: nothing to project.
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
    the nearest rotation ``P`` by setting its singular values to 1, and writes
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
def project_corotational_sigma(sigma: wp.vec2, mu: float, lam: float) -> wp.vec2:
    """Closed-form 2D corotational local projection on singular values.

    Minimises E(s) = mu*||s-I||^2 + (lam/2)*tr(s-I)^2  subject to PD's
    quadratic penalty (k/2)*||s-s0||^2 with k = 2*mu.  The resulting 2x2
    linear system has the closed-form solution below.  Both singular
    values share the same energy-form term (symmetric trace), which is
    the standard corotational formulation in elasticity references
    (Sifakis 2012, Bouaziz 2014, etc.): the trace term sums both singular
    values, ``tr(s-I) = (s_0-1) + (s_1-1)``.

    Args:
        sigma: Singular values of ``F``.
        mu: Second Lamé parameter (shear modulus) [Pa].
        lam: First Lamé parameter [Pa].

    Returns:
        Projected singular-value pair ``(sigma_proj_0, sigma_proj_1)``.
    """
    s0 = sigma[0]
    s1 = sigma[1]

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
    singular values with :func:`svd_3x2_frame`, applies the closed-form corotational
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

    u0, u1, v0, v1, s0, s1 = svd_3x2_frame(F)
    sigma_proj = project_corotational_sigma(wp.vec2(s0, s1), mu, lam)

    p_col0 = u0 * (sigma_proj[0] * v0[0]) + u1 * (sigma_proj[1] * v1[0])
    p_col1 = u0 * (sigma_proj[0] * v0[1]) + u1 * (sigma_proj[1] * v1[1])

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
    contributions to its three particles for the per-particle gather. The
    singular values of F from :func:`svd_3x2_frame` are cast fp32 -> fp64
    before L-BFGS; the projected sigma is cast back to fp32 for the P
    reconstruction.
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

    u0, u1, v0, v1, s0, s1 = svd_3x2_frame(F)
    # The Neo-Hookean energy diverges at a vanishing singular value.
    sigma_init = wp.vec2d(wp.float64(wp.max(s0, 1.0e-10)), wp.float64(wp.max(s1, 1.0e-10)))
    mu_d = wp.float64(mu)
    lam_d = wp.float64(lam)
    sigma_proj_d = project_neohookean_sigma2d_lbfgs(sigma_init, mu_d, lam_d)
    sp0 = wp.float32(sigma_proj_d[0])
    sp1 = wp.float32(sigma_proj_d[1])

    p_col0 = u0 * (sp0 * v0[0]) + u1 * (sp1 * v1[0])
    p_col1 = u0 * (sp0 * v0[1]) + u1 * (sp1 * v1[1])

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
    edge_norm: wp.array[wp.float32],  # rest curvature magnitude, from the edge's rest angle
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
        edge_norm: Per-edge rest curvature magnitude, from its ``edge_rest_angle``.
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
