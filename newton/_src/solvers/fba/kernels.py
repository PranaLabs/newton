# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics cloth solver)."""

import warp as wp

# 3x2 matrix type: Warp has no ``wp.mat32``.
mat32 = wp.types.matrix(shape=(3, 2), dtype=wp.float32)


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
