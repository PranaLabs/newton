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
    dst: wp.array[wp.float32],
):
    """dst[i] = src[i][component]."""
    tid = wp.tid()
    dst[tid] = src[tid][component]


@wp.kernel
def insert_component_kernel(
    src: wp.array[wp.float32],
    component: int,
    dst: wp.array[wp.vec3],
):
    """dst[i][component] = src[i]."""
    tid = wp.tid()
    v = dst[tid]
    v[component] = src[tid]
    dst[tid] = v


@wp.kernel
def scale_by_diag_kernel(
    src: wp.array[wp.float32],
    diag: wp.array[wp.float32],
    dst: wp.array[wp.float32],
):
    """dst[i] = diag[i] * src[i] — used for D-inverse scaling in the linear solve."""
    tid = wp.tid()
    dst[tid] = diag[tid] * src[tid]


@wp.kernel
def apply_permutation_scalar_kernel(
    src: wp.array[wp.float32],
    perm: wp.array[wp.int32],
    dst: wp.array[wp.float32],
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
    """x_inertia = x_prev + dt·v_prev + (dt²/m)·(f_ext + g·m)."""
    tid = wp.tid()
    w_idx = wp.max(particle_world[tid], 0)
    g = gravity[w_idx]
    im = inv_mass[tid]
    a = f_ext[tid] * im + g * wp.step(-im)  # gravity active only for free particles
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

    Computes the deformation gradient `F = Ds · Dm_inv` (3×2), projects to the
    nearest rotation `P` via SVD-clamp, and scatters the per-particle
    contribution `w · Dm_inv · P^T` into the RHS vector via atomic_add.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tri_indices: Flat triangle indices, shape ``[3 * tri_count]``.
        tri_rest_inv: Per-triangle 2×2 rest-pose inverse (``Dm_inv``).
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
