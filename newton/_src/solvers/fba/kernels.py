# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverFBA (projective-dynamics cloth solver)."""

import warp as wp


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
