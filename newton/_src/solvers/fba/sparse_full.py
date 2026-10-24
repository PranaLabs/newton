# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Instance-local exact-zero sparse storage for the experimental Full NSN path.

The full contact operator and fp64 nonzero entries are retained. Sparse
reductions can differ from dense reductions by floating-point roundoff.
"""

import functools
import inspect
import time
import weakref
from types import MethodType

import numpy as np
import warp as wp
import warp.sparse as wps

from . import kernels as K
from . import nsn_pcr_solver as P
from .kernels import _WI_TILE_M, compute_wi_kernel


@wp.kernel
def _count_rows(w: wp.array2d[wp.float64], n: int, counts: wp.array[wp.int32]):
    i = wp.tid()
    count = int(0)
    for j in range(n):
        if i == j or w[i, j] != wp.float64(0.0):
            count += 1
    counts[i] = count


@wp.kernel
def _fill_rows(
    w: wp.array2d[wp.float64],
    n: int,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
):
    i = wp.tid()
    k = offsets[i]
    for j in range(n):
        if i == j or w[i, j] != wp.float64(0.0):
            columns[k] = j
            values[k] = w[i, j]
            k += 1


@wp.kernel
def _regularize_diagonal(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], factor: wp.float64, values: wp.array[wp.float64]
):
    i = wp.tid()
    for k in range(offsets[i], offsets[i + 1]):
        if columns[k] == i:
            values[k] = factor * values[k]


def dense_to_exact_bsr(w, n):
    """Preserve all nonzero fp64 values, their signs and each zero diagonal."""
    counts = wp.empty(n, dtype=wp.int32, device=w.device)
    offsets = wp.zeros(n + 1, dtype=wp.int32, device=w.device)
    wp.launch(_count_rows, n, inputs=[w, n], outputs=[counts], device=w.device)
    wp.utils.array_scan(counts, offsets[1:], inclusive=True)
    nnz = int(offsets[n : n + 1].numpy()[0])
    out = wps.bsr_zeros(n, n, block_type=wp.float64, device=w.device)
    out.offsets = offsets
    out.columns = wp.empty(nnz, dtype=wp.int32, device=w.device)
    out.values = wp.empty(nnz, dtype=wp.float64, device=w.device)
    out.nnz = nnz
    wp.launch(_fill_rows, n, inputs=[w, n, offsets], outputs=[out.columns, out.values], device=w.device)
    return out


def _same_structure(w):
    a = wps.bsr_zeros(w.nrow, w.ncol, block_type=wp.float64, device=w.device)
    a.offsets, a.columns, a.nnz = w.offsets, w.columns, w.nnz
    a.values = wp.empty(w.nnz, dtype=wp.float64, device=w.device)
    return a


def solve_sparse_with_block(pcr, a, b, out, block_precond=None):
    """Native sparse PCR, plus the native dense path's joint preconditioner.

    With a callback, initialize z=P r identically to dense PCR and use the
    existing protected iteration, which already calls _block_precond. The
    iteration count, tolerance, chart-curvature cutoff and check cadence stay
    unchanged; sparse summation order may cause floating-point differences.
    """
    if block_precond is None:
        return pcr.solve_sparse(a, b, out)

    n = int(b.shape[0])
    if n <= 0 or n > pcr.max_n or a.shape != (n, n) or out.shape != (n,):
        raise ValueError("Sparse PCR dimensions disagree")
    dev = pcr.device
    pcr._x.zero_()
    dot_b = float(wp.utils.array_inner(b, b, count=n))
    if dot_b == 0.0:
        wp.copy(out, pcr._x, count=n)
        return 0
    diag = wps.bsr_get_diag(a, out=pcr._diag)
    wp.launch(P._precond_from_bsr_diag_kernel, n, inputs=[diag], outputs=[pcr._precond], device=dev)
    wp.copy(pcr._r, b, count=n)
    wp.launch(P._apply_precond_kernel, n, inputs=[pcr._r, pcr._precond], outputs=[pcr._s], device=dev)
    pcr._block_precond = block_precond
    block_precond(pcr._r, pcr._s)
    wp.copy(pcr._d, pcr._s, count=n)
    wps.bsr_mv(a, pcr._d[:n], pcr._q[:n])
    wp.utils.array_inner(pcr._s, pcr._q, out=pcr._rho_d, count=n)
    pcr._stopped_d.zero_()
    limit, count = min(pcr.max_iter, n), 0
    tolerance, residual = pcr.tol * pcr.tol * dot_b, dot_b
    while count < limit and residual > tolerance:
        end = min(count + pcr.check_every, limit)
        while count < end:
            pcr._pcr_iter_eager(None, n, n, dev, W_bsr=a, diag=diag)
            count += 1
        wp.utils.array_inner(pcr._r, pcr._r, out=pcr._r_norm_d, count=n)
        residual = float(pcr._r_norm_d.numpy()[0])
        if pcr._stopped_d.numpy()[0] != 0.0:
            break
    wp.copy(out, pcr._x, count=n)
    return count


def _driver(self, args, sparse_w, sparse_a):

    a = args
    m = a["m_contacts"]
    if m is None:
        m = len(a["mu"]) if a["mu"] is not None else (a["mu_device"].shape[0] if a["mu_device"] is not None else 0)
    m, bilateral = int(m), int(a["n_bilateral"])
    n = 3 * m + bilateral
    if not n:
        return tuple(np.zeros(0, np.float64) for _ in range(3))
    # Reuse the native vectors, but never request its dense A allocation.
    self._ensure_nsn_inner_buffers(n, alloc_2d=False)
    dense_a = getattr(self, "_nsn_a_schur_d", None)
    self._nsn_inner_has_2d = dense_a is not None and min(dense_a.shape) >= self._nsn_inner_n
    self._ensure_pcr_solver(n)
    dev, dt = self._device, wp.float64(a["dt"])

    def source(host_name, device_name, size):
        if size == 0:
            return wp.empty(0, dtype=wp.float64, device=dev)
        d = a[device_name]
        if d is not None and d.shape[0] >= size:
            return d[:size]
        return wp.array(np.asarray(a[host_name], np.float64)[:size], dtype=wp.float64, device=dev)

    mu, r, pene0 = source("mu", "mu_device", m), source("r", "r_device", n), source("pene0", "pene0_device", n)
    for host_name, device_name, target in (
        ("lam_init", "lam_init_device", self._nsn_lam_d),
        ("omega_init", "omega_init_device", self._nsn_omega_d),
    ):
        d, h = a[device_name], a[host_name]
        if d is not None and d.shape[0] >= n:
            wp.copy(target, d, count=n)
        elif h is not None and h.shape == (n,):
            wp.copy(target, wp.array(h, dtype=wp.float64, device=dev), count=n)
        else:
            target[:n].zero_()
        if target.size > n:
            target[n:].zero_()
    lam, omega, compliance = self._nsn_lam_d[:n], self._nsn_omega_d[:n], self._nsn_compliance_d[:n]
    h, pene, rhs, dlam = self._nsn_h_d[:n], self._nsn_penetration_d[:n], self._nsn_rhs_d[:n], self._nsn_dlam_d[:n]
    scratch = getattr(self, "_full_sparse_scratch", None)
    if scratch is None or scratch[0].size < n:
        scratch = tuple(wp.empty(n, dtype=wp.float64, device=dev) for _ in range(3))
        self._full_sparse_scratch = scratch
    ol, wol, diag = (x[:n] for x in scratch)
    wps.bsr_get_diag(sparse_w, out=diag)
    wp.launch(K.precond_from_bsr_diag_coulomb_kernel, n, inputs=[diag, dt], outputs=[self._nsn_precond_d], device=dev)
    counts = []
    for _ in range(int(a["max_iters"])):
        wp.launch(K.compute_omega_times_lam_kernel, n, inputs=[omega, lam], outputs=[ol], device=dev)
        wps.bsr_mv(sparse_w, ol, wol)
        wp.launch(K.compute_penetration_sparse_kernel, n, inputs=[r, wol, dt], outputs=[pene], device=dev)
        wp.launch(
            K.compute_frictional_fb_kernel,
            m,
            inputs=[pene, lam, mu, self._nsn_precond_d, pene0, dt],
            outputs=[omega, compliance, h],
            device=dev,
        )
        if bilateral:
            wp.launch(
                K.bilateral_law_kernel,
                bilateral,
                inputs=[pene0, wp.int32(3 * m)],
                outputs=[omega, compliance, h],
                device=dev,
            )
        wp.launch(
            K.build_a_schur_lite_bsr_kernel,
            n,
            inputs=[sparse_w.offsets, sparse_w.columns, sparse_w.values, omega, compliance],
            outputs=[sparse_a.values],
            device=dev,
        )
        if a["contact_regularization"] > 0.0:
            wp.launch(
                _regularize_diagonal,
                3 * m,
                inputs=[sparse_a.offsets, sparse_a.columns, wp.float64(1.0 + a["contact_regularization"])],
                outputs=[sparse_a.values],
                device=dev,
            )
        wp.launch(K.compute_omega_times_lam_kernel, n, inputs=[omega, lam], outputs=[ol], device=dev)
        wps.bsr_mv(sparse_w, ol, wol)
        wp.launch(K.compute_nsn_rhs_lite_bsr_kernel, n, inputs=[h, omega, pene0, r, wol, dt], outputs=[rhs], device=dev)
        # Unlike production's legacy catch-and-break, fail loudly in this
        # validation prototype so a failed sparse solve cannot look complete.
        counts.append(solve_sparse_with_block(self._pcr_solver, sparse_a, rhs, dlam, a["block_precond"]))
        wp.launch(K.axpy_lambda_kernel, n, inputs=[dlam], outputs=[lam], device=dev)
        wp.launch(K.coulomb_box_clamp_kernel, m, inputs=[mu], outputs=[lam], device=dev)
    if self.lambda_cap is not None:
        wp.launch(
            K.lambda_cap_clip_kernel,
            3 * m,
            inputs=[wp.float64(self.lambda_cap / (a["dt"] ** 2))],
            outputs=[lam],
            device=dev,
        )
    wp.launch(K.compute_lam_apply_kernel, n, inputs=[lam, omega, dt], outputs=[self._nsn_lam_apply_d], device=dev)
    self._full_sparse_last_stats.update(pcr_iterations=counts, block_preconditioner=a["block_precond"] is not None)
    if a["skip_download"]:
        return tuple(np.zeros(0, np.float64) for _ in range(3))
    return tuple(x.numpy()[:n].copy() for x in (self._nsn_lam_d, self._nsn_omega_d, self._nsn_lam_apply_d))


def _build_schur_exact_sparse(
    self,
    row_particle,
    row_particle_d,
    row_dir_d,
    row_alpha_d,
    total_rows,
    compute_wi_tiled_kernel,
    densify_S_iso_scaled_kernel,
    download=True,
):

    # The caller supplies the union of every primary, face, and bilateral slot.
    isodofs = np.unique(row_particle.astype(np.int64)).astype(np.int32)
    count = int(isodofs.size)
    tile_size = int(_WI_TILE_M)
    padded_count = ((count + tile_size - 1) // tile_size) * tile_size
    if padded_count == 0:
        return None if not download else np.zeros((total_rows, total_rows), dtype=np.float64)

    ranks = np.full(self.n, -1, dtype=np.int32)
    ranks[isodofs] = np.arange(count, dtype=np.int32)
    # Runtime A^-1 uses perm, despite the older sparse kernel's stale docstring.
    factor_columns = self._perm.numpy()[isodofs].astype(np.int32)

    # Retain Newton's capacity convention so the original method can be restored.
    if getattr(self, "_isodofs_cap", 0) < padded_count:
        capacity = max(padded_count, int(getattr(self, "_isodofs_cap", 0) * 1.5) + 1)
        capacity = ((capacity + tile_size - 1) // tile_size) * tile_size
        self._isodofs_cap = capacity
        self._isodof_perm_d = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self._Wi_device_d = wp.empty((capacity, capacity), dtype=wp.float64, device=self.device)
    if not hasattr(self, "_isodof_rank_d") or self._isodof_rank_d.shape[0] != self.n:
        self._isodof_rank_d = wp.empty(self.n, dtype=wp.int32, device=self.device)
    self._isodof_perm_d.assign(factor_columns)
    self._isodof_rank_d.assign(ranks)

    capacity = int(self._Wi_device_d.shape[0])
    wi = self._Wi_device_d
    if capacity != padded_count:
        wi = wp.array(
            ptr=wi.ptr,
            dtype=wp.float64,
            shape=(padded_count, padded_count),
            strides=(capacity * 8, 8),
            device=self.device,
        )
    # Preserve zero padding; active entries are all written by the exact kernel.
    wi.zero_()
    wp.launch(
        compute_wi_kernel,
        dim=(count, count),
        inputs=[
            self._ST_bsr.offsets,
            self._ST_bsr.columns,
            self._ST_bsr.values,
            self._Dinv,
            self._isodof_perm_d,
        ],
        outputs=[wi],
        device=self.device,
    )
    self._Wi_v = wi
    self._compose_schur()
    if not download:
        return None
    return self._W_device_d.numpy()[:total_rows, :total_rows].copy()


def configure_sparse_full(solver):
    """Configure one solver before its first step, without process-global patches."""
    original = type(solver)._solve_nsn_coulomb_gpu
    if getattr(original, "_full_sparse_prototype", False):
        raise RuntimeError("Full sparse NSN prototype already installed")
    signature = inspect.signature(original)
    revisions = {}
    generation = object()

    def changed(array):
        ptr = int(array.ptr)
        previous = revisions.get(ptr)
        revision = previous[1] + 1 if previous is not None else 1
        revisions[ptr] = (weakref.ref(array), revision)

    def wrap_change(obj, name, field):
        old = getattr(obj, name)

        @functools.wraps(old)
        def inner(*args, **kwargs):
            result = old(*args, **kwargs)
            value = getattr(obj, field, None)
            if value is not None:
                changed(value)
            return result

        setattr(obj, name, inner)

    def configure_linear(linear):
        linear._build_schur_isodof = MethodType(_build_schur_exact_sparse, linear)
        wrap_change(linear, "_compose_schur", "_W_device_d")
        wrap_change(linear, "build_schur_complement", "_W_device_d")

    solver._configure_sparse_linear = configure_linear
    wrap_change(solver, "_compose_rigid_delassus", "_rc_W_d")

    original_ensure = type(solver)._ensure_nsn_inner_buffers

    @functools.wraps(original_ensure)
    def ensure(self, n_rows, alloc_2d=True):
        # SolverFBA.step reserves scratch before it calls the inner driver.
        # Restrict this interception to the Full frictional path replaced here.
        if getattr(self, "friction", False) and getattr(self, "nsn_schur_mode", None) == "full":
            alloc_2d = False
        result = original_ensure(self, n_rows, alloc_2d=alloc_2d)
        dense_a = getattr(self, "_nsn_a_schur_d", None)
        self._nsn_inner_has_2d = dense_a is not None and min(dense_a.shape) >= self._nsn_inner_n
        return result

    solver._ensure_nsn_inner_buffers = MethodType(ensure, solver)

    @functools.wraps(original)
    def sparse(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        m = a["m_contacts"]
        if m is None:
            m = len(a["mu"]) if a["mu"] is not None else (a["mu_device"].shape[0] if a["mu_device"] is not None else 0)
        n = 3 * int(m) + int(a["n_bilateral"])
        if not n:
            return tuple(np.zeros(0, np.float64) for _ in range(3))
        dense = a["W_device"]
        if dense is None or min(dense.shape) < n:
            dense = wp.array(np.asarray(a["W"], np.float64), dtype=wp.float64, device=self._device)
        registered = revisions.get(int(dense.ptr))
        revision = registered[1] if registered is not None and registered[0]() is not None else None
        key = (generation, int(dense.ptr), n, tuple(dense.strides), revision)
        cache = getattr(self, "_full_sparse_cache", None)
        cached = key[-1] is not None and cache is not None and cache[0] == key
        started = time.perf_counter()
        if not cached:
            w = dense_to_exact_bsr(dense, n)
            cache = (key, w, _same_structure(w))
            self._full_sparse_cache = cache
        self._full_sparse_last_stats = {
            "rows": n,
            "nnz": int(cache[1].nnz),
            "exact_zero_fraction": 1.0 - cache[1].nnz / (n * n),
            "conversion_cached": cached,
            "conversion_wall_s": time.perf_counter() - started,
            "sparse_W_A_bytes": int(2 * cache[1].nnz * 8 + cache[1].nnz * 4 + (n + 1) * 4),
            "dense_W_retained": True,
            "threshold_pruning": False,
        }
        return _driver(self, a, cache[1], cache[2])

    solver._solve_nsn_coulomb_gpu = MethodType(sparse, solver)
