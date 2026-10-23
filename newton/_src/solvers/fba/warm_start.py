# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warm start of SolverFBA's full-operator contact NSN across steps.

Each step's NSN starts from the multipliers ``lambda`` and the weights
``omega`` that the previous step ended with on the same rows. Rows are matched
by identity: a particle contact by its particle and shape, a rigid contact by
its shape pair and its rank among that pair's contacts, and a joint row by its
index; the particle constraint rows carry over while their count stays. A
matched contact keeps its normal multiplier, its friction force re-expressed in
its new tangents, and the weights of its three rows. Contacts without such an
identity start from zero: face and self contacts, and particle contacts that
share their particle and shape with another contact of their step.
"""

import numpy as np
import warp as wp

# Key of a row without identity; it sorts after every other key.
_NO_KEY = wp.constant(wp.int64(9223372036854775807))


@wp.kernel
def particle_contact_keys_kernel(
    particle: wp.array[wp.int32],
    shape: wp.array[wp.int32],
    extra_particle: wp.array2d[wp.int32],
    has_extra: int,
    shape_count: int,
    # outputs
    keys: wp.array[wp.int64],
    order: wp.array[wp.int32],
):
    """Identity of each particle contact, its particle and shape, and the contact's index.

    Contacts on several particles (face contacts) and self contacts have none.
    """
    c = wp.tid()
    key = _NO_KEY
    if shape[c] >= 0:
        plain = int(1)
        if has_extra != 0:
            if extra_particle[c, 0] >= 0:
                plain = 0
        if plain != 0:
            key = wp.int64(particle[c]) * wp.int64(shape_count) + wp.int64(shape[c])
    keys[c] = key
    order[c] = c


@wp.kernel
def index_keys_kernel(
    keys_in: wp.array[wp.int64],
    # outputs
    keys: wp.array[wp.int64],
    order: wp.array[wp.int32],
):
    """Copy given keys and number their owners."""
    i = wp.tid()
    keys[i] = keys_in[i]
    order[i] = i


@wp.kernel
def unique_keys_kernel(
    sorted_keys: wp.array[wp.int64],
    order: wp.array[wp.int32],
    count: int,
    # outputs
    unique: wp.array[wp.int32],
):
    """Flag the owners of keys that occur once among the step's sorted keys."""
    i = wp.tid()
    key = sorted_keys[i]
    flag = int(1)
    if key == _NO_KEY:
        flag = 0
    if i > 0:
        if sorted_keys[i - 1] == key:
            flag = 0
    if i + 1 < count:
        if sorted_keys[i + 1] == key:
            flag = 0
    unique[order[i]] = flag


@wp.func
def _find(sorted_keys: wp.array[wp.int64], count: int, key: wp.int64) -> int:
    """Position of ``key`` among ``count`` sorted keys if it occurs there once, else -1."""
    lo = int(0)
    hi = int(count)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo >= count:
        return -1
    if sorted_keys[lo] != key:
        return -1
    if lo + 1 < count:
        if sorted_keys[lo + 1] == key:
            return -1
    return lo


@wp.kernel
def restore_contact_rows_kernel(
    keys: wp.array[wp.int64],
    unique: wp.array[wp.int32],
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
    prev_keys: wp.array[wp.int64],
    prev_order: wp.array[wp.int32],
    prev_count: int,
    prev_lam: wp.array[wp.float64],
    prev_omega: wp.array[wp.float64],
    prev_tangent1: wp.array[wp.vec3],
    prev_tangent2: wp.array[wp.vec3],
    row_offset: int,
    # outputs
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
):
    """Start the rows ``[n, t1, t2]`` of each contact from the previous step's rows of the same contact."""
    c = wp.tid()
    if unique[c] == 0:
        return
    k = _find(prev_keys, prev_count, keys[c])
    if k < 0:
        return
    p = prev_order[k]
    friction = prev_lam[3 * p + 1] * wp.vec3d(prev_tangent1[p]) + prev_lam[3 * p + 2] * wp.vec3d(prev_tangent2[p])
    r = row_offset + 3 * c
    lam[r] = prev_lam[3 * p]
    lam[r + 1] = wp.dot(friction, wp.vec3d(tangent1[c]))
    lam[r + 2] = wp.dot(friction, wp.vec3d(tangent2[c]))
    for j in range(3):
        omega[r + j] = prev_omega[3 * p + j]


@wp.kernel
def restore_rows_kernel(
    keys: wp.array[wp.int64],
    unique: wp.array[wp.int32],
    prev_keys: wp.array[wp.int64],
    prev_order: wp.array[wp.int32],
    prev_count: int,
    prev_lam: wp.array[wp.float64],
    prev_omega: wp.array[wp.float64],
    row_offset: int,
    # outputs
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
):
    """Start each single row from the previous step's row of the same key."""
    i = wp.tid()
    if unique[i] == 0:
        return
    k = _find(prev_keys, prev_count, keys[i])
    if k < 0:
        return
    p = prev_order[k]
    lam[row_offset + i] = prev_lam[p]
    omega[row_offset + i] = prev_omega[p]


def _grow(array: wp.array | None, size: int, dtype, device) -> wp.array:
    """``array`` if it holds ``size`` entries, else a new array with 1.5x headroom."""
    if array is not None and array.shape[0] >= size:
        return array
    return wp.empty(max(int(1.5 * size), 16), dtype=dtype, device=device)


class RowGroup:
    """Rows of one kind: this step's keys and the previous step's sorted keys, multipliers, weights and tangents.

    ``rows`` is the number of rows per key: 3 for contacts, whose tangents re-express their friction, 1 for joint
    rows.
    """

    def __init__(self, rows: int, device):
        self.rows, self.device = rows, device
        self.count = 0  # this step's keys
        self.keys = self.sorted = self.order = self.unique = None
        self.prev_count = 0
        self.prev_sorted = self.prev_order = self.prev_lam = self.prev_omega = None
        self.prev_tangent1 = self.prev_tangent2 = None

    def begin(self, count: int, write_keys) -> None:
        """Take this step's ``count`` keys, written by ``write_keys(keys, order)``, and sort a copy of them."""
        self.count = count
        if count == 0:
            return
        device = self.device
        self.keys = _grow(self.keys, count, wp.int64, device)
        self.sorted = _grow(self.sorted, 2 * count, wp.int64, device)
        self.order = _grow(self.order, 2 * count, wp.int32, device)
        self.unique = _grow(self.unique, count, wp.int32, device)
        write_keys(self.keys, self.order)
        wp.copy(self.sorted, self.keys, count=count)
        wp.utils.radix_sort_pairs(self.sorted, self.order, count)
        wp.launch(unique_keys_kernel, count, [self.sorted, self.order, count], [self.unique], device=device)

    def restore(self, lam: wp.array, omega: wp.array, row_offset: int, tangents=None) -> None:
        """Start this step's rows in ``lam`` and ``omega`` from ``row_offset`` on from the previous step's."""
        if self.count == 0 or self.prev_count == 0:
            return
        prev = [self.prev_sorted, self.prev_order, self.prev_count, self.prev_lam, self.prev_omega]
        if self.rows == 3:
            inputs = [self.keys, self.unique, *tangents, *prev, self.prev_tangent1, self.prev_tangent2, row_offset]
            wp.launch(restore_contact_rows_kernel, self.count, inputs, [lam, omega], device=self.device)
        else:
            inputs = [self.keys, self.unique, *prev, row_offset]
            wp.launch(restore_rows_kernel, self.count, inputs, [lam, omega], device=self.device)

    def store(self, lam: wp.array, omega: wp.array, row_offset: int, tangents=None) -> None:
        """Keep this step's final rows, from ``row_offset`` on in ``lam`` and ``omega``, for the next step."""
        count, device = self.count, self.device
        self.prev_count = count
        if count == 0:
            return
        n = self.rows * count
        self.prev_lam = _grow(self.prev_lam, n, wp.float64, device)
        self.prev_omega = _grow(self.prev_omega, n, wp.float64, device)
        wp.copy(self.prev_lam, lam, src_offset=row_offset, count=n)
        wp.copy(self.prev_omega, omega, src_offset=row_offset, count=n)
        if self.rows == 3:
            self.prev_tangent1 = _grow(self.prev_tangent1, count, wp.vec3, device)
            self.prev_tangent2 = _grow(self.prev_tangent2, count, wp.vec3, device)
            wp.copy(self.prev_tangent1, tangents[0], count=count)
            wp.copy(self.prev_tangent2, tangents[1], count=count)
        # This step's sorted keys become the previous step's; their arrays trade places.
        self.prev_sorted, self.sorted = self.sorted, self.prev_sorted
        self.prev_order, self.order = self.order, self.prev_order


class ContactWarmStart:
    """The previous step's NSN rows of SolverFBA, by kind: particle contacts, rigid contacts, joint rows and the
    particle constraint rows."""

    def __init__(self, device):
        self.device = device
        self.particle = RowGroup(3, device)
        self.rigid = RowGroup(3, device)
        self.joint = RowGroup(1, device)
        self.bilateral_count = 0
        self.bilateral_lam = self.bilateral_omega = None

    def clear(self) -> None:
        """Forget the previous step's rows, so that the next step starts from zero."""
        for group in (self.particle, self.rigid, self.joint):
            group.prev_count = 0
        self.bilateral_count = 0

    def restore_bilateral(self, lam: wp.array, omega: wp.array, row_offset: int, count: int) -> None:
        """Start the particle constraint rows from the previous step's if their count is unchanged."""
        if count and count == self.bilateral_count:
            wp.copy(lam, self.bilateral_lam, dest_offset=row_offset, count=count)
            wp.copy(omega, self.bilateral_omega, dest_offset=row_offset, count=count)

    def store_bilateral(self, lam: wp.array, omega: wp.array, row_offset: int, count: int) -> None:
        """Keep the particle constraint rows of this step."""
        self.bilateral_count = count
        if count == 0:
            return
        self.bilateral_lam = _grow(self.bilateral_lam, count, wp.float64, self.device)
        self.bilateral_omega = _grow(self.bilateral_omega, count, wp.float64, self.device)
        wp.copy(self.bilateral_lam, lam, src_offset=row_offset, count=count)
        wp.copy(self.bilateral_omega, omega, src_offset=row_offset, count=count)


def rigid_contact_keys(shape: np.ndarray, shape_count: int) -> np.ndarray:
    """Identity of each rigid contact from the shape pairs of the contacts in their deterministic order: the pair
    and the contact's rank among the pair's contacts.

    The collision pipeline sorts a pair's contacts, so the rank keeps a contact's identity while the pair keeps its
    contact count, as a resting object's contacts do; where the count changes, the ranks shift, and some contacts
    start from their neighbors' multipliers, which the step's iterations then correct.
    """
    pair = shape[:, 0].astype(np.int64) * shape_count + shape[:, 1]
    n = len(pair)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
    rank = np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))
    return pair * (1 << 20) + rank
