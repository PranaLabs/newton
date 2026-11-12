# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warm start of SolverFBA's full-operator contact NSN across steps.

Each step's NSN starts from the multipliers ``lambda`` and the weights
``omega`` that the previous step ended with on the same rows. Rows are matched
by identity: a particle contact by its particle and shape, a rigid contact by
its shape pair and its anchors (see :class:`RigidContactGroup`), and a joint row
by its index; particle constraint rows carry over while their world's rows stay
unchanged. A matched contact keeps its normal multiplier, its friction force
re-expressed in its new tangents, and the weights of its three rows. Contacts
without such an identity start from zero: face and self contacts, particle
contacts that share their particle and shape with another contact of their
step, and rigid contacts that find no previous contact of their pair to continue.
"""

import numpy as np
import warp as wp

from ...core.reset import reset_world_selected

# Key of a row without identity; it sorts after every other key.
_NO_KEY = wp.constant(wp.int64(9223372036854775807))

# Rigid contact keys hold the shape pair above this many bits and the contact's rank in the pair below.
_RANK_BITS = wp.constant(20)
_RANK_MASK = wp.constant(wp.int64((1 << 20) - 1))


@wp.kernel
def clear_group_worlds_kernel(
    keys: wp.array[wp.int64],
    order: wp.array[wp.int32],
    entity_world: wp.array[wp.int32],
    kind: int,
    shape_count: int,
    rows_per_key: int,
    mask: wp.array[wp.bool],
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
):
    i = wp.tid()
    key = keys[i]
    if key == _NO_KEY:
        return
    world = int(0)
    if kind == 0:
        world = entity_world[int(key // wp.int64(shape_count))]
    elif kind == 1:
        pair = key >> wp.int64(_RANK_BITS)
        world = wp.max(
            entity_world[int(pair // wp.int64(shape_count))], entity_world[int(pair % wp.int64(shape_count))]
        )
    else:
        world = entity_world[int(key)]
    if reset_world_selected(world, mask, mask.shape[0] - 1):
        for j in range(rows_per_key):
            r = rows_per_key * order[i] + j
            lam[r] = wp.float64(0.0)
            omega[r] = wp.float64(0.0)


@wp.kernel
def clear_bilateral_worlds_kernel(
    world: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
):
    i = wp.tid()
    if reset_world_selected(world[i], mask, mask.shape[0] - 1):
        lam[i] = wp.float64(0.0)
        omega[i] = wp.float64(0.0)


@wp.kernel
def remap_bilateral_kernel(
    previous_lam: wp.array[wp.float64],
    previous_omega: wp.array[wp.float64],
    rows: wp.array[wp.int32],
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
):
    i = wp.tid()
    lam[i] = wp.float64(0.0)
    omega[i] = wp.float64(0.0)
    if rows[i] >= 0:
        lam[i] = previous_lam[rows[i]]
        omega[i] = previous_omega[rows[i]]


@wp.kernel
def clear_anchor_particles_kernel(
    world: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    step: wp.array[wp.int32],
):
    i = wp.tid()
    if reset_world_selected(world[i], mask, mask.shape[0] - 1):
        step[i] = -2


@wp.kernel
def clear_anchor_contacts_kernel(
    keys: wp.array[wp.int64],
    world: wp.array[wp.int32],
    shape_count: int,
    mask: wp.array[wp.bool],
    pressed: wp.array[wp.int32],
):
    i = wp.tid()
    if keys[i] != _NO_KEY:
        p = int(keys[i] // wp.int64(shape_count + 1))
        if reset_world_selected(world[p], mask, mask.shape[0] - 1):
            pressed[i] = 0


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
def _lower_bound(sorted_keys: wp.array[wp.int64], count: int, key: wp.int64) -> int:
    """Position of the first of ``count`` sorted keys that is not below ``key``."""
    lo = int(0)
    hi = int(count)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.func
def _find(sorted_keys: wp.array[wp.int64], count: int, key: wp.int64) -> int:
    """Position of ``key`` among ``count`` sorted keys if it occurs there once, else -1."""
    lo = _lower_bound(sorted_keys, count, key)
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


# A rigid contact continues only a previous contact of its shape pair whose normal is within acos(0.8), about 37
# degrees, of its own.
_MATCH_NORMAL_DOT = 0.8

# A claim no contact has made: past every packed distance and contact.
_NO_CLAIM = wp.constant(wp.int64(9223372036854775807))

# The distance part of the claims that pair a contact with a leftover previous contact, past every nearest claim.
_LEFTOVER_CLAIM = wp.constant(wp.int64(2100000000))

# A pair with more than this many contact pairings leaves its leftover contacts to start from zero.
_MAX_LEFTOVER_WORK = wp.constant(1024)


@wp.func
def _anchor_distance(anchor: wp.array2d[wp.vec3], c: int, prev_anchor: wp.array2d[wp.vec3], p: int) -> float:
    """Distance of contact ``c``'s anchors from previous contact ``p``'s, summed over the two sides."""
    return wp.length(anchor[c, 0] - prev_anchor[p, 0]) + wp.length(anchor[c, 1] - prev_anchor[p, 1])


@wp.kernel
def match_rigid_contacts_kernel(
    keys: wp.array[wp.int64],
    anchor: wp.array2d[wp.vec3],
    normal: wp.array[wp.vec3],
    prev_keys: wp.array[wp.int64],
    prev_order: wp.array[wp.int32],
    prev_count: int,
    prev_anchor: wp.array2d[wp.vec3],
    prev_normal: wp.array[wp.vec3],
    min_normal_dot: float,
    # outputs
    match: wp.array[wp.int32],
    packed: wp.array[wp.int64],
    claim: wp.array[wp.int64],
):
    """Find each contact's nearest previous contact of its shape pair and claim it with the distance and the
    contact, so that the nearest of the contacts that find the same previous one wins it."""
    c = wp.tid()
    match[c] = -1
    pair = keys[c] >> wp.int64(_RANK_BITS)
    end = (pair + wp.int64(1)) << wp.int64(_RANK_BITS)
    k = _lower_bound(prev_keys, prev_count, pair << wp.int64(_RANK_BITS))
    best = int(-1)
    best_d = float(0.0)
    while k < prev_count:
        if prev_keys[k] >= end:
            break
        p = prev_order[k]
        if wp.dot(normal[c], prev_normal[p]) >= min_normal_dot:
            d = _anchor_distance(anchor, c, prev_anchor, p)
            if best < 0 or d < best_d:
                best = p
                best_d = d
        k += 1
    if best < 0:
        return
    # The distance in nanometers above the contact; equal distances go to the lower contact.
    q = wp.int64(wp.min(best_d * 1.0e9, 2.0e9))
    key = (q << wp.int64(32)) | wp.int64(c)
    match[c] = best
    packed[c] = key
    wp.atomic_min(claim, best, key)


@wp.kernel
def match_rigid_leftovers_kernel(
    keys: wp.array[wp.int64],
    count: int,
    anchor: wp.array2d[wp.vec3],
    normal: wp.array[wp.vec3],
    prev_keys: wp.array[wp.int64],
    prev_order: wp.array[wp.int32],
    prev_count: int,
    prev_anchor: wp.array2d[wp.vec3],
    prev_normal: wp.array[wp.vec3],
    min_normal_dot: float,
    # outputs
    match: wp.array[wp.int32],
    packed: wp.array[wp.int64],
    claim: wp.array[wp.int64],
):
    """Pair each shape pair's contacts that won no previous contact with the pair's previous contacts that no
    contact won, nearest first, in the thread of the pair's first contact."""
    c0 = wp.tid()
    if (keys[c0] & _RANK_MASK) != wp.int64(0):
        return
    pair = keys[c0] >> wp.int64(_RANK_BITS)
    c1 = c0 + 1
    while c1 < count:
        if (keys[c1] >> wp.int64(_RANK_BITS)) != pair:
            break
        c1 += 1
    p0 = _lower_bound(prev_keys, prev_count, pair << wp.int64(_RANK_BITS))
    p1 = _lower_bound(prev_keys, prev_count, (pair + wp.int64(1)) << wp.int64(_RANK_BITS))
    if p0 == p1 or (c1 - c0) * (p1 - p0) > _MAX_LEFTOVER_WORK:
        return
    while True:
        best_c = int(-1)
        best_p = int(-1)
        best_d = float(0.0)
        for c in range(c0, c1):
            m = match[c]
            won = False
            if m >= 0:
                won = claim[m] == packed[c]
            if not won:
                for k in range(p0, p1):
                    p = prev_order[k]
                    if claim[p] == _NO_CLAIM and wp.dot(normal[c], prev_normal[p]) >= min_normal_dot:
                        d = _anchor_distance(anchor, c, prev_anchor, p)
                        if best_c < 0 or d < best_d:
                            best_c = c
                            best_p = p
                            best_d = d
        if best_c < 0:
            break
        key = (_LEFTOVER_CLAIM << wp.int64(32)) | wp.int64(best_c)
        match[best_c] = best_p
        packed[best_c] = key
        claim[best_p] = key


@wp.kernel
def restore_matched_contact_rows_kernel(
    match: wp.array[wp.int32],
    packed: wp.array[wp.int64],
    claim: wp.array[wp.int64],
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
    prev_lam: wp.array[wp.float64],
    prev_omega: wp.array[wp.float64],
    prev_tangent1: wp.array[wp.vec3],
    prev_tangent2: wp.array[wp.vec3],
    row_offset: int,
    # outputs
    lam: wp.array[wp.float64],
    omega: wp.array[wp.float64],
):
    """Start the rows ``[n, t1, t2]`` of each contact that won its previous contact from that contact's rows."""
    c = wp.tid()
    p = match[c]
    if p < 0:
        return
    if claim[p] != packed[c]:
        return
    friction = prev_lam[3 * p + 1] * wp.vec3d(prev_tangent1[p]) + prev_lam[3 * p + 2] * wp.vec3d(prev_tangent2[p])
    r = row_offset + 3 * c
    lam[r] = prev_lam[3 * p]
    lam[r + 1] = wp.dot(friction, wp.vec3d(tangent1[c]))
    lam[r + 2] = wp.dot(friction, wp.vec3d(tangent2[c]))
    for j in range(3):
        omega[r + j] = prev_omega[3 * p + j]


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


class RigidContactGroup(RowGroup):
    """Rigid contact rows, keyed by :func:`rigid_contact_keys` and matched by their anchors.

    A shape pair's contacts change in number as a resting object's manifold gains or loses a point, and their order
    in the pair follows anchors that can tie, so a contact's rank does not keep its identity. Each contact instead
    continues the previous contact of its pair whose anchors lie nearest to its own, summed over both sides, unless
    a nearer contact of the step continues that one. The pair's other contacts then take its other previous
    contacts, nearest first, so that a manifold that swaps a point for another keeps its forces; the rest start from
    zero. A contact does not continue a previous one whose normal points more than about 37 degrees away.
    """

    def __init__(self, device):
        super().__init__(3, device)
        self.prev_anchor = self.prev_normal = None
        self.match = self.packed = self.claim = None

    def restore(self, lam, omega, row_offset, tangents=None, anchor=None, normal=None) -> None:
        """Start this step's rows in ``lam`` and ``omega`` from ``row_offset`` on from the previous step's, with the
        contacts' ``anchor`` per side and ``normal`` in their order."""
        if self.count == 0 or self.prev_count == 0:
            return
        device = self.device
        self.match = _grow(self.match, self.count, wp.int32, device)
        self.packed = _grow(self.packed, self.count, wp.int64, device)
        self.claim = _grow(self.claim, self.prev_count, wp.int64, device)
        self.claim.fill_(_NO_CLAIM)
        prev = [self.prev_sorted, self.prev_order, self.prev_count, self.prev_anchor, self.prev_normal]
        outputs = [self.match, self.packed, self.claim]
        inputs = [self.keys, anchor, normal, *prev, _MATCH_NORMAL_DOT]
        wp.launch(match_rigid_contacts_kernel, self.count, inputs, outputs, device=device)
        inputs = [self.keys, self.count, anchor, normal, *prev, _MATCH_NORMAL_DOT]
        wp.launch(match_rigid_leftovers_kernel, self.count, inputs, outputs, device=device)
        inputs = [self.match, self.packed, self.claim, *tangents, self.prev_lam, self.prev_omega]
        inputs += [self.prev_tangent1, self.prev_tangent2, row_offset]
        wp.launch(restore_matched_contact_rows_kernel, self.count, inputs, [lam, omega], device=device)

    def store(self, lam, omega, row_offset, tangents=None, anchor=None, normal=None) -> None:
        """Keep this step's final rows, from ``row_offset`` on in ``lam`` and ``omega``, and the contacts' anchors
        and normals for the next step."""
        super().store(lam, omega, row_offset, tangents)
        count, device = self.count, self.device
        if count == 0:
            return
        if self.prev_anchor is None or self.prev_anchor.shape[0] < count:
            self.prev_anchor = wp.empty((max(int(1.5 * count), 16), 2), dtype=wp.vec3, device=device)
        self.prev_normal = _grow(self.prev_normal, count, wp.vec3, device)
        wp.copy(self.prev_anchor, anchor, count=2 * count)
        wp.copy(self.prev_normal, normal, count=count)


class ContactWarmStart:
    """The previous step's NSN rows of SolverFBA, by kind: particle contacts, rigid contacts, joint rows and the
    particle constraint rows."""

    def __init__(self, device):
        self.device = device
        self.particle = RowGroup(3, device)
        self.rigid = RigidContactGroup(device)
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

    def remap_bilateral(self, rows: np.ndarray) -> None:
        """Move retained worlds' cached rows to their new indices; ``-1`` starts a row from zero."""
        if not np.any(rows >= 0):
            self.bilateral_count = 0
            return
        if np.array_equal(rows, np.arange(self.bilateral_count)):
            return
        lam = wp.empty(len(rows), dtype=wp.float64, device=self.device)
        omega = wp.empty_like(lam)
        wp.launch(
            remap_bilateral_kernel,
            len(rows),
            [self.bilateral_lam, self.bilateral_omega, wp.array(rows, dtype=wp.int32, device=self.device)],
            [lam, omega],
            device=self.device,
        )
        self.bilateral_lam, self.bilateral_omega = lam, omega
        self.bilateral_count = len(rows)

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
    """Keys of the rigid contacts in their deterministic order, sorted by shape pair: the pair and the contact's
    rank among the pair's contacts. :class:`RigidContactGroup` finds a contact's previous contacts by its pair and
    matches them by their anchors."""
    pair = shape[:, 0].astype(np.int64) * shape_count + shape[:, 1]
    n = len(pair)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
    rank = np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))
    return pair * (1 << _RANK_BITS) + rank
