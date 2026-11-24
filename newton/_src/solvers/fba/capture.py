# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph captures of SolverFBA with Python's garbage collector paused, and the arrays they launch with."""

import contextlib
import functools
import gc
import types

import warp as wp

# The modules of the objects whose attributes reachable_arrays enters: the solvers' helpers and Warp's sparse
# matrices. The caller's model, states and contacts are not entered.
_ENTERED_MODULES = ("newton._src.solvers.", "warp.sparse", "warp._src.sparse")


@contextlib.contextmanager
def gc_paused():
    """Pause Python's garbage collector inside the block.

    A collection during a CUDA graph capture frees the arrays of unreachable reference cycles, such as a dropped
    solver's, inside the capture: the capture records the free of memory allocated outside it, which then either
    breaks the capture or is freed again at each replay. SolverFBA's own captures, and its steps recorded into a
    caller's capture, run with the collector paused; the cycles are collected after the block.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@contextlib.contextmanager
def scoped_capture(device):
    """:class:`warp.ScopedCapture` on ``device`` with the garbage collector paused (:func:`gc_paused`)."""
    with gc_paused(), wp.ScopedCapture(device=device) as capture:
        yield capture


def reachable_arrays(root) -> dict[int, wp.array]:
    """The Warp arrays reachable from ``root``, by id, through its attributes, containers, partial functions and
    the attributes of the solvers' helper objects and Warp's sparse matrices: the arrays that a step of ``root``
    recorded into a CUDA graph may launch with. A view keeps the array it views."""
    found, seen, stack = {}, set(), [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, wp.array):
            found[id(obj)] = obj
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, functools.partial):
            stack.extend(obj.args)
            stack.extend(obj.keywords.values())
        elif obj is root or isinstance(obj, types.SimpleNamespace) or type(obj).__module__.startswith(_ENTERED_MODULES):
            stack.extend(getattr(obj, "__dict__", {}).values())
    return found
