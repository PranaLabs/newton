# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph captures of SolverFBA with Python's garbage collector paused."""

import contextlib
import gc

import warp as wp


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
