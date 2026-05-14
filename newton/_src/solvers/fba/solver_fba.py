# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


class SolverFBA(SolverBase):
    """Fast But Accurate projective-dynamics cloth solver (MVP stub)."""

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap"] = "arap",
    ) -> None:
        super().__init__(model)
        raise NotImplementedError("SolverFBA __init__ pending Task 10")

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        raise NotImplementedError("SolverFBA.step pending Task 11")

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        raise NotImplementedError(
            "Contact-aware FBA solver TBD; SolverFBA MVP supports gravity + pin only"
        )
