# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest


class TestSolverFBAImport(unittest.TestCase):
    def test_import_solver_fba(self):
        from newton._src.solvers.fba import SolverFBA  # noqa: F401


if __name__ == "__main__":
    unittest.main()
