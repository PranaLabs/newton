# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""RealSim CudaTests baseline drivers."""

from pathlib import Path

REALSIM_ROOT = Path("/home/ziqiu/work/RealSim_py/realsim_py")

DEMOS: dict[str, dict] = {
    "TwistingBar": {"scene": "simulation/config/CudaTests/TwistingBar/scene.json", "expected_frames": 810},
    "TwistingBarNH": {"scene": "simulation/config/CudaTests/TwistingBarNH/scene.json", "expected_frames": 810},
    "StretchingCloth": {"scene": "simulation/config/CudaTests/StretchingCloth/scene.json"},
    "PullingWooper": {"scene": "simulation/config/CudaTests/PullingWooper/scene.json", "expected_frames": 500},
    "CrossingGingerbreadman": {
        "scene": "simulation/config/CudaTests/CrossingGingerbreadman/scene.json",
        "expected_frames": 810,
    },
    "SqueezingBall": {"scene": "simulation/config/CudaTests/SqueezingBall/scene.json", "expected_frames": 600},
    "SharpCorner": {"scene": "simulation/config/CudaTests/SharpCorner/scene.json", "expected_frames": 200},
    "ClothOnKnives": {"scene": "simulation/config/CudaTests/ClothOnKnives/scene.json", "expected_frames": 900},
    "ParallelEnvTest": {"scene": "simulation/config/CudaTests/ParallelEnvTest/scene.json", "expected_frames": 300},
    "CableGrabRaptor": {"scene": "simulation/config/CudaTests/CableGrabRaptor/scene.json", "expected_frames": 6000},
}
