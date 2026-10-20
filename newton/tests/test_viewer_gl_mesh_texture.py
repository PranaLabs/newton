# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.viewer import viewer_gl


class _MeshProbe:
    """Stands in for ``MeshGL``: holds a texture id while the last update had a texture, as ``MeshGL`` does."""

    def __init__(self, num_points, num_indices, device, hidden=False, backface_culling=True, dynamic=False, **_):
        self.num_points, self.num_indices, self.dynamic = num_points, num_indices, dynamic
        self.hidden, self.backface_culling = hidden, backface_culling
        self.color = (1.0, 1.0, 1.0)
        self.material = (0.5, 0.0, 0.0, 0.0)
        self.texture_id = None

    def update(self, points, indices, normals, uvs, texture=None, opacity=None):
        self.texture_id = 1 if texture is not None else None

    def destroy(self):
        pass


class TestViewerGLMeshTexture(unittest.TestCase):
    def test_log_mesh_shows_its_texture(self):
        """Turn on the texture of a logged mesh exactly while ``log_mesh`` receives one, roughness or not."""
        viewer = SimpleNamespace(
            objects={},
            device="cpu",
            CudaInterop=viewer_gl.ViewerGL.CudaInterop,
            _cuda_interop_enabled=lambda kind: False,
            _qualify=lambda name: name,
        )
        points = wp.array(np.zeros((3, 3)), dtype=wp.vec3, device="cpu")
        indices = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
        uvs = wp.array(np.zeros((3, 2)), dtype=wp.vec2, device="cpu")
        texture = np.zeros((2, 2, 3), dtype=np.uint8)
        with mock.patch.object(viewer_gl, "MeshGL", _MeshProbe):
            viewer_gl.ViewerGL.log_mesh(viewer, "/mesh", points, indices, uvs=uvs, texture=texture)
            self.assertEqual(viewer.objects["/mesh"].material[3], 1.0)
            viewer_gl.ViewerGL.log_mesh(viewer, "/mesh", points, indices, uvs=uvs, texture=texture, roughness=0.2)
            self.assertEqual(viewer.objects["/mesh"].material, (0.2, 0.0, 0.0, 1.0))
            viewer_gl.ViewerGL.log_mesh(viewer, "/mesh", points, indices)
            self.assertEqual(viewer.objects["/mesh"].material[3], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
