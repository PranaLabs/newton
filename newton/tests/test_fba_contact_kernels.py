# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GPU contact bookkeeping kernels (NSN GPU port Step 5.5).

Covers the kernels that replaced the Python ``for`` loop in
:meth:`~newton._src.solvers.fba.solver_fba.SolverFBA.update_contacts`:

* :func:`~newton._src.solvers.fba.kernels.compute_world_anchor_kernel`
* :func:`~newton._src.solvers.fba.kernels.compute_normal_offset_kernel`
* :func:`~newton._src.solvers.fba.kernels.compute_tangent_basis_kernel`
* :func:`~newton._src.solvers.fba.kernels.compute_tangent_offsets_kernel`
* :func:`~newton._src.solvers.fba.kernels.compute_v_anchor_kernel`

Each kernel is exercised in isolation against a numpy reference; an
integration test runs ``update_contacts`` end-to-end on a small cloth +
sphere model and confirms the device-side ``_contact_offset_d`` /
``_contact_tangent*_offset_d`` arrays match the reference computation
that the CPU loop previously produced.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType
from newton._src.solvers.fba import SolverFBA
from newton._src.solvers.fba import kernels as K
from newton._src.solvers.fba.solver_fba import (
    _quat_rotate_z_axis,
    _transform_point,
    compute_tangent_basis,
)


def _make_transform(pos: tuple[float, float, float], quat: tuple[float, float, float, float]):
    """Build a Warp transform from python tuples (x, y, z) + (qx, qy, qz, qw)."""
    return wp.transform(wp.vec3(*pos), wp.quat(*quat))


class TestComputeWorldAnchorKernel(unittest.TestCase):
    """Verify ``compute_world_anchor_kernel`` against the CPU reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_world_anchor_static_passthrough(self):
        # Shape index -1 OR shape_body[s] = -1: world_anchor == body_pos.
        M = 4
        body_pos_np = np.array(
            [
                [1.0, 2.0, 3.0],
                [-0.5, 0.25, 0.0],
                [0.0, 0.0, 0.0],
                [4.0, -1.0, 2.5],
            ],
            dtype=np.float32,
        )
        # First two contacts have shape_idx -1 (no shape lookup);
        # last two reference shape 0 which is static (shape_body[0] = -1).
        shape_idx_np = np.array([-1, -1, 0, 0], dtype=np.int32)
        shape_body_np = np.array([-1, -1], dtype=np.int32)

        body_pos = wp.array(body_pos_np, dtype=wp.vec3)
        shape_idx = wp.array(shape_idx_np, dtype=wp.int32)
        shape_body = wp.array(shape_body_np, dtype=wp.int32)
        body_q = wp.empty(1, dtype=wp.transform)
        world_anchor = wp.zeros(M, dtype=wp.vec3)

        wp.launch(
            K.compute_world_anchor_kernel,
            dim=M,
            inputs=[body_pos, shape_idx, shape_body, body_q, wp.int32(0), wp.int32(1)],
            outputs=[world_anchor],
        )
        np.testing.assert_allclose(world_anchor.numpy(), body_pos_np, atol=1e-7)

    def test_world_anchor_dynamic_body(self):
        # Dynamic body with non-trivial translation + rotation. world_anchor
        # should equal _transform_point(body_q, body_pos).
        # Rotation: 90° about world Y (qx, qy, qz, qw) = (0, sin45, 0, cos45)
        s2 = float(np.sqrt(2.0) / 2.0)
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        quat = np.array([0.0, s2, 0.0, s2], dtype=np.float64)  # 90° about +Y
        bpos_np = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, -0.5, 0.25],
            ],
            dtype=np.float32,
        )
        M = bpos_np.shape[0]
        body_pos = wp.array(bpos_np, dtype=wp.vec3)
        shape_idx = wp.array(np.array([0, 0, 0, 0], dtype=np.int32), dtype=wp.int32)
        shape_body = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)  # shape 0 → body 0
        body_q = wp.array(
            [_make_transform(tuple(pos.tolist()), tuple(quat.tolist()))],
            dtype=wp.transform,
        )
        world_anchor = wp.zeros(M, dtype=wp.vec3)

        wp.launch(
            K.compute_world_anchor_kernel,
            dim=M,
            inputs=[body_pos, shape_idx, shape_body, body_q, wp.int32(1), wp.int32(1)],
            outputs=[world_anchor],
        )
        got = world_anchor.numpy()
        expected = np.stack(
            [_transform_point(pos, quat, bpos_np[i].astype(np.float64)) for i in range(M)],
            axis=0,
        )
        np.testing.assert_allclose(got, expected.astype(np.float32), atol=1e-5)


class TestComputeNormalOffsetKernel(unittest.TestCase):
    """Verify normal offset + sphere/cylinder cushion."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_normal_offset_sphere_cylinder_cushion(self):
        # 4 contacts: shape types plane (1), sphere (3), cylinder (6), mesh
        # (15 ≠ sphere or cylinder). Only sphere and cylinder receive
        # the -0.01 cushion.
        normals_np = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        anchors_np = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 2.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 0.0, 0.25],
            ],
            dtype=np.float32,
        )
        shape_idx_np = np.array([0, 1, 2, 3], dtype=np.int32)
        shape_type_np = np.array(
            [
                int(GeoType.PLANE),
                int(GeoType.SPHERE),
                int(GeoType.CYLINDER),
                15,  # arbitrary non-sphere/cylinder type
            ],
            dtype=np.int32,
        )

        M = 4
        normals = wp.array(normals_np, dtype=wp.vec3)
        anchors = wp.array(anchors_np, dtype=wp.vec3)
        shape_idx = wp.array(shape_idx_np, dtype=wp.int32)
        shape_type = wp.array(shape_type_np, dtype=wp.int32)
        offset = wp.zeros(M, dtype=wp.float64)

        wp.launch(
            K.compute_normal_offset_kernel,
            dim=M,
            inputs=[
                normals,
                anchors,
                shape_idx,
                shape_type,
                wp.int32(1),
                wp.int32(int(GeoType.SPHERE)),
                wp.int32(int(GeoType.CYLINDER)),
            ],
            outputs=[offset],
        )

        got = offset.numpy()
        # Plane: dot(n, anchor) = 0.5; no cushion → 0.5
        # Sphere: dot = 2.0 -0.01 → 1.99
        # Cylinder: dot = 1.5 -0.01 → 1.49
        # Mesh: dot = 0.25 (no cushion)
        np.testing.assert_allclose(got, np.array([0.5, 1.99, 1.49, 0.25]), atol=1e-7)

    def test_normal_offset_sentinel_shape_idx(self):
        # shape_idx = -1 → no cushion regardless of shape_type.
        normals = wp.array(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.vec3)
        anchors = wp.array(np.array([[0.0, 0.0, 0.3]], dtype=np.float32), dtype=wp.vec3)
        shape_idx = wp.array(np.array([-1], dtype=np.int32), dtype=wp.int32)
        shape_type = wp.array(np.array([int(GeoType.SPHERE)], dtype=np.int32), dtype=wp.int32)
        offset = wp.zeros(1, dtype=wp.float64)

        wp.launch(
            K.compute_normal_offset_kernel,
            dim=1,
            inputs=[
                normals,
                anchors,
                shape_idx,
                shape_type,
                wp.int32(1),
                wp.int32(int(GeoType.SPHERE)),
                wp.int32(int(GeoType.CYLINDER)),
            ],
            outputs=[offset],
        )
        self.assertAlmostEqual(float(offset.numpy()[0]), 0.3, delta=1e-7)


class TestComputeTangentBasisKernel(unittest.TestCase):
    """Verify tangent basis construction and spinning-cylinder override."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _check_orthonormal(self, n: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> None:
        self.assertAlmostEqual(float(np.dot(n, t1)), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(np.dot(n, t2)), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(np.dot(t1, t2)), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(t1)), 1.0, delta=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(t2)), 1.0, delta=1e-5)

    def test_tangent_basis_orthonormal_default(self):
        # Non-spinning shapes (has_shape_omega=0) should fall through to the
        # generic ``compute_tangent_basis`` branch.
        normals_np = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],  # nearly aligned with X (|nx| > 0.9)
                [0.6, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
        M = normals_np.shape[0]
        normals = wp.array(normals_np, dtype=wp.vec3)
        shape_idx = wp.array(np.full(M, -1, dtype=np.int32), dtype=wp.int32)
        shape_omega = wp.empty(1, dtype=wp.float64)
        shape_transform = wp.empty(1, dtype=wp.transform)
        t1 = wp.zeros(M, dtype=wp.vec3)
        t2 = wp.zeros(M, dtype=wp.vec3)
        is_spin = wp.zeros(M, dtype=wp.int32)

        wp.launch(
            K.compute_tangent_basis_kernel,
            dim=M,
            inputs=[
                normals,
                shape_idx,
                shape_omega,
                shape_transform,
                wp.int32(0),
                wp.int32(0),
            ],
            outputs=[t1, t2, is_spin],
        )

        t1_np = t1.numpy()
        t2_np = t2.numpy()
        is_spin_np = is_spin.numpy()
        for c in range(M):
            self._check_orthonormal(normals_np[c].astype(np.float64), t1_np[c], t2_np[c])
            self.assertEqual(int(is_spin_np[c]), 0)

        # Spot check: when |nx| > 0.9 we use ref = (0,1,0), else (1,0,0).
        # For n=(1,0,0) and ref=(0,1,0): t1 = cross(n, ref) = (0,0,1)
        np.testing.assert_allclose(t1_np[2], np.array([0.0, 0.0, 1.0]), atol=1e-6)

    def test_tangent_basis_matches_cpu_reference(self):
        # Compare against the CPU reference ``compute_tangent_basis``.
        normals_np = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.95, 0.31225, 0.0],  # |nx| > 0.9
                [0.6, 0.8, 0.0],
            ],
            dtype=np.float64,
        )
        for c in range(normals_np.shape[0]):
            normals_np[c] /= np.linalg.norm(normals_np[c])

        M = normals_np.shape[0]
        normals = wp.array(normals_np.astype(np.float32), dtype=wp.vec3)
        shape_idx = wp.array(np.full(M, -1, dtype=np.int32), dtype=wp.int32)
        shape_omega = wp.empty(1, dtype=wp.float64)
        shape_transform = wp.empty(1, dtype=wp.transform)
        t1 = wp.zeros(M, dtype=wp.vec3)
        t2 = wp.zeros(M, dtype=wp.vec3)
        is_spin = wp.zeros(M, dtype=wp.int32)

        wp.launch(
            K.compute_tangent_basis_kernel,
            dim=M,
            inputs=[
                normals,
                shape_idx,
                shape_omega,
                shape_transform,
                wp.int32(0),
                wp.int32(0),
            ],
            outputs=[t1, t2, is_spin],
        )

        t1_np = t1.numpy()
        t2_np = t2.numpy()
        for c in range(M):
            t1_ref, t2_ref = compute_tangent_basis(normals_np[c])
            np.testing.assert_allclose(t1_np[c], t1_ref, atol=1e-5)
            np.testing.assert_allclose(t2_np[c], t2_ref, atol=1e-5)

    def test_tangent_basis_spinning_cylinder(self):
        # Spinning cylinder: shape_omega ≠ 0, shape transform points the local
        # +Z axis along world +Y. Expected t1 = normalize(cross(n, axis_world)).
        # Choose normal along +Z so cross(n, axis) (= +Y) -> -X direction.
        s2 = float(np.sqrt(2.0) / 2.0)
        # Rotate 90° about +X: q = (sin45, 0, 0, cos45) maps local +Z → world +Y
        shape_quat = (s2, 0.0, 0.0, s2)
        normals_np = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        normals = wp.array(normals_np, dtype=wp.vec3)
        shape_idx = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)
        shape_omega = wp.array(np.array([5.0], dtype=np.float64), dtype=wp.float64)
        shape_transform = wp.array(
            [_make_transform((0.0, 0.0, 0.0), shape_quat)],
            dtype=wp.transform,
        )
        t1 = wp.zeros(1, dtype=wp.vec3)
        t2 = wp.zeros(1, dtype=wp.vec3)
        is_spin = wp.zeros(1, dtype=wp.int32)

        wp.launch(
            K.compute_tangent_basis_kernel,
            dim=1,
            inputs=[
                normals,
                shape_idx,
                shape_omega,
                shape_transform,
                wp.int32(1),
                wp.int32(1),
            ],
            outputs=[t1, t2, is_spin],
        )

        # Reference: axis_world = q * (0,0,1) = (0,1,0).
        # t1 = normalize(cross((0,0,1), (0,1,0))) = normalize((-1,0,0)) = (-1,0,0)
        axis_world = _quat_rotate_z_axis(np.array(shape_quat, dtype=np.float64))
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t1_ref = np.cross(n, axis_world)
        t1_ref /= np.linalg.norm(t1_ref)
        t2_ref = np.cross(n, t1_ref)
        t2_ref /= np.linalg.norm(t2_ref)

        np.testing.assert_allclose(t1.numpy()[0], t1_ref.astype(np.float32), atol=1e-5)
        np.testing.assert_allclose(t2.numpy()[0], t2_ref.astype(np.float32), atol=1e-5)
        self.assertEqual(int(is_spin.numpy()[0]), 1)


class TestComputeTangentOffsetsKernel(unittest.TestCase):
    """Verify ``t·world_anchor`` projection."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_tangent_offsets_match_dot_product(self):
        rng = np.random.default_rng(7)
        M = 6
        t1_np = rng.standard_normal((M, 3)).astype(np.float32)
        t2_np = rng.standard_normal((M, 3)).astype(np.float32)
        anchor_np = rng.standard_normal((M, 3)).astype(np.float32)
        t1 = wp.array(t1_np, dtype=wp.vec3)
        t2 = wp.array(t2_np, dtype=wp.vec3)
        anchor = wp.array(anchor_np, dtype=wp.vec3)
        t1_off = wp.zeros(M, dtype=wp.float64)
        t2_off = wp.zeros(M, dtype=wp.float64)

        wp.launch(
            K.compute_tangent_offsets_kernel,
            dim=M,
            inputs=[t1, t2, anchor],
            outputs=[t1_off, t2_off],
        )

        t1_off_ref = np.einsum("ij,ij->i", t1_np.astype(np.float64), anchor_np.astype(np.float64))
        t2_off_ref = np.einsum("ij,ij->i", t2_np.astype(np.float64), anchor_np.astype(np.float64))
        np.testing.assert_allclose(t1_off.numpy(), t1_off_ref, atol=1e-5)
        np.testing.assert_allclose(t2_off.numpy(), t2_off_ref, atol=1e-5)


class TestComputeVAnchorKernel(unittest.TestCase):
    """Verify rolling cylinder kinematic anchor velocity."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_v_anchor_zero_for_non_spinning(self):
        M = 3
        is_spin = wp.array(np.zeros(M, dtype=np.int32), dtype=wp.int32)
        shape_idx = wp.array(np.full(M, -1, dtype=np.int32), dtype=wp.int32)
        shape_omega = wp.empty(1, dtype=wp.float64)
        shape_transform = wp.empty(1, dtype=wp.transform)
        anchor = wp.array(np.random.default_rng(3).standard_normal((M, 3)).astype(np.float32), dtype=wp.vec3)
        v_anchor = wp.zeros(M, dtype=wp.vec3d)

        wp.launch(
            K.compute_v_anchor_kernel,
            dim=M,
            inputs=[is_spin, shape_idx, shape_omega, shape_transform, anchor],
            outputs=[v_anchor],
        )

        np.testing.assert_array_equal(v_anchor.numpy(), np.zeros((M, 3), dtype=np.float64))

    def test_v_anchor_rolling_cylinder(self):
        # Cylinder centered at (0.5, 0.0, 0.0); local +Z axis rotated to world
        # +Y. Anchor at (0.5, 0.0, 1.0) → r_local = (0, 0, 1). cross(axis, r)
        # = cross((0,1,0), (0,0,1)) = (1, 0, 0). v_anchor = -omega · (1, 0, 0).
        s2 = float(np.sqrt(2.0) / 2.0)
        shape_quat = (s2, 0.0, 0.0, s2)  # rotates local +Z → world +Y
        shape_pos = (0.5, 0.0, 0.0)
        omega = 3.5

        is_spin = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32)
        shape_idx = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)
        shape_omega = wp.array(np.array([omega], dtype=np.float64), dtype=wp.float64)
        shape_transform = wp.array([_make_transform(shape_pos, shape_quat)], dtype=wp.transform)
        anchor = wp.array(np.array([[0.5, 0.0, 1.0]], dtype=np.float32), dtype=wp.vec3)
        v_anchor = wp.zeros(1, dtype=wp.vec3d)

        wp.launch(
            K.compute_v_anchor_kernel,
            dim=1,
            inputs=[is_spin, shape_idx, shape_omega, shape_transform, anchor],
            outputs=[v_anchor],
        )

        axis_world = _quat_rotate_z_axis(np.array(shape_quat, dtype=np.float64))
        r_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cross_ar = np.cross(axis_world, r_local)
        expected = -omega * cross_ar
        got = v_anchor.numpy()[0]
        np.testing.assert_allclose(got, expected, atol=1e-6)


class TestUpdateContactsIntegration(unittest.TestCase):
    """End-to-end: ``update_contacts`` outputs match a numpy reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_cloth_over_plane(self):
        """Tiny cloth above a static plane — produces a known contact set."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.5, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 1.0),
                wp.vec3(1.0, 0.0, 1.0),
            ],
            indices=[0, 1, 2, 1, 3, 2],
            density=1.0,
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0,
            edge_kd=0.0,
        )
        # Static plane (y=0).
        builder.add_shape_plane(
            plane=(0.0, 1.0, 0.0, 0.0),
            width=10.0,
            length=10.0,
        )
        return builder.finalize()

    def test_update_contacts_offsets_match_reference(self):
        model = self._build_cloth_over_plane()
        solver = SolverFBA(
            model,
            iterations=1,
            pin_stiffness=1.0,
            friction=True,
        )
        state = model.state()
        # Push particles down so they overlap the plane.
        particle_q_np = state.particle_q.numpy()
        particle_q_np[:, 1] = -0.01  # below the plane
        state.particle_q.assign(particle_q_np)

        pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.5)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.update_contacts(contacts, state)

        M = solver._contact_count
        self.assertGreater(M, 0, "expected at least one contact below the plane")

        # Pull device outputs.
        offset_d = solver._contact_offset_d.numpy()[:M]
        normal_d = solver._contact_normal_d.numpy()[:M]
        shape_d = solver._contact_shape_d.numpy()[:M]
        world_anchor_d = solver._contact_world_anchor_d.numpy()[:M]

        # Reference: world_anchor = body_pos for static shapes (body_index<0).
        # Need the raw sorted ordering; reproduce the host-side lexsort.
        particle_raw = contacts.soft_contact_particle.numpy()
        shape_raw = contacts.soft_contact_shape.numpy()
        normal_raw = contacts.soft_contact_normal.numpy()
        body_pos_raw = contacts.soft_contact_body_pos.numpy()
        M_raw = int(contacts.soft_contact_count.numpy()[0])
        valid = particle_raw[:M_raw] >= 0
        particle = particle_raw[:M_raw][valid]
        shape = shape_raw[:M_raw][valid]
        normal = normal_raw[:M_raw][valid]
        bpos = body_pos_raw[:M_raw][valid]
        keys = (
            normal[:, 2].astype(np.float64),
            normal[:, 1].astype(np.float64),
            normal[:, 0].astype(np.float64),
            shape.astype(np.int64),
            particle.astype(np.int64),
        )
        order = np.lexsort(keys)
        normal_sorted = normal[order].astype(np.float64)
        bpos_sorted = bpos[order].astype(np.float64)
        shape_sorted = shape[order].astype(np.int64)

        # The plane is static so world_anchor = body_pos.
        np.testing.assert_allclose(world_anchor_d.astype(np.float64), bpos_sorted, atol=1e-5)

        # offset_ref = dot(n, anchor); apply cushion if sphere or cylinder.
        # In this model the only shape is a plane → no cushion.
        offset_ref = np.einsum("ij,ij->i", normal_sorted, bpos_sorted)
        np.testing.assert_allclose(offset_d, offset_ref, atol=1e-6)
        # Normals must round-trip through device storage with full fidelity.
        np.testing.assert_allclose(normal_d.astype(np.float64), normal_sorted, atol=1e-6)
        # Shape index for plane is 0 here.
        self.assertTrue(np.all(shape_d == shape_sorted.astype(np.int32)))

        # Tangent offsets: dot(t1, anchor), dot(t2, anchor).
        t1_d = solver._contact_tangent1_d.numpy()[:M].astype(np.float64)
        t2_d = solver._contact_tangent2_d.numpy()[:M].astype(np.float64)
        t1_off_d = solver._contact_tangent1_offset_d.numpy()[:M]
        t2_off_d = solver._contact_tangent2_offset_d.numpy()[:M]

        t1_off_ref = np.einsum("ij,ij->i", t1_d, bpos_sorted)
        t2_off_ref = np.einsum("ij,ij->i", t2_d, bpos_sorted)
        np.testing.assert_allclose(t1_off_d, t1_off_ref, atol=1e-5)
        np.testing.assert_allclose(t2_off_d, t2_off_ref, atol=1e-5)

        # v_anchor must be zero for non-spinning static plane contacts.
        v_anchor = solver._contact_v_anchor_d.numpy()[:M]
        np.testing.assert_allclose(v_anchor, np.zeros_like(v_anchor), atol=1e-12)

        # Tangents must be orthonormal w.r.t. each normal.
        for c in range(M):
            self.assertAlmostEqual(float(np.dot(normal_sorted[c], t1_d[c])), 0.0, delta=1e-5)
            self.assertAlmostEqual(float(np.dot(normal_sorted[c], t2_d[c])), 0.0, delta=1e-5)
            self.assertAlmostEqual(float(np.dot(t1_d[c], t2_d[c])), 0.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
