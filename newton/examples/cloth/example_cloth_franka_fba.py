# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka (FBA port)
#
# Same coupled robot-cloth simulation as ``example_cloth_franka.py`` but with
# ``SolverFBA(nsn_schur_mode="lite")`` standing in for ``SolverVBD``.
#
# The contacts carry the 0.8 cm particle radius as a contact thickness, so
# the shirt rests on the table like SolverVBD's. The hard contacts do not let
# the closing fingertips sink into the shirt and pinch it, as SolverVBD's
# penalty contacts do, so the fingertips come down only until their pads rest
# on the shirt. As they close, rows of the solver take the shirt under the gap
# between the pads and lift it into the gap, like a positional controller,
# until the fingertips open.
#
# Command: python -m newton.examples cloth_franka_fba
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFBA, SolverFeatherstone

# Fingertip pads of the Franka hand at scale 100 [cm], in the fingertip frame: the inner face of
# each sits at the opening that its finger joint states, and the pads reach PAD_HALF_WIDTH to
# either side and PAD_HALF_LENGTH about PAD_CENTER along the axis out of the fingertips.
PAD_HALF_WIDTH = 0.875
PAD_HALF_LENGTH = 0.925
PAD_CENTER = -0.935
# Time [s] over which the taken particles rise into the gap between the pads.
GRASP_RISE = 0.5


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(
        pos_diff[0],
        pos_diff[1],
        pos_diff[2],
        ang_diff[0],
        ang_diff[1],
        ang_diff[2],
    )


class Example:
    def __init__(self, viewer, args):
        # Centimeter scale (matches the VBD reference).
        self.add_cloth = True
        self.add_robot = True
        # Three substeps of 1/180 s for the robot and the cloth. The lite NSN's contacts take a particle's
        # mass alone for its response, which overstates it more the longer the step, so they close their
        # violations over several iterations.
        self.sim_substeps = 3
        # PD local-global iterations, each with one NSN step.
        self.pd_iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viz_scale = 0.01  # cm → m for the viewer

        # Body-cloth contact (cm): shape contacts stop the particle radius out, so the shirt rests 0.8 cm over the
        # table as with SolverVBD, and are detected up to the margin farther out.
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        # Distance the shirt keeps each particle from its nearest triangle (cm). At 0.8 cm the layers of the
        # dropping shirt cross, 72 triangle pairs for good; 1.2 cm keeps them apart.
        self.cloth_self_contact_margin = 1.2

        # Robot-side contact materials (cm scale, copied from VBD reference;
        # FBA doesn't actually consume ke/kd directly — it uses the rigid
        # contact pipeline's soft_contact_* arrays and reads shape_material_mu).
        self.robot_contact_mu = 1.5

        # Cloth elasticity (cm scale, mirrors VBD reference).  Tuned for
        # FBA's ARAP stretching + isometric bending below.
        self.tri_ke = 5000.0
        self.tri_ka = 1.0e4
        # ARAP stretching handles the elastic response; tri_kd here mirrors
        # the VBD reference's stretch-damping coefficient and gets surfaced
        # via the PD damping channel when set.
        self.tri_kd = 1.5e-6

        # Per unit edge_ke, FBA's isometric bending of a hinge of this mesh (1 cm edges) is 3.7 times as stiff
        # as SolverVBD's dihedral bending. Three times the reference's 5 still halves the creasing of the
        # resting shirt: its hinges bend 9 deg on average, against 14 deg at 5 and 18 deg in the reference.
        self.bending_ke = 15.0
        # FBA's PD prefactor lacks an explicit damping channel; mirror the VBD
        # reference's bending kd anyway (read by other code paths) but keep
        # tri_kd at 0 since the ARAP stretch step takes its place.
        self.bending_kd = 1.0e-2

        self.viewer = viewer

        self.scene = ModelBuilder(gravity=(0.0, 0.0, -981.0))

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)
            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # Static table (cm scale).
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            xform=wp.transform(self.table_pos_cm, wp.quat_identity()),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        # T-shirt from USD asset.
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        # Even out the asset's 0.6 to 2.8 cm edges to at most 1.48 cm at about its vertex count, so that the
        # 0.8 cm vertex-face self contacts cover every crossing edge.
        shirt_mesh = newton.utils.remesh_mesh(shirt_mesh, method="isotropic", edge_length=1.11)
        mesh_points = shirt_mesh.vertices
        mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            # Cloth pose: ``pos.z = 42`` keeps the entire shirt bbox above
            # the table top (z=20).  The asset's local z range is roughly
            # [-20, +7]; offsetting by 42 lands the cloth bbox at z ≈
            # [22, 49], so no initial interpenetration with the table.
            # FBA's first-substep contact resolution is far less forgiving
            # of frame-0 inter-penetration than VBD's relaxation.
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(0.0, 70.0, 42.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.02,
                scale=1.0,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )
            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table primitive from the viewer (it bakes scale into mesh
        # and ignores ``shape_scale`` — same trick used by the VBD reference);
        # we render the table manually in ``render()`` at meter scale.
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Meter-scale table viz data.
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        # Push the robot contact mu into every shape so soft_contact_mu picks
        # it up regardless of which finger / link the cloth touches.
        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[...] = self.robot_contact_mu
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        # The T-shirt asset is folded in its rest pose; zero the bending rest
        # angle so the bending term doesn't try to restore each edge to its
        # folded configuration (which produces enormous initial bending
        # forces).  Same trick as the VBD reference.
        if self.add_cloth and self.model.edge_count > 0:
            self.model.edge_rest_angle.zero_()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        # Contact pipeline shared with FBA — same custom margin as VBD.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_gap=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Robot solver.
        self.robot_solver = SolverFeatherstone(
            self.model,
            update_mass_matrix_interval=self.sim_substeps,
        )
        self.set_up_control()

        # Cloth solver: SolverFBA with the lite NSN, whose per-particle kernel
        # handles several contacts per particle from the gripper fingers and
        # the table.
        self.cloth_solver: SolverFBA | None = None
        if self.add_cloth:
            # Friction for cloth-vs-robot is taken from ``shape_material_mu``
            # (set to ``robot_contact_mu`` above), no per-pair override needed.
            self.cloth_solver = SolverFBA(
                self.model,
                iterations=self.pd_iterations,
                friction=True,
                stretching_model="arap",
                pin_stiffness=1.0e10,
                nsn_schur_mode="lite",
                # The robot is stepped by Featherstone before the cloth.
                integrate_with_external_rigid_solver=True,
                # Cloth squeezed between the closing rigid fingers or between them
                # and the table has no solution; this force limit per contact row,
                # 200 N in this cm-kg-s scene, bounds it. A particle resting on the
                # table needs 0.14 N, a landing one up to 50 N; two self contacts of
                # the landing layers reach the limit for a step.
                lambda_cap=2.0e4,
                # The shirt keeps each particle its self-contact margin from its nearest triangle, but
                # not from triangles within one ring or within that distance at rest, so folds close
                # only between parts that start apart.
                particle_enable_self_contact=True,
                particle_self_contact_margin=self.cloth_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=self.cloth_self_contact_margin,
                nsn_linear_iterations=10,
            )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Meter-scale viz state (same trick as VBD reference: physics in cm,
        # render in m).
        self.viz_state = self.model.state()
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale
        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)
        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # Gravity arrays for swap-during-substep (cloth phase) and zero-out
        # (during Featherstone phase).
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # The particles held between the fingertips, and whether the next close takes new ones.
        self.grasped = np.zeros(0, dtype=np.int32)
        self.grasp_armed = True

    # ----- Grasp -----
    def fingertip_frame(self, state: State):
        """Origin, axes and half opening of the fingertip pads.

        The axes are the columns: the fingers close along the first, the second spans the pads, and the third
        points out of the fingertips.
        """
        tf = wp.transform(*state.body_q.numpy()[self.endeffector_id]) * self.endeffector_offset
        rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(tf))).reshape(3, 3)
        opening = float(state.joint_q.numpy()[-2:].mean())
        return np.array(wp.transform_get_translation(tf)), rotation @ self.fingertip_axes, opening

    def grasp(self, state: State):
        """Take the particles under the gap that the pads close to, a contact thickness inside each, down to the
        second layer of the shirt, with one rise straight up for all that brings the lowest into the gap, in the
        fingertip frame. Held there, they keep their spacing and their contact thickness from the pads."""
        origin, axes, _ = self.fingertip_frame(state)
        local = (state.particle_q.numpy() - origin) @ axes
        up = axes.T @ np.array([0.0, 0.0, 1.0])
        # The fingers close to half openings of four times the key poses' clamp.
        half = np.array([4.0 * self.clamp_close - self.cloth_particle_radius, PAD_HALF_WIDTH, PAD_HALF_LENGTH])
        # Where each particle's way up enters and leaves the box of the gap.
        with np.errstate(divide="ignore", invalid="ignore"):
            ends = ((np.array([0.0, 0.0, PAD_CENTER]) + np.stack([-half, half]))[:, None] - local) / up
        enter, leave = np.nanmax(ends.min(axis=0), axis=1), np.nanmin(ends.max(axis=0), axis=1)
        reach = 2.0 * self.cloth_particle_radius + self.cloth_self_contact_margin
        under = np.flatnonzero((enter <= leave) & (leave >= 0.0) & (enter <= reach))
        # The solver holds up to 21 particles at points: keep the nearest to the gap.
        self.grasped = under[np.argsort(enter[under])[:21]].astype(np.int32)
        self.grasp_from = local[self.grasped]
        rise = min(enter[self.grasped].max(initial=0.0) + 0.25, leave[self.grasped].min(initial=np.inf))
        self.grasp_to = self.grasp_from + rise * up
        self.grasp_time = self.sim_time

    def hold_grasp(self, state: State):
        """Hold the taken particles at points of the fingertip frame that rise into the gap between the pads."""
        origin, axes, _ = self.fingertip_frame(state)
        rise = min((self.sim_time - self.grasp_time) / GRASP_RISE, 1.0)
        targets = origin + (self.grasp_from + rise * (self.grasp_to - self.grasp_from)) @ axes.T
        n = len(self.grasped)
        self.cloth_solver.set_particle_constraints(
            np.repeat(self.grasped, 3)[:, None], np.ones((3 * n, 1)), targets.reshape(-1), np.tile(np.eye(3), (n, 1))
        )

    def update_grasp(self):
        """Take the particles once the fingertips have closed to within 5 mm, and let them go as they open."""
        if self.target[-1] > 0.5 * (self.clamp_open + self.clamp_close):
            if len(self.grasped):
                self.grasped = self.grasped[:0]
                self.cloth_solver.set_particle_constraints(None, None, None)
            self.grasp_armed = True
        elif self.grasp_armed and self.fingertip_frame(self.state_0)[2] < 4.0 * self.clamp_close + 0.5:
            self.grasp(self.state_0)
            self.grasp_armed = False

    # ----- Robot control (identical to VBD reference) -----
    def set_up_control(self):
        self.control = self.model.control()
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            return wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_com: wp.array[wp.vec3],
            body_out: wp.array[float],
        ):
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(*self.endeffector_offset.p))
            X_wb = body_q[ee_id]
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd)
            v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]
            body_out[1] = v_tip[1]
            body_out[2] = v_tip[2]
            body_out[3] = omega[0]
            body_out[4] = omega[1]
            body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=100,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        # Closed, the pads stay 1.4 cm apart from the middle, so the shirt held within 0.6 cm of it keeps its
        # contact thickness from them.
        clamp_close = self.clamp_close = 0.35
        clamp_open = self.clamp_open = 0.8
        # Fingertip frame axes in the end effector body frame, as columns.
        self.fingertip_axes = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)]]) / np.sqrt(2.0)

        # Over the 20 cm table top, with the pads' lowest edge 0.6 cm under the end effector in the grasp pose.
        grasp_z = 20.0 + 2.0 * self.cloth_particle_radius + self.cloth_self_contact_margin + 0.1 + 0.6
        # The first fold of the SolverVBD reference: the later folds stack layers of the shirt, which needs
        # cloth self-contact.
        self.robot_key_poses = np.array(
            [
                # descend
                [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
                # approach until the pads rest on the top-left corner, their lowest edge 1 mm past the
                # contact thickness of the second layer of the shirt, a self-contact radius over the
                # first on the 20 cm table top; then grasp it
                [4, 31.0, -60.0, grasp_z, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
                [2, 31.0, -60.0, grasp_z, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
                # lift
                [2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
                [2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
                # carry across
                [3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
                # release
                [1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
                # retreat
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform([0.0, 0.0, 22.0], wp.quat_identity())

    def compute_body_jacobian(
        self, model: Model, joint_q: wp.array, joint_qd: wp.array, include_rotation: bool = False
    ):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True
        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel,
                1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                ],
                outputs=[self.body_out],
            )
        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(self, state_in: State):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return
        current = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current]
        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )
        self.compute_body_jacobian(self.model, state_in.joint_q, state_in.joint_qd, include_rotation=True)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J
        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        delta_q = J_inv @ delta_target + N @ delta_q_null
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]
        self.target_joint_qd.assign(delta_q)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.add_cloth:
            self.update_grasp()
        self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                # Disable particle simulation during the Featherstone step
                # (same pattern as VBD reference: rigid step does FK only).
                particle_count = self.model.particle_count
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)
                self.state_0.joint_qd.assign(self.target_joint_qd)
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0.particle_f.zero_()
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # Cloth step (FBA in lite mode).
            self.collision_pipeline.collide(self.state_0, self.contacts)
            if self.add_cloth and len(self.grasped):
                self.hold_grasp(self.state_1)
            if self.add_cloth:
                self.cloth_solver.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        self.viewer.end_frame()
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def test_final(self):
        p_lower = wp.vec3(-60.0, -100.0, -10.0)
        p_upper = wp.vec3(60.0, 80.0, 80.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 500.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # The key poses finish at 20 s.
    parser.set_defaults(num_frames=1200)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
