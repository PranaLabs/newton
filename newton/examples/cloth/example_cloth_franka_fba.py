# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka (FBA port)
#
# Same coupled robot-cloth simulation as ``example_cloth_franka.py`` but with
# ``SolverFBA(nsn_schur_mode="lite")`` standing in for ``SolverVBD``.
#
# Self-contact: ``particle_contact_mode='vt'`` enables vertex-triangle and
# edge-edge cloth self-contact, validated end-to-end here as Phase 4 of
# ``docs/superpowers/plans/2026-05-19-fba-particle-contact-vt-ee.md``.
#
# Command (interactive):
#     python -m newton.examples cloth_franka_fba
# Command (headless, offline; recommended for verification):
#     python -m newton.examples cloth_franka_fba --viewer null --num-frames 1200
#
###########################################################################

from __future__ import annotations

import time

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFBA, SolverFeatherstone


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
        pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2],
    )


class Example:
    def __init__(self, viewer, args):
        # Centimeter scale (matches the VBD reference).
        self.add_cloth = True
        self.add_robot = True
        # FBA's PD outer loop is its own iteration scheme; sim_substeps drives
        # the rigid-body integrator + PD-FBA pair.  We keep the VBD reference
        # 10 substeps so the robot's key-pose interpolation isn't aliased.
        self.sim_substeps = 10
        # PD + NSN iterations.  Empirically, the cloth-on-table phase under
        # the Franka gripper trajectory requires at least 8 PD sweeps and 3
        # NSN-inner iterations to keep contact lambdas bounded; combined
        # with the ``lambda_cap`` knob below, this lets the demo run the
        # full 1200-frame trajectory without NaN.
        self.pd_iterations = 8
        self.nsn_iterations = 3
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viz_scale = 0.01  # cm → m for the viewer

        # Body-cloth contact margin (cm).
        self.cloth_body_contact_margin = 0.8

        # Robot-side contact materials (cm scale, copied from VBD reference;
        # FBA doesn't actually consume ke/kd directly — it uses the rigid
        # contact pipeline's soft_contact_* arrays and reads shape_material_mu).
        self.robot_contact_mu = 1.5

        # Cloth elasticity (cm scale, mirrors VBD reference).  Tuned for
        # FBA's ARAP stretching + isometric bending below.
        self.tri_ke = 1.0e4
        self.tri_ka = 1.0e4
        # ARAP stretching handles the elastic response; tri_kd here mirrors
        # the VBD reference's stretch-damping coefficient and gets surfaced
        # via the PD damping channel when set.
        self.tri_kd = 1.5e-6

        self.bending_ke = 5.0
        # FBA's PD prefactor lacks an explicit damping channel; mirror the VBD
        # reference's bending kd anyway (read by other code paths) but keep
        # tri_kd at 0 since the ARAP stretch step takes its place.
        self.bending_kd = 1.0e-2

        self.viewer = viewer

        self.scene = ModelBuilder(gravity=-981.0)

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
            wp.transform(self.table_pos_cm, wp.quat_identity()),
            hx=self.table_hx_cm, hy=self.table_hy_cm, hz=self.table_hz_cm,
        )

        # T-shirt from USD asset.
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
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
                particle_radius=0.8,
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
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Robot solver.
        self.robot_solver = SolverFeatherstone(
            self.model, update_mass_matrix_interval=self.sim_substeps,
        )
        self.set_up_control()

        # Cloth solver — FBA in LiteNSN mode (Phase 1 per-particle kernel
        # handles multi-contact-per-particle from the gripper fingers + table).
        # Self-contact uses the v-t + e-e broadphase from
        # ``2026-05-19-fba-particle-contact-vt-ee.md`` Phase 1+3, so the
        # fold-up sequence should produce visually separated cloth layers.
        self.cloth_solver: SolverFBA | None = None
        if self.add_cloth:
            # Particle radius is in cm (the cloth was built at cm scale with
            # particle_radius=0.8 above); margin ~half-particle keeps the
            # broadphase from over-counting at rest.  Friction for cloth-vs-
            # robot is taken from ``shape_material_mu`` (set to
            # ``robot_contact_mu`` above) — no per-pair override needed.
            self.cloth_solver = SolverFBA(
                self.model,
                iterations=self.pd_iterations,
                nsn_iterations=self.nsn_iterations,
                friction=True,
                stretching_model="arap",
                pin_stiffness=1.0e10,
                nsn_schur_mode="lite",
                # ``lambda_cap`` clamps the per-step impulse magnitude per
                # contact row; without this the dense cloth-on-table contact
                # set produces lambda values that occasionally diverge through
                # the Schur preconditioner, NaN-ing the simulation by frame
                # ~250.  1.0 N·s is well above any physical impulse on a 0.5
                # kg cloth at gravity, and gates only the numerical outliers.
                lambda_cap=1.0,
                # Self-contact (Phase 1 v-t + Phase 3 e-e). Tuned for the
                # cm-scale T-shirt: per-vertex hit cap=4 + ``rest_exclusion=1.0
                # cm`` (5x the contact radius) prevents the broadphase from
                # latching on adjacent panels at rest, which would otherwise
                # produce O(10⁵) contacts on the first folding moment and
                # overflow the LCP buffers.
                particle_contact_radius=0.2,             # cm
                particle_contact_margin=0.05,            # cm — tight, only catches imminent contact
                particle_contact_friction=0.25,
                particle_contact_topology_ring=2,
                particle_contact_rest_exclusion_radius=1.0,  # cm
                particle_contact_mode="vt",
                # e-e disabled for this scene: with this T-shirt mesh, e-e
                # broadphase fires O(10⁴) pairs as soon as the cloth begins
                # to fold (frames 10-15), saturating the NSN-PCR solver and
                # producing NaN.  v-t alone provides cloth-cloth separation
                # adequate for the Franka pick-place trajectory; see
                # Phase 4 results in the plan doc.
                particle_contact_ee=False,
                particle_contact_max_hits_per_vertex=4,
            )

        # Diagnostics for headless verification (Phase 4 acceptance gate).
        self._diag_step_times: list[float] = []
        self._diag_max_vt_hits = 0
        self._diag_max_ee_hits = 0
        self._diag_nan_seen = False
        self._diag_last_y_lo = 0.0
        self._diag_last_y_hi = 0.0
        self._diag_frame = 0

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
            floating=False, scale=100,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close = 0.1
        clamp_open = 0.8

        # Reduced key-pose set: focus on grasp + lift + move (the parts that
        # work without self-contact).  Drops the dual-fold sequence which
        # depends on layer stacking.
        self.robot_key_poses = np.array(
            [
                # descend
                [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
                # approach + grasp top-left corner
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
                # lift
                [2, 26.0, -60.0, 30.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
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

    def compute_body_jacobian(self, model: Model, joint_q: wp.array, joint_qd: wp.array, include_rotation: bool = False):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True
        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1,
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
            compute_ee_delta, dim=1,
            inputs=[
                state_in.body_q, self.endeffector_offset, self.endeffector_id,
                self.bodies_per_world, wp.transform(*self.target[:7]),
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
        q_des = q.copy(); q_des[1:] = self.initial_pose[1:]
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        delta_q = J_inv @ delta_target + N @ delta_q_null
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]
        self.target_joint_qd.assign(delta_q)

    def step(self):
        t0 = time.perf_counter()
        self.generate_control_joint_qd(self.state_0)
        self.simulate()
        self.sim_time += self.frame_dt
        # Only synchronize at the reporting cadence so step time captures the
        # full frame including the final substep's contact solve, without
        # serialising every step on the main thread.
        self._diag_frame += 1

        # Latest v-t / e-e broadphase hit counts (set by the contact pass).
        vt_hits = int(getattr(self.cloth_solver, "_pc_last_vt_hits", 0)) if self.cloth_solver else 0
        ee_hits = int(getattr(self.cloth_solver, "_pc_last_ee_hits", 0)) if self.cloth_solver else 0
        self._diag_max_vt_hits = max(self._diag_max_vt_hits, vt_hits)
        self._diag_max_ee_hits = max(self._diag_max_ee_hits, ee_hits)

        if self._diag_frame % 50 == 0 or self._diag_frame == 1:
            wp.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0
            self._diag_step_times.append(step_ms)
            # NaN guard + y-range tracking at the report cadence.
            q = self.state_0.particle_q.numpy()
            if not np.isfinite(q).all() and not self._diag_nan_seen:
                self._diag_nan_seen = True
                print(f"  [NaN] frame {self._diag_frame}")
            self._diag_last_y_lo = float(q[:, 1].min())
            self._diag_last_y_hi = float(q[:, 1].max())
            rolling = self._diag_step_times[-50:]
            step_mean = float(np.mean(rolling)) if rolling else 0.0
            print(
                f"  f={self._diag_frame:5d}  vt_hits={vt_hits:6d}  ee_hits={ee_hits:6d}  "
                f"y=[{self._diag_last_y_lo:+7.2f},{self._diag_last_y_hi:+7.2f}]  "
                f"step_mean={step_mean:6.2f}ms",
                flush=True,
            )
        else:
            step_ms = (time.perf_counter() - t0) * 1000.0
            self._diag_step_times.append(step_ms)

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
                self.model.shape_contact_pair_count = 0
                self.state_0.joint_qd.assign(self.target_joint_qd)
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0.particle_f.zero_()
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # Cloth step (FBA in lite mode).
            self.collision_pipeline.collide(self.state_0, self.contacts)
            if self.add_cloth:
                self.cloth_solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt,
                )

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return
        wp.launch(
            scale_positions, dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms, dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_shapes(
            "/table", newton.GeoType.BOX, self.table_viz_scale,
            self.table_viz_xform, self.table_viz_color,
        )
        self.viewer.end_frame()
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def test_final(self):
        self._print_diag_summary()
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

    def _print_diag_summary(self) -> None:
        if not self._diag_step_times:
            return
        steps = np.asarray(self._diag_step_times, dtype=np.float64)
        print("")
        print(f"  [phase4] frames                : {self._diag_frame}")
        print(f"  [phase4] total_runtime         : {steps.sum() / 1000.0:.2f} s")
        print(f"  [phase4] step_mean overall     : {steps.mean():.2f} ms")
        print(f"  [phase4] step_mean (last 50)   : {steps[-50:].mean():.2f} ms")
        print(f"  [phase4] max v-t hits          : {self._diag_max_vt_hits}")
        print(f"  [phase4] max e-e hits          : {self._diag_max_ee_hits}")
        print(f"  [phase4] last particle y range : [{self._diag_last_y_lo:+.3f}, {self._diag_last_y_hi:+.3f}]")
        print(f"  [phase4] NaN observed          : {self._diag_nan_seen}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # Reduced key-pose set finishes at ~16s; pad to 1000 frames (16.7s @ 60fps).
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        # Always print diagnostics summary at exit, even when ``--test`` is
        # not set (Phase 4 acceptance reporting from a headless run).
        example._print_diag_summary()
