# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Helpers of ``cloth_dual_franka_fba`` and ``shirt_fold_fba``: force-limited joint drives of the arms, whose bodies
:class:`~newton.solvers.SolverFBA` integrates in the fully coupled mode (it treats joints as bilateral constraints
only, so the drives act on the bodies as external wrenches), and the options, action noise and view of parallel
worlds."""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class JointDrive:
    """Stable PD drives of the revolute and prismatic joints of fixed-base articulations, with gravity and Coriolis
    compensation, clipped at the joints' effort limits and applied to each joint's child body and, as the reaction,
    to its parent, as :attr:`~newton.State.body_f`.

    The joint positions and velocities are measured from the body states with :func:`~newton.eval_ik`, and the
    mass matrix ``M``, gravity and Coriolis forces of the measured configuration come from
    :func:`~newton.eval_inverse_dynamics_passive`. Each joint's stiffness and damping are ``M_jj w^2`` and
    ``2 zeta M_jj w`` for its natural frequency ``w`` and damping ratio ``zeta``, and the PD force anticipates the
    step as in stable PD (Tan, Liu and Turk, 2011): it acts on the position at the step's end and on the velocity
    after the drives' own acceleration, so that the stiff gains of light fingers stay stable over the step. Loads the
    model does not contain, such as the contacts of the cloth on the fingers, are not compensated: they show in the
    tracking error and the forces.
    """

    def __init__(self, model, frequency, damping_ratio=1.0, effort_limit=None):
        """
        Args:
            model: The model; every joint is revolute or prismatic.
            frequency: Natural frequency of each joint's drive [rad/s], scalar or per DOF.
            damping_ratio: Damping ratio of each joint's drive, scalar or per DOF.
            effort_limit: Force [N] or torque [N·m] limit per DOF, by default the model's effort limits.
        """
        self.model = model
        types = model.joint_type.numpy()
        # one DOF per joint, so that joint j drives DOF j
        if not np.isin(types, (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)).all():
            raise ValueError("JointDrive drives revolute and prismatic joints only")
        n = model.joint_dof_count
        self.frequency = np.broadcast_to(np.asarray(frequency, dtype=float), (n,))
        self.damping_ratio = np.broadcast_to(np.asarray(damping_ratio, dtype=float), (n,))
        self.limit = model.joint_effort_limit.numpy().astype(float) if effort_limit is None else effort_limit
        self.revolute = types == newton.JointType.REVOLUTE
        self.parent, self.child = model.joint_parent.numpy(), model.joint_child.numpy()
        self.X_p, self.axis = model.joint_X_p.numpy(), model.joint_axis.numpy()
        self.com = model.body_com.numpy()
        start = model.joint_qd_start.numpy()
        self.articulation = np.searchsorted(model.articulation_start.numpy(), np.arange(model.joint_count), "right") - 1
        self.local = start[:-1] - start[model.articulation_start.numpy()[self.articulation]]
        count, dofs = model.articulation_count, model.max_dofs_per_articulation
        self.mass_matrix = wp.zeros((count, dofs, dofs), dtype=wp.float32, device=model.device)
        self.gravity = wp.zeros(n, dtype=wp.float32, device=model.device)
        self.coriolis = wp.zeros(n, dtype=wp.float32, device=model.device)
        self.q = self.qd = self.tau = None

    def measure(self, state):
        """Joint positions [m or rad] and velocities [m/s or rad/s] of ``state``'s bodies, into ``state.joint_q`` and
        ``state.joint_qd``."""
        newton.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        self.q, self.qd = state.joint_q.numpy().astype(float), state.joint_qd.numpy().astype(float)
        return self.q, self.qd

    def apply(self, state, q_target, qd_target, dt, force=None):
        """Measure ``state`` and add the drive wrenches toward the targets at the end of the step to
        ``state.body_f``.

        Args:
            state: The state whose bodies the drives push.
            q_target: Target joint positions at the step's end, shape [joint_coord_count].
            qd_target: Target joint velocities at the step's end, shape [joint_dof_count].
            dt: The step [s].
            force: Optional feedforward force [N or N·m] per DOF, added to the PD force before the limit, such as
                the squeeze of fingers closed on an object, which the PD alone would expect to move them.

        Returns:
            The applied joint forces [N or N·m], shape [joint_dof_count].
        """
        q, qd = self.measure(state)
        newton.eval_inverse_dynamics_passive(
            self.model, state, mass_matrix=self.mass_matrix, gravity_force=self.gravity, coriolis_force=self.coriolis
        )
        tau = self.gravity.numpy() + self.coriolis.numpy()
        if force is not None:
            tau += force
        M = self.mass_matrix.numpy().astype(float)
        for a in range(M.shape[0]):
            dofs = np.flatnonzero(self.articulation == a)
            H = M[a][np.ix_(self.local[dofs], self.local[dofs])]
            w, zeta = self.frequency[dofs], self.damping_ratio[dofs]
            K, D = np.diag(H) * w * w, 2.0 * zeta * np.diag(H) * w
            pd = K * (np.asarray(q_target)[dofs] - q[dofs] - dt * qd[dofs]) + D * (
                np.asarray(qd_target)[dofs] - qd[dofs]
            )
            qdd = np.linalg.solve(H + dt * np.diag(D), pd)
            tau[dofs] += pd - dt * D * qdd
        self.tau = np.clip(tau, -self.limit, self.limit)

        # Each joint turns or pushes its child about or along its axis and its parent back: a prismatic joint's
        # force acts on both at the child's COM, so that the pair takes no net moment.
        body_q = state.body_q.numpy()
        wrench = state.body_f.numpy()
        com = np.array(
            [wp.transform_point(wp.transform(*body_q[b]), wp.vec3(*self.com[b])) for b in range(len(body_q))]
        )
        for j in range(self.model.joint_count):
            p, c = self.parent[j], self.child[j]
            X = wp.transform(*self.X_p[j])
            if p >= 0:
                X = wp.transform(*body_q[p]) * X
            a = np.array(wp.quat_rotate(X.q, wp.vec3(*self.axis[j])))
            if self.revolute[j]:
                w = np.r_[np.zeros(3), self.tau[j] * a]
                wrench[c] += w
                if p >= 0:
                    wrench[p] -= w
            else:
                f = self.tau[j] * a
                wrench[c, :3] += f
                if p >= 0:
                    wrench[p, :3] -= f
                    wrench[p, 3:] -= np.cross(com[c] - com[p], f)
        state.body_f.assign(wrench)
        return self.tau


def joint_residual(model, body_q):
    """Largest anchor separation [m] and largest locked-axis misalignment [rad] of the revolute and prismatic
    joints: how far the bodies are from satisfying their joints."""
    X_p, X_c, axis = model.joint_X_p.numpy(), model.joint_X_c.numpy(), model.joint_axis.numpy()
    gap = turn = 0.0
    for j, (p, c, kind) in enumerate(
        zip(model.joint_parent.numpy(), model.joint_child.numpy(), model.joint_type.numpy(), strict=True)
    ):
        A = wp.transform(*X_p[j]) if p < 0 else wp.transform(*body_q[p]) * wp.transform(*X_p[j])
        B = wp.transform(*body_q[c]) * wp.transform(*X_c[j])
        a = np.array(wp.quat_rotate(A.q, wp.vec3(*axis[j])))
        d = np.array(B.p) - np.array(A.p)
        if kind == newton.JointType.PRISMATIC:
            d -= (d @ a) * a  # free along its axis
            r = np.array(wp.quat_inverse(A.q) * B.q, dtype=float)
            turn = max(turn, 2.0 * float(np.arctan2(np.linalg.norm(r[:3]), abs(r[3]))))
        else:
            b = np.array(wp.quat_rotate(B.q, wp.vec3(*axis[j])))
            turn = max(turn, float(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))
        gap = max(gap, float(np.linalg.norm(d)))
    return gap, turn


def add_world_args(parser, spacing=2.0):
    """Add ``--world-count``, ``--action-noise`` and ``--world-spacing`` to ``parser``."""
    newton.examples.add_world_count_arg(parser)
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Half width [rad] of a fixed uniform bias of each world's arm joint targets, drawn once (seed 42).",
    )
    parser.add_argument(
        "--world-spacing",
        type=float,
        default=spacing,
        help="Spacing [m] of the worlds' grid in the viewer, along both horizontal axes.",
    )
    return parser


def world_args(args, spacing=2.0):
    """The checked world count, action noise [rad] and world spacing [m] of ``args``; without them, one world, no
    noise."""
    count = getattr(args, "world_count", 1)
    noise = float(getattr(args, "action_noise", 0.0))
    spacing = float(getattr(args, "world_spacing", spacing))
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError(f"--world-count must be a positive integer, got {count}")
    if not np.isfinite(noise) or noise < 0.0:
        raise ValueError(f"--action-noise must be finite and non-negative, got {noise}")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"--world-spacing must be finite and positive, got {spacing}")
    return int(count), noise, spacing


class ArmNoise:
    """A fixed bias of each world's arm joint targets, uniform in [-amplitude, amplitude] [rad], drawn once from a
    local generator; the finger targets keep none."""

    def __init__(self, model, fingers, world_count, amplitude, seed=42):
        """
        Args:
            model: The model of one world's arms, whose joint limits bound the biased targets.
            fingers: First coordinate of each arm's two finger joints.
            world_count: Number of worlds.
            amplitude: Half width of the bias [rad].
            seed: Seed of the generator.
        """
        n = model.joint_coord_count
        self.arm = np.setdiff1d(np.arange(n), [f + k for f in fingers for k in (0, 1)])
        self.lower = model.joint_limit_lower.numpy()[self.arm]
        self.upper = model.joint_limit_upper.numpy()[self.arm]
        self.amplitude = amplitude
        rng = np.random.default_rng(seed)
        self.bias = rng.uniform(-amplitude, amplitude, (world_count, len(self.arm)))

    def __call__(self, q):
        """``q`` [world_count, joint_coord_count] with the bias on its arm joints, within their limits, in place."""
        if self.amplitude > 0.0:
            q[:, self.arm] = np.clip(q[:, self.arm] + self.bias, self.lower, self.upper)
        return q


def view_worlds(viewer, model, spacing, center, radius, pitch, yaw):
    """Spread the worlds over a grid ``spacing`` [m] apart in the viewer and point the camera at them all, at
    ``pitch`` and ``yaw`` [deg]; ``center`` and ``radius`` [m] frame one world's scene."""
    spacing = (spacing, spacing, 0.0)
    viewer.set_world_offsets(spacing)
    offsets = newton.utils.compute_world_offsets(model.world_count, spacing, model.up_axis)
    lo, hi = offsets.min(axis=0), offsets.max(axis=0)
    middle = np.asarray(center) + 0.5 * (lo + hi)
    radius += 0.5 * float(np.linalg.norm(hi - lo))
    p, y = np.radians(pitch), np.radians(yaw)
    front = np.array((np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)))
    # nine tenths of the distance at which a sphere of the radius fills a 45 degree view: the grid's scenes fill
    # the frame with some padding around them
    distance = 0.9 * radius / np.sin(np.radians(22.5))
    viewer.set_camera(pos=wp.vec3(*(middle - front * distance)), pitch=pitch, yaw=yaw)
