# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coupling of hinged bodies that SolverFBA integrates to the material of SolverImplicitMPM, and a source of the
material's particles, for the examples.

The two solvers exchange the material's impulse on the bodies once per FBA step:

1. Before the FBA step, :meth:`HingeCoupling.apply_impulses` sets each body's impulse from the last MPM step as a
   constant force over the step.
2. After it, :meth:`HingeCoupling.update_colliders` gives MPM each body at its pose at the step's start, turning with
   FBA's velocity at its end less the spin that the applied impulse gave it. MPM sees the body as a collider that turns
   about its hinge with its moment of inertia about it (:meth:`HingeCoupling.collider_arrays`), so that each MPM step
   solves the material and the bodies' response to it together.
3. After the MPM step, :meth:`HingeCoupling.collect_impulses` sums the collider impulses on each body for the next
   FBA step.

:class:`ParticleSource` lets particles join the material over time, as from a spout, and :class:`ParticleDrain` takes
those that leave a box out of it, through the solver's public API.
"""

import numpy as np
import warp as wp

import newton


@wp.kernel
def collect_impulses_kernel(
    impulses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    collider_ids: wp.array[int],
    collider_body: wp.array[int],
    coupled_index: wp.array[int],
    hinge_point: wp.array[wp.vec3],
    # outputs
    out: wp.array2d[wp.vec3],
):
    """Add each MPM collider impulse to its body's ``out[k, 0]``, the linear impulse, and ``out[k, 1]``, the angular
    impulse about the body's hinge, ``k`` the body's index among the coupled bodies."""
    i = wp.tid()
    c = collider_ids[i]
    if c < 0 or c >= collider_body.shape[0]:
        return
    b = collider_body[c]
    if b < 0:
        return
    k = coupled_index[b]
    if k < 0:
        return
    j = impulses[i]
    wp.atomic_add(out, k, 0, j)
    wp.atomic_add(out, k, 1, wp.cross(positions[i] - hinge_point[k], j))


@wp.kernel
def apply_impulses_kernel(
    impulse: wp.array2d[wp.vec3],
    bodies: wp.array[int],
    hinge_point: wp.array[wp.vec3],
    inv_dt: float,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    applied: wp.array2d[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Set each coupled body's impulse of the last MPM step as a constant force over the FBA step, a force at its
    center of mass and the torque about it, and keep the impulse as the one applied."""
    k = wp.tid()
    b = bodies[k]
    p = impulse[k, 0]
    l_hinge = impulse[k, 1]
    applied[k, 0] = p
    applied[k, 1] = l_hinge
    com = wp.transform_point(body_q[b], body_com[b])
    body_f[b] = body_f[b] + wp.spatial_vector(p * inv_dt, (l_hinge - wp.cross(com - hinge_point[k], p)) * inv_dt)


@wp.kernel
def update_colliders_kernel(
    bodies: wp.array[int],
    hinge_point: wp.array[wp.vec3],
    hinge_axis: wp.array[wp.vec3],
    inv_axis_inertia: wp.array[float],
    applied: wp.array2d[wp.vec3],
    body_q_start: wp.array[wp.transform],
    body_q_end: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # outputs
    collider_q: wp.array[wp.transform],
    collider_qd: wp.array[wp.spatial_vector],
):
    """Each coupled body as MPM's collider sees it over a step: its pose at the step's start, and FBA's velocity at its
    end less the spin that the applied impulse gave it, taken at the hinge, the collider's reference point."""
    k = wp.tid()
    b = bodies[k]
    com = wp.transform_point(body_q_end[b], body_com[b])
    w = wp.spatial_bottom(body_qd[b])
    v_hinge = wp.spatial_top(body_qd[b]) + wp.cross(w, hinge_point[k] - com)
    axis = hinge_axis[k]
    w = w - axis * (wp.dot(axis, applied[k, 1]) * inv_axis_inertia[k])
    collider_q[b] = body_q_start[b]
    collider_qd[b] = wp.spatial_vector(v_hinge, w)


class HingeCoupling:
    """Couple bodies of ``model`` that turn about hinges fixed in the world to SolverImplicitMPM.

    Args:
        model: The model of the bodies, integrated by SolverFBA.
        state: A state of ``model`` at the bodies' initial poses.
        hinges: Per coupled body, ``(body, point, axis)``: a point of its hinge's axis and the axis's direction, in the
            world frame.
    """

    def __init__(self, model: newton.Model, state: newton.State, hinges):
        self.model = model
        self.count = len(hinges)
        device = model.device
        body_q = state.body_q.numpy()
        mass, com, inertia = model.body_mass.numpy(), model.body_com.numpy(), model.body_inertia.numpy()
        index = np.full(model.body_count, -1, dtype=np.int32)
        points, axes, inv_axis_inertia = [], [], []
        # MPM's per-body collider data: the hinge in the body frame as the reference point, a mass beyond any
        # material's (no translation) and the inverse moment of inertia about the axis alone.
        self.collider_com = np.zeros((model.body_count, 3), dtype=np.float32)
        self.collider_mass = np.zeros(model.body_count, dtype=np.float32)
        self.collider_inv_inertia = np.zeros((model.body_count, 3, 3), dtype=np.float32)
        for k, (body, hinge_point, hinge_axis) in enumerate(hinges):
            index[body] = k
            point = np.asarray(hinge_point, dtype=np.float64)
            axis = np.asarray(hinge_axis, dtype=np.float64) / np.linalg.norm(hinge_axis)
            xform = wp.transform(wp.vec3(*body_q[body][:3]), wp.quat(*body_q[body][3:]))
            local_point = np.array(wp.transform_point(wp.transform_inverse(xform), wp.vec3(*point)))
            local_axis = np.array(wp.quat_rotate_inv(wp.transform_get_rotation(xform), wp.vec3(*axis)))
            r = com[body] - local_point
            # The moment of inertia about the axis: about the center of mass, plus the mass at its distance from
            # the axis.
            moment = local_axis @ inertia[body] @ local_axis + mass[body] * (r @ r - (r @ local_axis) ** 2)
            points.append(point)
            axes.append(axis)
            inv_axis_inertia.append(1.0 / moment)
            self.collider_com[body] = local_point
            self.collider_mass[body] = 1.0e9
            self.collider_inv_inertia[body] = np.outer(local_axis, local_axis) / moment
        self.coupled_index = wp.array(index, dtype=int, device=device)
        self.bodies = wp.array([h[0] for h in hinges], dtype=int, device=device)
        self.hinge_point = wp.array(np.array(points), dtype=wp.vec3, device=device)
        self.hinge_axis = wp.array(np.array(axes), dtype=wp.vec3, device=device)
        self.inv_axis_inertia = wp.array(np.array(inv_axis_inertia), dtype=float, device=device)
        self.impulse = wp.zeros((self.count, 2), dtype=wp.vec3, device=device)
        self.applied = wp.zeros((self.count, 2), dtype=wp.vec3, device=device)
        self.body_q_start = wp.clone(state.body_q)
        self.collider_q = wp.clone(state.body_q)
        self.collider_qd = wp.zeros_like(state.body_qd)

    def setup_collider_args(self) -> dict:
        """The per-body arguments of :meth:`SolverImplicitMPM.setup_collider` for the coupled bodies."""
        device = self.model.device
        return {
            "body_com": wp.array(self.collider_com, dtype=wp.vec3, device=device),
            "body_mass": wp.array(self.collider_mass, dtype=float, device=device),
            "body_inv_inertia": wp.array(self.collider_inv_inertia, dtype=wp.mat33, device=device),
            "body_q": self.collider_q,
        }

    def attach(self, mpm_state: newton.State) -> None:
        """Let ``mpm_state``, the material's state, carry the colliders' poses and velocities."""
        mpm_state.body_q = self.collider_q
        mpm_state.body_qd = self.collider_qd

    def apply_impulses(self, state: newton.State, dt: float) -> None:
        """Before an FBA step of ``dt``: keep the bodies' start poses and add the material's last impulse on them to
        ``state.body_f``."""
        wp.copy(self.body_q_start, state.body_q)
        wp.launch(
            apply_impulses_kernel,
            dim=self.count,
            inputs=[self.impulse, self.bodies, self.hinge_point, 1.0 / dt, state.body_q, self.model.body_com],
            outputs=[self.applied, state.body_f],
        )

    def hold(self, state: newton.State) -> None:
        """Instead of an FBA step: the bodies stay at rest where ``state`` has them, and the material's last impulse
        on them is dropped."""
        self.impulse.zero_()
        self.applied.zero_()
        wp.copy(self.body_q_start, state.body_q)

    def update_colliders(self, state: newton.State) -> None:
        """After the FBA step: the colliders that the MPM step sees, from the step's start and end states."""
        wp.launch(
            update_colliders_kernel,
            dim=self.count,
            inputs=[
                *(self.bodies, self.hinge_point, self.hinge_axis, self.inv_axis_inertia, self.applied),
                *(self.body_q_start, state.body_q, state.body_qd, self.model.body_com),
            ],
            outputs=[self.collider_q, self.collider_qd],
        )

    def collect_impulses(self, mpm: newton.solvers.SolverImplicitMPM, mpm_state: newton.State) -> None:
        """After the MPM step: sum the collider impulses on each body for the next FBA step."""
        impulses, positions, ids = mpm.collect_collider_impulses(mpm_state)
        self.impulse.zero_()
        if ids.shape[0] > 0:
            wp.launch(
                collect_impulses_kernel,
                dim=ids.shape[0],
                inputs=[impulses, positions, ids, mpm.collider_body_index, self.coupled_index, self.hinge_point],
                outputs=[self.impulse],
            )


@wp.kernel
def escaping_kernel(
    q: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    lo: wp.vec3,
    hi: wp.vec3,
    # outputs
    count: wp.array[int],
):
    """Count the active particles outside the box ``[lo, hi]`` or no longer finite."""
    i = wp.tid()
    if (flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return
    p = q[i]
    inside = p[0] > lo[0] and p[1] > lo[1] and p[2] > lo[2] and p[0] < hi[0] and p[1] < hi[1] and p[2] < hi[2]
    if not inside or not (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])):
        wp.atomic_add(count, 0, 1)


@wp.kernel
def restart_kernel(
    particles: wp.array[wp.int64],
    # outputs
    qd_grad: wp.array[wp.mat33],
    elastic_strain: wp.array[wp.mat33],
    transform: wp.array[wp.mat33],
    stress: wp.array[wp.mat33],
    jp: wp.array[float],
):
    """Start the particles ``particles`` undeformed and with a uniform velocity, whatever they did before."""
    i = particles[wp.tid()]
    qd_grad[i] = wp.mat33(0.0)
    elastic_strain[i] = wp.identity(n=3, dtype=float)
    transform[i] = wp.identity(n=3, dtype=float)
    stress[i] = wp.mat33(0.0)
    jp[i] = 1.0


class ParticleDrain:
    """Takes the SolverImplicitMPM particles that leave a box out of the material, through the solver's public API.

    A particle beyond a fixed grid makes the MPM step's collider impulses non-finite, so the box lies inside the
    grid and :meth:`drain` runs after every step. Taking particles out and putting them back goes through the model's
    particle flags and :meth:`SolverImplicitMPM.notify_model_changed`, so a step that changes the particles reads them
    back to the host.

    Args:
        mpm: The material's solver.
        model: The material's model.
        state: The material's state.
        drain_lo: The drain box's lower corner [m].
        drain_hi: The drain box's upper corner [m].
        hidden: Where the particles wait [m] once they leave.
    """

    def __init__(self, mpm, model, state, drain_lo, drain_hi, hidden):
        self.mpm, self.model, self.state = mpm, model, state
        self.drain_lo, self.drain_hi = np.asarray(drain_lo), np.asarray(drain_hi)
        self.hidden = np.asarray(hidden)
        self.flags = model.particle_flags.numpy()
        self.escaping = wp.zeros(1, dtype=int, device=model.device)
        self.free = []  # the particles out of the material

    def active(self) -> np.ndarray:
        """Whether each particle is in the material."""
        return (self.flags & int(newton.ParticleFlags.ACTIVE)) != 0

    def _write(self, q, qd) -> None:
        self.state.particle_q.assign(q)
        self.state.particle_qd.assign(qd)
        self.model.particle_flags.assign(self.flags)
        self.mpm.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def retire(self, particles) -> None:
        """Take the particles ``particles`` out of the material."""
        q, qd = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        q[particles], qd[particles] = self.hidden, 0.0
        self.flags[particles] &= ~int(newton.ParticleFlags.ACTIVE)
        self.free.extend(np.asarray(particles, dtype=np.int64).tolist())
        self._write(q, qd)

    def drain(self) -> None:
        """Take out the particles outside the drain box or no longer finite, after a step that moved some there."""
        self.escaping.zero_()
        inputs = [self.state.particle_q, self.model.particle_flags, wp.vec3(*self.drain_lo), wp.vec3(*self.drain_hi)]
        wp.launch(escaping_kernel, self.model.particle_count, inputs=inputs, outputs=[self.escaping])
        if not self.escaping.numpy()[0]:
            return
        q = self.state.particle_q.numpy()
        inside = np.all((q > self.drain_lo) & (q < self.drain_hi), axis=1) & np.all(np.isfinite(q), axis=1)
        self.retire(np.flatnonzero(self.active() & ~inside))


class ParticleSource(ParticleDrain):
    """A stream of SolverImplicitMPM particles that join the material, as from a spout, and leave it past a box.

    The stream leaves ``origin`` in layers of the points ``layer`` (relative to ``origin``), one layer every
    ``interval``, each particle placed and moving where its flight from its layer's point puts it when it joins, and
    undeformed. It takes its particles from a pool of inactive ones, and the particles that leave the drain box
    (:class:`ParticleDrain`) go back to the pool.

    Args:
        mpm: The material's solver.
        model: The material's model.
        state: The material's state.
        pool: The inactive particles that the stream takes.
        origin: The stream's origin [m].
        layer: The points of a layer [m], relative to ``origin``, shape ``[count, 3]``.
        velocity: The stream's velocity [m/s].
        interval: The time between two layers [s].
        drain_lo: The drain box's lower corner [m].
        drain_hi: The drain box's upper corner [m].
        hidden: Where the particles wait [m], before they join and after they leave.
        gravity: The gravity [m/s^2] of the particles' flight from the stream's origin.
    """

    def __init__(self, mpm, model, state, pool, origin, layer, velocity, interval, drain_lo, drain_hi, hidden, gravity):
        super().__init__(mpm, model, state, drain_lo, drain_hi, hidden)
        self.free = list(np.asarray(pool, dtype=np.int64)[::-1])  # taken from the end
        self.origin, self.layer = np.asarray(origin, dtype=np.float64), np.asarray(layer, dtype=np.float64)
        self.velocity, self.interval = np.asarray(velocity, dtype=np.float64), float(interval)
        self.gravity = np.asarray(gravity)
        self.next_layer = 0
        self.flowing = True

    def emit(self, time: float, duration: float) -> None:
        """Let the layers due before ``time + duration`` join, placed and moving as their flight from the stream's
        origin puts them at ``time``. The stream stops when the pool runs out."""
        times = []
        while self.flowing and self.next_layer * self.interval < time + duration:
            if len(self.free) < len(self.layer):
                self.flowing = False
                break
            times.append(self.next_layer * self.interval)
            self.next_layer += 1
        if not times:
            return
        joining = np.array([self.free.pop() for _ in range(len(times) * len(self.layer))], dtype=np.int64)
        tau = np.repeat(time - np.array(times), len(self.layer))[:, None]
        q, qd = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        start = self.origin + np.tile(self.layer, (len(times), 1))
        q[joining] = start + self.velocity * tau + 0.5 * tau * tau * self.gravity
        qd[joining] = self.velocity + tau * self.gravity
        self.flags[joining] |= int(newton.ParticleFlags.ACTIVE)
        mpm_state = self.state.mpm
        wp.launch(
            restart_kernel,
            len(joining),
            inputs=[wp.array(joining, dtype=wp.int64, device=self.model.device)],
            outputs=[
                *(mpm_state.particle_qd_grad, mpm_state.particle_elastic_strain, mpm_state.particle_transform),
                *(mpm_state.particle_stress, mpm_state.particle_Jp),
            ],
        )
        self._write(q, qd)


def stream_layout(direction, spacing, flow, side_count):
    """A square stream for :class:`ParticleSource`: layers of ``side_count`` x ``side_count`` points ``spacing`` apart
    across ``direction``, the first half a spacing out along it, at the speed that carries ``flow`` [m^3/s].

    Returns:
        A layer's points relative to the stream's origin, the stream's velocity [m/s] and the time between layers [s].
    """
    along = np.asarray(direction, dtype=np.float64)
    along /= np.linalg.norm(along)
    side = np.cross(along, (0.0, 0.0, 1.0))
    if np.linalg.norm(side) < 1.0e-6:
        side = np.cross(along, (1.0, 0.0, 0.0))
    side /= np.linalg.norm(side)
    normal = np.cross(along, side)
    speed = flow / (side_count * side_count * spacing**2)
    offsets = (np.arange(side_count) - 0.5 * (side_count - 1)) * spacing
    layer = np.array([0.5 * spacing * along + a * side + b * normal for a in offsets for b in offsets])
    return layer, speed * along, spacing / speed
