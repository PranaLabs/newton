# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth on Knives FBA
#
# A 6 x 5.4 m cloth drops onto five blades with V notches, after RealSim's
# cloth on knives scene. Each blade is a closed mesh that tapers from 20 mm
# at its base to 1 mm at its tips, so particle contacts see both of its
# sides. The cloth triangles are larger than the contact margin, and face
# contacts keep the blade tips and edges out of them.
#
# Command: uv run -m newton.examples cloth_on_knives_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

BLADE_Y = (-2.0, -1.0, 0.0, 1.0, 2.0)
OUTLINE = np.array([(-0.5, 0.0), (3.5, 0.0), (3.5, 4.0), (1.5, 2.0), (-0.5, 4.0)])  # (x, z), V notch at index 3
THICKNESS_TIP, THICKNESS_BASE = 1.0e-3, 2.0e-2
CONTACT_THICKNESS = 0.01


def blade_thickness(z):
    return THICKNESS_TIP + (THICKNESS_BASE - THICKNESS_TIP) * (4.0 - z) / 4.0


def knife_mesh(n=20):
    """Five closed blades: the outline split into three triangles around the notch, each into ``n^2``.

    Warp's sign query treats faces within 1e-3 of the mean edge length as equally close, so the edges
    must stay well below 1 m for the sign to hold across the 1 mm tips.
    """
    bl, br, tr, notch, tl = OUTLINE
    pts, tris = [], []
    for a, b, c in ((notch, bl, br), (notch, br, tr), (notch, tl, bl)):
        index = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                index[i, j] = len(pts)
                pts.append((i * b + j * c + (n - i - j) * a) / n)
        for i in range(n):
            for j in range(n - i):
                tris.append((index[i, j], index[i + 1, j], index[i, j + 1]))
                if i + j < n - 1:
                    tris.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    pts, remap = np.unique(np.round(pts, 9), axis=0, return_inverse=True)
    tris = remap.reshape(-1)[np.array(tris)]
    e1, e2 = pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]]
    clockwise = e1[:, 0] * e2[:, 1] < e1[:, 1] * e2[:, 0]
    tris[clockwise] = tris[clockwise][:, ::-1]
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    a, b = edges[counts[inverse.reshape(-1)] == 1].T  # boundary edges, counterclockwise around the outline
    v, half = len(pts), 0.5 * blade_thickness(pts[:, 1])
    sides = [np.stack([a, b, b + v], axis=1), np.stack([a, b + v, a + v], axis=1)]
    blade_tris = np.vstack([tris[:, ::-1], tris + v, *sides])
    verts, faces = [], []
    for y in BLADE_Y:
        faces.append(blade_tris + len(verts) * 2 * v)
        verts.append(np.vstack([np.column_stack((pts[:, 0], y + s * half, pts[:, 1])) for s in (1.0, -1.0)]))
    return newton.Mesh(np.vstack(verts), np.vstack(faces).ravel())


def cloth_mesh():
    """RealSim's square_5101P: a 51 x 51 grid with a center particle per cell, four triangles per cell."""
    g = np.linspace(-1.0, 1.0, 51)
    c = 0.5 * (g[1:] + g[:-1])
    uv = np.vstack([np.stack(np.meshgrid(a, a, indexing="ij"), -1).reshape(-1, 2) for a in (g, c)])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(50), np.arange(50), indexing="ij"))
    quad = np.stack([i * 51 + j, (i + 1) * 51 + j, (i + 1) * 51 + j + 1, i * 51 + j + 1], axis=1)
    center = 2601 + i * 50 + j
    tris = np.concatenate([np.stack([quad[:, k], quad[:, (k + 1) % 4], center], axis=1) for k in range(4)])
    return np.column_stack((3.0 * uv[:, 0], -(2.7 * uv[:, 1] + 0.1), np.full(len(uv), 5.0))), tris


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        builder.add_shape_mesh(-1, mesh=knife_mesh(), cfg=newton.ModelBuilder.ShapeConfig(mu=0.3))

        # RealSim's membrane, young 37500 and poisson 0.4; 1000 kg spread evenly over the particles.
        young, poisson = 37500.0, 0.4
        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        pos, tris = cloth_mesh()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(*p) for p in pos],
            indices=tris.ravel().tolist(),
            density=1.0,
            tri_ke=mu,
            tri_ka=lam,
            edge_ke=10.0,
            particle_radius=CONTACT_THICKNESS,
        )
        self.model = builder.finalize()
        self.model.particle_mass.fill_(1000.0 / self.model.particle_count)
        self.model.particle_inv_mass.fill_(self.model.particle_count / 1000.0)

        self.solver = SolverFBA(
            self.model,
            iterations=5,
            stretching_model="neohookean",
            face_contacts=True,
            face_contact_margin=0.03,
            nsn_linear_iterations=10,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.025)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-7.5, -11.0, 8.5), pitch=-21.0, yaw=54.0)

    def simulate(self):
        # SolverFBA reads contacts back to the host every step, so the loop is not graph-captured.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        # The free cloth edges whip past 15 m/s as the cloth wraps the tips; a contact fling is much faster.
        speed = np.abs(self.state_0.particle_qd.numpy()).max()
        assert speed < 25.0, f"particle speed {speed:.1f} m/s at t={self.sim_time:.2f} s"

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(x)), "non-finite particle positions"
        assert x[:, 2].max() > 3.9, "the cloth slid off the blade tips"
        assert x[:, 2].min() > 0.0, "the cloth reached the ground"
        # Particles beside a blade keep at least half the contact thickness from its faces.
        beside = (x[:, 0] > -0.5) & (x[:, 0] < 3.5) & (x[:, 2] < 2.0 + np.abs(x[:, 0] - 1.5))
        gap = np.abs(x[:, 1] - np.clip(np.rint(x[:, 1]), -2.0, 2.0)) - 0.5 * blade_thickness(x[:, 2])
        assert gap[beside].min() > 0.5 * CONTACT_THICKNESS, f"a particle is {gap[beside].min():.4f} m from a blade"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
