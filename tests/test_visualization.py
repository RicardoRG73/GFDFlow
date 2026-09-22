"""Unit tests for the visualization module."""

import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GFDFlow.visualization import (
    plot_solution_2d,
    plot_solution_3d,
    plot_solution_comparison_3d,
    plot_nodes,
    plot_normal_vectors,
    plot_phreatic_surface,
)


class TestVisualization(unittest.TestCase):
    """Test visualization functions for correctness and non-blocking operation."""

    def setUp(self):
        # Create a simple grid
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 6)
        X, Y = np.meshgrid(x, y)
        self.coords = np.vstack([X.ravel(), Y.ravel()]).T
        self.u = np.sin(np.pi * self.coords[:, 0]) * np.cos(np.pi * self.coords[:, 1])
        self.u_exact = self.u * 1.05

        # Connectivity (Delaunay)
        from scipy.spatial import Delaunay
        tri = Delaunay(self.coords)
        self.triangles = tri.simplices

        # Sample normals
        self.normal_vecs = np.zeros((len(self.coords), 2))
        boundary = np.where((self.coords[:, 0] == 0) | (self.coords[:, 0] == 1))[0]
        self.boundary = boundary
        self.normal_vecs[boundary, 0] = 1.0

    def tearDown(self):
        plt.close("all")

    def test_plot_solution_2d(self):
        fig, ax = plot_solution_2d(
            self.coords,
            self.u,
            levels=10,
            cmap="viridis",
            colorbar=True,
            colorbar_label="u",
            contour_lines=True,
            clabel=True,
            overlay_nodes={"Boundary": self.boundary},
            title="Test 2D",
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_solution_2d_with_triangles(self):
        fig, ax = plot_solution_2d(
            self.coords,
            self.u,
            triangles=self.triangles,
            levels=10,
            overlay_nodes=[("Bnd", self.boundary)],
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_solution_3d(self):
        fig, ax = plot_solution_3d(
            self.coords,
            self.u,
            triangles=self.triangles,
            cmap="plasma",
            colorbar=True,
            view_init=(30, -60),
            title="Test 3D",
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_solution_comparison_3d(self):
        fig, ax = plot_solution_comparison_3d(
            self.coords,
            self.u,
            self.u_exact,
            triangles=self.triangles,
            num_label="Numerical",
            exact_label="Exact",
            view_init=(20, -50),
            title="Comparison 3D",
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_nodes(self):
        node_groups = {
            "Interior": np.arange(10),
            "Boundary": self.boundary,
        }
        fig, ax = plot_nodes(self.coords, node_groups, point_size=15, title="Test Nodes")
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_normal_vectors(self):
        fig, ax = plot_normal_vectors(
            self.coords,
            self.normal_vecs,
            boundary_nodes=self.boundary,
            quiver_color="red",
            title="Test Normals",
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_phreatic_surface(self):
        fig, ax = plt.subplots()
        plot_phreatic_surface(ax, self.coords, self.u, triangles=self.triangles)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
