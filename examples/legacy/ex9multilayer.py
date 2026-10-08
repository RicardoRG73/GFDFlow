"""Example 09: Multi-Layered Domain with Three Materials and Central Vertex Intersection.

This example solves a 2D Poisson equation in a square domain partitioned into
three angular material sectors that meet at the origin (node index 5).

Governing Equation
------------------
.. math::
    \\nabla^2 u = f = -100, \\quad (x, y) \\in [-1, 1] \\times [-1, 1]

Material Sectors
----------------
* Material 0: :math:`k_0 = 5.0`
* Material 1: :math:`k_1 = 500.0`
* Material 2: :math:`k_2 = 100.0`

Boundary Conditions
-------------------
* Exterior edges: Dirichlet, :math:`u(x, y) = e^{x + y}`
* Interfaces: Continuous normal flux, :math:`\\beta = 0`
* Central node (node 5): Multi-wedge equilibrium intersection
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d

# Configure visualization style
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh9.json", "r") as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
faces = np.array(mesh_data["faces"])
normal_vecs = np.array(mesh_data["normal_vecs"])
mat0_nodes = np.array(mesh_data["mat0_nodes"])
mat1_nodes = np.array(mesh_data["mat1_nodes"])
mat2_nodes = np.array(mesh_data["mat2_nodes"])
interf0_nodes = np.array(mesh_data["interf0_nodes"])
interf1_nodes = np.array(mesh_data["interf1_nodes"])
interf2_nodes = np.array(mesh_data["interf2_nodes"])
dirichlet_nodes = np.array(mesh_data["dirichlet_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
source = lambda p: -100.0

k0 = lambda p: 5.0      # Mat0 permeability
k1 = lambda p: 500.0    # Mat1 permeability
k2 = lambda p: 100.0    # Mat2 permeability

fd = lambda p: np.exp(p[0] + p[1])  # Exterior Dirichlet condition
fi_intersc = lambda p: 0.0          # Interface and intersection flux balance

# =============================================================================
# 3. GFDM Problem Initialization and Material Setup
# =============================================================================
problem = gfdmi(
    coords,
    faces,
    normal_vecs,
    L,
    source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

problem.material("mat0", k0, mat0_nodes)
problem.material("mat1", k1, mat1_nodes)
problem.material("mat2", k2, mat2_nodes)

problem.dirichlet_boundary("dir", dirichlet_nodes, fd)

problem.interface("interf0", k2, k0, interf0_nodes, None, fi_intersc, None, mat2_nodes, mat0_nodes)
problem.interface("interf1", k0, k1, interf1_nodes, None, fi_intersc, None, mat0_nodes, mat1_nodes)
problem.interface("interf2", k1, k2, interf2_nodes, None, fi_intersc, None, mat1_nodes, mat2_nodes)

# Configure multi-wedge angular balance at the center intersection vertex
center_node = 5
problem.intersection("intersection1", center_node, "interf0", "interf1", "mat0", fi_intersc)
problem.intersection("intersection2", center_node, "interf1", "interf2", "mat1", fi_intersc)
problem.intersection("intersection3", center_node, "interf2", "interf0", "mat2", fi_intersc)

# =============================================================================
# 4. Continuous System Assembly and Solution
# =============================================================================
K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 5. Visualization and Post-Processing
# =============================================================================
plot_solution_2d(
    coords,
    U,
    triangles=faces,
    levels=25,
    cmap="inferno",
    colorbar_label="total head",
    title="Steady State Solution",
    figsize=(7, 7),
    linewidths=1,
    line_alpha=0.5,
    overlay_nodes={
        "interface0": interf0_nodes,
        "interface1": interf1_nodes,
        "interface2": interf2_nodes,
        "center": [5],
    },
    savepath="examples/legacy/figures/ex9/contourf.png",
)

plot_solution_3d(
    coords,
    U,
    triangles=faces,
    cmap="inferno",
    edge_color="k",
    alpha=0.7,
    title="3D Solution",
    figsize=(7, 7),
    savepath="examples/legacy/figures/ex9/3dplot.png",
)

plt.show()