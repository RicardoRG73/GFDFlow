"""Example 05: 2D Porous Media Flow with Curved Circular Inclusions.

This example solves a 2D Poisson equation in a composite medium composed of
sand and rock zones, including an embedded low-permeability circular inclusion
and an internal circular Dirichlet boundary.

Governing Equation
------------------
.. math::
    k_i \\nabla^2 u = 0

Material Properties
-------------------
* Sand (left subdomain): :math:`k_{\\text{sand}} = 1.0`
* Rock (right subdomain): :math:`k_{\\text{rock}} = 0.3`
* Circular inclusion: :math:`k_{\\text{circ}} = 0.1`

Boundary Conditions
-------------------
* Left: Dirichlet, :math:`u = 1.0`
* Right: Dirichlet, :math:`u = 0.0`
* Internal Right Circle: Dirichlet, :math:`u(x) = 0.75 - x/3`
* Top / Bottom: Neumann, :math:`\\partial u / \\partial n = 0`
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

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
with open("examples/legacy/meshes/mesh5.json", "r") as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
faces = np.array(mesh_data["faces"])
normal_vecs = np.array(mesh_data["normal_vecs"])
left_half_nodes = np.array(mesh_data["left_half_nodes"])
right_half_nodes = np.array(mesh_data["right_half_nodes"])
left_circ_mat_nodes = np.array(mesh_data["left_circ_mat_nodes"])
left_top_nodes = np.array(mesh_data["left_top_nodes"])
left_bottom_nodes = np.array(mesh_data["left_bottom_nodes"])
right_top_nodes = np.array(mesh_data["right_top_nodes"])
right_bottom_nodes = np.array(mesh_data["right_bottom_nodes"])
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
right_circ_nodes = np.array(mesh_data["right_circ_nodes"])
left_circ_nodes = np.array(mesh_data["left_circ_nodes"])
interface_nodes = np.array(mesh_data["interface_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
k_sand = lambda p: 1.0
k_rock = lambda p: 0.3
k_rock_circ = lambda p: 0.1
source = lambda p: 0.0

neumann_cond = lambda p: 0.0
left_dirichlet = lambda p: 1.0
right_dirichlet = lambda p: 0.0
beta = lambda p: 0.0

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

# Material assignments
problem.material("sand", k_sand, left_half_nodes)
problem.material("rock", k_rock, right_half_nodes)
problem.material("rock_circ", k_rock_circ, left_circ_mat_nodes)

# Neumann boundaries
problem.neumann_boundary("left_top", k_sand, left_top_nodes, neumann_cond)
problem.neumann_boundary("left_bottom", k_sand, left_bottom_nodes, neumann_cond)
problem.neumann_boundary("right_top", k_rock, right_top_nodes, neumann_cond)
problem.neumann_boundary("right_bottom", k_rock, right_bottom_nodes, neumann_cond)

# Dirichlet boundaries
problem.dirichlet_boundary("izq", left_nodes, left_dirichlet)
problem.dirichlet_boundary("der", right_nodes, right_dirichlet)
problem.dirichlet_boundary("right_circle", right_circ_nodes, lambda p: 0.75 - p[0] / 3.0)

# Planar and circular interface conditions
problem.interface(
    "left_circle",
    k_sand,
    k_rock_circ,
    left_circ_nodes,
    None,
    beta,
    None,
    left_half_nodes,
    left_circ_mat_nodes,
)
problem.interface(
    "line",
    k_sand,
    k_rock,
    interface_nodes,
    None,
    beta,
    None,
    left_half_nodes,
    right_half_nodes,
)

# =============================================================================
# 4. Continuous System Assembly and Solution
# =============================================================================
K, F = problem.continuous_discretization()
U = spsolve(K, F)

# =============================================================================
# 5. Visualization and Post-Processing
# =============================================================================
overlay_interfaces = {
    "Left circle": left_circ_nodes,
    "Right circle": right_circ_nodes,
    "Interface": interface_nodes,
}

plot_solution_2d(
    coords,
    U,
    triangles=faces,
    levels=50,
    cmap="jet",
    figsize=(10, 4),
    overlay_nodes=overlay_interfaces,
    title="Poisson 2D Solution with Interfaces",
    savepath="examples/legacy/figures/ex5/contourf.png",
)

plot_solution_3d(
    coords,
    U,
    triangles=faces,
    cmap="jet",
    view_init=(30, -70),
    figsize=(6, 5),
    title="3D Solution",
    savepath="examples/legacy/figures/ex5/3dplot.png",
)

plt.show()
