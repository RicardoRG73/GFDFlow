"""Example 01: 2D Poisson Equation with Material Interface and Solution Jump.

This example solves a 2D Poisson equation across two rectangular subdomains
with differing permeabilities and a prescribed potential jump across the
vertical interface separating them.

Governing Equation
------------------
.. math::
    \\nabla^2 u = u_{xx} + u_{yy} = -1

Interface Conditions
--------------------
* Solution jump: :math:`u_{\\text{mat0}} - u_{\\text{mat1}} = \\alpha = 0.5`
* Flux balance: :math:`k_0 \\frac{\\partial u_0}{\\partial n} - k_1 \\frac{\\partial u_1}{\\partial n} = \\beta = 0`

Boundary Conditions
-------------------
* Left: Dirichlet, :math:`u = 1 - y^2`
* Right: Dirichlet, :math:`u = 1`
* Top / Bottom: Neumann, :math:`\\partial u / \\partial n = 0`
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d

# Configure visualization style
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh1.json", "r") as file:
    loaded_data = json.load(file)

coords = np.array(loaded_data["coords"])
triangles = np.array(loaded_data["triangles"])
normal_vecs = np.array(loaded_data["normal_vecs"])
left_nodes = np.array(loaded_data["left_nodes"])
right_nodes = np.array(loaded_data["right_nodes"])
bottom_left_nodes = np.array(loaded_data["bottom_left_nodes"])
top_left_nodes = np.array(loaded_data["top_left_nodes"])
bottom_right_nodes = np.array(loaded_data["bottom_right_nodes"])
top_right_nodes = np.array(loaded_data["top_right_nodes"])
left_interface_nodes = np.array(loaded_data["left_interface_nodes"])
right_interface_nodes = np.array(loaded_data["right_interface_nodes"])
interior_material_0_nodes = np.array(loaded_data["interior_material_0_nodes"])
interior_material_1_nodes = np.array(loaded_data["interior_material_1_nodes"])

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
# Operator coefficients: L = [A, B, C, 2D, E, 2F]
L = np.array([0, 0, 0, 1, 0, 1])
permeability_mat0 = lambda p: 1.0
permeability_mat1 = lambda p: 0.1
source = lambda p: -1.0
left_condition = lambda p: 1.0 - p[1] ** 2
right_condition = lambda p: 1.0
bottom_condition = lambda p: 0.0
top_condition = lambda p: 0.0

# Interface conditions
flux_difference = lambda p: 0.0       # du/dn|_{mat0} - du/dn|_{mat1} = beta
solution_difference = lambda p: 0.5   # u_{mat0} - u_{mat1} = alpha

# =============================================================================
# 3. GFDM Problem Initialization and Interface Setup
# =============================================================================
problem = gfdmi(coords, triangles, normal_vecs, L, source)

# Material assignments
problem.material("material0", permeability_mat0, interior_material_0_nodes)
problem.material("material1", permeability_mat1, interior_material_1_nodes)

# Neumann boundaries
problem.neumann_boundary("bottom_left", permeability_mat0, bottom_left_nodes, bottom_condition)
problem.neumann_boundary("top_left", permeability_mat0, top_left_nodes, top_condition)
problem.neumann_boundary("top_right", permeability_mat1, top_right_nodes, top_condition)
problem.neumann_boundary("bottom_right", permeability_mat1, bottom_right_nodes, bottom_condition)

# Dirichlet boundaries
problem.dirichlet_boundary("left", left_nodes, left_condition)
problem.dirichlet_boundary("right", right_nodes, right_condition)

# Discontinuous interface condition
problem.interface(
    "interface",
    permeability_mat0,
    permeability_mat1,
    left_interface_nodes,
    right_interface_nodes,
    flux_difference,
    solution_difference,
    interior_material_0_nodes,
    interior_material_1_nodes,
)

# =============================================================================
# 4. Discontinuous System Assembly and Solution
# =============================================================================
# Assemble sparse linear system K U = F with discontinuous interface equations
K, F = problem.discontinuous_discretization()

# Solve sparse linear system
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 5. Visualization and Post-Processing
# =============================================================================
plot_solution_2d(
    coords,
    U,
    cmap="inferno",
    levels=11,
    clabel=True,
    savepath="examples/legacy/figures/ex1/contourf.jpg",
)

plot_solution_3d(
    coords,
    U,
    cmap="inferno",
    view_init=(30, -120),
    savepath="examples/legacy/figures/ex1/3dplot.jpg",
)

plt.show()