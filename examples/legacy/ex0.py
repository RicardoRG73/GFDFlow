"""Example 00: 2D Poisson Equation on an Irregular Curved Domain.

This example solves a 2D Poisson equation with mixed Dirichlet and Neumann
boundary conditions on a bounded domain featuring a circular arc boundary.
The spatial discretization is performed using the Generalized Finite Difference
Method (GFDM) with ghost nodes enforcing directional flux on the curved boundary.

Governing Equation
------------------
.. math::
    \\nabla^2 u = u_{xx} + u_{yy} = -2

Boundary Conditions
-------------------
* Left (:math:`x = 0`): Dirichlet, :math:`u = 0`
* Bottom (:math:`y = 0`): Dirichlet, :math:`u = 0.5 x`
* Top (:math:`y = 1`): Dirichlet, :math:`u = x`
* Right (circular arc): Neumann, :math:`\\partial u / \\partial n = 0`
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
# 1. Load Mesh and Precomputed Stencils
# =============================================================================
with open("examples/legacy/meshes/mesh0.json", "r") as file:
    loaded_data = json.load(file)

coords = np.array(loaded_data["coords"])
triangles = np.array(loaded_data["triangles"])
normal_vectors = np.array(loaded_data["normal_vectors"])
left_nodes = np.array(loaded_data["left_nodes"])
right_nodes = np.array(loaded_data["right_nodes"])
bottom_nodes = np.array(loaded_data["bottom_nodes"])
top_nodes = np.array(loaded_data["top_nodes"])
interior_nodes = np.array(loaded_data["interior_nodes"])
support_stencils = {int(k): np.array(v) for k, v in loaded_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in loaded_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
# Differential operator coefficients: L = [A, B, C, 2D, E, 2F]
# Approximating: L u = A*u + B*u_x + C*u_y + D*u_xx + E*u_xy + F*u_yy
L = np.array([0, 0, 0, 1, 0, 1])
permeability = lambda p: 1.0
source = lambda p: -2.0
left_condition = lambda p: 0.0
right_condition = lambda p: 0.0
bottom_condition = lambda p: p[0] * 0.5
top_condition = lambda p: p[0]

# =============================================================================
# 3. GFDM Problem Initialization and Boundary Setup
# =============================================================================
problem = gfdmi(coords, triangles, normal_vectors, L, source, support_stencils, M_pinv)

# Material assignment
problem.material("0", permeability, interior_nodes)

# Boundary condition assignment
problem.neumann_boundary("right", permeability, right_nodes, right_condition)
problem.dirichlet_boundary("left", left_nodes, left_condition)
problem.dirichlet_boundary("top", top_nodes, top_condition)
problem.dirichlet_boundary("bottom", bottom_nodes, bottom_condition)

# =============================================================================
# 4. System Assembly and Solution
# =============================================================================
# Assemble sparse system K U = F
K, F = problem.continuous_discretization()

# Solve linear system
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
    savepath="examples/legacy/figures/ex0/contourf.png",
)

plot_solution_3d(
    coords,
    U,
    cmap="inferno",
    view_init=(30, -130),
    savepath="examples/legacy/figures/ex0/3dplot.png",
)

plt.show()