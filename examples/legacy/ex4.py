"""Example 04: Multi-Material Seepage with Interface Triple Junctions.

This example solves steady-state 2D potential flow across three geological
materials (rock, clay, mixed) converging at multiple triple-junction interface
intersection nodes.

Governing Equation
------------------
.. math::
    \\nabla^2 u = 0

Materials and Conductivities
----------------------------
* Rock: :math:`k_r = 1.0`
* Clay: :math:`k_c = 0.1`
* Mixed: :math:`k_m = 0.5`

Boundary Conditions
-------------------
* Left: Dirichlet, :math:`u = 8.0`
* Right: Dirichlet, :math:`u = 0.0`
* Top / Bottom: Neumann, :math:`\\partial u / \\partial n = 0`
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_phreatic_surface, plot_solution_2d

# Configure visualization style
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# =============================================================================
# 1. Load Mesh and Topology Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh4.json", "r") as f:
    mesh_data = json.load(f)

left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
bottom_nodes = np.array(mesh_data["bottom_nodes"])
top_nodes = np.array(mesh_data["top_nodes"])
interface_a_nodes = np.array(mesh_data["interface_a_nodes"])
interface_b_nodes = np.array(mesh_data["interface_b_nodes"])
interface_c_nodes = np.array(mesh_data["interface_c_nodes"])
interface_d_nodes = np.array(mesh_data["interface_d_nodes"])
interface_e_nodes = np.array(mesh_data["interface_e_nodes"])
interface_f_nodes = np.array(mesh_data["interface_f_nodes"])
rock_nodes = np.array(mesh_data["rock_nodes"])
clay_nodes = np.array(mesh_data["clay_nodes"])
mixed_nodes = np.array(mesh_data["mixed_nodes"])
normal_vecs = np.array(mesh_data["normal_vecs"])
coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
kr = lambda p: 1.0       # Rock conductivity
kc = lambda p: 1e-1      # Clay conductivity
km = lambda p: 0.5       # Mixed conductivity
source = lambda p: 0.0

neumann_zero = lambda p: 0.0
left_dirichlet = lambda p: 8.0
right_dirichlet = lambda p: 0.0
beta = lambda p: 0.0

# =============================================================================
# 3. GFDM Problem Initialization and Domain Materials
# =============================================================================
problem = gfdmi(
    coords,
    triangles,
    normal_vecs,
    L,
    source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

problem.material("rock", kr, rock_nodes)
problem.material("clay", kc, clay_nodes)
problem.material("mixed", km, mixed_nodes)

problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)
problem.neumann_boundary("bottom", kr, bottom_nodes, neumann_zero)
problem.neumann_boundary("top", kr, top_nodes, neumann_zero)

# =============================================================================
# 4. Interface and Multi-Material Intersection Definitions
# =============================================================================
# 6 interface segments separating rock, clay, and mixed lithologies
problem.interface("interface_a", kr, km, interface_a_nodes, None, beta, None, rock_nodes, mixed_nodes)
problem.interface("interface_b", kr, km, interface_b_nodes, None, beta, None, rock_nodes, mixed_nodes)
problem.interface("interface_c", kc, kr, interface_c_nodes, None, beta, None, clay_nodes, rock_nodes)
problem.interface("interface_d", kc, km, interface_d_nodes, None, beta, None, clay_nodes, mixed_nodes)
problem.interface("interface_e", kc, kr, interface_e_nodes, None, beta, None, clay_nodes, rock_nodes)
problem.interface("interface_f", kc, km, interface_f_nodes, None, beta, None, clay_nodes, mixed_nodes)

# Multi-wedge angular intersections at central nodes 1 and 2
center_node_1 = 1
problem.intersection("inters_1", center_node_1, "interface_a", "interface_d", "mixed", beta)
problem.intersection("inters_2", center_node_1, "interface_d", "interface_c", "clay", beta)
problem.intersection("inters_3", center_node_1, "interface_c", "interface_a", "rock", beta)

center_node_2 = 2
problem.intersection("inters_4", center_node_2, "interface_b", "interface_e", "rock", beta)
problem.intersection("inters_5", center_node_2, "interface_e", "interface_f", "clay", beta)
problem.intersection("inters_6", center_node_2, "interface_f", "interface_b", "mixed", beta)

# =============================================================================
# 5. System Assembly and Solution
# =============================================================================
K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 6. Visualization and Post-Processing
# =============================================================================
fig, ax = plot_solution_2d(
    coords,
    U,
    triangles=triangles,
    levels=50,
    cmap="inferno",
    colorbar_label="total head",
    title="Steady State Solution",
    figsize=(7, 3),
    line_alpha=0.3,
    savepath="examples/legacy/figures/ex4/contourf.png",
)
plot_phreatic_surface(ax, coords, U, triangles=triangles, color="b", linewidths=2.0)

plt.show()