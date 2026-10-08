"""Example 11: Seepage Analysis in a Tailings Dam and Multi-Zone Foundation.

This example simulates steady-state 2D geotechnical seepage through an
engineered tailings dam embankment, impounded tailings beach, and underlying
bedrock foundation, featuring severe conductivity contrasts and a triple junction.

Governing Equation
------------------
.. math::
    \\nabla^2 u = u_{xx} + u_{yy} = 0

Material Properties
-------------------
* Impounded Tailings: :math:`k_{\\text{tailing}} = 1.0`
* Dam Embankment: :math:`k_{\\text{dam}} = 10^{-2}`
* Foundation Rock: :math:`k_{\\text{rock}} = 10^{-4}`

Boundary Conditions
-------------------
* Upstream Reservoir (Left): Dirichlet, :math:`u = 300.0`
* Downstream Discharge (Right): Dirichlet, :math:`u = 200.0`
* Impermeable Bedrock (Bottom): Neumann, :math:`\\partial u / \\partial n = 0`
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdm
from GFDFlow.visualization import plot_solution_2d

start_time = time.perf_counter()

# Configure visualization style
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
mesh_file = "examples/legacy/meshes/mesh11.json"
with open(mesh_file, "r") as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])
normal_vecs = np.array(mesh_data["normal_vecs"])
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
bottom_nodes = np.array(mesh_data["bottom_nodes"])
top_dam_nodes = np.array(mesh_data["top_dam_nodes"])
top_tailing_nodes = np.array(mesh_data["top_tailing_nodes"])
interface_a_nodes = np.array(mesh_data["interface_a_nodes"])
interface_b_nodes = np.array(mesh_data["interface_b_nodes"])
interface_c_nodes = np.array(mesh_data["interface_c_nodes"])
rock_nodes = np.array(mesh_data["rock_nodes"])
dam_nodes = np.array(mesh_data["dam_nodes"])
tailing_nodes = np.array(mesh_data["tailing_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Material Conductivities
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
kdam = lambda p: 1e-2        # Conductivity of dam embankment
ktailing = lambda p: 1.0     # Conductivity of tailings
krock = lambda p: 1e-4       # Conductivity of bedrock
source = lambda p: 0.0

neumann_0 = lambda p: 0.0
neumann_1 = lambda p: 0.0
left_dirichlet = lambda p: 300.0
right_dirichlet = lambda p: 200.0
beta = lambda p: 0.0

# =============================================================================
# 3. GFDM Problem Initialization and Material Setup
# =============================================================================
problem = gfdm(
    coords,
    triangles,
    normal_vecs,
    L,
    source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

# Material assignments
problem.material("dam", kdam, dam_nodes)
problem.material("tailing", ktailing, tailing_nodes)
problem.material("rock", krock, rock_nodes)

# Boundary condition assignments
problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)
problem.neumann_boundary("bottom", krock, bottom_nodes, neumann_0)
problem.neumann_boundary("top_dam", kdam, top_dam_nodes, neumann_0)
problem.neumann_boundary("top_tailing", ktailing, top_tailing_nodes, neumann_1)

# Geological interfaces
problem.interface("interface_a", ktailing, krock, interface_a_nodes, None, beta, None, tailing_nodes, rock_nodes)
problem.interface("interface_b", ktailing, kdam, interface_b_nodes, None, beta, None, tailing_nodes, dam_nodes)
problem.interface("interface_c", kdam, krock, interface_c_nodes, None, beta, None, dam_nodes, rock_nodes)

# Triple intersection at node 11
center_node = 11
problem.intersection("inters_1_tailing", center_node, "interface_a", "interface_b", "tailing", beta)
problem.intersection("inters_1_rock", center_node, "interface_a", "interface_c", "rock", beta)
problem.intersection("inters_1_dam", center_node, "interface_b", "interface_c", "dam", beta)

# =============================================================================
# 4. Continuous System Assembly and Solution
# =============================================================================
K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 5. Visualization and Post-Processing
# =============================================================================
interface_overlay = {
    "Interface A": interface_a_nodes,
    "Interface B": interface_b_nodes,
    "Interface C": interface_c_nodes,
    "Intersection": np.array([center_node]),
}

plot_solution_2d(
    coords,
    U,
    triangles=triangles,
    levels=25,
    cmap="inferno",
    colorbar_label="Hydraulic Head (m)",
    title="Tailings Dam Hydraulic Head Distribution",
    xlabel="X (m)",
    ylabel="Y (m)",
    figsize=(10, 10),
    overlay_nodes=interface_overlay,
    savepath="examples/legacy/figures/ex11/contourf.png",
)

plt.show()
