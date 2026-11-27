"""Stationary Saturated Groundwater Flow in a Five-Layer Dam.

This module solves the 2D steady-state saturated potential seepage problem
for the five-layered embankment dam with central clay core and foundation
described in Section 4.2.3 (Table 2) of Chavez et al. (2023), using the
Generalized Finite Difference Method (GFDM) via ``GFDMI_2D_problem``.

Governing Equation
------------------
Within each homogeneous material subdomain :math:`\\Omega_i`:

.. math::
    k_i \\nabla^2 h = f_i \\quad \\text{in } \\Omega_i

where :math:`h(x, y) = \\psi + y` is the total hydraulic head (m) and
:math:`k_i` is the isotropic saturated hydraulic conductivity (m/s).

Material Properties
-------------------
- **Rockfill shells** (upstream and downstream): :math:`k_{\\text{rock}} = 10^{-4}\\text{ m/s}`
- **Clay core & cutoff wall**: :math:`k_{\\text{clay}} = 10^{-6}\\text{ m/s}`
- **Soil foundation**: :math:`k_{\\text{foundation}} = 10^{-4}\\text{ m/s}`

Boundary & Interface Conditions
-------------------------------
- **Upstream submerged face & left foundation wall**: Dirichlet, :math:`h = 8.0\\text{ m}`
- **Downstream underdrain & ground surface**: Dirichlet, :math:`h = 0.0\\text{ m}`
- **Bedrock base & dry dam slopes/crest**: Homogeneous Neumann, :math:`\\partial h / \\partial n = 0`
- **Internal interfaces** (``interface_a`` to ``interface_f``): Continuous head and
  normal Darcy flux balance (:math:`k_0 \\partial h / \\partial n + k_1 \\partial h / \\partial n = 0`)
- **Triple junctions** (``tj1`` at ``[20, 0]`` and ``tj2`` at ``[30, 0]``):
  Three-wedge angular flux equilibrium across converging material interfaces.

Inputs
------
- ``examples/paper/mesh_data.json`` : Precomputed mesh coordinates, connectivity,
  boundary/interface node groups, outward normals, support stencils, and ``M_pinv``.

Outputs
-------
- ``examples/paper/figures/sat_st.png`` : Total hydraulic head contour map,
  equipotential isolines, and extracted phreatic surface (:math:`h = y`).

References
----------
- Chávez-Negrete, C., Domínguez-Mota, F. J., & Román-Gutiérrez, R. (2023/2024).
  "Interface formulation for generalized finite difference method for solving
  groundwater flow." *Computers and Geotechnics*, Vol. 166, 105990.
  DOI: 10.1016/j.compgeo.2023.105990 (indexed under 2023 in some databases
  and published in Vol. 166 in 2024).
"""

# ==============================================================================
# 1. Imports and Plot Configuration
# ==============================================================================
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from GFDFlow.GFDM import GFDMI_2D_problem
from GFDFlow.visualization import plot_phreatic_surface

# Configure Matplotlib publication style
plt.style.use(["seaborn-v0_8-paper"])
plt.rcParams["figure.autolayout"] = True

# ==============================================================================
# 2. Load Precomputed Mesh Data and GFDM Stencils
# ==============================================================================
mesh_path = "examples/paper/mesh_data.json"
with open(mesh_path, "r") as f:
    mesh_data = json.load(f)

coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])
normal_vecs = np.array(mesh_data["normal_vecs"])

# Outer boundary node groups
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
bottom_nodes = np.array(mesh_data["bottom_nodes"])
top_nodes = np.array(mesh_data["top_nodes"])

# Internal two-material interface node groups
ia_nodes = np.array(mesh_data["interface_a_nodes"])
ib_nodes = np.array(mesh_data["interface_b_nodes"])
ic_nodes = np.array(mesh_data["interface_c_nodes"])
id_nodes = np.array(mesh_data["interface_d_nodes"])
ie_nodes = np.array(mesh_data["interface_e_nodes"])
if_nodes = np.array(mesh_data["interface_f_nodes"])

# Strictly interior material subdomain node groups
rock_nodes = np.array(mesh_data["rock_nodes"])
clay_nodes = np.array(mesh_data["clay_nodes"])
foundation_nodes = np.array(mesh_data["foundation_nodes"])

# Triple-junction singularity node indices at (20, 0) and (30, 0)
tj1 = int(mesh_data["tj1"][0])
tj2 = int(mesh_data["tj2"][0])

# Convert JSON string keys back to integer node indices
support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# ==============================================================================
# 3. Problem Parameters and GFDM System Assembly
# ==============================================================================
# Saturated hydraulic conductivities from Table 2 of Chavez et al. (2023)
k_rock = lambda p: 1e-4         # Rockfill shells: 0.0001 m/s
k_clay = lambda p: 1e-6         # Clay core & cutoff: 1e-06 m/s
k_foundation = lambda p: 1e-4   # Soil foundation: 0.0001 m/s

# Zero volumetric source term
h_source = lambda p: 0.0

# Boundary and interface condition callables
h_left = lambda p: 8.0          # Upstream reservoir level (m)
h_right = lambda p: 0.0         # Downstream drain head (m)
h_neumann_zero = lambda p: 0.0  # Impermeable / zero-flux boundary
h_interface_source = lambda p: 0.0

# Second-order PDE operator coefficients [A, B, C, D, E, F] -> u_xx + u_yy = 0
L = np.array([0, 0, 0, 1, 0, 1])

# Initialize GFDM problem with precomputed stencils and pseudo-inverses
problem = GFDMI_2D_problem(
    coords,
    triangles,
    normal_vecs,
    L,
    h_source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

# Assign material conductivities to interior subdomain nodes
problem.material("rock", k_rock, rock_nodes)
problem.material("clay", k_clay, clay_nodes)
problem.material("foundation", k_foundation, foundation_nodes)

# Assign outer Dirichlet and homogeneous Neumann boundary conditions
problem.dirichlet_boundary("left", left_nodes, h_left)
problem.dirichlet_boundary("right", right_nodes, h_right)
problem.neumann_boundary("bottom", k_foundation, bottom_nodes, h_neumann_zero)
problem.neumann_boundary("top", k_rock, top_nodes, h_neumann_zero)

# Assign internal material interfaces with continuous normal flux balance
# Note: On interface_e and interface_f, outward normal_vecs point from
# foundation/rock into clay, so m0 is foundation/rock and m1 is clay.
problem.interface("interface_a", k_rock, k_foundation, ia_nodes, None, h_interface_source, None, rock_nodes, foundation_nodes)
problem.interface("interface_b", k_rock, k_foundation, ib_nodes, None, h_interface_source, None, rock_nodes, foundation_nodes)
problem.interface("interface_c", k_clay, k_rock, ic_nodes, None, h_interface_source, None, clay_nodes, rock_nodes)
problem.interface("interface_d", k_clay, k_foundation, id_nodes, None, h_interface_source, None, clay_nodes, foundation_nodes)
problem.interface("interface_e", k_foundation, k_clay, ie_nodes, None, h_interface_source, None, foundation_nodes, clay_nodes)
problem.interface("interface_f", k_rock, k_clay, if_nodes, None, h_interface_source, None, rock_nodes, clay_nodes)

# Assign triple-junction multi-wedge angular flux intersections
# Node 1 at (20, 0): interfaces A, D, C meeting rockfill, clay, and foundation
problem.intersection("inters_1", tj1, "interface_a", "interface_d", "foundation", h_interface_source)
problem.intersection("inters_2", tj1, "interface_d", "interface_c", "clay", h_interface_source)
problem.intersection("inters_3", tj1, "interface_c", "interface_a", "rock", h_interface_source)

# Node 2 at (30, 0): interfaces B, F, E meeting rockfill, clay, and foundation
problem.intersection("inters_4", tj2, "interface_b", "interface_f", "rock", h_interface_source)
problem.intersection("inters_5", tj2, "interface_e", "interface_f", "clay", h_interface_source)
problem.intersection("inters_6", tj2, "interface_e", "interface_b", "foundation", h_interface_source)

# Assemble global sparse system K * U = F and solve directly via sparse LU
K, F = problem.continuous_discretization()
U = spsolve(K, F)

# ==============================================================================
# 4. Visualization and Post-Processing
# ==============================================================================
# Render filled total head contours, equipotential isolines, and phreatic line (h = y)
fig1, ax1 = plt.subplots(figsize=(10, 3.5), layout="constrained")
levels = np.linspace(0.0, 8.0, 21)
tcf = ax1.tricontourf(
    coords[:, 0],
    coords[:, 1],
    triangles,
    U,
    levels=levels,
    cmap="viridis",
)
ax1.tricontour(
    coords[:, 0],
    coords[:, 1],
    triangles,
    U,
    levels=levels,
    colors="k",
    linewidths=0.5,
    alpha=0.6,
)
plot_phreatic_surface(
    ax1,
    coords,
    U,
    triangles=triangles,
    color="#002451",
    linestyle="--",
    linewidths=1.5,
    label="Phreatic surface ($h = y$)",
)
cbar1 = fig1.colorbar(tcf, ax=ax1, shrink=0.9, pad=0.02)
cbar1.set_label("Total head $h$ (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)")
ax1.set_title("Steady State Potential Flow in Five-Layer Dam")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.axis("equal")

plt.savefig("examples/paper/figures/sat_st.png", dpi=300, bbox_inches="tight")
plt.show()
