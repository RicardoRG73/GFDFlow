"""Example 03: Stationary Groundwater Flow and Transient Diffusion with Phreatic Line.

This example models steady-state seepage and transient diffusion through
a layered stratigraphic medium (rock-clay-rock). It extracts the phreatic
surface and performs time-integration of the diffusion equation.

Governing Equations
-------------------
* Stationary Seepage:
    .. math:: \\nabla^2 u = 0
* Transient Diffusion:
    .. math:: \\frac{\\partial u}{\\partial t} = \\nabla^2 u

Stratigraphy and Materials
--------------------------
* Rock: :math:`k_r = 1.0`
* Clay: :math:`k_c = 0.1`
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import (
    plot_phreatic_surface,
    plot_solution_2d,
    plot_solution_3d,
)

# Configure visualization style
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh3.json", "r") as f:
    mesh_data = json.load(f)

coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])
normal_vecs = np.array(mesh_data["normal_vecs"])
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
bottom_nodes = np.array(mesh_data["bottom_nodes"])
top_nodes = np.array(mesh_data["top_nodes"])
rock_nodes = np.array(mesh_data["rock_nodes"])
clay_nodes = np.array(mesh_data["clay_nodes"])
left_interface_nodes = np.array(mesh_data["left_interface_nodes"])
right_interface_nodes = np.array(mesh_data["right_interface_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
kr = lambda p: 1.0
kc = lambda p: 1e-1
source = lambda p: 0.0
neumann_cond = lambda p: 0.0
left_dirichlet = lambda p: 8.0
right_dirichlet = lambda p: 0.0
beta = lambda p: 0.0

# =============================================================================
# 3. GFDM Problem Initialization and Stationary Solution
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

# Material assignments
problem.material("rock", kr, rock_nodes)
problem.material("clay", kc, clay_nodes)

# Boundary condition assignments
problem.neumann_boundary("bottom", kr, bottom_nodes, neumann_cond)
problem.neumann_boundary("top", kr, top_nodes, neumann_cond)
problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)

# Continuous interface flux balance
problem.interface(
    "left_interface", kr, kc, left_interface_nodes, None, beta, None, rock_nodes, clay_nodes
)
problem.interface(
    "right_interface", kc, kr, right_interface_nodes, None, beta, None, clay_nodes, rock_nodes
)

# Assemble and solve stationary system K U = F
K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 4. Stationary Flow Visualization & Phreatic Surface
# =============================================================================
plot_solution_3d(
    coords,
    U,
    cmap="inferno",
    title=r"Stationary solution $\nabla^2 u = 0$",
    savepath="examples/legacy/figures/ex3/3dplot_stationary.png",
)

fig, ax = plot_solution_2d(
    coords,
    U,
    cmap="inferno",
    levels=20,
    title=r"Stationary solution $\nabla^2 u = 0$",
    savepath="examples/legacy/figures/ex3/contourf_stationary.png",
)
plot_phreatic_surface(ax, coords, U, color="b")

# =============================================================================
# 5. Transient Diffusion Problem (IVP)
# =============================================================================
# Semi-discrete system: du/dt = K U - F
t_span = [0, 80]
fun = lambda t, U_vec: K @ U_vec - F
U0 = np.zeros(coords.shape[0])
U0[left_nodes] = 8.0
U0[right_nodes] = 0.0

# Plot initial condition
plot_solution_3d(
    coords,
    U0,
    cmap="inferno",
    title="Initial Condition $U_0$",
    savepath="examples/legacy/figures/ex3/3dplot_u0.png",
)

# Solve initial value problem
sol = solve_ivp(fun, t_span, U0)
U_diffusion = sol.y

# =============================================================================
# 6. Transient Results Visualization
# =============================================================================
fig = plt.figure()
final_index = sol.t.shape[0] - 1
times_index = [0, final_index // 10, final_index // 3, final_index]

for i, t_i in enumerate(times_index):
    ax = plt.subplot(2, 2, i + 1)
    plot_solution_2d(
        coords,
        U_diffusion[:, t_i],
        levels=20,
        cmap="inferno",
        colorbar=False,
        contour_lines=False,
        ax=ax,
        title=f"$t = {sol.t[t_i]:1.2f}$",
    )
    plot_phreatic_surface(
        ax, coords, U_diffusion[:, t_i], color="k", linewidths=0.5, label=None
    )

fig.savefig("examples/legacy/figures/ex3/diffusion_steps.png", dpi=300, bbox_inches="tight")

plot_solution_3d(
    coords,
    U_diffusion[:, final_index],
    cmap="inferno",
    title=f"Solution $U$ at time $t={sol.t[-1]:1.2f}$",
    savepath="examples/legacy/figures/ex3/3dplot.png",
)

# Report matrix condition number
print("\n\n Condition number cond(K): %1.3e" % np.linalg.cond(K.toarray()))

plt.show()