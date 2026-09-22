#%%
# =============================================================================
# Importing needed libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use("seaborn-v0_8")

from scipy.integrate import solve_ivp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_phreatic_surface, plot_solution_2d


# loading mesh data
import json
with open("examples/legacy/meshes/mesh4.json","r") as f:
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


#%%
# =============================================================================
# Problem parameters
# =============================================================================
L = np.array([0,0,0,1,0,1])
kr = lambda p: 1        # conductivity of rock
kc = lambda p: 1e-1     # conductivity of clay
km = lambda p: 0.5      # conductivity of mixed

# source term 
source = lambda p: 0

# boundary conditions
neumann_zero = lambda p: 0
left_dirichlet = lambda p: 8
right_dirichlet = lambda p: 0
beta = lambda p: 0

#%%
# =============================================================================
# Assembling and solving system KU=F
# =============================================================================
problem = gfdmi(
    coords,
    triangles,
    normal_vecs,
    L,
    source
)

# material domains
problem.material("rock", kr, rock_nodes)
problem.material("clay", kc, clay_nodes)
problem.material("mixed", km, mixed_nodes)

# dirichlet boaundaries
problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)

# neumann boaundaries
problem.neumann_boundary("bottom", kr, bottom_nodes, neumann_zero)
problem.neumann_boundary("top", kr, top_nodes, neumann_zero)

# interfaces
problem.interface("interface_a", kr, km, interface_a_nodes, None, beta, None, rock_nodes, mixed_nodes)
problem.interface("interface_b", kr, km, interface_b_nodes, None, beta, None, rock_nodes, mixed_nodes)
problem.interface("interface_c", kc, kr, interface_c_nodes, None, beta, None, clay_nodes, rock_nodes)
problem.interface("interface_d", kc, km, interface_d_nodes, None, beta, None, clay_nodes, mixed_nodes)
problem.interface("interface_e", kc, kr, interface_e_nodes, None, beta, None, clay_nodes, rock_nodes)
problem.interface("interface_f", kc, km, interface_f_nodes, None, beta, None, clay_nodes, mixed_nodes)

# interface intersection
center_node = 1
#                   [center_node, interface1, interface2, material_between, source_center]
problem.intersection("inters_1", center_node, "interface_a", "interface_d", "mixed", beta)
problem.intersection("inters_2", center_node, "interface_d", "interface_c", "clay", beta)
problem.intersection("inters_3", center_node, "interface_c", "interface_a", "rock", beta)
center_node = 2
problem.intersection("inters_4", center_node, "interface_b", "interface_e", "rock", beta)
problem.intersection("inters_5", center_node, "interface_e", "interface_f", "clay", beta)
problem.intersection("inters_6", center_node, "interface_f", "interface_b", "mixed", beta)

#%%
# ====
# Solution
# ====
K, F = problem.continuous_discretization()
np.linalg.cond(K.toarray())

U = sp.linalg.spsolve(K,F)

#%%
# =====
# Plotting solution
# =====
# 2D contour plot
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