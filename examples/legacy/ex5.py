# %% [markdown]
# Example 06
# 
# Solution of the Poisson equation in 2D
# 
# $$
# k_i \nabla^2 u = f
# $$
# 
# with $k_i$ discontinuous due to materials with different properties, but with $u$ continuous.

# %% [markdown]
# # Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

from scipy.sparse.linalg import spsolve

from GFDFlow import (
    GFDMI_2D_problem as gfdmi,
    plot_geometry,
    plot_mesh,
    plot_nodes,
    plot_normal_vectors,
    plot_solution_2d,
    plot_solution_3d,
)

import json
with open('examples/legacy/meshes/mesh5.json', 'r') as file:
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


# %% [markdown]
# # Problem parameters

# %%
# coeffitients L = [A, B, C, D, E, F] in differential operator
# Au + Bu_x + Cu_y + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
# permeability / conductivity
k_sand = lambda p: 1
k_rock = lambda p: 0.3
k_rock_circ = lambda p: 0.1
# source
source = lambda p: 0
# boundaries conditions
neumann_cond = lambda p: 0
left_dirichlet = lambda p: 1
right_dirichlet = lambda p: 0

# interface flux. u_n|_{M0} - u_n|_{M1} = beta
beta = lambda p: 0

# Problem Assembling

problem = gfdmi(
    coords,
    faces,
    normal_vecs,
    L,
    source,
    M_pinv=M_pinv,
    support_stencils=support_stencils
)

# interior nodes
problem.material("sand", k_sand, left_half_nodes)
problem.material("rock", k_rock, right_half_nodes)
problem.material("rock_circ", k_rock_circ, left_circ_mat_nodes)

# neumann boundaries
problem.neumann_boundary("left_top", k_sand, left_top_nodes, neumann_cond)
problem.neumann_boundary("left_bottom", k_sand, left_bottom_nodes, neumann_cond)
problem.neumann_boundary("right_top", k_rock, right_top_nodes, neumann_cond)
problem.neumann_boundary("right_bottom", k_rock, right_bottom_nodes, neumann_cond)

# dirichlet boundaries
problem.dirichlet_boundary("izq", left_nodes, left_dirichlet)
problem.dirichlet_boundary("der", right_nodes, right_dirichlet)
problem.dirichlet_boundary("right_circle", right_circ_nodes, lambda p: 0.75 - p[0]/3)

# interfaces
problem.interface(
    "left_circle",
    k_sand,
    k_rock_circ,
    left_circ_nodes,
    None,
    beta,
    None,
    left_half_nodes,
    left_circ_mat_nodes
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
    right_half_nodes
)

# %% [markdown]
# # Problem solution
# 
# For problems with a continuous solution $u$ ($u_{M_0} - u_{M_1} = \alpha$, with $\alpha = 0$), it is preferable to use the `create_system_K_F_cont_U` function from the `GFDMI` module.

# %%
K,F = problem.continuous_discretization()

U = spsolve(K,F)

# %% [markdown]
# # Plotting solution

# %%
# 2D contour plot with interface nodes overlay
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

# %%
# 3D surface plot
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
