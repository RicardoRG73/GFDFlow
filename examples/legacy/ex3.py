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
from GFDFlow.visualization import (
    plot_phreatic_surface,
    plot_solution_2d,
    plot_solution_3d,
)


# loading mesh data
import json
with open("examples/legacy/meshes/mesh3.json","r") as f:
    mesh_data = json.load(f)

for key in mesh_data.keys():
    globals()[key] = np.array(mesh_data[key])

#%%
# =============================================================================
# Problem parameters
# =============================================================================
L = np.array([0,0,0,1,0,1])
kr = lambda p: 1
kc = lambda p: 1e-1
source = lambda p: 0
neumann_cond = lambda p: 0
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

problem.material("rock", kr, rock_nodes)
problem.material("clay", kc, clay_nodes)

problem.neumann_boundary("bottom", kr, bottom_nodes, neumann_cond)
problem.neumann_boundary("top", kr, top_nodes, neumann_cond)

problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)

problem.interface(
    "left_interface",
    kr,
    kc,
    left_interface_nodes,
    None,
    beta,
    None,
    rock_nodes,
    clay_nodes
)

problem.interface(
    "right_interface",
    kc,
    kr,
    right_interface_nodes,
    None,
    beta,
    None,
    clay_nodes,
    rock_nodes
)


#%% system KU=F assembling
K,F = problem.continuous_discretization()

#%% system KU=F solution
U = sp.linalg.spsolve(K,F)

#%%
# =============================================================================
# Plotting U
# =============================================================================
# 3D
plot_solution_3d(
    coords,
    U,
    cmap="inferno",
    title=r"Stationary solution $\nabla^2 u = 0$",
    savepath="examples/legacy/figures/ex3/3dplot_stationary.png",
)

#%% contourf with phreatic line
fig, ax = plot_solution_2d(
    coords,
    U,
    cmap="inferno",
    levels=20,
    title=r"Stationary solution $\nabla^2 u = 0$",
    savepath="examples/legacy/figures/ex3/contourf_stationary.png",
)
plot_phreatic_surface(ax, coords, U, color="b")

#%%
# =============================================================================
# Difusion equation
# \nabla^2 u + f = du/dt
# =============================================================================
t = [0,80]
fun = lambda t,U: K@U - F
U0 = np.zeros(coords.shape[0])
U0[left_nodes] = 8
U0[right_nodes] = 0

#%% initial condition plot
plot_solution_3d(
    coords,
    U0,
    cmap="inferno",
    title="Initial Condition $U_0$",
    savepath="examples/legacy/figures/ex3/3dplot_u0.png",
)

#%% solution
sol = solve_ivp(fun, t, U0)

U_difussion = sol.y

#%% plots
fig = plt.figure()

final_index = sol.t.shape[0] - 1
times_index = [0, final_index//10, final_index//3, final_index]

for i,t_i in enumerate(times_index):
    ax = plt.subplot(2,2,i+1)
    plot_solution_2d(
        coords,
        U_difussion[:,t_i],
        levels=20,
        cmap="inferno",
        colorbar=False,
        contour_lines=False,
        ax=ax,
        title=f"$t = {sol.t[t_i]:1.2f}$",
    )
    plot_phreatic_surface(ax, coords, U_difussion[:,t_i], color="k", linewidths=0.5, label=None)

fig.savefig("examples/legacy/figures/ex3/diffusion_steps.png", dpi=300, bbox_inches="tight")

#%% 3d plot at final time
plot_solution_3d(
    coords,
    U_difussion[:,final_index],
    cmap="inferno",
    title=f"Solution $U$ at time $t={sol.t[-1]:1.2f}$",
    savepath="examples/legacy/figures/ex3/3dplot.png",
)

# condition number
print("\n\n Condition number cond(K): %1.3e" %np.linalg.cond(K.toarray()))

plt.show()