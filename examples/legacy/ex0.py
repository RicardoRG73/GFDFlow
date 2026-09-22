import json
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d

with open('examples/legacy/meshes/mesh0.json', 'r') as file:
    loaded_data = json.load(file)

left_nodes = np.array(loaded_data["left_nodes"])
right_nodes = np.array(loaded_data["right_nodes"])
bottom_nodes = np.array(loaded_data["bottom_nodes"])
top_nodes = np.array(loaded_data["top_nodes"])
interior_nodes = np.array(loaded_data["interior_nodes"])
coords = np.array(loaded_data["coords"])
triangles = np.array(loaded_data["triangles"])
normal_vectors = np.array(loaded_data["normal_vectors"])
support_stencils = {int(k): np.array(v) for k, v in loaded_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in loaded_data["M_pinv"].items()}


#%% Problem parameters
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
permeability = lambda p: 1
source = lambda p: -2
left_condition = lambda p: 0
right_condition = lambda p: 0
bottom_condition = lambda p: p[0] * 0.5
top_condition = lambda p: p[0]

# problem definition
problem = gfdmi(coords,triangles, normal_vectors, L, source, support_stencils, M_pinv)

problem.material('0', permeability, interior_nodes)

problem.neumann_boundary('right', permeability, right_nodes, right_condition)

problem.dirichlet_boundary('left', left_nodes, left_condition)
problem.dirichlet_boundary('top', top_nodes, top_condition)
problem.dirichlet_boundary('bottom', bottom_nodes, bottom_condition)


#%% System `KU=F` assembling
K,F = problem.continuous_discretization()

#%% Solution to KU=F
U = sp.linalg.spsolve(K,F)

#%% Visualizations using GFDFlow.visualization
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