#%% Importing needed libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d

#%% Loading mesh from file
import json
with open('examples/legacy/meshes/mesh1.json', 'r') as file:
    loaded_data = json.load(file)

for key in loaded_data.keys():
    globals()[key] = np.array(loaded_data[key])

#%% Problem parameters
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
permeability_mat0 = lambda p: 1
permeability_mat1 = lambda p: 0.1
source = lambda p: -1
left_condition = lambda p: 1 - p[1]**2
right_condition = lambda p: 1
bottom_condition = lambda p: 0
top_condition = lambda p: 0
# flux difference at interface du/dn|_{mat0} - du/dn|_{mat1} = beta
flux_difference = lambda p: 0
# solution diference at interface u_{mat0} - u_{mat1} = alpha
solution_difference = lambda p: 0.5


#%% problem definition
problem = gfdmi(coords,triangles,normal_vecs,L,source)

problem.material('material0', permeability_mat0, interior_material_0_nodes)
problem.material('material1', permeability_mat1, interior_material_1_nodes)

problem.neumann_boundary('bottom_left', permeability_mat0, bottom_left_nodes, bottom_condition)
problem.neumann_boundary('top_left', permeability_mat0, top_left_nodes, top_condition)
problem.neumann_boundary('top_right', permeability_mat1, top_right_nodes, top_condition)
problem.neumann_boundary('bottom_right', permeability_mat1, bottom_right_nodes, bottom_condition)

problem.dirichlet_boundary('left', left_nodes, left_condition)
problem.dirichlet_boundary('right', right_nodes, right_condition)

problem.interface(
    'interface',
    permeability_mat0,
    permeability_mat1,
    left_interface_nodes,
    right_interface_nodes,
    flux_difference,
    solution_difference,
    interior_material_0_nodes,
    interior_material_1_nodes
)

#%% System `KU=F` assembling
K,F = problem.discontinuous_discretization()

#%% Solution
U = sp.linalg.spsolve(K,F)

#%% Visualizations using GFDFlow.visualization
plot_solution_2d(
    coords,
    U,
    cmap="plasma",
    levels=11,
    clabel=True,
    savepath="examples/legacy/figures/ex1_contourf.jpg",
)

plot_solution_3d(
    coords,
    U,
    cmap="plasma",
    view_init=(30, -120),
    savepath="examples/legacy/figures/ex1-3d.jpg",
)

plt.show()