"""
Solution to the Poisson equation:
\nabla^2 u = f

in the square domain: x in [-1,1] and y in [-1,1]

Implementing 3 different materials
with 3 interfaces passing through the center node
(coordinates (0,0) and index 5)
"""


# =====
# Importing needed libraries
# =====
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import json
with open('examples/legacy/meshes/mesh9.json', 'r') as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
faces = np.array(mesh_data["faces"])
normal_vecs = np.array(mesh_data["normal_vecs"])
mat0_nodes = np.array(mesh_data["mat0_nodes"])
mat1_nodes = np.array(mesh_data["mat1_nodes"])
mat2_nodes = np.array(mesh_data["mat2_nodes"])
interf0_nodes = np.array(mesh_data["interf0_nodes"])
interf1_nodes = np.array(mesh_data["interf1_nodes"])
interf2_nodes = np.array(mesh_data["interf2_nodes"])
dirichlet_nodes = np.array(mesh_data["dirichlet_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =====
# Problem parameters
# =====
k0 = lambda p: 5                                    # mat0 permeability
k1 = lambda p: 500                                     # mat1 permeability
k2 = lambda p: 100
fd = lambda p: np.exp(p[0] + p[1])                           # Dirichlet condition
fi0 = lambda p: 0                            # interface condition
fi1 = lambda p: 0
fi2 = lambda p: 0
fi_intersc = lambda p: 0

source = lambda p: -100
L=np.array([0,0,0,1,0,1])

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d
problem = gfdmi(coords, faces, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material("mat0", k0, mat0_nodes)
problem.material("mat1", k1, mat1_nodes)
problem.material("mat2", k2, mat2_nodes)

problem.dirichlet_boundary("dir", dirichlet_nodes, fd)

problem.interface("interf0", k2, k0, interf0_nodes, None, fi_intersc, None, mat2_nodes, mat0_nodes)
problem.interface("interf1", k0, k1, interf1_nodes, None, fi_intersc, None, mat0_nodes, mat1_nodes)
problem.interface("interf2", k1, k2, interf2_nodes, None, fi_intersc, None, mat1_nodes, mat2_nodes)

center_node = 5
#                   [center_node, interface1, interface2, material_between, source_center]
problem.intersection("intersection1", center_node, "interf0", "interf1", "mat0", fi_intersc)
problem.intersection("intersection2", center_node, "interf1", "interf2", "mat1", fi_intersc)
problem.intersection("intersection3", center_node, "interf2", "interf0", "mat2", fi_intersc)

# ====
# Solution
# ====
K, F = problem.continuous_discretization()

U = sp.linalg.spsolve(K,F)

# =====
# Plotting solution
# =====
# 2D contour plot
plot_solution_2d(
    coords,
    U,
    triangles=faces,
    levels=25,
    cmap="inferno",
    colorbar_label="total head",
    title="Steady State Solution",
    figsize=(7, 7),
    linewidths=1,
    line_alpha=0.5,
    overlay_nodes= {
        "interface0": interf0_nodes,
        "interface1": interf1_nodes,
        "interface2": interf2_nodes,
        "center": [5]
    },
    savepath="examples/legacy/figures/ex9/contourf.png",
)

plot_solution_3d(
    coords,
    U,
    triangles=faces,
    cmap="inferno",
    edge_color="k",
    alpha=0.7,
    title="3D Solution",
    figsize=(7, 7),
    savepath="examples/legacy/figures/ex9/3dplot.png",
)

plt.show()