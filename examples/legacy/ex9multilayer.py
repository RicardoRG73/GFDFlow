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

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

from GFDFlow.utils import compute_normal_vectors
from GFDFlow.visualization import (
    plot_geometry,
    plot_mesh,
    plot_nodes,
    plot_normal_vectors,
    plot_solution_2d,
    plot_solution_3d,
)


# =====
# Geometry creation
# =====
g = cfg.Geometry()          # geometry object

# points
refinement = 0.1
g.point([-1, -1])         # 0
g.point([1, -1])         # 1
g.point([1, 1])         # 2
g.point([0, 1])       # 3
g.point([-1, 1])         # 4
g.point([0, 0])     # 5 : Center point


# lines
dir = 10
g.line([0, 1], marker=dir)      # 0
g.line([1, 2], marker=dir)      # 1
g.line([2, 3], marker=dir)      # 2
g.line([3, 4], marker=dir)      # 3
g.line([4, 0], marker=dir)      # 4


interf0 = 11
interf1 = 12
interf2 = 13
g.line([5, 0], marker=interf0)  # 5
g.line([5, 1], marker=interf1)  # 6
g.line([5, 3], marker=interf2)  # 7



# surfaces
mat0 = 100
mat1 = 101
mat2 = 102
g.surface([0, 6, 5], marker=mat0)       # 0
g.surface([1, 2, 7, 6], marker=mat1)    # 1
g.surface([3, 4, 5, 7], marker=mat2)    # 2

# geometry plot
plot_geometry(g, title="Geometry", figsize=(7, 7), savepath="examples/legacy/figures/ex9/geometry.png")

# =====
# Mesh creation from geometry object
# =====
mesh = cfm.GmshMesh(g)

mesh.el_type = 2                # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.05

coords, edof, dofs, bdofs, elementmarkers = mesh.create()       # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(coords, edof, mesh.dofs_per_node, mesh.el_type)  # coordinate-edges to vertices-faces

# mesh plot
plot_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True,
    figsize=(7, 7),
    title="Mesh",
    savepath="examples/legacy/figures/ex9/mesh.png",
)

# =====
# Detection of boundary and interior nodes
# =====
dirichlet_nodes = np.asarray(bdofs[dir]) - 1
interf0_nodes = np.asarray(bdofs[interf0]) - 1
interf0_nodes = np.setdiff1d(interf0_nodes,[5,0])
interf1_nodes = np.asarray(bdofs[interf1]) - 1
interf1_nodes = np.setdiff1d(interf1_nodes,[5,1])
interf2_nodes = np.asarray(bdofs[interf2]) - 1
interf2_nodes = np.setdiff1d(interf2_nodes,[5,3])

B = np.hstack((dirichlet_nodes,interf0_nodes,interf1_nodes,interf2_nodes,[5]))

elementmarkers = np.asarray(elementmarkers)

mat0_nodes = faces[elementmarkers == mat0]
mat0_nodes = mat0_nodes.flatten()
mat0_nodes = np.setdiff1d(mat0_nodes,B)

mat1_nodes = faces[elementmarkers == mat1]
mat1_nodes = mat1_nodes.flatten()
mat1_nodes = np.setdiff1d(mat1_nodes,B)

mat2_nodes = faces[elementmarkers == mat2]
mat2_nodes = mat2_nodes.flatten()
mat2_nodes = np.setdiff1d(mat2_nodes,B)

plot_nodes(
    coords,
    {
        "mat0": mat0_nodes,
        "mat1": mat1_nodes,
        "mat2": mat2_nodes,
        "interface0": interf0_nodes,
        "interface1": interf1_nodes,
        "interface2": interf2_nodes,
        "center": [5],
        "dirichlet boundary": dirichlet_nodes,
    },
    figsize=(7, 7),
    point_size=10,
    alpha=0.7,
    savepath="examples/legacy/figures/ex9/nodes.png",
)

# normal vectors
normal_vecs = np.zeros((len(coords), 2))

normal_vecs[interf0_nodes] = compute_normal_vectors(interf0_nodes, coords)
normal_vecs[interf1_nodes] = compute_normal_vectors(interf1_nodes, coords)
normal_vecs[interf2_nodes] = compute_normal_vectors(interf2_nodes, coords)
normal_vecs[[5]] = np.array([1,1])/np.sqrt(2)

# vectors plot
plot_normal_vectors(
    coords,
    normal_vecs,
    [interf0_nodes, interf1_nodes, interf2_nodes, [5]],
    savepath="examples/legacy/figures/ex9/normal_vectors.png",
)

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
problem = gfdmi(coords, faces, normal_vecs, L, source)

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