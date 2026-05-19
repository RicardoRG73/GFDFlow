"""
Solution to the Poisson equation
\nabla^2 u = f
in the square domain: `x in [0,2]` and `y in [0,2]`
using 3 different materials
and 3 interfaces
"""


# =====
# Importing needed libraries
# =====
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'src'))
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
cfv.figure(fig_size=(7,7))
cfv.title('Geometry')
cfv.draw_geometry(g)

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
cfv.figure(fig_size=(7,7))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

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

plt.figure(figsize=(7,7))
for label, nodes in zip(
    ["mat0","mat1","mat2","interface0","interface1","interface2","center","dirichlet boundary"],
    [mat0_nodes,mat1_nodes,mat2_nodes,interf0_nodes,interf1_nodes,interf2_nodes,[5],dirichlet_nodes]
):
    plt.scatter(coords[nodes,0], coords[nodes,1], s=10, alpha=0.7, label=label)
plt.axis("equal")
plt.legend()


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
problem = gfdmi(coords, faces, L, source)

problem.material("mat0", k0, mat0_nodes)
problem.material("mat1", k1, mat1_nodes)
problem.material("mat2", k2, mat2_nodes)

problem.dirichlet_boundary("dir", dirichlet_nodes, fd)

problem.interface("interf0", k2, k0, interf0_nodes, None, fi_intersc, None, mat2_nodes, mat0_nodes)
problem.interface("interf1", k0, k1, interf1_nodes, None, fi_intersc, None, mat0_nodes, mat1_nodes)
problem.interface("interf2", k1, k2, interf2_nodes, None, fi_intersc, None, mat1_nodes, mat2_nodes)

center_node = 5
#                   [center_node, interface1, interface2, material_between, source_center]
problem.intersection("interf0", center_node, "interf0", "interf1", "mat0", fi_intersc)
problem.intersection("interf1", center_node, "interf1", "interf2", "mat1", fi_intersc)
problem.intersection("interf2", center_node, "interf2", "interf0", "mat2", fi_intersc)

# ====
# Solution
# ====
K, F = problem.continuous_discretization()

U = sp.linalg.spsolve(K,F)

# =====
# Plotting solution
# =====
# 2D contour plot
plt.figure(figsize=(7,7))
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U,
    levels=25,
    cmap="plasma"
)
plt.colorbar(label="total head")
plt.title("Steady State Solution")
plt.tricontour(
    coords[:,0],
    coords[:,1],
    U,
    triangles=faces,
    levels=25,
    colors="k",
    linewidths=1,
    alpha=0.5
)
plt.axis("equal")

# 3D plot
plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.plot_trisurf(coords[:, 0], coords[:, 1], U, cmap='viridis', edgecolor='k', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U')
plt.title('3D Solution')

plt.show()