"""Mesh 10: Stationary and Transient Diffusion with Circular Inclusion.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 10. The domain is a square [-1, 1] x [-1, 1] containing
a centered circular inclusion of radius r = 0.5. Four planar outer edges form Dirichlet
boundaries, while four circular arc segments compose the material interface separating
the outer matrix from the inner inclusion. Outward unit normal vectors are computed
along the circular interface, and local GFDM support stencils and pseudo-inverse
matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Bottom Boundary (marker 10): Dirichlet outer boundary, y = -1
* Right Boundary (marker 11): Dirichlet outer boundary, x = 1
* Top Boundary (marker 12): Dirichlet outer boundary, y = 1
* Left Boundary (marker 13): Dirichlet outer boundary, x = -1
* Circular Interface (marker 14): Four circular arcs enclosing the inclusion, r = 0.5

Material Subdomains
-------------------
* Matrix Domain (marker 100): Outer host medium, r > 0.5
* Inclusion Domain (marker 101): Inner circular inclusion, r < 0.5

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh10.json`
* Figures: `examples/legacy/figures/ex10/{geometry,mesh,nodes,normal_vectors}.png`
"""

import json
import os

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np

from GFDFlow.utils import (
    compute_M_matrix,
    compute_M_matrix_neumann,
    compute_normal_vectors,
    get_support_nodes_2D,
)
from GFDFlow.visualization import (
    plot_geometry,
    plot_mesh,
    plot_nodes,
    plot_normal_vectors,
)

# Execution flags
save_mesh_to_file = True
show_plots = True
save_figures = True

# Configure visualization style
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# =============================================================================
# 1. Geometry Definition
# =============================================================================
g = cfg.Geometry()

# Outer square corners
interface_elsize = 0.5
g.point([-1, -1])  # P0: Bottom-left
g.point([1, -1])   # P1: Bottom-right
g.point([1, 1])    # P2: Top-right
g.point([-1, 1])   # P3: Top-left

# Circular inclusion control points (center (0, 0), r = 0.5)
g.point([0, 0])                              # P4: Circle center
g.point([0, -0.5], el_size=interface_elsize)  # P5: Circle bottom
g.point([0.5, 0], el_size=interface_elsize)   # P6: Circle right
g.point([0, 0.5], el_size=interface_elsize)   # P7: Circle top
g.point([-0.5, 0], el_size=interface_elsize)  # P8: Circle left

# Outer boundary curve markers
dird = 10
dirr = 11
diru = 12
dirl = 13

# Outer boundary straight lines
g.line([0, 1], marker=dird)
g.line([1, 2], marker=dirr)
g.line([2, 3], marker=diru)
g.line([3, 0], marker=dirl)

# Circular interface arcs
interf = 14
g.circle([5, 4, 6], marker=interf)
g.circle([6, 4, 7], marker=interf)
g.circle([7, 4, 8], marker=interf)
g.circle([8, 4, 5], marker=interf)

# Material surfaces
mat0 = 100  # Host matrix with hole
mat1 = 101  # Inner circular inclusion
g.surface([0, 1, 2, 3], [[4, 5, 6, 7]], marker=mat0)
g.surface([4, 5, 6, 7], marker=mat1)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type,
)

# =============================================================================
# 3. Node Identification and Boundary Extraction
# =============================================================================
b0 = np.asarray(bdofs[dird]) - 1
b1 = np.asarray(bdofs[dirr]) - 1
b1 = np.setdiff1d(b1, [1, 2])
b2 = np.asarray(bdofs[diru]) - 1
b3 = np.asarray(bdofs[dirl]) - 1
b3 = np.setdiff1d(b3, [3, 0])
bi = np.asarray(bdofs[interf]) - 1

all_boundaries = np.hstack((b0, b1, b2, b3, bi))

elementmarkers = np.asarray(elementmarkers)
bm0 = faces[elementmarkers == mat0].flatten()
bm0 = np.setdiff1d(bm0, all_boundaries)

bm1 = faces[elementmarkers == mat1].flatten()
bm1 = np.setdiff1d(bm1, all_boundaries)

nodes_to_plot = (b0, b1, b2, b3, bi, bm0, bm1)
labels = ("b0", "b1", "b2", "b3", "bi", "bm0", "bm1")

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
normal_vecs[bi] = compute_normal_vectors(bi, coords)

# =============================================================================
# 5. GFDM Stencil Selection and Matrix Precomputation
# =============================================================================
M_pinv = {}
support_stencils = {}

for i in range(coords.shape[0]):
    support_stencils[i] = get_support_nodes_2D(i, faces)
    if np.linalg.norm(normal_vecs[i]) > 1e-10:
        M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vecs[i])
    else:
        M = compute_M_matrix(i, support_stencils[i], coords)
    M_pinv[i] = np.linalg.pinv(M)

# =============================================================================
# 6. Export Mesh and Stencil Data to JSON
# =============================================================================
if save_mesh_to_file:
    data_to_save = {}
    for b, label in zip(nodes_to_plot, labels):
        data_to_save[label + "_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["faces"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh10.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex10", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(5, 5),
        savepath="examples/legacy/figures/ex10/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(5, 5),
        title="Mesh",
        savepath="examples/legacy/figures/ex10/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=12,
        figsize=(5, 5),
        savepath="examples/legacy/figures/ex10/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=bi,
        quiver_alpha=0.4,
        title="Normal Vectors",
        figsize=(5, 5),
        savepath="examples/legacy/figures/ex10/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")
