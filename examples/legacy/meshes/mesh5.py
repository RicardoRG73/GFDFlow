"""Mesh 05: 2D Porous Media Flow with Curved Circular Inclusions.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 05. The domain [0, 2] x [0, 1] represents a composite
porous medium partitioned by a vertical interface at x = 1. The left subdomain contains
an embedded circular inclusion of low permeability, while the right subdomain contains
an internal circular cavity boundary with prescribed Dirichlet potential. Normal vectors
are computed along Neumann boundaries, the vertical interface, and circular curves.
GFDM support stencils and local pseudo-inverse matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 10): Dirichlet boundary, x = 0
* Right Boundary (marker 11): Dirichlet boundary, x = 2
* Left-Top Boundary (marker 12): Neumann boundary, y = 1, x < 1
* Right-Top Boundary (marker 13): Neumann boundary, y = 1, x > 1
* Left-Bottom Boundary (marker 15): Neumann boundary, y = 0, x < 1
* Right-Bottom Boundary (marker 16): Neumann boundary, y = 0, x > 1
* Interface (marker 17): Vertical material interface, x = 1
* Left Circle (marker 18): Interface of embedded circular inclusion
* Right Circle (marker 19): Internal circular Dirichlet boundary

Material Subdomains
-------------------
* Left Subdomain (marker 0): Sand matrix
* Right Subdomain (marker 1): Rock matrix
* Circular Inclusion (marker 2): Low-permeability inclusion

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh5.json`
* Figures: `examples/legacy/figures/ex5/{geometry,mesh,nodes,normal_vectors}.png`
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

# Outer domain corners
g.point([0, 0])  # P0
g.point([2, 0])  # P1
g.point([2, 1])  # P2
g.point([0, 1])  # P3

# Interface intersection points
g.point([1, 0])  # P4
g.point([1, 1])  # P5

# Left circle control points (center at (0.5, 0.5), r = 0.2)
g.point([0.5, 0.5])  # P6: Center
g.point([0.7, 0.5])  # P7
g.point([0.5, 0.7])  # P8
g.point([0.3, 0.5])  # P9
g.point([0.5, 0.3])  # P10

# Right circle control points (center at (1.5, 0.5), r = 0.2)
g.point([1.5, 0.5])  # P11: Center
g.point([1.7, 0.5])  # P12
g.point([1.5, 0.7])  # P13
g.point([1.3, 0.5])  # P14
g.point([1.5, 0.3])  # P15

# Boundary and curve markers
left_marker = 10
right_marker = 11
left_top_marker = 12
right_top_marker = 13
left_bottom_marker = 15
right_bottom_marker = 16
interface_marker = 17
left_circ_mark = 18
right_circ_mark = 19

# Left subdomain boundaries
g.spline([0, 4], marker=left_bottom_marker)
g.spline([4, 5], marker=interface_marker)
g.spline([5, 3], marker=left_top_marker)
g.spline([3, 0], marker=left_marker)

# Right subdomain boundaries
g.spline([4, 1], marker=right_bottom_marker)
g.spline([1, 2], marker=right_marker)
g.spline([2, 5], marker=right_top_marker)

# Left circular inclusion curves
g.circle([7, 6, 8], marker=left_circ_mark)
g.circle([8, 6, 9], marker=left_circ_mark)
g.circle([9, 6, 10], marker=left_circ_mark)
g.circle([10, 6, 7], marker=left_circ_mark)

# Right circular cavity curves
g.circle([12, 11, 13], marker=right_circ_mark)
g.circle([13, 11, 14], marker=right_circ_mark)
g.circle([14, 11, 15], marker=right_circ_mark)
g.circle([15, 11, 12], marker=right_circ_mark)

# Material surfaces
left_half_mark = 0
right_half_mark = 1
left_circ_surf_mark = 2

g.surface(outer_loop=[0, 1, 2, 3], holes=[[7, 8, 9, 10]], marker=left_half_mark)
g.surface([7, 8, 9, 10], marker=left_circ_surf_mark)
g.surface([4, 5, 6, 1], [[11, 12, 13, 14]], marker=right_half_mark)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g, el_size_factor=0.05)

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
left_nodes = np.asarray(bdofs[left_marker]) - 1
right_nodes = np.asarray(bdofs[right_marker]) - 1
interface_nodes = np.asarray(bdofs[interface_marker]) - 1

left_top_nodes = np.asarray(bdofs[left_top_marker]) - 1
left_top_nodes = np.setdiff1d(left_top_nodes, [5, 3])

left_bottom_nodes = np.asarray(bdofs[left_bottom_marker]) - 1
left_bottom_nodes = np.setdiff1d(left_bottom_nodes, [0, 4])

right_bottom_nodes = np.asarray(bdofs[right_bottom_marker]) - 1
right_bottom_nodes = np.setdiff1d(right_bottom_nodes, [4, 1])

right_top_nodes = np.asarray(bdofs[right_top_marker]) - 1
right_top_nodes = np.setdiff1d(right_top_nodes, [2, 5])

left_circ_nodes = np.asarray(bdofs[left_circ_mark]) - 1
right_circ_nodes = np.asarray(bdofs[right_circ_mark]) - 1

boundary_nodes = np.hstack((
    left_nodes,
    right_nodes,
    interface_nodes,
    left_top_nodes,
    left_bottom_nodes,
    right_top_nodes,
    right_bottom_nodes,
    left_circ_nodes,
    right_circ_nodes,
))

# Interior nodes by material subdomain
elementmarkers = np.asarray(elementmarkers)

left_half_nodes = faces[elementmarkers == left_half_mark].flatten()
left_half_nodes = np.setdiff1d(left_half_nodes, boundary_nodes)

left_circ_mat_nodes = faces[elementmarkers == left_circ_surf_mark].flatten()
left_circ_mat_nodes = np.setdiff1d(left_circ_mat_nodes, boundary_nodes)

right_half_nodes = faces[elementmarkers == right_half_mark].flatten()
right_half_nodes = np.setdiff1d(right_half_nodes, boundary_nodes)

nodes_to_plot = (
    left_nodes,
    right_nodes,
    interface_nodes,
    left_top_nodes,
    left_bottom_nodes,
    right_top_nodes,
    right_bottom_nodes,
    left_circ_nodes,
    right_circ_nodes,
    left_half_nodes,
    left_circ_mat_nodes,
    right_half_nodes,
)
labels = (
    "left",
    "right",
    "interface",
    "left_top",
    "left_bottom",
    "right_top",
    "right_bottom",
    "left_circ",
    "right_circ",
    "left_half",
    "left_circ_mat",
    "right_half",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
nodes_to_compute = (
    left_top_nodes,
    left_bottom_nodes,
    right_top_nodes,
    right_bottom_nodes,
    left_circ_nodes,
    interface_nodes,
)
for nodes in nodes_to_compute:
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

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

    with open("examples/legacy/meshes/mesh5.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex5", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex5/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(8, 4),
        title="Mesh",
        savepath="examples/legacy/figures/ex5/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=10,
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex5/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=nodes_to_compute,
        quiver_alpha=0.3,
        title="Normal Vectors",
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex5/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")
