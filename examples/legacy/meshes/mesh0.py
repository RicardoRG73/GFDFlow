"""Mesh 00: 2D Poisson Equation on an Irregular Curved Domain.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 00. The domain features a circular arc right boundary
with Neumann conditions, and planar left, bottom, and top boundaries with Dirichlet
conditions. It precomputes normal vectors for the curved Neumann boundary and calculates
GFDM support stencils and local pseudo-inverse matrices (M_pinv) for all nodes.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 11): Dirichlet boundary, x = 0
* Right Boundary (marker 12): Neumann circular arc centered at (1, 0) from (2, 0) to (1, 1)
* Bottom Boundary (marker 13): Dirichlet boundary, y = 0
* Top Boundary (marker 14): Dirichlet boundary, y = 1

Material Subdomains
-------------------
* Domain Surface (marker 100): Single isotropic medium (mat0)

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh0.json`
* Figures: `examples/legacy/figures/ex0/{geometry,mesh,nodes,normal_vectors}.png`
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
geometry = cfg.Geometry()

# Control points
geometry.point([0, 0])  # P0
geometry.point([1, 0])  # P1
geometry.point([2, 0])  # P2
geometry.point([1, 1])  # P3
geometry.point([0, 1])  # P4

# Boundary curve markers
left = 11    # Left boundary (Dirichlet)
right = 12   # Right boundary circular arc (Neumann)
bottom = 13  # Bottom boundary (Dirichlet)
top = 14     # Top boundary (Dirichlet)

# Boundary splines and circular arc
geometry.spline([0, 1], marker=bottom)
geometry.spline([1, 2], marker=bottom)
geometry.circle([2, 1, 3], marker=right)
geometry.spline([3, 4], marker=top)
geometry.spline([4, 0], marker=left)

# Surface definition
mat0 = 100
geometry.surface([0, 1, 2, 3, 4], marker=mat0)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(geometry)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.08

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
# 0-based node indices
left_nodes = np.asarray(bdofs[left]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
right_nodes = np.setdiff1d(right_nodes, [2, 3])
bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, [0])
top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, [4])

all_boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
))

elementmarkers = np.asarray(elementmarkers)
interior_nodes = faces[elementmarkers == mat0].flatten()
interior_nodes = np.setdiff1d(interior_nodes, all_boundaries)

nodes = (left_nodes, right_nodes, bottom_nodes, top_nodes, interior_nodes)
labels = ("Left", "Right", "Bottom", "Top", "Interior")

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
# Compute outward unit normal vectors for the curved Neumann boundary
normal_vectors_right = compute_normal_vectors(right_nodes, coords)
normal_vectors = np.zeros((coords.shape[0], 2))
normal_vectors[right_nodes] = normal_vectors_right

# =============================================================================
# 5. GFDM Stencil Selection and Matrix Precomputation
# =============================================================================
M_pinv = {}
support_stencils = {}

# Standard Taylor expansion stencils for Dirichlet and interior nodes
for boundary in (left_nodes, bottom_nodes, top_nodes, interior_nodes):
    for i in boundary:
        support_stencils[i] = get_support_nodes_2D(i, faces)
        M = compute_M_matrix(i, support_stencils[i], coords)
        M_pinv[i] = np.linalg.pinv(M)

# Ghost-node Neumann formulation for curved boundary nodes
for i in right_nodes:
    support_stencils[i] = get_support_nodes_2D(i, faces)
    M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vectors[i])
    M_pinv[i] = np.linalg.pinv(M)

# =============================================================================
# 6. Export Mesh and Stencil Data to JSON
# =============================================================================
if save_mesh_to_file:
    data_to_save = {}
    for b, label in zip(nodes, labels):
        data_to_save[label.lower() + "_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["normal_vectors"] = normal_vectors.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh0.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex0", exist_ok=True)

    # Geometry plot
    plot_geometry(
        geometry,
        title="Geometry",
        savepath="examples/legacy/figures/ex0/geometry.png",
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
        savepath="examples/legacy/figures/ex0/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes),
        title=f"$N = {coords.shape[0]}$",
        savepath="examples/legacy/figures/ex0/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vectors,
        boundary_nodes=right_nodes,
        quiver_color="red",
        quiver_alpha=0.5,
        title="Normal vectors",
        savepath="examples/legacy/figures/ex0/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")