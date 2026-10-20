"""Mesh 01: 2D Poisson Equation with Material Interface and Solution Jump.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 01. The domain consists of two adjacent rectangular
subdomains separated by a vertical interface with duplicate nodes to support
discontinuous potential jumps and normal flux balance. Unit normal vectors are
computed for all Neumann boundaries and interface lines.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 100): Dirichlet boundary, x = -1
* Bottom-Left Boundary (marker 101): Neumann boundary, y = 0, x < 0
* Left Interface Line (marker 102): Interface boundary on material 0 side
* Top-Left Boundary (marker 103): Neumann boundary, y = 1, x < 0
* Right Boundary (marker 104): Dirichlet boundary, x = 1
* Top-Right Boundary (marker 105): Neumann boundary, y = 1, x > 0
* Right Interface Line (marker 106): Interface boundary on material 1 side
* Bottom-Right Boundary (marker 107): Neumann boundary, y = 0, x > 0

Material Subdomains
-------------------
* Material 0 Surface (marker 10): Left rectangular subdomain [-1, 0] x [0, 1]
* Material 1 Surface (marker 11): Right rectangular subdomain [0, 1] x [0, 1]

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh1.json`
* Figures: `examples/legacy/figures/ex1/{geometry,mesh,nodes,normal_vectors}.png`
"""

import json
import os

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np

from GFDFlow.utils import compute_normal_vectors
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

# Outer corner points
geometry.point([-1, 0])  # P0
geometry.point([1, 0])   # P1
geometry.point([1, 1])   # P2
geometry.point([-1, 1])  # P3

# Interface duplicate points (small geometric separation delta)
delta = 0.0005
interface_offset = 0.2
el_size_scale_factor = 1.0

geometry.point([-interface_offset - delta, 0], el_size=el_size_scale_factor)  # P4
geometry.point([-interface_offset + delta, 0], el_size=el_size_scale_factor)  # P5
geometry.point([interface_offset - delta, 1], el_size=el_size_scale_factor)   # P6
geometry.point([interface_offset + delta, 1], el_size=el_size_scale_factor)   # P7

# Boundary and interface curve markers
left = 100
neumann_bottom_left = 101
left_interface = 102
neumann_top_left = 103

right = 104
neumann_top_right = 105
right_interface = 106
neumann_bottom_right = 107

# Boundary lines: Material 0 (left subdomain)
geometry.spline([3, 0], marker=left)
geometry.spline([0, 4], marker=neumann_bottom_left)
geometry.spline([4, 6], marker=left_interface)
geometry.spline([6, 3], marker=neumann_top_left)

# Boundary lines: Material 1 (right subdomain)
geometry.spline([1, 2], marker=right)
geometry.spline([2, 7], marker=neumann_top_right)
geometry.spline([5, 7], marker=right_interface)
geometry.spline([5, 1], marker=neumann_bottom_right)

# Material surfaces
mat0 = 10
mat1 = 11
geometry.surface([0, 1, 2, 3], marker=mat0)
geometry.surface([4, 5, 6, 7], marker=mat1)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(geometry)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.04

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
# Dirichlet boundary nodes
left_nodes = np.asarray(bdofs[left]) - 1
right_nodes = np.asarray(bdofs[right]) - 1

# Neumann boundary nodes
bottom_left_nodes = np.asarray(bdofs[neumann_bottom_left]) - 1
bottom_left_nodes = np.setdiff1d(bottom_left_nodes, 0)

top_left_nodes = np.asarray(bdofs[neumann_top_left]) - 1
top_left_nodes = np.setdiff1d(top_left_nodes, 3)

top_right_nodes = np.asarray(bdofs[neumann_top_right]) - 1
top_right_nodes = np.setdiff1d(top_right_nodes, 2)

bottom_right_nodes = np.asarray(bdofs[neumann_bottom_right]) - 1
bottom_right_nodes = np.setdiff1d(bottom_right_nodes, 1)

# Interface duplicate nodes
left_interface_nodes = np.asarray(bdofs[left_interface]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4, 6])

right_interface_nodes = np.asarray(bdofs[right_interface]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [5, 7])

# Interior nodes by material subdomain
elementmarkers = np.asarray(elementmarkers)
boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_left_nodes,
    top_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes,
))

interior_mat0 = faces[elementmarkers == mat0].flatten()
interior_mat0 = np.setdiff1d(interior_mat0, boundaries)

interior_nodes_mat1 = faces[elementmarkers == mat1].flatten()
interior_nodes_mat1 = np.setdiff1d(interior_nodes_mat1, boundaries)

nodes = (
    left_nodes,
    right_nodes,
    bottom_left_nodes,
    top_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes,
    interior_mat0,
    interior_nodes_mat1,
)
labels = (
    "Left",
    "Right",
    "Bottom-Left",
    "Top-Left",
    "Top-Right",
    "Bottom-Right",
    "Left Interface",
    "Right Interface",
    "Interior Material 0",
    "Interior Material 1",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
nodes_to_compute_normals = (
    top_left_nodes,
    bottom_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes,
)
for boundary in nodes_to_compute_normals:
    normals_compact = compute_normal_vectors(boundary, coords)
    normal_vecs[boundary] = normals_compact

# =============================================================================
# 5. Export Mesh Data to JSON
# =============================================================================
if save_mesh_to_file:
    data_to_save = {}
    for b, label in zip(nodes, labels):
        data_to_save[label.replace(" ", "_").replace("-", "_").lower() + "_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()

    with open("examples/legacy/meshes/mesh1.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 6. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex1", exist_ok=True)

    # Geometry plot
    plot_geometry(
        geometry,
        title="Geometry",
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex1/geometry.png",
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
        savepath="examples/legacy/figures/ex1/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.5,
        point_size=20,
        savepath="examples/legacy/figures/ex1/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=nodes_to_compute_normals,
        quiver_alpha=0.5,
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex1/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")