"""Mesh 06: Coupled Variable-Density Seawater Intrusion (Henry Problem).

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 06. The physical domain represents a rectangular coastal
confined aquifer [0, 2] x [0, 1] configured for the classic Henry saltwater intrusion
benchmark. Inward freshwater enters through the left boundary while seawater hydrostatic
pressure acts on the right boundary, bounded above and below by impermeable units.
Normal vectors are computed for all boundaries and corners, and local GFDM support
stencils and pseudo-inverse matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 10): Freshwater inflow boundary, x = 0
* Right Boundary (marker 11): Coastal hydrostatic seawater boundary, x = 2
* Top Boundary (marker 12): Impermeable confining unit, y = 1
* Bottom Boundary (marker 13): Impermeable bedrock base, y = 0

Material Subdomains
-------------------
* Aquifer Domain (marker 0): Homogeneous isotropic porous medium

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh6.json`
* Figures: `examples/legacy/figures/ex6/{geometry,mesh,nodes,normal_vectors}.png`
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
show_figures = show_plots
save_figures = True

# Configure visualization style
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Geometry Definition
# =============================================================================
geometry = cfg.Geometry()
geometria = geometry  # Backward-compatibility alias

# Corner control points
geometry.point([0, 0])  # P0: Bottom-left
geometry.point([2, 0])  # P1: Bottom-right
geometry.point([2, 1])  # P2: Top-right
geometry.point([0, 1])  # P3: Top-left

# Boundary curve markers
left = 10
right = 11
top = 12
bottom = 13

# Boundary lines
geometry.line([0, 1], marker=bottom)
geometry.line([1, 2], marker=right)
geometry.line([2, 3], marker=top)
geometry.line([3, 0], marker=left)

# Surface definition
mat0 = 0
geometry.surface([0, 1, 2, 3], marker=mat0)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
# Discretization resolution reference:
# | el_size_factor |   N   |
# |     0.1        |  274  |
# |     0.05       |  998  |
# |     0.03       | 2748  |
# |     0.02       | 5969  |

mesh = cfm.GmshMesh(geometry)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.03

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
left_nodes = np.asarray(bdofs[left]) - 1
left_nodes = np.setdiff1d(left_nodes, [0, 3])

right_nodes = np.asarray(bdofs[right]) - 1
right_nodes = np.setdiff1d(right_nodes, [1, 2])

bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, [0, 1])

top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, [2, 3])

corner_nodes = np.array([0, 1, 2, 3])

boundary_nodes = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    corner_nodes,
))
interior_nodes = np.setdiff1d(np.arange(coords.shape[0]), boundary_nodes)

nodes_to_plot = (
    interior_nodes,
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    corner_nodes,
)
labels = (
    "Interior",
    "Left",
    "Right",
    "Bottom",
    "Top",
    "Corner",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
nodes_to_compute = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    corner_nodes,
)
for b in nodes_to_compute:
    normal_vecs[b] = compute_normal_vectors(b, coords)

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
        data_to_save[label.replace(" ", "_").replace("-", "_").lower() + "_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["boundary_nodes"] = boundary_nodes.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh6.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_figures:
    os.makedirs("examples/legacy/figures/ex6", exist_ok=True)

    # Geometry plot
    plot_geometry(
        geometry,
        title="Geometry",
        font_size=16,
        figsize=(8, 5),
        savepath="examples/legacy/figures/ex6/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(7, 4),
        title=f"Mesh $N={coords.shape[0]}$",
        savepath="examples/legacy/figures/ex6/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.5,
        point_size=10,
        figsize=(7, 4),
        savepath="examples/legacy/figures/ex6/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=nodes_to_compute,
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex6/normal_vectors.png",
    )

    if show_figures:
        plt.show()
    else:
        plt.close("all")
