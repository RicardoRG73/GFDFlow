"""Mesh 07: Buoyancy-Driven Convection and Salt Fingering (Elder Problem).

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 07. The domain represents an aspect-ratio 4 closed porous
cavity [0, 4] x [0, 1] configured for the Elder benchmark of transient buoyancy-driven
natural convection and salt fingering. The central segment of the upper boundary [1, 3] x {1}
acts as the brine infiltration source, with local nodal refinement. Exact outward normal
vectors are assigned along all external boundaries, and local GFDM support stencils and
pseudo-inverse matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Bottom Boundary (marker 10): Impermeable cavity base, y = 0
* Right Boundary (marker 11): Impermeable lateral edge, x = 4
* Top-Right Boundary (marker 12): Impermeable top segment, 3 < x <= 4, y = 1
* Top-Middle Boundary (marker 13): Constant concentration brine source, 1 <= x <= 3, y = 1
* Top-Left Boundary (marker 14): Impermeable top segment, 0 <= x < 1, y = 1
* Left Boundary (marker 15): Impermeable lateral edge, x = 0

Material Subdomains
-------------------
* Cavity Domain: Homogeneous isotropic porous medium

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh7.json`
* Figures: `examples/legacy/figures/ex7/{geometry,mesh,nodes,normal_vectors}.png`
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
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Geometry Definition
# =============================================================================
g = cfg.Geometry()

# Control points: [0, 4] x [0, 1] with refinement on top central source segment
g.point([0, 0])  # P0: Bottom-left
g.point([4, 0])  # P1: Bottom-right
g.point([4, 1])  # P2: Top-right
g.point([3, 1], el_size=0.3)  # P3: Source right end
g.point([1, 1], el_size=0.3)  # P4: Source left end
g.point([0, 1])  # P5: Top-left

# Boundary markers
bottom = 10
right = 11
top_right = 12
top_middle = 13
top_left = 14
left = 15

# Boundary lines
g.line([0, 1], marker=bottom)
g.line([1, 2], marker=right)
g.line([2, 3], marker=top_right)
g.line([3, 4], marker=top_middle)
g.line([4, 5], marker=top_left)
g.line([5, 0], marker=left)

# Cavity surface
g.surface([0, 1, 2, 3, 4, 5])

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g, el_size_factor=0.1)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type,
)

print(f"Number of nodes N = {coords.shape[0]}")

# =============================================================================
# 3. Node Identification and Boundary Extraction
# =============================================================================
bottom_nodes = np.asarray(bdofs[bottom]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
top_right_nodes = np.asarray(bdofs[top_right]) - 1
top_middle_nodes = np.asarray(bdofs[top_middle]) - 1
top_left_nodes = np.asarray(bdofs[top_left]) - 1
left_nodes = np.asarray(bdofs[left]) - 1

# Disambiguate shared boundary intersection nodes
top_right_nodes = np.setdiff1d(top_right_nodes, top_middle_nodes)
top_left_nodes = np.setdiff1d(top_left_nodes, top_middle_nodes)
right_nodes = np.setdiff1d(right_nodes, top_right_nodes)
right_nodes = np.setdiff1d(right_nodes, bottom_nodes)
left_nodes = np.setdiff1d(left_nodes, top_left_nodes)
left_nodes = np.setdiff1d(left_nodes, bottom_nodes)

boundary_nodes = np.hstack((
    bottom_nodes,
    right_nodes,
    top_right_nodes,
    top_middle_nodes,
    top_left_nodes,
    left_nodes,
))

N = coords.shape[0]
interior_nodes = np.setdiff1d(np.arange(N), boundary_nodes)

nodes_to_plot = (
    interior_nodes,
    bottom_nodes,
    right_nodes,
    top_right_nodes,
    top_middle_nodes,
    top_left_nodes,
    left_nodes,
)
labels = (
    "interior",
    "bottom",
    "right",
    "top_right",
    "top_middle",
    "top_left",
    "left",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
normal_vecs[bottom_nodes] = np.array([0.0, -1.0])
normal_vecs[right_nodes] = np.array([1.0, 0.0])
normal_vecs[top_right_nodes] = np.array([0.0, 1.0])
normal_vecs[top_middle_nodes] = np.array([0.0, 1.0])
normal_vecs[top_left_nodes] = np.array([0.0, 1.0])
normal_vecs[left_nodes] = np.array([-1.0, 0.0])

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
    data_to_save["triangles"] = faces.tolist()
    data_to_save["boundary_nodes"] = boundary_nodes.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh7.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_figures:
    os.makedirs("examples/legacy/figures/ex7", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(8, 3),
        savepath="examples/legacy/figures/ex7/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(8, 3),
        title="Mesh",
        suptitle=f"el_size_factor={mesh.el_size_factor}, N={coords.shape[0]} nodes",
        savepath="examples/legacy/figures/ex7/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=1.0,
        point_size=20,
        figsize=(8, 3),
        savepath="examples/legacy/figures/ex7/nodes.png",
    )

    # Normal vectors plot
    b_nodes_tuple = (
        bottom_nodes,
        right_nodes,
        top_right_nodes,
        top_middle_nodes,
        top_left_nodes,
        left_nodes,
    )
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=b_nodes_tuple,
        title="Normal vectors",
        savepath="examples/legacy/figures/ex7/normal_vectors.png",
    )

    if show_figures:
        plt.show()
    else:
        plt.close("all")