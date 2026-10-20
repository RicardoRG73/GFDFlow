"""Mesh 03: Stationary Groundwater Flow and Transient Diffusion with Phreatic Line.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 03. The physical domain represents a zoned embankment
dam consisting of three distinct zones: an upstream permeable rockfill shell, a central
low-permeability clay core, and a downstream permeable rockfill shell. Normal vectors
are computed along Neumann boundaries and internal interfaces, and local GFDM stencils
and pseudo-inverse matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 108): Upstream reservoir boundary (Dirichlet)
* Right Boundary (marker 100): Downstream seepage face (Dirichlet)
* Bottom Boundary (marker 101): Impermeable dam foundation (Neumann)
* Top Boundary (marker 102): Crest and upper atmospheric surface (Neumann)
* Left Interface (marker 110): Upstream rockfill-clay contact
* Right Interface (marker 111): Clay-downstream rockfill contact

Material Subdomains
-------------------
* Rockfill (marker 100): Upstream and downstream permeable shells
* Clay (marker 102): Central impermeable core

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh3.json`
* Figures: `examples/legacy/figures/ex3/{geometry,mesh,nodes,normal_vectors}.png`
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

# Control points defining dam profile and core boundaries
g.point([3, 0])    # P0: Upstream toe
g.point([21, 0])   # P1: Upstream core base
g.point([29, 0])   # P2: Downstream core base
g.point([40, 0])   # P3: Downstream toe
g.point([47, 0])   # P4: Downstream tailwater limit
g.point([27, 10])  # P5: Downstream crest edge
g.point([23, 10])  # P6: Upstream crest edge
g.point([19, 8])   # P7: Upstream slope shoulder

# Boundary and interface curve markers
left_b = 108
right_b = 100
bottom_b = 101
top_b = 102

left_interface = 110
right_interface = 111

# Boundary curves
g.spline([0, 1], marker=bottom_b)
g.spline([1, 2], marker=bottom_b)
g.spline([2, 3], marker=bottom_b)
g.spline([3, 4], marker=right_b, el_on_curve=15)
g.spline([4, 5], marker=top_b)
g.spline([5, 6], marker=top_b)
g.spline([6, 7], marker=top_b, el_on_curve=10)
g.spline([7, 0], marker=left_b, el_on_curve=15)

# Internal interface curves
g.spline([1, 6], marker=left_interface)
g.spline([2, 5], marker=right_interface)

# Material surfaces
Rockfill = 100
Clay = 102

g.surface([0, 8, 6, 7], marker=Rockfill)  # Upstream rockfill
g.surface([1, 9, 5, 8], marker=Clay)      # Central clay core
g.surface([2, 3, 4, 9], marker=Rockfill)  # Downstream rockfill

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 2.0

coords, edof, dofs, bdofs, element_markers = mesh.create()

# Convert element degree of freedom table to 0-based triangle indices
triangles = np.zeros(edof.shape, dtype=int)
for i, elem in enumerate(edof):
    triangles[i, :] = elem[1], elem[0], elem[2]
triangles = triangles - 1
bdofs = {boundary: np.array(bdofs[boundary]) - 1 for boundary in bdofs}

# =============================================================================
# 3. Node Identification and Boundary Extraction
# =============================================================================
left_nodes = np.asarray(bdofs[left_b])
right_nodes = np.asarray(bdofs[right_b])

bottom_nodes = np.asarray(bdofs[bottom_b])
bottom_nodes = np.setdiff1d(bottom_nodes, np.intersect1d(bottom_nodes, left_nodes))
bottom_nodes = np.setdiff1d(bottom_nodes, np.intersect1d(bottom_nodes, right_nodes))

top_nodes = np.asarray(bdofs[top_b])
top_nodes = np.setdiff1d(top_nodes, np.intersect1d(top_nodes, left_nodes))
top_nodes = np.setdiff1d(top_nodes, np.intersect1d(top_nodes, right_nodes))

left_interface_nodes = np.asarray(bdofs[left_interface])
left_interface_nodes = np.setdiff1d(left_interface_nodes, np.intersect1d(left_interface_nodes, top_nodes))
left_interface_nodes = np.setdiff1d(left_interface_nodes, np.intersect1d(left_interface_nodes, bottom_nodes))

right_interface_nodes = np.asarray(bdofs[right_interface])
right_interface_nodes = np.setdiff1d(right_interface_nodes, np.intersect1d(right_interface_nodes, top_nodes))
right_interface_nodes = np.setdiff1d(right_interface_nodes, np.intersect1d(right_interface_nodes, bottom_nodes))

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    left_interface_nodes,
    right_interface_nodes,
))

element_markers = np.array(element_markers)
rock_nodes = triangles[element_markers == Rockfill].flatten()
rock_nodes = np.setdiff1d(rock_nodes, boundaries)

clay_nodes = triangles[element_markers == Clay].flatten()
clay_nodes = np.setdiff1d(clay_nodes, boundaries)

nodes_to_plot = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    left_interface_nodes,
    right_interface_nodes,
    rock_nodes,
    clay_nodes,
)
labels = (
    "Left",
    "Right",
    "Bottom",
    "Top",
    "Left Interface",
    "Right Interface",
    "Rock",
    "Clay",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
for nodes in (bottom_nodes, top_nodes, left_interface_nodes, right_interface_nodes):
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

# =============================================================================
# 5. GFDM Stencil Selection and Matrix Precomputation
# =============================================================================
M_pinv = {}
support_stencils = {}

for i in range(coords.shape[0]):
    support_stencils[i] = get_support_nodes_2D(i, triangles)
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
    data_to_save["triangles"] = triangles.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh3.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex3", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(6, 4),
        savepath="examples/legacy/figures/ex3/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(6, 4),
        title="Mesh",
        savepath="examples/legacy/figures/ex3/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=10,
        savepath="examples/legacy/figures/ex3/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=(bottom_nodes, top_nodes, left_interface_nodes, right_interface_nodes),
        quiver_color="k",
        quiver_alpha=0.3,
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex3/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")