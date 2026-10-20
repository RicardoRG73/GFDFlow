"""Mesh 08: 2D Confined Seepage Under a Dam with a Sheet Pile.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 08. The domain represents an alluvial permeable foundation
[0, 90] x [0, 30] beneath a concrete dam structure equipped with an impermeable vertical
cutoff sheet pile wall. Fine nodal discretization is applied around the sheet pile tip
and corners where hydraulic gradients are steepest. Outward unit normal vectors are
computed for all impermeable Neumann boundaries, and local GFDM support stencils and
pseudo-inverse matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 10): Upstream permeable riverbed, Dirichlet head h1
* Right Boundary (marker 11): Downstream permeable riverbed, Dirichlet head h2
* Neumann Boundary (marker 12): Impermeable bottom, lateral edges, dam base, and sheet pile wall

Material Subdomains
-------------------
* Foundation Domain: Homogeneous permeable soil foundation

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh8.json`
* Figures: `examples/legacy/figures/ex8/{geometry,mesh,nodes,normal_vectors}.png`
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

# Foundation and dam profile control points with localized refinement
g.point([0, 0])                    # P0: Deep bedrock bottom-left
g.point([90, 0])                   # P1: Deep bedrock bottom-right
g.point([90, 30])                  # P2: Downstream ground surface edge
g.point([60, 30], el_size=0.4)     # P3: Downstream dam toe
g.point([60, 16], el_size=0.1)     # P4: Downstream sheet pile tip
g.point([59, 16], el_size=0.1)     # P5: Downstream sheet pile base
g.point([59, 27], el_size=0.6)     # P6: Dam downstream base
g.point([31, 27], el_size=0.6)     # P7: Dam upstream base
g.point([31, 12], el_size=0.1)     # P8: Upstream sheet pile base
g.point([30, 12], el_size=0.1)     # P9: Upstream sheet pile tip
g.point([30, 30], el_size=0.4)     # P10: Upstream dam heel
g.point([0, 30])                   # P11: Upstream ground surface edge

# Boundary markers
left = 10
right = 11
neumann = 12

# Boundary splines
g.spline([0, 1], marker=neumann)     # Foundation bottom bedrock
g.spline([1, 2], marker=neumann)     # Downstream vertical boundary
g.spline([2, 3], marker=right)       # Downstream permeable riverbed (Dirichlet)
g.spline([3, 4], marker=neumann)     # Sheet pile right wall
g.spline([4, 5], marker=neumann)     # Sheet pile tip
g.spline([5, 6], marker=neumann)     # Sheet pile left wall
g.spline([6, 7], marker=neumann)     # Concrete dam base
g.spline([7, 8], marker=neumann)     # Upstream cutoff wall right edge
g.spline([8, 9], marker=neumann)     # Upstream cutoff wall tip
g.spline([9, 10], marker=neumann)    # Upstream cutoff wall left edge
g.spline([10, 11], marker=left)      # Upstream permeable riverbed (Dirichlet)
g.spline([11, 0], marker=neumann)    # Upstream vertical boundary

# Permeable foundation surface
g.surface([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g, el_size_factor=2.0)

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
right_nodes = np.asarray(bdofs[right]) - 1
neumann_nodes = np.asarray(bdofs[neumann]) - 1

# Eliminate shared corner nodes
neumann_nodes = np.setdiff1d(neumann_nodes, right_nodes)
neumann_nodes = np.setdiff1d(neumann_nodes, left_nodes)

N = coords.shape[0]
boundary_nodes = np.hstack((left_nodes, right_nodes, neumann_nodes))
interior_nodes = np.setdiff1d(np.arange(N), boundary_nodes)

nodes_to_plot = (
    interior_nodes,
    left_nodes,
    right_nodes,
    neumann_nodes,
)
labels = (
    "interior",
    "left",
    "right",
    "neumann",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs_compact = compute_normal_vectors(neumann_nodes, coords)
normal_vecs = np.zeros((N, 2))
normal_vecs[neumann_nodes] = normal_vecs_compact

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

    with open("examples/legacy/meshes/mesh8.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex8", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex8/geometry.png",
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
        savepath="examples/legacy/figures/ex8/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=10,
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex8/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=neumann_nodes,
        quiver_alpha=0.3,
        title="Normal Vectors",
        figsize=(8, 4),
        savepath="examples/legacy/figures/ex8/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")
