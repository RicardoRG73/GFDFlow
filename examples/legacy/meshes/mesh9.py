"""Mesh 09: Multi-Layered Domain with Three Materials and Central Vertex Intersection.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 09. The domain is a square [-1, 1] x [-1, 1] partitioned
into three angular material sectors that converge at a central singular vertex (node 5
at (0, 0)). Internal interface segments separate the three distinct material conductivity
zones. Unit normal vectors are computed along all interface segments, and a directional
normal is assigned to the central triple-junction node. Local GFDM support stencils
and pseudo-inverse matrices (M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Outer Boundaries (marker 10): External square Dirichlet boundaries
* Interface 0 (marker 11): Diagonal ray from (0, 0) to (-1, -1)
* Interface 1 (marker 12): Diagonal ray from (0, 0) to (1, -1)
* Interface 2 (marker 13): Vertical ray from (0, 0) to (0, 1)

Material Subdomains
-------------------
* Sector 0 (marker 100): Lower triangular wedge
* Sector 1 (marker 101): Right polygonal sector
* Sector 2 (marker 102): Left/upper polygonal sector

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh9.json`
* Figures: `examples/legacy/figures/ex9/{geometry,mesh,nodes,normal_vectors}.png`
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

# Control points: outer perimeter and central vertex
g.point([-1, -1])  # P0: Bottom-left
g.point([1, -1])   # P1: Bottom-right
g.point([1, 1])    # P2: Top-right
g.point([0, 1])    # P3: Top-middle
g.point([-1, 1])   # P4: Top-left
g.point([0, 0])    # P5: Central triple-junction vertex

# Outer boundary lines
dir_marker = 10
g.line([0, 1], marker=dir_marker)  # L0
g.line([1, 2], marker=dir_marker)  # L1
g.line([2, 3], marker=dir_marker)  # L2
g.line([3, 4], marker=dir_marker)  # L3
g.line([4, 0], marker=dir_marker)  # L4

# Internal interface rays converging at P5
interf0 = 11
interf1 = 12
interf2 = 13
g.line([5, 0], marker=interf0)  # L5: Interface 0
g.line([5, 1], marker=interf1)  # L6: Interface 1
g.line([5, 3], marker=interf2)  # L7: Interface 2

# Material sector surfaces
mat0 = 100
mat1 = 101
mat2 = 102
g.surface([0, 6, 5], marker=mat0)     # Sector 0
g.surface([1, 2, 7, 6], marker=mat1)  # Sector 1
g.surface([3, 4, 5, 7], marker=mat2)  # Sector 2

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.05

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
dirichlet_nodes = np.asarray(bdofs[dir_marker]) - 1

interf0_nodes = np.asarray(bdofs[interf0]) - 1
interf0_nodes = np.setdiff1d(interf0_nodes, [5, 0])

interf1_nodes = np.asarray(bdofs[interf1]) - 1
interf1_nodes = np.setdiff1d(interf1_nodes, [5, 1])

interf2_nodes = np.asarray(bdofs[interf2]) - 1
interf2_nodes = np.setdiff1d(interf2_nodes, [5, 3])

boundaries = np.hstack((
    dirichlet_nodes,
    interf0_nodes,
    interf1_nodes,
    interf2_nodes,
    [5],
))

elementmarkers = np.asarray(elementmarkers)
mat0_nodes = faces[elementmarkers == mat0].flatten()
mat0_nodes = np.setdiff1d(mat0_nodes, boundaries)

mat1_nodes = faces[elementmarkers == mat1].flatten()
mat1_nodes = np.setdiff1d(mat1_nodes, boundaries)

mat2_nodes = faces[elementmarkers == mat2].flatten()
mat2_nodes = np.setdiff1d(mat2_nodes, boundaries)

nodes_to_plot = (
    mat0_nodes,
    mat1_nodes,
    mat2_nodes,
    interf0_nodes,
    interf1_nodes,
    interf2_nodes,
    dirichlet_nodes,
)
labels = (
    "mat0",
    "mat1",
    "mat2",
    "interf0",
    "interf1",
    "interf2",
    "dirichlet",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
normal_vecs = np.zeros((coords.shape[0], 2))
normal_vecs[interf0_nodes] = compute_normal_vectors(interf0_nodes, coords)
normal_vecs[interf1_nodes] = compute_normal_vectors(interf1_nodes, coords)
normal_vecs[interf2_nodes] = compute_normal_vectors(interf2_nodes, coords)
normal_vecs[[5]] = np.array([1.0, 1.0]) / np.sqrt(2.0)

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

    with open("examples/legacy/meshes/mesh9.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex9", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(6, 5),
        savepath="examples/legacy/figures/ex9/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(6, 5),
        title="Mesh",
        savepath="examples/legacy/figures/ex9/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=12,
        figsize=(6, 5),
        savepath="examples/legacy/figures/ex9/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=(interf0_nodes, interf1_nodes, interf2_nodes, np.array([5])),
        quiver_alpha=0.4,
        title="Normal Vectors",
        figsize=(6, 5),
        savepath="examples/legacy/figures/ex9/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")
