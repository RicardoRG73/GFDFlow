"""Mesh 11: Seepage Analysis in a Tailings Dam and Multi-Zone Foundation.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 11. The domain models a full-scale geotechnical tailings dam
cross-section [-400, 1365] x [-100, 360] comprising impounded tailings, an engineered
dam embankment, and underlying bedrock foundation. Internal interfaces meet at a critical
triple-junction node (node 11 at (40, 280)). Unit normal vectors are computed for
impermeable boundaries and internal material interfaces, with a dedicated directional normal
assigned to the triple junction. Local GFDM support stencils and pseudo-inverse matrices
(M_pinv) are precomputed.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 10): Upstream reservoir boundary (Dirichlet)
* Bottom Boundary (marker 11): Impermeable deep bedrock base (Neumann)
* Right Boundary (marker 12): Downstream boundary (Dirichlet)
* Top Tailings (marker 13): Tailings beach surface
* Top Dam (marker 14): Dam crest, slope, and downstream ground surface
* Interface A (marker 15): Upstream tailings-bedrock contact
* Interface B (marker 16): Tailings-dam embankment contact
* Interface C (marker 17): Dam embankment-bedrock foundation contact

Material Subdomains
-------------------
* Tailings Domain (marker 0): Impounded mine tailings
* Rock Domain (marker 1): Foundation bedrock
* Dam Domain (marker 2): Engineered embankment shell

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh11.json`
* Figures: `examples/legacy/figures/ex11/{geometry,mesh,nodes,normal_vectors}.png`
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
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Geometry Definition
# =============================================================================
g = cfg.Geometry()

# Outer and internal control points defining dam, beach, and foundation
g.point([-400, 355])  # P0: Upstream crest edge
g.point([-400, -100]) # P1: Foundation bottom-left
g.point([1365, -100]) # P2: Foundation bottom-right
g.point([1365, 110])  # P3: Downstream boundary base
g.point([1365, 130])  # P4: Downstream ground edge
g.point([1150, 140])  # P5
g.point([1120, 148])  # P6
g.point([1080, 148])  # P7
g.point([1070, 152])  # P8
g.point([1050, 152])  # P9
g.point([280, 360])   # P10: Main dam crest

# Embankment-foundation internal profile points
g.point([40, 280])    # P11: Triple-junction node
g.point([120, 250])   # P12
g.point([300, 250])   # P13
g.point([340, 230])   # P14
g.point([360, 225])   # P15
g.point([410, 200])   # P16
g.point([460, 190])   # P17
g.point([510, 200])   # P18
g.point([1080, 120])  # P19
g.point([1230, 110])  # P20

# Curve markers
left = 10
bottom = 11
right = 12
top_tailing = 13
top_dam = 14
interface_a = 15
interface_b = 16
interface_c = 17

# Outer boundary lines
g.line([0, 1], marker=left)
g.line([1, 2], marker=bottom)
g.line([2, 3], marker=right)
g.line([3, 4], marker=right)
g.line([4, 5], marker=top_dam)
g.line([5, 6], marker=top_dam)
g.line([6, 7], marker=top_dam)
g.line([7, 8], marker=top_dam)
g.line([8, 9], marker=top_dam)
g.line([9, 10], marker=top_dam)
g.line([10, 0], marker=top_tailing)

# Internal interface lines
g.line([0, 11], marker=interface_a)
g.line([11, 10], marker=interface_b)
g.line([11, 12], marker=interface_c)
g.line([12, 13], marker=interface_c)
g.line([13, 14], marker=interface_c)
g.line([14, 15], marker=interface_c)
g.line([15, 16], marker=interface_c)
g.line([16, 17], marker=interface_c)
g.line([17, 18], marker=interface_c)
g.line([18, 19], marker=interface_c)
g.line([19, 20], marker=interface_c)
g.line([20, 3], marker=interface_c)

# Material surfaces
tailing = 0
rock = 1
dam = 2

g.surface([10, 11, 12], marker=tailing)
g.surface([3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], marker=dam)
g.surface([0, 1, 2, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11], marker=rock)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g, el_size_factor=20.0)

coords, edof, dofs, bdofs, element_markers = mesh.create()
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
bottom_nodes = np.asarray(bdofs[bottom]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
top_tailing_nodes = np.asarray(bdofs[top_tailing]) - 1
top_dam_nodes = np.asarray(bdofs[top_dam]) - 1
interface_a_nodes = np.asarray(bdofs[interface_a]) - 1
interface_b_nodes = np.asarray(bdofs[interface_b]) - 1
interface_c_nodes = np.asarray(bdofs[interface_c]) - 1

# Disambiguate shared boundary corners
bottom_nodes = np.setdiff1d(bottom_nodes, left_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, right_nodes)

top_tailing_nodes = np.setdiff1d(top_tailing_nodes, left_nodes)
top_dam_nodes = np.setdiff1d(top_dam_nodes, top_tailing_nodes)
top_dam_nodes = np.setdiff1d(top_dam_nodes, right_nodes)

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_tailing_nodes,
    top_dam_nodes,
))

interface_a_nodes = np.setdiff1d(interface_a_nodes, boundaries)
interface_b_nodes = np.setdiff1d(interface_b_nodes, boundaries)
interface_c_nodes = np.setdiff1d(interface_c_nodes, boundaries)

boundaries = np.hstack((
    boundaries,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
))

# Isolate triple junction node (node 11)
interface_a_nodes = np.setdiff1d(interface_a_nodes, [11])
interface_b_nodes = np.setdiff1d(interface_b_nodes, [11])
interface_c_nodes = np.setdiff1d(interface_c_nodes, [11])

element_markers = np.array(element_markers)
rock_nodes = faces[element_markers == rock].flatten()
rock_nodes = np.setdiff1d(rock_nodes, boundaries)

dam_nodes = faces[element_markers == dam].flatten()
dam_nodes = np.setdiff1d(dam_nodes, boundaries)

tailing_nodes = faces[element_markers == tailing].flatten()
tailing_nodes = np.setdiff1d(tailing_nodes, boundaries)

nodes_to_plot = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_tailing_nodes,
    top_dam_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    rock_nodes,
    dam_nodes,
    tailing_nodes,
)
labels = (
    "Left",
    "Right",
    "Bottom",
    "Top tailing",
    "Top dam",
    "Interface A",
    "Interface B",
    "Interface C",
    "Rock",
    "Dam",
    "Tailing",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
boundaries_with_normals = (
    bottom_nodes,
    top_tailing_nodes,
    top_dam_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
)
normal_vecs = np.zeros((coords.shape[0], 2))
for nodes in boundaries_with_normals:
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

# Prescribe normal direction at triple junction node
normal_vecs[[11]] = np.array([0.0, -1.0])

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
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh11.json", "w") as f:
        json.dump(data_to_save, f, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_figures:
    os.makedirs("examples/legacy/figures/ex11", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(10, 4),
        savepath="examples/legacy/figures/ex11/geometry.png",
    )

    # Mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(10, 4),
        title="Mesh",
        suptitle=f"el_size_factor={mesh.el_size_factor}, N={coords.shape[0]} nodes",
        savepath="examples/legacy/figures/ex11/mesh.png",
    )

    # Node distribution plot
    colors = [
        "#2b5c8f",  # Blue
        "#d95f02",  # Orange
        "#7570b3",  # Purple
        "#1b9e77",  # Teal
        "#e7298a",  # Pink
        "#e6ab02",  # Yellow
        "#a6761d",  # Brown
        "#666666",  # Gray
        "#1f78b4",  # Light blue
        "#33a02c",  # Light green
        "#fb9a99",  # Light pink
        "#b2df8a",  # Light green
    ]
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        colors=colors,
        alpha=0.5,
        figsize=(10, 4),
        title="Nodes Distribution",
        savepath="examples/legacy/figures/ex11/nodes.png",
    )

    # Normal vectors plot
    all_normal_nodes = boundaries_with_normals + (np.array([11]),)
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=all_normal_nodes,
        quiver_alpha=0.3,
        figsize=(10, 4),
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex11/normal_vectors.png",
    )

    if show_figures:
        plt.show()
    else:
        plt.close("all")
