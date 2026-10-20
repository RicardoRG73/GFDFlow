"""Mesh 04: Multi-Material Seepage in a Five-Layered Dam with Triple Junctions.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 04. The domain models a five-layered embankment dam
with six internal material interfaces that intersect at two central triple-junction
nodes (nodes 1 and 2). Unit normal vectors are computed for all Neumann boundaries,
internal interface segments, and triple junctions. GFDM support stencils and
pseudo-inverse matrices (M_pinv) are precomputed for all nodes.

Geometric Boundaries and Markers
--------------------------------
* Left Boundary (marker 10): Upstream reservoir boundary (Dirichlet)
* Right Boundary (marker 11): Downstream boundary (Dirichlet)
* Bottom Boundary (marker 12): Impermeable bedrock base (Neumann)
* Top Boundary (marker 13): Crest and slope boundaries (Neumann)
* Interface A (marker 14): Internal interface segment A
* Interface B (marker 15): Internal interface segment B
* Interface C (marker 16): Internal interface segment C
* Interface D (marker 17): Internal interface segment D
* Interface E (marker 18): Internal interface segment E
* Interface F (marker 19): Internal interface segment F

Material Subdomains
-------------------
* Rock (marker 1): Subdomains 1 and 2 (permeable shells)
* Clay (marker 2): Subdomain 5 (impermeable clay core)
* Mixed (marker 3): Subdomains 3 and 4 (transition layers)

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh4.json`
* Figures: `examples/legacy/figures/ex4/{geometry,mesh,nodes,normal_vectors}.png`
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

# Control points for the dam profile, foundation, and internal layers
g.point([0, 0])     # P0
g.point([15, 0])    # P1: Triple junction 1
g.point([35, 0])    # P2: Triple junction 2
g.point([42, 0])    # P3
g.point([47, 0])    # P4
g.point([27, 10])   # P5: Crest right
g.point([23, 10])   # P6: Crest left
g.point([19, 8])    # P7
g.point([0, -5])    # P8
g.point([23, -5])   # P9
g.point([27, -5])   # P10
g.point([55, -5])   # P11
g.point([55, 0])    # P12

# Boundary curve markers
left = 10
right = 11
bottom = 12
top = 13

# Interface curve markers
interface_a = 14
interface_b = 15
interface_c = 16
interface_d = 17
interface_e = 18
interface_f = 19

# Spline curves: Boundaries and internal interfaces
g.spline([0, 1], marker=interface_a)
g.spline([2, 3], marker=interface_b)
g.spline([3, 4], marker=right)
g.spline([4, 5], marker=top)
g.spline([5, 6], marker=top)
g.spline([6, 7], marker=top)
g.spline([7, 0], marker=left)

g.spline([0, 8], marker=bottom)
g.spline([8, 9], marker=bottom)
g.spline([9, 10], marker=bottom)
g.spline([10, 11], marker=bottom)
g.spline([11, 12], marker=bottom)
g.spline([12, 4], marker=right)

g.spline([6, 1], marker=interface_c)
g.spline([1, 9], marker=interface_d)

g.spline([5, 2], marker=interface_f)
g.spline([2, 10], marker=interface_e)

# Material surfaces
rock = 1
clay = 2
mixed = 3

g.surface([0, 13, 5, 6], marker=rock)               # Subdomain 1 (rock)
g.surface([1, 2, 3, 15], marker=rock)               # Subdomain 2 (rock)
g.surface([7, 8, 14, 0], marker=mixed)              # Subdomain 3 (mixed)
g.surface([10, 11, 12, 2, 1, 16], marker=mixed)    # Subdomain 4 (mixed)
g.surface([13, 14, 9, 16, 15, 4], marker=clay)     # Subdomain 5 (clay core)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(g)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.5

coords, edof, dofs, bdofs, element_markers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type,
)
triangles = faces.copy()

# =============================================================================
# 3. Node Identification and Boundary Extraction
# =============================================================================
left_nodes = np.asarray(bdofs[left]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
top_nodes = np.asarray(bdofs[top]) - 1
bottom_nodes = np.asarray(bdofs[bottom]) - 1

interface_a_nodes = np.asarray(bdofs[interface_a]) - 1
interface_b_nodes = np.asarray(bdofs[interface_b]) - 1
interface_c_nodes = np.asarray(bdofs[interface_c]) - 1
interface_d_nodes = np.asarray(bdofs[interface_d]) - 1
interface_e_nodes = np.asarray(bdofs[interface_e]) - 1
interface_f_nodes = np.asarray(bdofs[interface_f]) - 1

# Eliminate boundary corner duplicates
top_nodes = np.setdiff1d(top_nodes, left_nodes)
top_nodes = np.setdiff1d(top_nodes, right_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, left_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, right_nodes)

outer_boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
))

interface_a_nodes = np.setdiff1d(interface_a_nodes, outer_boundaries)
interface_b_nodes = np.setdiff1d(interface_b_nodes, outer_boundaries)
interface_c_nodes = np.setdiff1d(interface_c_nodes, outer_boundaries)
interface_d_nodes = np.setdiff1d(interface_d_nodes, outer_boundaries)
interface_e_nodes = np.setdiff1d(interface_e_nodes, outer_boundaries)
interface_f_nodes = np.setdiff1d(interface_f_nodes, outer_boundaries)

all_boundaries = np.hstack((
    outer_boundaries,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    interface_d_nodes,
    interface_e_nodes,
    interface_f_nodes,
))

element_markers = np.array(element_markers)
rock_nodes = triangles[element_markers == rock].flatten()
rock_nodes = np.setdiff1d(rock_nodes, all_boundaries)

clay_nodes = triangles[element_markers == clay].flatten()
clay_nodes = np.setdiff1d(clay_nodes, all_boundaries)

mixed_nodes = triangles[element_markers == mixed].flatten()
mixed_nodes = np.setdiff1d(mixed_nodes, all_boundaries)

# Isolate triple junction intersection nodes
interface_a_nodes = np.setdiff1d(interface_a_nodes, [1])
interface_c_nodes = np.setdiff1d(interface_c_nodes, [1])
interface_d_nodes = np.setdiff1d(interface_d_nodes, [1])
interface_b_nodes = np.setdiff1d(interface_b_nodes, [2])
interface_e_nodes = np.setdiff1d(interface_e_nodes, [2])
interface_f_nodes = np.setdiff1d(interface_f_nodes, [2])

nodes_to_plot = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    interface_d_nodes,
    interface_e_nodes,
    interface_f_nodes,
    rock_nodes,
    clay_nodes,
    mixed_nodes,
)
labels = (
    "Left",
    "Right",
    "Bottom",
    "Top",
    "Interface A",
    "Interface B",
    "Interface C",
    "Interface D",
    "Interface E",
    "Interface F",
    "Rock",
    "Clay",
    "Mixed",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
boundaries_with_normals = (
    bottom_nodes,
    top_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    interface_d_nodes,
    interface_e_nodes,
    interface_f_nodes,
)
normal_vecs = np.zeros((coords.shape[0], 2))
for nodes in boundaries_with_normals:
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

# Prescribe normal direction for triple junction nodes
normal_vecs[[1]] = np.array([-1.0, 0.0])
normal_vecs[[2]] = np.array([-1.0, 0.0])

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

    with open("examples/legacy/meshes/mesh4.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex4", exist_ok=True)

    # Geometry plot
    plot_geometry(
        g,
        title="Geometry",
        figsize=(6, 4),
        savepath="examples/legacy/figures/ex4/geometry.png",
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
        savepath="examples/legacy/figures/ex4/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=10,
        savepath="examples/legacy/figures/ex4/nodes.png",
    )

    # Normal vectors plot
    all_normal_nodes = boundaries_with_normals + (np.array([1, 2]),)
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=all_normal_nodes,
        quiver_color="k",
        quiver_alpha=0.3,
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex4/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")