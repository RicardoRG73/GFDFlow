"""Mesh Generation and GFDM Precomputation for Five-Layer Dam Seepage Problem.

This script constructs the 2D computational geometry, generates an unstructured
triangular mesh via CalfEM/Gmsh, classifies topological node groups, and
precomputes the Generalized Finite Difference Method (GFDM) operators for the
five-layered embankment dam with central clay core and foundation described in
Section 4.2.3 of Chavez et al. (2023).

Geometric & Material Setup
--------------------------
- **Rockfill shells** (marker 1): Upstream and downstream pervious shoulders.
- **Clay core & cutoff wall** (marker 2): Impervious central barrier.
- **Soil foundation** (marker 3): Underlying alluvial stratum (y in [-4, 0] m).
- **Outer boundaries**:
  - `left` (10): Submerged upstream face and left foundation wall (Dirichlet).
  - `right` (11): Downstream underdrain/ground surface and right wall (Dirichlet).
  - `bottom` (12): Impermeable bedrock base at y = -4 m (Neumann).
  - `top` (13): Dry upstream slope, dam crest, and downstream slope (Neumann).
- **Internal interfaces**: `interface_a` (14) through `interface_f` (19), plus
  two triple-junction singularities (`tj1` at [20, 0] and `tj2` at [30, 0]).

Outputs
-------
- JSON data: `examples/paper/mesh_data.json` (coordinates, triangles, normals,
  node groups, support stencils, and precomputed `M_pinv` matrices).
- Diagnostic figures: `examples/paper/figures/{geometry,mesh,nodes,normal_vectors}.png`.

References
----------
- Chávez-Negrete, C., Domínguez-Mota, F. J., & Román-Gutiérrez, R. (2023/2024).
  "Interface formulation for generalized finite difference method for solving
  groundwater flow." *Computers and Geotechnics*, Vol. 166, 105990.
  DOI: 10.1016/j.compgeo.2023.105990 (indexed under 2023 in some databases
  and published in Vol. 166 in 2024).
"""

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
import json

# Configure global Matplotlib styling for publication-ready diagnostic figures
plt.style.use(['seaborn-v0_8-paper'])
plt.rcParams["figure.autolayout"] = True

from GFDFlow.visualization import plot_nodes, plot_normal_vectors
from GFDFlow.utils import (
    compute_normal_vectors,
    get_support_nodes_2D,
    compute_M_matrix,
    compute_M_matrix_neumann,
)

# ==============================================================================
# 1. Geometry Definition
# ==============================================================================
g = cfg.Geometry()

# Control points defining dam profile, crest, foundation, and cutoff trench
g.point([3, 0])      # P0: Upstream toe
g.point([20, 0])     # P1: Triple junction 1 (upstream core-foundation)
g.point([30, 0])     # P2: Triple junction 2 (downstream core-foundation)
g.point([40, 0])     # P3: Downstream filter/drain start
g.point([47, 0])     # P4: Downstream dam toe
g.point([27, 10])    # P5: Crest right corner
g.point([23, 10])    # P6: Crest left corner
g.point([19, 8])     # P7: Upstream reservoir water surface level
g.point([3, -4])     # P8: Foundation bottom-left corner
g.point([24, -4])    # P9: Core cutoff base left
g.point([26, -4])    # P10: Core cutoff base right
g.point([60, -4])    # P11: Foundation bottom-right corner
g.point([60, 0])     # P12: Foundation top-right ground surface

# Semantic integer markers for outer boundaries, internal interfaces, and materials
markers = {
    "left": 10,
    "right": 11,
    "bottom": 12,
    "top": 13,
    "interface_a": 14,
    "interface_b": 15,
    "interface_c": 16,
    "interface_d": 17,
    "interface_e": 18,
    "interface_f": 19,
    "rock": 1,
    "clay": 2,
    "foundation": 3,
}

# Splines: Dam outer boundaries and horizontal rock-foundation interfaces
g.spline([0, 1], marker=markers["interface_a"])   # S0: Upstream rockfill-foundation contact
g.spline([2, 3], marker=markers["interface_b"])   # S1: Downstream rockfill-foundation contact
g.spline([3, 4], marker=markers["right"])         # S2: Downstream horizontal underdrain
g.spline([4, 5], marker=markers["top"])           # S3: Downstream slope (2:1)
g.spline([5, 6], marker=markers["top"])           # S4: Dam crest (x = 23 to 27 m)
g.spline([6, 7], marker=markers["top"])           # S5: Dry upstream slope above reservoir
g.spline([7, 0], marker=markers["left"])          # S6: Submerged upstream slope

# Foundation outer boundaries
g.spline([0, 8], marker=markers["left"])          # S7: Left foundation vertical wall
g.spline([8, 9], marker=markers["bottom"])        # S8: Upstream bedrock base
g.spline([9, 10], marker=markers["bottom"])       # S9: Cutoff trench base
g.spline([10, 11], marker=markers["bottom"])      # S10: Downstream bedrock base
g.spline([11, 12], marker=markers["bottom"])      # S11: Right foundation vertical wall
g.spline([12, 4], marker=markers["right"])        # S12: Downstream ground surface

# Core internal sloping interfaces
g.spline([6, 1], marker=markers["interface_c"])   # S13: Upstream rockfill-clay contact
g.spline([1, 9], marker=markers["interface_d"])   # S14: Upstream foundation-cutoff contact
g.spline([5, 2], marker=markers["interface_f"])   # S15: Downstream rockfill-clay contact
g.spline([2, 10], marker=markers["interface_e"])  # S16: Downstream foundation-cutoff contact

# Closed surfaces defining the five material subdomains
g.surface([0, 13, 5, 6], marker=markers["rock"])               # Subdomain 1: Rockfill upstream
g.surface([1, 2, 3, 15], marker=markers["rock"])               # Subdomain 2: Rockfill downstream
g.surface([7, 8, 14, 0], marker=markers["foundation"])         # Subdomain 3: Foundation upstream
g.surface([10, 11, 12, 2, 1, 16], marker=markers["foundation"])# Subdomain 4: Foundation downstream
g.surface([13, 14, 9, 16, 15, 4], marker=markers["clay"])      # Subdomain 5: Clay core & cutoff

# Render and save the geometric model diagram
plt.figure(figsize=(10, 4))
cfv.draw_geometry(g, title="Geometry")
plt.savefig("examples/paper/figures/geometry.png", dpi=300, bbox_inches="tight")

# ==============================================================================
# 2. Unstructured Triangular Mesh Generation
# ==============================================================================
mesh = cfm.GmshMesh(g)
mesh.el_type = 2          # 3-node linear triangle
mesh.dofs_per_node = 1    # Scalar hydraulic head DOF per node
mesh.el_size_factor = 1   # Global element size scaling factor

coords, edof, dofs, bdofs, element_markers = mesh.create()

# Convert 1-indexed CalfEM element DOFs (edof) into 0-indexed vertex connectivity
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type,
)

triangles = faces.copy()

# Render and save the generated triangular mesh
plt.figure(figsize=(10, 4))
cfv.draw_mesh(coords, edof, mesh.dofs_per_node, mesh.el_type, title="Mesh", filled=True)
plt.savefig("examples/paper/figures/mesh.png", dpi=300, bbox_inches="tight")

# ==============================================================================
# 3. Node Group Classification & Boundary Deduplication
# ==============================================================================
# Convert 1-indexed CalfEM boundary DOFs to 0-indexed NumPy arrays
left_nodes = np.asarray(bdofs[markers["left"]]) - 1
right_nodes = np.asarray(bdofs[markers["right"]]) - 1
top_nodes = np.asarray(bdofs[markers["top"]]) - 1
bottom_nodes = np.asarray(bdofs[markers["bottom"]]) - 1

ia_nodes = np.asarray(bdofs[markers["interface_a"]]) - 1
ib_nodes = np.asarray(bdofs[markers["interface_b"]]) - 1
ic_nodes = np.asarray(bdofs[markers["interface_c"]]) - 1
id_nodes = np.asarray(bdofs[markers["interface_d"]]) - 1
ie_nodes = np.asarray(bdofs[markers["interface_e"]]) - 1
if_nodes = np.asarray(bdofs[markers["interface_f"]]) - 1

# Remove corner duplicates so Dirichlet nodes take precedence over Neumann boundaries
top_nodes = np.setdiff1d(top_nodes, left_nodes)
top_nodes = np.setdiff1d(top_nodes, right_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, left_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, right_nodes)

outer_boundaries = np.hstack((left_nodes, right_nodes, bottom_nodes, top_nodes))

# Strip outer boundary nodes from internal interfaces to ensure mutually exclusive sets
ia_nodes = np.setdiff1d(ia_nodes, outer_boundaries)
ib_nodes = np.setdiff1d(ib_nodes, outer_boundaries)
ic_nodes = np.setdiff1d(ic_nodes, outer_boundaries)
id_nodes = np.setdiff1d(id_nodes, outer_boundaries)
ie_nodes = np.setdiff1d(ie_nodes, outer_boundaries)
if_nodes = np.setdiff1d(if_nodes, outer_boundaries)

all_boundaries = np.hstack((
    outer_boundaries,
    ia_nodes,
    ib_nodes,
    ic_nodes,
    id_nodes,
    ie_nodes,
    if_nodes,
))

# Extract strictly interior material nodes by excluding all boundary and interface nodes
elem_markers_arr = np.array(element_markers)
rock_nodes = np.setdiff1d(
    triangles[elem_markers_arr == markers["rock"]].flatten(),
    all_boundaries,
)
clay_nodes = np.setdiff1d(
    triangles[elem_markers_arr == markers["clay"]].flatten(),
    all_boundaries,
)
foundation_nodes = np.setdiff1d(
    triangles[elem_markers_arr == markers["foundation"]].flatten(),
    all_boundaries,
)

# Detect triple-junction nodes where three material subdomains meet: (20, 0) and (30, 0)
tj1_idx = int(np.where(np.linalg.norm(coords - np.array([20.0, 0.0]), axis=1) < 1e-4)[0][0])
tj2_idx = int(np.where(np.linalg.norm(coords - np.array([30.0, 0.0]), axis=1) < 1e-4)[0][0])

# Exclude triple-junction nodes from standard two-material interface arrays
ia_nodes = np.setdiff1d(ia_nodes, [tj1_idx])
ic_nodes = np.setdiff1d(ic_nodes, [tj1_idx])
id_nodes = np.setdiff1d(id_nodes, [tj1_idx])

ib_nodes = np.setdiff1d(ib_nodes, [tj2_idx])
ie_nodes = np.setdiff1d(ie_nodes, [tj2_idx])
if_nodes = np.setdiff1d(if_nodes, [tj2_idx])

node_groups = {
    "left_nodes": left_nodes,
    "right_nodes": right_nodes,
    "bottom_nodes": bottom_nodes,
    "top_nodes": top_nodes,
    "interface_a_nodes": ia_nodes,
    "interface_b_nodes": ib_nodes,
    "interface_c_nodes": ic_nodes,
    "interface_d_nodes": id_nodes,
    "interface_e_nodes": ie_nodes,
    "interface_f_nodes": if_nodes,
    "rock_nodes": rock_nodes,
    "clay_nodes": clay_nodes,
    "foundation_nodes": foundation_nodes,
    "tj1": np.array([tj1_idx]),
    "tj2": np.array([tj2_idx]),
}

# Visualize classified node groups (materials, Dirichlet, Neumann, and interfaces)
nodes_to_plot = (
    node_groups["clay_nodes"],
    node_groups["rock_nodes"],
    node_groups["foundation_nodes"],
    np.hstack((node_groups["left_nodes"], node_groups["right_nodes"])),
    np.hstack((node_groups["bottom_nodes"], node_groups["top_nodes"])),
    np.hstack((
        node_groups["interface_a_nodes"],
        node_groups["interface_b_nodes"],
        node_groups["interface_c_nodes"],
        node_groups["interface_d_nodes"],
        node_groups["interface_e_nodes"],
        node_groups["interface_f_nodes"],
        node_groups["tj1"],
        node_groups["tj2"],
    )),
)
labels = (
    "Clay Core (Layer 1)",
    "Rockfill (Layer 2)",
    "Soil Foundation (Layer 3)",
    "Dirichlet BC (NFronD)",
    "Neumann BC (NFronN)",
    "Interfaces (NFronInter)",
)

plot_nodes(
    coords,
    zip(labels, nodes_to_plot),
    title=f"Mesh Node Distribution ($N = {coords.shape[0]}$)",
    alpha=0.85,
    point_size=12,
    figsize=(10, 4),
    savepath="examples/paper/figures/nodes.png",
)

# ==============================================================================
# 4. Outward Normal Vectors & GFDM Star Matrix Precomputation
# ==============================================================================
boundaries_with_normals = (
    node_groups["bottom_nodes"],
    node_groups["top_nodes"],
    node_groups["interface_a_nodes"],
    node_groups["interface_b_nodes"],
    node_groups["interface_c_nodes"],
    node_groups["interface_d_nodes"],
    node_groups["interface_e_nodes"],
    node_groups["interface_f_nodes"],
)

# Compute outward unit normal vectors along Neumann boundaries and internal interfaces
normal_vecs = np.zeros((coords.shape[0], 2))
for nodes in boundaries_with_normals:
    if len(nodes) > 0:
        normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

# Prescribed reference normal direction for triple-junction angular balance nodes
tj1_idx = int(node_groups["tj1"][0])
tj2_idx = int(node_groups["tj2"][0])
normal_vecs[tj1_idx] = np.array([-1.0, 0.0])
normal_vecs[tj2_idx] = np.array([-1.0, 0.0])

support_stencils = {}
M_pinv = {}

# Build 1-ring Delaunay support stencils and precompute Moore-Penrose pseudo-inverses
for i in range(coords.shape[0]):
    support_stencils[i] = get_support_nodes_2D(i, triangles)
    if np.linalg.norm(normal_vecs[i]) > 1e-10:
        # Use Neumann-augmented Taylor matrix for boundary/interface nodes
        M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vecs[i])
    else:
        # Use standard 2nd-order Taylor matrix for interior/Dirichlet nodes
        M = compute_M_matrix(i, support_stencils[i], coords)
    M_pinv[i] = np.linalg.pinv(M)

# Plot outward normal vectors for visual verification
all_normal_nodes = (
    node_groups["bottom_nodes"],
    node_groups["top_nodes"],
    node_groups["interface_a_nodes"],
    node_groups["interface_b_nodes"],
    node_groups["interface_c_nodes"],
    node_groups["interface_d_nodes"],
    node_groups["interface_e_nodes"],
    node_groups["interface_f_nodes"],
    node_groups["tj1"],
    node_groups["tj2"],
)

plot_normal_vectors(
    coords,
    normal_vecs,
    boundary_nodes=all_normal_nodes,
    quiver_color="k",
    quiver_alpha=0.4,
    title="Boundary & Interface Normal Vectors",
    figsize=(10, 4),
    savepath="examples/paper/figures/normal_vectors.png",
)

# ==============================================================================
# 5. Serialization of Mesh Data & Precomputed GFDM Operators
# ==============================================================================
data = {
    "coords": coords.tolist(),
    "triangles": triangles.tolist(),
    "normal_vecs": normal_vecs.tolist(),
    "support_stencils": {str(k): v.tolist() for k, v in support_stencils.items()},
    "M_pinv": {str(k): v.tolist() for k, v in M_pinv.items()},
}

for k, v in node_groups.items():
    data[k] = v.tolist()

filepath = "examples/paper/mesh_data.json"
with open(filepath, "w") as f:
    json.dump(data, f, indent=2)
print(f"Mesh data successfully saved to: {filepath}")


plt.show()
