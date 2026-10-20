"""Mesh 02: Laplace Equation with Sinusoidal Interface and Analytical Verification.

This script constructs the computational geometry and generates the unstructured
triangular mesh for Example 02. The unit square [0, 1] x [0, 1] is bisected by a
curved sinusoidal interface x(y) = 0.5 + 0.1 sin(2 pi y) separating two subdomains
(Omega^+ on the left and Omega^- on the right). Duplicate interface curves are offset
by a small tolerance (delta_interface = 0.01) to accommodate potential and flux jump
discontinuities. Analytical normal vectors are computed along the interface.

Geometric Boundaries and Markers
--------------------------------
* Dirichlet Outer Boundaries (marker 10): External square boundaries
* Left Interface Line (marker 11): Interface boundary on subdomain Omega^+
* Right Interface Line (marker 12): Interface boundary on subdomain Omega^-

Material Subdomains
-------------------
* Subdomain Omega^+ (marker 0): Left subdomain x < 0.5 + 0.1 sin(2 pi y)
* Subdomain Omega^- (marker 1): Right subdomain x > 0.5 + 0.1 sin(2 pi y)

Mesh and Precomputed Outputs
----------------------------
* JSON: `examples/legacy/meshes/mesh2.json`
* Figures: `examples/legacy/figures/ex2/{geometry,mesh,nodes,normal_vectors}.png`
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

# Outer square corners
geometry.point([0, 0])  # P0
geometry.point([1, 0])  # P1
geometry.point([1, 1])  # P2
geometry.point([0, 1])  # P3

# Interface-boundary intersection points
delta_interface = 0.01

geometry.point([0.5 - delta_interface, 0])  # P4
geometry.point([0.5 - delta_interface, 1])  # P5
geometry.point([0.5 + delta_interface, 0])  # P6
geometry.point([0.5 + delta_interface, 1])  # P7

# Outer Dirichlet boundaries
dirichlet = 10
geometry.spline([5, 3], marker=dirichlet)  # Top left
geometry.spline([3, 0], marker=dirichlet)  # Left
geometry.spline([0, 4], marker=dirichlet)  # Bottom left

geometry.spline([6, 1], marker=dirichlet)  # Bottom right
geometry.spline([1, 2], marker=dirichlet)  # Right
geometry.spline([2, 7], marker=dirichlet)  # Top right

# Sinusoidal interface: Left side (Omega^+)
interface_left = 11
N = 11
delta_y = 1.0 / (N + 1)
y = 0.0

for i in range(N):
    y += delta_y
    x = -delta_interface + 0.5 + 0.1 * np.sin(2.0 * np.pi * y)
    geometry.point([x, y])

for i in range(N - 1):
    geometry.spline([8 + i, 9 + i], marker=interface_left)

geometry.spline([4, 8], marker=interface_left)
geometry.spline([7 + N, 5], marker=interface_left)

# Sinusoidal interface: Right side (Omega^-)
interface_right = 12
y = 0.0
for i in range(N):
    y += delta_y
    x = delta_interface + 0.5 + 0.1 * np.sin(2.0 * np.pi * y)
    geometry.point([x, y])

for i in range(N - 1):
    geometry.spline([8 + N + i, 9 + N + i], marker=interface_right)

geometry.spline([6, 8 + N], marker=interface_right)
geometry.spline([7 + 2 * N, 7], marker=interface_right)

# Material surfaces
left_domain = 0
left_surf_index = np.hstack((
    np.array([0, 1, 2]),
    np.array([5 + N]),
    np.arange(5 + 1, 5 + N),
    np.array([5 + N + 1]),
))
geometry.surface(left_surf_index, marker=left_domain)

right_domain = 1
right_surf_index = np.hstack((
    np.array([5 + 2 * N + 1]),
    np.arange(5 + N + 2, 5 + 2 * N + 1),
    np.array([5 + 2 * N + 2]),
    np.array([5, 4, 3]),
))
geometry.surface(right_surf_index, marker=right_domain)

# =============================================================================
# 2. Mesh Generation
# =============================================================================
mesh = cfm.GmshMesh(geometry)
mesh.el_type = 2          # 2 = 3-node triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.1

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
# Dirichlet outer nodes
dirichlet_nodes = np.asarray(bdofs[dirichlet]) - 1

# Interface duplicate nodes
left_interface_nodes = np.asarray(bdofs[interface_left]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4, 5])

right_interface_nodes = np.asarray(bdofs[interface_right]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [6, 7])

# Interior nodes by subdomain
elementmarkers = np.asarray(elementmarkers)
boundaries = np.hstack((dirichlet_nodes, left_interface_nodes, right_interface_nodes))

left_interior_nodes = faces[elementmarkers == left_domain].flatten()
left_interior_nodes = np.setdiff1d(left_interior_nodes, boundaries)

right_interior_nodes = faces[elementmarkers == right_domain].flatten()
right_interior_nodes = np.setdiff1d(right_interior_nodes, boundaries)

nodes = (
    dirichlet_nodes,
    left_interface_nodes,
    right_interface_nodes,
    left_interior_nodes,
    right_interior_nodes,
)
labels = (
    "Dirichlet",
    "Interface left",
    "Interface right",
    "Omega_plus",
    "Omega_minus",
)

# =============================================================================
# 4. Normal Vector Computation
# =============================================================================
def compute_analytical_interface_normals(node_indices):
    """Compute normalized analytical normal vectors along sinusoidal interface."""
    normals = np.empty((node_indices.shape[0], 2))
    normals[:, 0] = 1.0
    normals[:, 1] = -0.2 * np.pi * np.cos(2.0 * np.pi * coords[node_indices, 1])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norms

normal_vecs_left_interface = compute_analytical_interface_normals(left_interface_nodes)
normal_vecs_right_interface = compute_analytical_interface_normals(right_interface_nodes)

normal_vecs = np.zeros((coords.shape[0], 2))
normal_vecs[left_interface_nodes, :] = normal_vecs_left_interface
normal_vecs[right_interface_nodes, :] = normal_vecs_right_interface

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
    for b, label in zip(nodes, labels):
        data_to_save[label.replace(" ", "_").replace("-", "_").lower() + "_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open("examples/legacy/meshes/mesh2.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    print("\n============\n Mesh saved \n============")

# =============================================================================
# 7. Visualization and Diagnostic Plots
# =============================================================================
if save_figures or show_plots:
    os.makedirs("examples/legacy/figures/ex2", exist_ok=True)

    # Geometry plot
    plot_geometry(
        geometry,
        title="Geometry",
        figsize=(4, 4),
        savepath="examples/legacy/figures/ex2/geometry.png",
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
        savepath="examples/legacy/figures/ex2/mesh.png",
    )

    # Node distribution plot
    plot_nodes(
        coords,
        zip(labels, nodes),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.5,
        savepath="examples/legacy/figures/ex2/nodes.png",
    )

    # Normal vectors plot
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=[left_interface_nodes, right_interface_nodes],
        quiver_alpha=0.3,
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex2/normal_vectors.png",
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")