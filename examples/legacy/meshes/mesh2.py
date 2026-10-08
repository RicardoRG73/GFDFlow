show_plots = False
save_mesh_to_file = True

#%%
# =============================================================================
# Importing nedeed libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

from GFDFlow.utils import get_support_nodes_2D, compute_M_matrix, compute_M_matrix_neumann

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

#%%
# =============================================================================
# Creating geometry object
# =============================================================================
geometry = cfg.Geometry()

# points: square domain
geometry.point([0,0])       # 0
geometry.point([1,0])       # 1
geometry.point([1,1])       # 2
geometry.point([0,1])       # 3

# points: interface-boundaries intersection
delta_interface = 0.01

geometry.point([ 0.5 - delta_interface , 0 ])     # 4
geometry.point([ 0.5 - delta_interface , 1 ])     # 5
geometry.point([ 0.5 + delta_interface , 0 ])     # 6
geometry.point([ 0.5 + delta_interface , 1 ])     # 7

# lines: square domain
dirichlet = 10
geometry.spline([5,3], marker=dirichlet)    # 0
geometry.spline([3,0], marker=dirichlet)    # 1
geometry.spline([0,4], marker=dirichlet)    # 2

geometry.spline([6,1], marker=dirichlet)    # 3
geometry.spline([1,2], marker=dirichlet)    # 4
geometry.spline([2,7], marker=dirichlet)    # 5

# interface
## left interface
interface_left = 11
### points
N = 11
delta_y = 1/(N+1)
y = 0

for i in range(N):
    y += delta_y
    x = - delta_interface + 0.5 + 0.1 * np.sin(6.28 * y)
    geometry.point([x, y])

### lines
for i in range(N-1):
    geometry.spline([8+i,9+i], marker=interface_left)

### left interface lines, conecting interface and boundaries
geometry.spline([4,8], marker=interface_left)
geometry.spline([7+N,5], marker=interface_left)

## right interface
interface_right = 12
### points
y = 0
for i in range(N):
    y += delta_y
    x = delta_interface + 0.5 + 0.1 * np.sin(6.28 * y)
    geometry.point([x, y])

### lines
for i in range(N-1):
    geometry.spline([8+N+i,9+N+i], marker=interface_right)

### left interface lines, conecting interface and boundaries
geometry.spline([6,8+N], marker=interface_right)
geometry.spline([7+2*N,7], marker=interface_right)


# surfaces
## \Omega^+ : left side
left_domain = 0
left_surf_index = np.hstack((
    np.array([0,1,2]),
    np.array([5+N]),
    np.arange(5+1,5+N),
    np.array([5+N+1])
))
geometry.surface(left_surf_index, marker=left_domain)

## \Omega^- : right side
right_domain = 1
left_surf_index = np.hstack((
    np.array([5+2*N+1]),
    np.arange(5+N+2,5+2*N+1),
    np.array([5+2*N+2]),
    np.array([5,4,3])
))
geometry.surface(left_surf_index, marker=right_domain)

#%%
# =============================================================================
# Creating mesh
# =============================================================================
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.1

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%%
# =============================================================================
# Nodes indexing separated by boundary conditions
# =============================================================================
# Dirichlet nodes
dirichlet_nodes = np.asarray(bdofs[dirichlet]) - 1

# Interface nodes
left_interface_nodes = np.asarray(bdofs[interface_left]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4,5])
right_interface_nodes = np.asarray(bdofs[interface_right]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [6,7])

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
boundaries = np.hstack((dirichlet_nodes,left_interface_nodes,right_interface_nodes))

left_interior_nodes = faces[elementmarkers == left_domain]
left_interior_nodes = left_interior_nodes.flatten()
left_interior_nodes = np.setdiff1d(left_interior_nodes,boundaries)

right_interior_nodes = faces[elementmarkers == right_domain]
right_interior_nodes = right_interior_nodes.flatten()
right_interior_nodes = np.setdiff1d(right_interior_nodes,boundaries)

nodes = (
    dirichlet_nodes,
    left_interface_nodes,
    right_interface_nodes,
    left_interior_nodes,
    right_interior_nodes
)
labels=(
    "Dirichlet",
    "Interface left",
    "Interface right",
    "Omega_plus",
    "Omega_minus"
)

# =============================================================================
# Normal vectors, support stencils and M_pinv
# =============================================================================
# normal vectors at interface left nodes
def compute_normal_vecs(b):
    normal_vecs = np.empty((b.shape[0], 2))
    normal_vecs[:, 0] = 1.0
    normal_vecs[:, 1] = -0.628 * np.cos(6.28 * coords[b, 1])
    norms = np.linalg.norm(normal_vecs, axis=1, keepdims=True)
    return normal_vecs / norms

normal_vecs_left_interface = compute_normal_vecs(left_interface_nodes)
normal_vecs_right_interface = compute_normal_vecs(right_interface_nodes)

normal_vecs = np.zeros((coords.shape[0],2))
normal_vecs[left_interface_nodes,:] = normal_vecs_left_interface
normal_vecs[right_interface_nodes,:] = normal_vecs_right_interface

M_pinv = {}
support_stencils = {}

for i in range(coords.shape[0]):
    support_stencils[i] = get_support_nodes_2D(i, faces)
    if np.linalg.norm(normal_vecs[i]) > 1e-10:
        M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vecs[i])
    else:
        M = compute_M_matrix(i, support_stencils[i], coords)
    M_pinv[i] = np.linalg.pinv(M)

if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}
    with open('examples/legacy/meshes/mesh2.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")


if show_plots:
    from GFDFlow.visualization import (
        plot_geometry,
        plot_mesh,
        plot_nodes,
    )

    # geometry plot
    plot_geometry(geometry, title="Geometry", figsize=(4, 4), savepath="examples/legacy/figures/ex2/geometry.png")

    # mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(8, 4),
        title="Mesh",
        savepath="examples/legacy/figures/ex2/mesh.png"
    )

    # plotting boundaries in different colors
    plot_nodes(
        coords,
        zip(labels, nodes),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.5,
        savepath="examples/legacy/figures/ex2/nodes.png"
    )

    plt.show()