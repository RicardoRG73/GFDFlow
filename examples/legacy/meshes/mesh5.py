show_plots = False
save_mesh_to_file = True

import numpy as np
import matplotlib.pyplot as plt
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
from GFDFlow.utils import compute_normal_vectors, get_support_nodes_2D, compute_M_matrix, compute_M_matrix_neumann

g = cfg.Geometry()

# points
#   corners
g.point([0,0])
g.point([2,0])
g.point([2,1])
g.point([0,1])

#   interface
g.point([1,0])
g.point([1,1])

#   left-circle
g.point([0.5,0.5])
g.point([0.7,0.5])
g.point([0.5,0.7])
g.point([0.3,0.5])
g.point([0.5,0.3])

#   right-circle
g.point([1.5,0.5])
g.point([1.7,0.5])
g.point([1.5,0.7])
g.point([1.3,0.5])
g.point([1.5,0.3])

# lines
left_marker = 10
right_marker = 11
left_top_marker = 12
right_top_marker = 13
left_bottom_marker = 15
right_bottom_marker = 16
interface_marker = 17
left_circ_mark = 18
right_circ_mark = 19

#   left-half
g.spline([0,4], marker=left_bottom_marker)
g.spline([4,5], marker=interface_marker)
g.spline([5,3], marker=left_top_marker)
g.spline([3,0], marker=left_marker)

#   right-half
g.spline([4,1], marker=right_bottom_marker)
g.spline([1,2], marker=right_marker)
g.spline([2,5], marker=right_top_marker)

#   left-circle
g.circle([7,6,8], marker=left_circ_mark)
g.circle([8,6,9], marker=left_circ_mark)
g.circle([9,6,10], marker=left_circ_mark)
g.circle([10,6,7], marker=left_circ_mark)

#   right-circle
g.circle([12,11,13], marker=right_circ_mark)
g.circle([13,11,14], marker=right_circ_mark)
g.circle([14,11,15], marker=right_circ_mark)
g.circle([15,11,12], marker=right_circ_mark)

# surfaces
left_half_mark = 0
right_half_mark = 1
left_circ_surf_mark = 2
#   left-half
g.surface(outer_loop=[0,1,2,3], holes=[[7,8,9,10]], marker=left_half_mark)
#   left-circle
g.surface([7,8,9,10], marker=left_circ_surf_mark)
#   right-half
g.surface([4,5,6,1], [[11,12,13,14]], marker=right_half_mark)


# Mesh
mesh = cfm.GmshMesh(g,el_size_factor=0.05)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# Identify the indices of the different boundaries and interfaces

# boundaries
left_nodes = np.asarray(bdofs[left_marker]) - 1
right_nodes = np.asarray(bdofs[right_marker]) - 1
interface_nodes = np.asarray(bdofs[interface_marker]) - 1

left_top_nodes = np.asarray(bdofs[left_top_marker]) - 1
left_top_nodes = np.setdiff1d(left_top_nodes, [5,3])

left_bottom_nodes = np.asarray(bdofs[left_bottom_marker]) - 1
left_bottom_nodes = np.setdiff1d(left_bottom_nodes, [0,4])

right_bottom_nodes = np.asarray(bdofs[right_bottom_marker]) - 1
right_bottom_nodes = np.setdiff1d(right_bottom_nodes, [4,1])

right_top_nodes = np.asarray(bdofs[right_top_marker]) - 1
right_top_nodes = np.setdiff1d(right_top_nodes, [2,5])

left_circ_nodes = np.asarray(bdofs[left_circ_mark]) - 1
right_circ_nodes = np.asarray(bdofs[right_circ_mark]) - 1

boundary_nodes = np.hstack((
    left_nodes, right_nodes, interface_nodes, left_top_nodes,
    left_bottom_nodes, right_top_nodes, right_bottom_nodes,
    left_circ_nodes, right_circ_nodes
))

# interior nodes
elementmarkers = np.asarray(elementmarkers)

left_half_nodes = faces[elementmarkers == left_half_mark]
left_half_nodes = left_half_nodes.flatten()
left_half_nodes = np.setdiff1d(left_half_nodes, boundary_nodes)

left_circ_mat_nodes = faces[elementmarkers == left_circ_surf_mark]
left_circ_mat_nodes = left_circ_mat_nodes.flatten()
left_circ_mat_nodes = np.setdiff1d(left_circ_mat_nodes, boundary_nodes)

right_half_nodes = faces[elementmarkers == right_half_mark]
right_half_nodes = right_half_nodes.flatten()
right_half_nodes = np.setdiff1d(right_half_nodes, boundary_nodes)


# computing normal vectors
normal_vecs = np.zeros((coords.shape[0],2))
nodes_to_compute = (
    left_top_nodes,
    left_bottom_nodes,
    right_top_nodes,
    right_bottom_nodes,
    left_circ_nodes,
    interface_nodes
)
for nodes in nodes_to_compute:
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)

M_pinv = {}
support_stencils = {}

for i in range(coords.shape[0]):
    support_stencils[i] = get_support_nodes_2D(i, faces)
    if np.linalg.norm(normal_vecs[i]) > 1e-10:
        M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vecs[i])
    else:
        M = compute_M_matrix(i, support_stencils[i], coords)
    M_pinv[i] = np.linalg.pinv(M)

nodes_to_plot = (
    left_nodes,
    right_nodes,
    interface_nodes,
    left_top_nodes,
    left_bottom_nodes,
    right_top_nodes,
    right_bottom_nodes,
    left_circ_nodes,
    right_circ_nodes,
    left_half_nodes,
    left_circ_mat_nodes,
    right_half_nodes
)
labels = (
    "left",
    "right",
    "interface",
    "left_top",
    "left_bottom",
    "right_top",
    "right_bottom",
    "left_circ",
    "right_circ",
    "left_half",
    "left_circ_mat",
    "right_half"
)

if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["faces"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}
    with open('examples/legacy/meshes/mesh5.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")
