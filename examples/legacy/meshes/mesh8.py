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
g.point([0,0])                  # 0
g.point([90,0])                 # 1
g.point([90,30])                # 2
g.point([60,30], el_size=0.4)                # 3
g.point([60,16], el_size=0.1)   # 4
g.point([59,16], el_size=0.1)   # 5
g.point([59,27], el_size=0.6)   # 6
g.point([31,27], el_size=0.6)   # 7
g.point([31,12], el_size=0.1)   # 8
g.point([30,12], el_size=0.1)   # 9
g.point([30,30], el_size=0.4)                # 10
g.point([0,30])                 # 11

# lines
left = 10
right = 11
neumann = 12
g.spline([0,1], marker=neumann)
g.spline([1,2], marker=neumann)
g.spline([2,3], marker=right)
g.spline([3,4], marker=neumann)
g.spline([4,5], marker=neumann)
g.spline([5,6], marker=neumann)
g.spline([6,7], marker=neumann)
g.spline([7,8], marker=neumann)
g.spline([8,9], marker=neumann)
g.spline([9,10], marker=neumann)
g.spline([10,11], marker=left)
g.spline([11,0], marker=neumann)

# surfaces
g.surface([0,1,2,3,4,5,6,7,8,9,10,11])

# mesh generation
mesh = cfm.GmshMesh(g,el_size_factor=2)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# nodes identification
left_nodes = np.asarray(bdofs[left]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
neumann_nodes = np.asarray(bdofs[neumann]) - 1

# elination of repited nodes
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

# Normal vectors computation
normal_vecs_compact = compute_normal_vectors(neumann_nodes, coords)
normal_vecs = np.zeros((N, 2))
normal_vecs[neumann_nodes] = normal_vecs_compact

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
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["faces"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    data_to_save["M_pinv"] = {str(k): v.tolist() for k, v in M_pinv.items()}
    data_to_save["support_stencils"] = {str(k): v.tolist() for k, v in support_stencils.items()}

    with open('examples/legacy/meshes/mesh8.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")
