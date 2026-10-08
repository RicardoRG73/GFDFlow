show_plots = False
save_mesh_to_file = True

import numpy as np
import matplotlib.pyplot as plt
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
from GFDFlow.utils import compute_normal_vectors, get_support_nodes_2D, compute_M_matrix, compute_M_matrix_neumann

# =====
# Geometry creation
# =====
g = cfg.Geometry()          # geometry object

# points
refinement = 0.1
g.point([-1, -1])         # 0
g.point([1, -1])         # 1
g.point([1, 1])         # 2
g.point([0, 1])       # 3
g.point([-1, 1])         # 4
g.point([0, 0])     # 5 : Center point


# lines
dir = 10
g.line([0, 1], marker=dir)      # 0
g.line([1, 2], marker=dir)      # 1
g.line([2, 3], marker=dir)      # 2
g.line([3, 4], marker=dir)      # 3
g.line([4, 0], marker=dir)      # 4


interf0 = 11
interf1 = 12
interf2 = 13
g.line([5, 0], marker=interf0)  # 5
g.line([5, 1], marker=interf1)  # 6
g.line([5, 3], marker=interf2)  # 7



# surfaces
mat0 = 100
mat1 = 101
mat2 = 102
g.surface([0, 6, 5], marker=mat0)       # 0
g.surface([1, 2, 7, 6], marker=mat1)    # 1
g.surface([3, 4, 5, 7], marker=mat2)    # 2

# =====
# Mesh creation from geometry object
# =====
mesh = cfm.GmshMesh(g)

mesh.el_type = 2                # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.05

coords, edof, dofs, bdofs, elementmarkers = mesh.create()       # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(coords, edof, mesh.dofs_per_node, mesh.el_type)  # coordinate-edges to vertices-faces

# =====
# Detection of boundary and interior nodes
# =====
dirichlet_nodes = np.asarray(bdofs[dir]) - 1
interf0_nodes = np.asarray(bdofs[interf0]) - 1
interf0_nodes = np.setdiff1d(interf0_nodes,[5,0])
interf1_nodes = np.asarray(bdofs[interf1]) - 1
interf1_nodes = np.setdiff1d(interf1_nodes,[5,1])
interf2_nodes = np.asarray(bdofs[interf2]) - 1
interf2_nodes = np.setdiff1d(interf2_nodes,[5,3])

B = np.hstack((dirichlet_nodes,interf0_nodes,interf1_nodes,interf2_nodes,[5]))

elementmarkers = np.asarray(elementmarkers)

mat0_nodes = faces[elementmarkers == mat0]
mat0_nodes = mat0_nodes.flatten()
mat0_nodes = np.setdiff1d(mat0_nodes,B)

mat1_nodes = faces[elementmarkers == mat1]
mat1_nodes = mat1_nodes.flatten()
mat1_nodes = np.setdiff1d(mat1_nodes,B)

mat2_nodes = faces[elementmarkers == mat2]
mat2_nodes = mat2_nodes.flatten()
mat2_nodes = np.setdiff1d(mat2_nodes,B)

# normal vectors
normal_vecs = np.zeros((len(coords), 2))

normal_vecs[interf0_nodes] = compute_normal_vectors(interf0_nodes, coords)
normal_vecs[interf1_nodes] = compute_normal_vectors(interf1_nodes, coords)
normal_vecs[interf2_nodes] = compute_normal_vectors(interf2_nodes, coords)
normal_vecs[[5]] = np.array([1,1])/np.sqrt(2)

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
    mat0_nodes, mat1_nodes, mat2_nodes,
    interf0_nodes, interf1_nodes, interf2_nodes,
    dirichlet_nodes
)
labels = (
    "mat0", "mat1", "mat2",
    "interf0", "interf1", "interf2",
    "dirichlet"
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

    with open('examples/legacy/meshes/mesh9.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")
