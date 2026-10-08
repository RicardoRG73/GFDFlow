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
interface_elsize = 0.5
g.point([-1, -1])         # 0
g.point([1, -1])         # 1
g.point([1, 1])         # 2
g.point([-1, 1])         # 3

g.point([0, 0])         # 4 : center
g.point([0, -0.5], el_size=interface_elsize)       # 5
g.point([0.5, 0], el_size=interface_elsize)       # 6
g.point([0, 0.5], el_size=interface_elsize)       # 7
g.point([-0.5, 0], el_size=interface_elsize)      # 8

# lines
    # boundary
dird = 10
dirr = 11
diru = 12
dirl = 13
g.line([0,1], marker=dird)       # 0
g.line([1,2], marker=dirr)       # 1
g.line([2,3], marker=diru)       # 2
g.line([3,0], marker=dirl)       # 3

    # circle
interf = 14
g.circle([5,4,6], marker=interf)    # 4
g.circle([6,4,7], marker=interf)    # 5
g.circle([7,4,8], marker=interf)    # 6
g.circle([8,4,5], marker=interf)    # 7

# surfaces
mat0 = 100      # marker for nodes on material 1
mat1 = 101      # marker for nodes on material 2
g.surface([0, 1, 2, 3], [[4, 5, 6, 7]], marker=mat0)    # 0
g.surface([4, 5, 6, 7], marker=mat1)    # 1


# =====
# Mesh creation from geometry object
# =====
mesh = cfm.GmshMesh(g)

mesh.el_type = 2                # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()       # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(coords, edof, mesh.dofs_per_node, mesh.el_type)  # coordinate-edges to vertices-faces


# =====
# Detection of boundary nodes index
# =====
b0 = np.asarray(bdofs[dird]) - 1            # index nodes in down boundary
b1 = np.asarray(bdofs[dirr]) - 1            # index nodes in right boundary
b1 = np.setdiff1d(b1,[1,2])
b2 = np.asarray(bdofs[diru]) - 1            # index nodes in up boundary
b3 = np.asarray(bdofs[dirl]) - 1            # index nodes in left boundary
b3 = np.setdiff1d(b3,[3,0])
bi = np.asarray(bdofs[interf]) - 1          # index of nodes on the interface

B = np.hstack((b0,b1,b2,b3,bi))

elementmarkers = np.asarray(elementmarkers)

bm0 = faces[elementmarkers == mat0]
bm0 = bm0.flatten()
bm0 = np.setdiff1d(bm0,B)

bm1 = faces[elementmarkers == mat1]
bm1 = bm1.flatten()
bm1 = np.setdiff1d(bm1,B)

# normal vectors
normal_vecs = np.zeros((coords.shape[0],2))
normal_vecs[bi] = compute_normal_vectors(bi, coords)

M_pinv = {}
support_stencils = {}

for i in range(coords.shape[0]):
    support_stencils[i] = get_support_nodes_2D(i, faces)
    if np.linalg.norm(normal_vecs[i]) > 1e-10:
        M = compute_M_matrix_neumann(i, support_stencils[i], coords, normal_vecs[i])
    else:
        M = compute_M_matrix(i, support_stencils[i], coords)
    M_pinv[i] = np.linalg.pinv(M)

nodes_to_plot = (b0, b1, b2, b3, bi, bm0, bm1)
labels = ("b0", "b1", "b2", "b3", "bi", "bm0", "bm1")


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

    with open('examples/legacy/meshes/mesh10.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")
