show_figures = True
save_mesh_to_file = True

#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
g.point([0, 0])
g.point([4, 0])
g.point([4, 1])
g.point([3, 1], el_size=0.3)
g.point([1, 1], el_size=0.3)
g.point([0, 1])

# lines
bottom = 10
right = 11
top_right = 12
top_middle = 13
top_left = 14
left = 15
g.line([0, 1], marker=bottom)
g.line([1, 2], marker=right)
g.line([2, 3], marker=top_right)
g.line([3, 4], marker=top_middle)
g.line([4, 5], marker=top_left)
g.line([5, 0], marker=left)

# surfaces
g.surface([0, 1, 2, 3, 4, 5])

#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=0.1)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

print(f"Number of nodes N={coords.shape[0]}")

#%%
# =============================================================================
# Node identification
# =============================================================================
bottom_nodes = np.asarray(bdofs[bottom]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
top_right_nodes = np.asarray(bdofs[top_right]) - 1
top_middle_nodes = np.asarray(bdofs[top_middle]) - 1
top_left_nodes = np.asarray(bdofs[top_left]) - 1
left_nodes = np.asarray(bdofs[left]) - 1

# elimination of intersection nodes
top_right_nodes = np.setdiff1d(top_right_nodes, top_middle_nodes)
top_left_nodes = np.setdiff1d(top_left_nodes, top_middle_nodes)
right_nodes = np.setdiff1d(right_nodes, top_right_nodes)
right_nodes = np.setdiff1d(right_nodes, bottom_nodes)
left_nodes = np.setdiff1d(left_nodes, top_left_nodes)
left_nodes = np.setdiff1d(left_nodes, bottom_nodes)

boundary_nodes = np.hstack((
    bottom_nodes,
    right_nodes,
    top_right_nodes,
    top_middle_nodes,
    top_left_nodes,
    left_nodes
))

N = coords.shape[0]
interior_nodes = np.setdiff1d(np.arange(N), boundary_nodes)

nodes_to_plot = (
    interior_nodes,
    bottom_nodes,
    right_nodes,
    top_right_nodes,
    top_middle_nodes,
    top_left_nodes,
    left_nodes
)
labels = (
    "interior",
    "bottom",
    "right",
    "top_right",
    "top_middle",
    "top_left",
    "left"
)

normal_vecs = np.zeros((coords.shape[0], 2))
normal_vecs[bottom_nodes]       = np.array([0,-1])
normal_vecs[right_nodes]        = np.array([1,0])
normal_vecs[top_right_nodes]    = np.array([0,1])
normal_vecs[top_middle_nodes]   = np.array([0,1])
normal_vecs[top_left_nodes]     = np.array([0,1])
normal_vecs[left_nodes]         = np.array([-1,0])



if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["boundary_nodes"] = boundary_nodes.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()

    with open('examples/legacy/meshes/mesh7.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")




if show_figures:
    from GFDFlow.visualization import (
        plot_geometry,
        plot_mesh,
        plot_nodes,
        plot_normal_vectors,
    )

    # geometry plot
    plot_geometry(g, title="Geometry", figsize=(8, 3), savepath="examples/legacy/figures/ex7/geometry.png")

    # mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(8, 3),
        title="Mesh",
        suptitle=f"el_size_factor={mesh.el_size_factor}, N={coords.shape[0]} nodes",
        savepath="examples/legacy/figures/ex7/mesh.png",
    )

    # nodes by color plot
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=1.0,
        point_size=20,
        figsize=(8, 3),
        savepath="examples/legacy/figures/ex7/nodes.png",
    )

    # normal vectors plot
    b_nodes_tuple = (
        bottom_nodes,
        right_nodes,
        top_right_nodes,
        top_middle_nodes,
        top_left_nodes,
        left_nodes,
    )
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=b_nodes_tuple,
        title="Normal vectors",
        savepath="examples/legacy/figures/ex7/normal_vectors.png",
    )

    plt.show()