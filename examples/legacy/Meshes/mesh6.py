show_figures = False
save_mesh_to_file = True

#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
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
geometria = cfg.Geometry()

# points
geometria.point([0,0])      # 0
geometria.point([2,0])      # 1
geometria.point([2,1])      # 2
geometria.point([0,1])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 13

geometria.line([0,1], marker=bottom)    # 0
geometria.line([1,2], marker=right)     # 1
geometria.line([2,3], marker=top)       # 2
geometria.line([3,0], marker=left)      # 3

# surfaces
mat0 = 0
geometria.surface([0,1,2,3], marker=mat0)

#%%
# =============================================================================
# Mesh
# =============================================================================
# | el_size_factor |   N   |
# |     0.1        |  274  |
# |     0.05       |  998  |
# |     0.03       | 2748  |
# |     0.02       | 5969  |

mesh = cfm.GmshMesh(geometria)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.03

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%%
# =============================================================================
# Nodes identification by color
# =============================================================================
left_nodes = np.asarray(bdofs[left]) - 1
left_nodes = np.setdiff1d(left_nodes, [0,3])
right_nodes = np.asarray(bdofs[right]) - 1
right_nodes = np.setdiff1d(right_nodes, [1,2])
bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, [0,1])
top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, [2,3])
corner_nodes = np.array([0,1,2,3])

boundary_nodes = np.hstack((
    left_nodes, right_nodes, bottom_nodes, top_nodes, corner_nodes
))
interior_nodes = np.setdiff1d(np.arange(coords.shape[0]) , boundary_nodes)
nodes_to_plot = (interior_nodes, left_nodes, right_nodes, bottom_nodes, top_nodes, corner_nodes)
labels = (
    "Interior",
    "Left",
    "Right",
    "Bottom",
    "Top",
    "Corner"
)

#%%
# =============================================================================
# Plot figures
# =============================================================================
if show_figures:
    # geometry plot
    cfv.figure(fig_size=(8,5))
    cfv.title('Geometría')
    cfv.draw_geometry(geometria, font_size=16, draw_axis=True)

    # mesh plot
    plt.figure(figsize=(7,4))
    cfv.title('Malla $N=%d' %coords.shape[0] +'$')
    cfv.draw_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True
    )

    # nodes plot by color
    plt.figure(figsize=(7,4))
    for nodes, label in zip(nodes_to_plot, labels):
        plt.scatter(
            coords[nodes, 0],
            coords[nodes, 1],
            label=label,
            s=10,
            alpha=0.5
        )
    plt.axis('equal')
    plt.legend()


    plt.show()

#%%
# =============================================================================
# Save mesh to file
# =============================================================================
if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["boundary_nodes"] = boundary_nodes.tolist()

    with open('Meshes/mesh6.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")