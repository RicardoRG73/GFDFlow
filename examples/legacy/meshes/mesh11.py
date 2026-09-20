show_figures = True
save_mesh_to_file = True

#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

from GFDFlow.utils import compute_normal_vectors

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
g.point([-400,355])     # 0
g.point([-400,-100])    # 1
g.point([1365,-100])    # 2
g.point([1365,110])     # 3
g.point([1365,130])     # 4
g.point([1150,140])     # 5
g.point([1120,148])     # 6
g.point([1080,148])     # 7
g.point([1070,152])     # 8
g.point([1050,152])     # 9
g.point([280,360])      # 10

g.point([40,280])       # 11
g.point([120,250])      # 12
g.point([300,250])      # 13
g.point([340,230])      # 14
g.point([360,225])      # 15
g.point([410,200])      # 16
g.point([460,190])      # 17
g.point([510,200])      # 18
g.point([1080,120])     # 19
g.point([1230,110])     # 20

# lines markers
left = 10
bottom = 11
right = 12
top_tailing = 13
top_dam = 14
interface_a = 15
interface_b = 16
interface_c = 17


# lines
g.line([0,1], marker=left)   # 0
g.line([1,2], marker=bottom)   # 1
g.line([2,3], marker=right)   # 2
g.line([3,4], marker=right)   # 3
g.line([4,5], marker=top_dam)   # 4
g.line([5,6], marker=top_dam)   # 5
g.line([6,7], marker=top_dam)   # 6
g.line([7,8], marker=top_dam)   # 7
g.line([8,9], marker=top_dam)   # 8
g.line([9,10], marker=top_dam)  # 9
g.line([10,0], marker=top_tailing)  # 10
g.line([0,11], marker=interface_a)  # 11
g.line([11,10], marker=interface_b) # 12
g.line([11,12], marker=interface_c) # 13
g.line([12,13], marker=interface_c) # 14
g.line([13,14], marker=interface_c) # 15
g.line([14,15], marker=interface_c) # 16
g.line([15,16], marker=interface_c) # 17
g.line([16,17], marker=interface_c) # 18
g.line([17,18], marker=interface_c) # 19
g.line([18,19], marker=interface_c) # 20
g.line([19,20], marker=interface_c) # 21
g.line([20,3], marker=interface_c)  # 22

# surfaces
tailing = 0
rock = 1
dam = 2
g.surface([10,11,12], marker=tailing)   # 0
g.surface([3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,22], marker=dam)     # 1
g.surface([0,1,2,22,21,20,19,18,17,16,15,14,13,11], marker=rock)  # 2



#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=20)

coords, edof, dofs, bdofs, element_markers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%%
# =============================================================================
# Nodes identification
# =============================================================================
left_nodes = np.asarray(bdofs[left]) - 1
bottom_nodes = np.asarray(bdofs[bottom]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
top_tailing_nodes = np.asarray(bdofs[top_tailing]) - 1
top_dam_nodes = np.asarray(bdofs[top_dam]) - 1
interface_a_nodes = np.asarray(bdofs[interface_a]) - 1
interface_b_nodes = np.asarray(bdofs[interface_b]) - 1
interface_c_nodes = np.asarray(bdofs[interface_c]) - 1

bottom_nodes = np.setdiff1d(bottom_nodes,left_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes,right_nodes)

top_tailing_nodes = np.setdiff1d(top_tailing_nodes, left_nodes)
top_dam_nodes = np.setdiff1d(top_dam_nodes, top_tailing_nodes)
top_dam_nodes = np.setdiff1d(top_dam_nodes, right_nodes)

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_tailing_nodes,
    top_dam_nodes
))

interface_a_nodes = np.setdiff1d(interface_a_nodes, boundaries)
interface_b_nodes = np.setdiff1d(interface_b_nodes, boundaries)
interface_c_nodes = np.setdiff1d(interface_c_nodes, boundaries)

boundaries = np.hstack((
    boundaries,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes
))

interface_a_nodes = np.setdiff1d(interface_a_nodes,[11])
interface_b_nodes = np.setdiff1d(interface_b_nodes,[11])
interface_c_nodes = np.setdiff1d(interface_c_nodes,[11])

element_markers = np.array(element_markers)

rock_nodes = faces[element_markers == rock]
rock_nodes = rock_nodes.flatten()
rock_nodes = np.setdiff1d(rock_nodes, boundaries)

dam_nodes = faces[element_markers == dam]
dam_nodes = dam_nodes.flatten()
dam_nodes = np.setdiff1d(dam_nodes, boundaries)

tailing_nodes = faces[element_markers == tailing]
tailing_nodes = tailing_nodes.flatten()
tailing_nodes = np.setdiff1d(tailing_nodes, boundaries)

nodes_to_plot = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_tailing_nodes,
    top_dam_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    rock_nodes,
    dam_nodes,
    tailing_nodes
)

labels = (
    "Left",
    "Right",
    "Bottom",
    "Top tailing",
    "Top dam",
    "Interface A",
    "Interface B",
    "Interface C",
    "Rock",
    "Dam",
    "Tailing"
)

# normal vectors
boundaries_with_normals = (
    bottom_nodes,
    top_tailing_nodes,
    top_dam_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
)
normal_vecs = np.zeros((coords.shape[0],2))
for nodes in boundaries_with_normals:
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)
normal_vecs[[11]] = np.array([0,-1])

# save data
if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()

    with open("examples/legacy/meshes/mesh11.json", "w") as f:
        json.dump(data_to_save, f, indent=4)
    print("============")
    print("Mesh saved ")
    print("============")



#%% Plots
if show_figures:
    # geometry plot
    plt.figure()
    cfv.draw_geometry(g)

    # mesh plot
    plt.figure()
    cfv.draw_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True
    )
    plt.title(f"Mesh")
    plt.suptitle(f"el_size_factor={mesh.el_size_factor}, N={coords.shape[0]} nodes", fontsize=8, y=0.90)

    # nodes identification plot
    colors = [
        "#2b5c8f",  # Blue
        "#d95f02",  # Orange
        "#7570b3",  # Purple
        "#1b9e77",  # Teal
        "#e7298a",  # Pink
        "#e6ab02",  # Yellow
        "#a6761d",  # Brown
        "#666666",  # Gray
        "#1f78b4",  # Light blue
        "#33a02c",  # Light green
        "#fb9a99",  # Light pink
        "#b2df8a",  # Light green
    ]
    plt.figure()
    for nodes, label, color in zip(nodes_to_plot, labels, colors):
        plt.scatter(coords[nodes, 0], coords[nodes, 1], label=label, alpha=0.5, color=color)
    plt.axis("equal")
    plt.legend()

    # normal vectors plot
    plt.figure()
    for nodes in boundaries_with_normals:
        plt.scatter(coords[nodes, 0], coords[nodes, 1])
        plt.quiver(coords[nodes, 0], coords[nodes, 1], normal_vecs[nodes, 0], normal_vecs[nodes, 1], alpha=0.3)
    plt.scatter(coords[[11],0],coords[[11],1])
    plt.quiver(coords[[11],0],coords[[11],1],normal_vecs[[11],0],normal_vecs[[11],1],alpha=0.3)
    plt.axis("equal")
    plt.show()

