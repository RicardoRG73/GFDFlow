
show_plots = True
save_mesh_to_file = True
#%%
# =============================================================================
# Importing needed libraries
# =============================================================================
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn-v0_8")

from GFDFlow.utils import compute_normal_vectors

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()  # Create a GeoData object that holds the geometry.
g.point([0, 0]) #0
g.point([15, 0]) #1
g.point([35, 0]) #2
g.point([42,0])
g.point([47, 0]) #3
g.point([27, 10]) #4
g.point([23, 10]) #5
g.point([19, 8]) #6
g.point([0, -5]) #7
g.point([23, -5]) #8
g.point([27, -5]) #9
g.point([55, -5]) #10
g.point([55,0]) #11

# line markers
left = 10
right = 11
bottom = 12
top = 13
interface_a = 14
interface_b = 15
interface_c = 16
interface_d = 17
interface_e = 18
interface_f = 19

# lines
g.spline([0, 1], marker=interface_a) #0
g.spline([2, 3], marker=interface_b) #1
g.spline([3, 4], marker=right)
g.spline([4, 5], marker=top) #2
g.spline([5, 6], marker=top) #3
g.spline([6, 7], marker=top) #4
g.spline([7, 0], marker=left) #5

g.spline([0,8], marker=bottom) #6
g.spline([8,9], marker=bottom) #7
g.spline([9,10], marker=bottom) #8
g.spline([10,11], marker=bottom) #9
g.spline([11,12], marker=bottom) #10
g.spline([12,4], marker=right) #11

g.spline([6,1], marker=interface_c) #12
g.spline([1,9], marker=interface_d) #13

g.spline([5,2], marker=interface_f) #14
g.spline([2,10], marker=interface_e) #15

# surface markers
rock = 1
clay = 2
mixed = 3

# surfaces
g.surface([0,13,5,6], marker=rock) #0
g.surface([1,2,3,15], marker=rock) #1
g.surface([7,8,14,0], marker=mixed) #2
g.surface([10,11,12,2,1,16], marker=mixed) #3
g.surface([13,14,9,16,15,4], marker=clay) #4

#%% mesh creation
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g)

mesh.el_type = 2
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.5

coords, edof, dofs, bdofs, element_markers = mesh.create()

verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

triangles = faces.copy()

#%%
# =============================================================================
# Nodes index
# =============================================================================
left_nodes = np.asarray(bdofs[left]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
top_nodes = np.asarray(bdofs[top]) - 1
bottom_nodes = np.asarray(bdofs[bottom]) - 1
interface_a_nodes = np.asarray(bdofs[interface_a]) - 1
interface_b_nodes = np.asarray(bdofs[interface_b]) - 1
interface_c_nodes = np.asarray(bdofs[interface_c]) - 1
interface_d_nodes = np.asarray(bdofs[interface_d]) - 1
interface_e_nodes = np.asarray(bdofs[interface_e]) - 1
interface_f_nodes = np.asarray(bdofs[interface_f]) - 1

# elimination of duplicated nodes
top_nodes = np.setdiff1d(top_nodes, left_nodes)
top_nodes = np.setdiff1d(top_nodes, right_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, left_nodes)
bottom_nodes = np.setdiff1d(bottom_nodes, right_nodes)

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
))

interface_a_nodes = np.setdiff1d(interface_a_nodes, boundaries)
interface_b_nodes = np.setdiff1d(interface_b_nodes, boundaries)
interface_c_nodes = np.setdiff1d(interface_c_nodes, boundaries)
interface_d_nodes = np.setdiff1d(interface_d_nodes, boundaries)
interface_e_nodes = np.setdiff1d(interface_e_nodes, boundaries)
interface_f_nodes = np.setdiff1d(interface_f_nodes, boundaries)

boundaries = np.hstack((
    boundaries,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    interface_d_nodes,
    interface_e_nodes,
    interface_f_nodes
))

element_markers = np.array(element_markers)

rock_nodes = triangles[element_markers == rock]
rock_nodes = rock_nodes.flatten()
rock_nodes = np.setdiff1d(rock_nodes, boundaries)

clay_nodes = triangles[element_markers == clay]
clay_nodes = clay_nodes.flatten()
clay_nodes = np.setdiff1d(clay_nodes, boundaries)

mixed_nodes = triangles[element_markers == mixed]
mixed_nodes = mixed_nodes.flatten()
mixed_nodes = np.setdiff1d(mixed_nodes, boundaries)

# deleting interface intersection nodes
interface_a_nodes = np.setdiff1d(interface_a_nodes, [1])
interface_c_nodes = np.setdiff1d(interface_c_nodes, [1])
interface_d_nodes = np.setdiff1d(interface_d_nodes, [1])
interface_b_nodes = np.setdiff1d(interface_b_nodes, [2])
interface_e_nodes = np.setdiff1d(interface_e_nodes, [2])
interface_f_nodes = np.setdiff1d(interface_f_nodes, [2])


nodes_to_plot = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    interface_d_nodes,
    interface_e_nodes,
    interface_f_nodes,
    rock_nodes,
    clay_nodes,
    mixed_nodes
)
labels = (
    "Left",
    "Right",
    "Bottom",
    "Top",
    "Interface A",
    "Interface B",
    "Interface C",
    "Interface D",
    "Interface E",
    "Interface F",
    "Rock",
    "Clay",
    "Mixed"
)

# computing normal vectors
boundaries_with_normals = (
    bottom_nodes,
    top_nodes,
    interface_a_nodes,
    interface_b_nodes,
    interface_c_nodes,
    interface_d_nodes,
    interface_e_nodes,
    interface_f_nodes
)
normal_vecs = np.zeros((coords.shape[0],2))
for nodes in boundaries_with_normals:
    normal_vecs[nodes] = compute_normal_vectors(nodes, coords)
normal_vecs[[1]] = np.array([-1,0])
normal_vecs[[2]] = np.array([-1,0])

if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = triangles.tolist()
    data_to_save["normal_vecs"] = normal_vecs.tolist()
    with open('examples/legacy/meshes/mesh4.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")


if show_plots:
    from GFDFlow.visualization import (
        plot_geometry,
        plot_mesh,
        plot_nodes,
        plot_normal_vectors,
    )

    # geometry plot
    plot_geometry(g, title="Geometry", figsize=(6, 4), savepath="examples/legacy/figures/ex4/geometry.png")

    # mesh plot
    plot_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True,
        figsize=(6, 4),
        title="Mesh",
        savepath="examples/legacy/figures/ex4/mesh.png",
    )

    # plotting nodes by color
    plot_nodes(
        coords,
        zip(labels, nodes_to_plot),
        title=f"$N = {coords.shape[0]}$",
        alpha=0.75,
        point_size=10,
        savepath="examples/legacy/figures/ex4/nodes.png",
    )

    # normal vectors plot
    all_normal_nodes = boundaries_with_normals + (np.array([1, 2]),)
    plot_normal_vectors(
        coords,
        normal_vecs,
        boundary_nodes=all_normal_nodes,
        quiver_color="k",
        quiver_alpha=0.3,
        title="Normal Vectors",
        savepath="examples/legacy/figures/ex4/normal_vectors.png",
    )

    plt.show()