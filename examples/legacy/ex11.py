#%%
# -- import libraries --
import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import json

from GFDFlow.GFDM import GFDMI_2D_problem as gfdm

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

#%%
# -- reading mesh data --
mesh_file = 'examples/legacy/meshes/mesh11.json'
with open(mesh_file, 'r') as file:
    mesh_data = json.load(file)
    coords = np.array(mesh_data["coords"])
    triangles = np.array(mesh_data["triangles"])
    normal_vecs = np.array(mesh_data["normal_vecs"])
    left_nodes = np.array(mesh_data["left_nodes"])
    right_nodes = np.array(mesh_data["right_nodes"])
    bottom_nodes = np.array(mesh_data["bottom_nodes"])
    top_dam_nodes = np.array(mesh_data["top_dam_nodes"])
    top_tailing_nodes = np.array(mesh_data["top_tailing_nodes"])
    interface_a_nodes = np.array(mesh_data["interface_a_nodes"])
    interface_b_nodes = np.array(mesh_data["interface_b_nodes"])
    interface_c_nodes = np.array(mesh_data["interface_c_nodes"])
    rock_nodes = np.array(mesh_data["rock_nodes"])
    dam_nodes = np.array(mesh_data["dam_nodes"])
    tailing_nodes = np.array(mesh_data["tailing_nodes"])

#%%
# -- domain properties --
L = np.array([0,0,0,1,0,1])
kdam = lambda p: 1e-2        # conductivity of dam
ktailing = lambda p: 1     # conductivity of tailing
krock = lambda p: 1e-4      # conductivity of rock

# source term 
source = lambda p: 0

# boundary conditions
neumann_0 = lambda p: 0
neumann_1 = lambda p: 0
left_dirichlet = lambda p: 300
right_dirichlet = lambda p: 200
beta = lambda p: 0

#%%
# -- assembling system KU = F --
problem = gfdm(coords, triangles, normal_vecs, L, source)

# material domains
problem.material("dam", kdam, dam_nodes)
problem.material("tailing", ktailing, tailing_nodes)
problem.material("rock", krock, rock_nodes)

# dirichlet boaundaries
problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)

# neumann boundaries
problem.neumann_boundary("bottom", krock,bottom_nodes, neumann_0)
problem.neumann_boundary("top_dam", kdam, top_dam_nodes, neumann_0)
problem.neumann_boundary("top_tailing", ktailing, top_tailing_nodes, neumann_1)

# interfaces
problem.interface("interface_a", ktailing, krock, interface_a_nodes, None, beta, None, tailing_nodes, rock_nodes)
problem.interface("interface_b", ktailing, kdam, interface_b_nodes, None, beta, None, tailing_nodes, dam_nodes)
problem.interface("interface_c", kdam, krock, interface_c_nodes, None, beta, None, dam_nodes, rock_nodes)

# intersection
center_node = 11
problem.intersection("inters_1_tailing", center_node, "interface_a", "interface_b", "tailing", beta)
problem.intersection("inters_1_rock", center_node, "interface_a", "interface_c", "rock", beta)
problem.intersection("inters_1_dam", center_node, "interface_b", "interface_c", "dam", beta)

# -- solving system KU = F --
K,F = problem.continuous_discretization()
U = sp.linalg.spsolve(K,F)

#%%
# -- post processing --
plt.figure(figsize=(10, 10))
tri = plt.tricontourf(coords[:,0], coords[:,1], triangles, U, cmap='viridis', levels=25)
plt.colorbar(tri, label=' Concentration (mg/L)')
plt.tricontour(coords[:,0], coords[:,1], triangles, U, colors='k', levels=25, linewidths=0.5)
plt.scatter(coords[interface_a_nodes,0], coords[interface_a_nodes,1], c='k', s=5, alpha=0.2)
plt.scatter(coords[interface_b_nodes,0], coords[interface_b_nodes,1], c='k', s=5, alpha=0.2)
plt.scatter(coords[interface_c_nodes,0], coords[interface_c_nodes,1], c='k', s=5, alpha=0.2)
plt.scatter(coords[center_node,0], coords[center_node,1], c='k', s=5, alpha=0.2)
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Concentration Distribution')
plt.show()

