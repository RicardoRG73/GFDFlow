#%%
import time
start_time = time.perf_counter()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import json
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d

with open('examples/legacy/meshes/mesh8.json', 'r') as file:
    mesh_data = json.load(file)

for key in mesh_data.keys():
    if key not in ["M_pinv", "support_stencils"]:
        globals()[key] = np.array(mesh_data[key])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

N = coords.shape[0]

#%% Problem Discretization
# Paramters laplacian
L = np.array([0,0,0,1,0,1])
source = lambda p: 0
k = lambda p: 0.5
neumann_condition = lambda p: 0

problem = gfdmi(coords, faces, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material("interior", k, interior_nodes)
problem.dirichlet_boundary("left", left_nodes, lambda p: 50)
problem.dirichlet_boundary("right", right_nodes, lambda p: 30)
problem.neumann_boundary("neumann", k, neumann_nodes, neumann_condition)

K, F = problem.continuous_discretization()

import scipy.sparse as sp
U = sp.linalg.spsolve(K, F)

# Plot solution
fig, ax = plot_solution_2d(
    coords,
    U,
    triangles=faces,
    levels=25,
    cmap="inferno",
    colorbar_label="h",
    figsize=(10, 4),
    linewidths=1,
    savepath="examples/legacy/figures/ex8/contourf.png",
)

# pile sheet region
pile_sheet_nodes = np.array([3,4,5,6,7,8,9,10])
plt.fill(
    coords[pile_sheet_nodes,0],
    coords[pile_sheet_nodes,1],
    color="gray"
)

# dam region
xs_dam = np.array([
    30.        , 31.57894737, 33.15789474, 34.73684211, 36.31578947,
    37.89473684, 39.47368421, 41.05263158, 42.63157895, 44.21052632,
    45.78947368, 47.36842105, 48.94736842, 50.52631579, 52.10526316,
    53.68421053, 55.26315789, 56.84210526, 58.42105263, 60.        , 60., 30.
])
ys_dam = np.array([
    45.        , 44.77443609, 44.54887218, 44.32330827, 44.09774436,
    43.27302632, 41.99013158, 40.70723684, 39.42434211, 38.14144737,
    37.34817814, 37.04453441, 36.74089069, 36.43724696, 36.13360324,
    35.82995951, 35.52631579, 35.22267206, 33.94736842, 30.        , 29., 29
])

plt.fill(
    xs_dam,
    ys_dam,
    color="gray"
)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

# plt.savefig("figures/ex8.png", dpi=300, bbox_inches="tight")
plt.show()