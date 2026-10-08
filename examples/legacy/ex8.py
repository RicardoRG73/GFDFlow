"""Example 08: 2D Confined Seepage Under a Dam with a Sheet Pile.

This example simulates steady-state 2D confined seepage through a permeable
foundation beneath a concrete dam structure equipped with a vertical sheet pile wall.

Governing Equation
------------------
.. math::
    \\nabla^2 u = u_{xx} + u_{yy} = 0

Parameters
----------
* Permeability: :math:`k = 0.5`
* Upstream reservoir head: :math:`h_1 = 50.0`
* Downstream tailwater head: :math:`h_2 = 30.0`
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d

start_time = time.perf_counter()

# Configure visualization style
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh8.json", "r") as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
faces = np.array(mesh_data["faces"])
normal_vecs = np.array(mesh_data["normal_vecs"])
interior_nodes = np.array(mesh_data["interior_nodes"])
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
neumann_nodes = np.array(mesh_data["neumann_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
source = lambda p: 0.0
k = lambda p: 0.5
neumann_condition = lambda p: 0.0

# =============================================================================
# 3. GFDM Problem Initialization and Boundary Setup
# =============================================================================
problem = gfdmi(
    coords,
    faces,
    normal_vecs,
    L,
    source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

problem.material("interior", k, interior_nodes)
problem.dirichlet_boundary("left", left_nodes, lambda p: 50.0)
problem.dirichlet_boundary("right", right_nodes, lambda p: 30.0)
problem.neumann_boundary("neumann", k, neumann_nodes, neumann_condition)

# =============================================================================
# 4. System Assembly and Solution
# =============================================================================
K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 5. Visualization with Sheet Pile and Dam Geometry Overlay
# =============================================================================
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

# Overlay sheet pile wall geometry
pile_sheet_nodes = np.array([3, 4, 5, 6, 7, 8, 9, 10])
plt.fill(coords[pile_sheet_nodes, 0], coords[pile_sheet_nodes, 1], color="gray")

# Overlay concrete dam cross-section
xs_dam = np.array([
    30.0, 31.57894737, 33.15789474, 34.73684211, 36.31578947,
    37.89473684, 39.47368421, 41.05263158, 42.63157895, 44.21052632,
    45.78947368, 47.36842105, 48.94736842, 50.52631579, 52.10526316,
    53.68421053, 55.26315789, 56.84210526, 58.42105263, 60.0, 60.0, 30.0,
])
ys_dam = np.array([
    45.0, 44.77443609, 44.54887218, 44.32330827, 44.09774436,
    43.27302632, 41.99013158, 40.70723684, 39.42434211, 38.14144737,
    37.34817814, 37.04453441, 36.74089069, 36.43724696, 36.13360324,
    35.82995951, 35.52631579, 35.22267206, 33.94736842, 30.0, 29.0, 29.0,
])
plt.fill(xs_dam, ys_dam, color="gray")

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")

plt.show()