"""Example 10: Stationary and Transient Diffusion with Circular Inclusion.

This example solves stationary and transient diffusion across a domain with
a centered circular inclusion (:math:`r = 0.25`). It evaluates steady-state
behavior and integrates time evolution using the implicit Crank-Nicolson scheme.

Governing Equations
-------------------
* Stationary Problem:
    .. math:: \\nabla^2 u = 0, \\quad (x, y) \\in [-1, 1] \\times [-1, 1]
* Transient Diffusion:
    .. math:: \\frac{\\partial u}{\\partial t} = \\nabla^2 u

Materials and Conductivities
----------------------------
* Matrix (:math:`r > 0.25`): :math:`k_0 = 100.0`
* Inclusion (:math:`r < 0.25`): :math:`k_1 = 1.0`
"""

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh10.json", "r") as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
faces = np.array(mesh_data["faces"])
normal_vecs = np.array(mesh_data["normal_vecs"])
b0 = np.array(mesh_data["b0_nodes"])
b1 = np.array(mesh_data["b1_nodes"])
b2 = np.array(mesh_data["b2_nodes"])
b3 = np.array(mesh_data["b3_nodes"])
bi = np.array(mesh_data["bi_nodes"])
bm0 = np.array(mesh_data["bm0_nodes"])
bm1 = np.array(mesh_data["bm1_nodes"])

support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
k0 = lambda p: 100.0                                    # Mat0 permeability
k1 = lambda p: 1.0                                      # Mat1 permeability
fd0 = lambda p: 0.0                                     # Dirichlet condition bottom
fd1 = lambda p: np.sin(np.pi * (p[1] + 1) / 4)          # Dirichlet condition right
fd2 = lambda p: np.sin(np.pi * (p[0] + 1) / 4)          # Dirichlet condition top
fd3 = lambda p: 0.0                                     # Dirichlet condition left
fi = lambda p: 0.0                                      # Interface flux balance
fs = lambda p: 0.0                                      # Source term

L = np.array([0, 0, 0, 2, 0, 2])

# =============================================================================
# 3. GFDM Problem Initialization and Stationary Solution
# =============================================================================
problem = gfdmi(
    coords,
    faces,
    normal_vecs,
    L,
    fs,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)
problem.material("mat0", k0, bm0)
problem.material("mat1", k1, bm1)

problem.dirichlet_boundary("down", b0, fd0)
problem.dirichlet_boundary("right", b1, fd1)
problem.dirichlet_boundary("up", b2, fd2)
problem.dirichlet_boundary("left", b3, fd3)

problem.interface("interf", k0, k1, bi, None, fi, None, bm0, bm1)

K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 4. Stationary Visualization
# =============================================================================
plot_solution_3d(
    coords,
    U,
    triangles=faces,
    cmap="inferno",
    edge_color="k",
    alpha=0.7,
    title="3D Solution",
    figsize=(7, 7),
    savepath="examples/legacy/figures/ex10/3dplot_steady.png",
)

plot_solution_2d(
    coords,
    U,
    levels=20,
    cmap="inferno",
    title="Contour Solution",
    figsize=(7, 7),
    overlay_nodes=bi,
    savepath="examples/legacy/figures/ex10/contourf_steady.png",
)

# =============================================================================
# 5. Crank-Nicolson Implicit Time-Stepping
# =============================================================================
T = 0.1
dt = 0.0001
m = round(T / dt)

beta = np.ones(len(F))
beta[np.hstack((b0, b1, b2, b3))] = 0.0  # Zero out Dirichlet boundary rows
A = sp.eye(len(F)) - dt / 2 * sp.diags(beta) @ K
B = sp.eye(len(F)) + dt / 2 * sp.diags(beta) @ K

# Initialize time-step solution matrix
U2 = np.zeros((m, len(F)))
for i in b1:
    U2[0, i] = fd1(coords[i])
for i in b2:
    U2[0, i] = fd2(coords[i])

F[b1] = 0.0
F[b2] = 0.0

# March forward in time
for i in range(m - 1):
    U2[i + 1] = sp.linalg.spsolve(A, B @ U2[i] + dt * F)

# =============================================================================
# 6. Transient Visualization at Final Time
# =============================================================================
plot_solution_3d(
    coords,
    U2[-1],
    triangles=faces,
    cmap="inferno",
    edge_color="k",
    alpha=0.7,
    view_init=(35, -127),
    title=f"Crank-Nicolson, $t={T}$",
    figsize=(7, 7),
    savepath="examples/legacy/figures/ex10/3dplot.png",
)

plot_solution_2d(
    coords,
    U2[-1],
    levels=20,
    cmap="inferno",
    title=f"Crank-Nicolson, $t={T}$",
    figsize=(7, 7),
    overlay_nodes=bi,
    savepath="examples/legacy/figures/ex10/contourf.png",
)

# =============================================================================
# 7. Animation Generation and Export
# =============================================================================
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)


def update(frame: int):
    """Update animation frames for 3D surface and 2D contour subplots.

    Parameters
    ----------
    frame : int
        Current time step index.

    Returns
    -------
    tuple
        Tuple of matplotlib artists (cont1, cont2).
    """
    ax1.clear()
    ax2.clear()

    cont1 = ax1.plot_trisurf(coords[:, 0], coords[:, 1], U2[frame], cmap="inferno")
    ax1.set_title("3D Solution")
    ax1.axis("equal")

    cont2 = ax2.tricontourf(coords[:, 0], coords[:, 1], U2[frame], cmap="inferno", levels=20)
    ax2.set_title("Contour Solution")
    ax2.axis("equal")
    fig.suptitle(f"Crank-Nicolson, t = {frame * dt:.4f}")
    print(f"t = {frame * dt:.4f}", flush=True)

    return cont1, cont2


ani = FuncAnimation(fig, update, frames=range(0, U2.shape[0], 10), blit=False, interval=24)
ani.save("examples/legacy/figures/ex10/solution.gif", writer="pillow", fps=24)

plt.savefig(f"examples/legacy/figures/ex10/solution_t={T}.png", dpi=300, bbox_inches="tight")
plt.show()