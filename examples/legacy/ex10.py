"""
Solution to the Poisson equation
\nabla^2 u = f
in domain: `x in [-1,1]` and `y in [-1,1]`
interface  in  `x**2 + y**2 == 0.25**2`
material 0 in  `x**2 + y**2 >  0.25**2`
material 1 in  `x**2 + y**2 <  0.25**2`

stationary and non-stationary solutions
\nabla^2 u + f = du/dt
"""

# =====
# Importing needed libraries
# =====
import numpy as np
import matplotlib.pyplot as plt
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import (
    plot_geometry,
    plot_mesh,
    plot_nodes,
    plot_normal_vectors,
    plot_solution_2d,
    plot_solution_3d,
)

import json
with open('examples/legacy/meshes/mesh10.json', 'r') as file:
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


# =====
# Problem parameters
# =====
k0 = lambda p: 100                                    # mat0 permeability
k1 = lambda p: 1                                      # mat1 permeability
fd0 = lambda p: 0                # Dirichlet condition down
fd1 = lambda p: np.sin(np.pi*(p[1]+1)/4)                 # Dirichlet condition right
fd2 = lambda p: np.sin(np.pi*(p[0]+1)/4)                 # dirichlet condition up
fd3 = lambda p: 0                 # dirichlet condition left
fi = lambda p: 0                            # interface condition
delta = 0.01 * 0.2 * 0.5
def fs(p):                                  # sourse
    out = 0
    return out
L = np.array([0,0,0,2,0,2])                 # coefitients vector

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

problem = gfdmi(coords, faces, normal_vecs, L, fs, M_pinv=M_pinv, support_stencils=support_stencils)
problem.material("mat0", k0, bm0)
problem.material("mat1", k1, bm1)

problem.dirichlet_boundary("down", b0, fd0)
problem.dirichlet_boundary("right", b1, fd1)
problem.dirichlet_boundary("up", b2, fd2)
problem.dirichlet_boundary("left", b3, fd3)

problem.interface("interf", k0, k1, bi, None, fi, None, bm0, bm1)

# ====
# Solution
# ====
K, F = problem.continuous_discretization()

U = sp.linalg.spsolve(K,F)

# =====
# Plotting solution
# =====
plot_solution_3d(
    coords, U,
    triangles=faces,
    cmap="inferno",
    edge_color="k",
    alpha=0.7,
    title="3D Solution",
    figsize=(7, 7),
    savepath="examples/legacy/figures/ex10/3dplot_steady.png",
)

plot_solution_2d(
    coords, U,
    levels=20,
    cmap="inferno",
    title="Contour Solution",
    figsize=(7, 7),
    overlay_nodes=bi,
    savepath="examples/legacy/figures/ex10/contourf_steady.png",
)



# =====
# Crank-Nicolson
# =====
T = 0.1
dt = 0.0001
m = round(T/dt)

beta = np.ones(len(F))
beta[np.hstack((b0,b1,b2,b3))] = 0  # Dirichlet boundaries
A = sp.eye(len(F)) - dt/2 * sp.diags(beta) @ K
B = sp.eye(len(F)) + dt/2 * sp.diags(beta) @ K

# first time step solution
U2 = np.zeros((m,len(F)))
for i in b1:
    U2[0,i] = fd1(coords[i])
for i in b2:
    U2[0,i] = fd2(coords[i])

F[b1] = 0
F[b2] = 0

# next solutions
for i in range(m-1):
    U2[i+1] = sp.linalg.spsolve(A, B@U2[i] + dt*F)

plot_solution_3d(
    coords, U2[-1],
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
    coords, U2[-1],
    levels=20,
    cmap="inferno",
    title=f"Crank-Nicolson, $t={T}$",
    figsize=(7, 7),
    overlay_nodes=bi,
    savepath="examples/legacy/figures/ex10/contourf.png",
)


# animated plot
from matplotlib.animation import FuncAnimation 
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1, projection="3d")
ax2 = fig.add_subplot(1,2,2)

index = U2.shape[0] - 1
cont1 = ax1.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U2[-1],
    cmap="inferno"
)
fig.colorbar(cont1)
ax1.set_title("3D Solution")
ax1.axis("equal")

cont2 = ax2.tricontourf(
    coords[:,0],
    coords[:,1],
    U2[-1],
    cmap="inferno",
    levels=20
)
fig.colorbar(cont2)
ax2.set_title("Contour Solution")
ax2.axis("equal")
fig.suptitle("Crank-Nicolson, t = {T:.4f}")

zlims = (-10, 10)
def update(frame):
    ax1.clear()
    ax2.clear()

    cont1 = ax1.plot_trisurf(
        coords[:,0],
        coords[:,1],
        U2[frame],
        cmap="inferno"
    )
    ax1.set_title("3D Solution")
    ax1.axis("equal")

    cont2 = ax2.tricontourf(
        coords[:,0],
        coords[:,1],
        U2[frame],
        cmap="inferno",
        levels=20
    )
    ax2.set_title("Contour Solution")
    ax2.axis("equal")
    fig.suptitle(f"Crank-Nicolson, t = {frame*dt:.4f}")
    print(f"t = {frame*dt:.4f}", flush=True)

    return cont1, cont2

ani = FuncAnimation(fig, update, frames=range(0, U2.shape[0], 10), blit=False, interval=24)

ani.save("examples/legacy/figures/ex10/solution.gif", writer='pillow', fps=24)
plt.savefig(f"examples/legacy/figures/ex10/solution_t={T}.png", dpi=300, bbox_inches="tight")
plt.show()