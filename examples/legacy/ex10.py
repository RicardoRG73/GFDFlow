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

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

# import plots

# =====
# Geometry creation
# =====
g = cfg.Geometry()          # geometry object

# points
    # square
g.point([-1, -1])    # 0
g.point([1, -1])    # 1
g.point([1, 1])   # 2
g.point([-1, 1])   # 3

    # circle
radius = 0.5
interface_elsize = 0.5
g.point([0, 0], el_size=interface_elsize)     # 4
g.point([radius, 0], el_size=interface_elsize)    # 5
g.point([0, radius], el_size=interface_elsize)    # 6
g.point([-radius, 0], el_size=interface_elsize)    # 7
g.point([0, -radius], el_size=interface_elsize)  # 8

# lines
    # square
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

# geometry plot
cfv.figure(fig_size=(7,7))
cfv.title('Geometry')
cfv.draw_geometry(g)

# =====
# Mesh creation from geometry object
# =====
mesh = cfm.GmshMesh(g)

mesh.el_type = 2                # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()       # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(coords, edof, mesh.dofs_per_node, mesh.el_type)  # coordinate-edges to vertices-faces

# mesh plot
cfv.figure(fig_size=(7,7))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

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


# plots.plot_normal_vectors(coords, bi)

B = np.hstack((b0,b1,b2,b3,bi))

elementmarkers = np.asarray(elementmarkers)

bm0 = faces[elementmarkers == mat0]
bm0 = bm0.flatten()
bm0 = np.setdiff1d(bm0,B)

bm1 = faces[elementmarkers == mat1]
bm1 = bm1.flatten()
bm1 = np.setdiff1d(bm1,B)

# plot of nodes
plt.figure(figsize=(7,7))
for label, nodes in zip(
    ('Down', 'Right', 'Up', 'Left', 'Interface', 'Material 0', 'Material 1'),
    (b0, b1, b2, b3, bi, bm0, bm1)
):
    plt.scatter(coords[nodes,0], coords[nodes,1], s=15, alpha=0.7, label=label)
plt.axis("equal")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Nodes')


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
delta = 0.01 * mesh.el_size_factor * interface_elsize
def fs(p):                                  # sourse
    out = 0
    return out
L = np.array([0,0,0,2,0,2])                 # coefitients vector

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'src'))
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

problem = gfdmi(coords, faces, L, fs)
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
fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.plot_trisurf(coords[:, 0], coords[:, 1], U, cmap='plasma', edgecolor='k', alpha=0.7)
plt.title('3D Solution')

plt.figure(figsize=(7,7))
plt.tricontourf(coords[:,0], coords[:,1], U, levels=20, cmap="plasma")
plt.colorbar()
plt.tricontour(coords[:,0], coords[:,1], U, levels=20, colors="k", linewidths=0.5)
plt.scatter(coords[bi,0], coords[bi,1], s=10, c='#000000', alpha=0.5)
plt.title('Contour Solution')



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

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.plot_trisurf(coords[:, 0], coords[:, 1], U2[-1], cmap='plasma', edgecolor='k', alpha=0.7)
ax.view_init(elev=35, azim=-127)
plt.title('Crank-Nicolson' + ', $t=' + str(T) + '$')

plt.figure(figsize=(7,7))
plt.tricontourf(coords[:,0], coords[:,1], U2[-1], levels=20, cmap="plasma")
plt.colorbar()
plt.tricontour(coords[:,0], coords[:,1], U2[-1], levels=20, colors="k", linewidths=0.5)
plt.scatter(coords[bi,0], coords[bi,1], s=10, c='#000000', alpha=0.5)
plt.title('Crank-Nicolson' + ', $t=' + str(T) + '$')


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
    cmap="plasma"
)
fig.colorbar(cont1)
ax1.set_title("3D Solution")
ax1.axis("equal")

cont2 = ax2.tricontourf(
    coords[:,0],
    coords[:,1],
    U2[-1],
    cmap="plasma",
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
        cmap="plasma"
    )
    ax1.set_title("3D Solution")
    ax1.axis("equal")

    cont2 = ax2.tricontourf(
        coords[:,0],
        coords[:,1],
        U2[frame],
        cmap="plasma",
        levels=20
    )
    ax2.set_title("Contour Solution")
    ax2.axis("equal")
    fig.suptitle(f"Crank-Nicolson, t = {frame*dt:.4f}")
    print(f"t = {frame*dt:.4f}", flush=True)

    return cont1, cont2

ani = FuncAnimation(fig, update, frames=range(0, U2.shape[0], 10), blit=False, interval=24)

# optional: save gif
# ani.save("figures/ex10.gif", writer='pillow', fps=24)
# plt.savefig("figures/ex10.png", dpi=300, bbox_inches="tight")
plt.show()