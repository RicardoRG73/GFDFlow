import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'src'))
#%%
# -- libraries --
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import numpy as np
import json
from scipy.integrate import solve_ivp



#%%
# -- reading mesh data --
mesh_file = 'Meshes/mesh6.json'
with open(mesh_file, 'r') as file:
    mesh_data = json.load(file)
    coords = np.array(mesh_data["coords"])
    faces = np.array(mesh_data["triangles"])
    boundary_nodes = np.array(mesh_data["boundary_nodes"])
    interior_nodes = np.array(mesh_data["interior_nodes"])
    left_nodes = np.array(mesh_data["left_nodes"])
    right_nodes = np.array(mesh_data["right_nodes"])
    bottom_nodes = np.array(mesh_data["bottom_nodes"])
    top_nodes = np.array(mesh_data["top_nodes"])
    corner_nodes = np.array(mesh_data["corner_nodes"])

#%%
# -- problem parameters --
# Henry: a=0.2637, b=0.1
# Pinder: a=0.2637, b=0.035
# Modified: a=0.1315, b=0.2

a = 0.2637
b = 0.035
k = lambda p: 1
source = lambda p: 0

#%%
# -- D matrices for flow function \Psi --
Psit = lambda p: 1
Psib = lambda p: 0
Psil = lambda p: 0
Psir = lambda p: 0

L2 = np.array([0,0,0,1,0,1])

problem = gfdmi(coords, faces, L2, source)

problem.material("0", k, interior_nodes)

problem.dirichlet_boundary("top", top_nodes, Psit)
problem.dirichlet_boundary("bottom", bottom_nodes, Psib)
problem.dirichlet_boundary("esquinas_top", [2,3], Psit)
problem.dirichlet_boundary("esquinas_bottom", [0,1], Psib)

problem.neumann_boundary("left", k, left_nodes, Psil)
problem.neumann_boundary("right", k, right_nodes, Psir)

D2psi, F2psi = problem.continuous_discretization()

Lx = np.array([0,1,0,0,0,0])
problem.L = Lx
Dxpsi, Fxpsi = problem.continuous_discretization()

Ly = np.array([0,0,1,0,0,0])
problem.L = Ly
Dypsi, Fypsi = problem.continuous_discretization()

#%%
# -- D matrices for concentration C --
Cl = lambda p: 0
Cr = lambda p: 1
Ct = lambda p: 0
Cb = lambda p: 0

problem = gfdmi(coords, faces, L2, source)

problem.material("0", k, interior_nodes)

problem.dirichlet_boundary("left", left_nodes, Cl)
problem.dirichlet_boundary("right", right_nodes, Cr)
problem.dirichlet_boundary("esquinas_left", [0,3], Cl)
problem.dirichlet_boundary("esquinas_right", [1,2], Cr)

problem.neumann_boundary("top", k, top_nodes, Ct)
problem.neumann_boundary("bottom", k, bottom_nodes, Cb)

D2c, F2c = problem.continuous_discretization()

problem.L = Lx
Dxc, Fxc = problem.continuous_discretization()

problem.L = Ly
Dyc, Fyc = problem.continuous_discretization()

#%%
# -- IVP assembly --
import scipy.sparse as sp

Dxcpsi = Dxc.copy()
Dxcpsi = sp.lil_matrix(Dxcpsi)
Fxcpsi = Fxc.copy()

Dxcpsi[boundary_nodes,:] = 0
Fxcpsi[boundary_nodes] = 0

Dypsic = Dypsi.copy()
Dypsic = sp.lil_matrix(Dypsic)
Fypsic = Fypsi.copy()

Dypsic[boundary_nodes,:] = 0
Fypsic[boundary_nodes] = 0

Dxpsic = Dxpsi.copy()
Dxpsic = sp.lil_matrix(Dxpsic)
Fxpsic = Fxpsi.copy()

Dxpsic[boundary_nodes,:] = 0
Fxpsic[boundary_nodes] = 0

# linear matrix A
N = coords.shape[0]
print("N = ", N)
A = sp.vstack((
    sp.hstack((
        D2psi, -1/a * Dxcpsi
    )),
    sp.hstack((
        np.zeros((N,N)), D2c
    ))
))

# linear vector F
F = np.hstack((
    - F2psi  +  1/a * Fxcpsi,
    - F2c
))

# nonlinear vector B
def B(U):
    term1 = (Dypsic@U[:N] - Fypsic) * (Dxc@U[N:] - Fxc)
    term2 = (Dxpsic@U[:N] - Fxpsic) * (Dyc@U[N:] - Fyc)
    vec2 = -1/b * (term1 - term2)
    vec1 = np.zeros(N)
    vec = np.hstack((vec1, vec2))
    return vec

# right hand side
fun = lambda t,U: A@U + F + B(U)

# initial conditions
C0 = np.zeros(N)
Psi0 = np.zeros(N)
for i in left_nodes:
    C0[i] = Cl(coords[i,:])
    Psi0[i] = Psil(coords[i,:])
for i in right_nodes:
    C0[i] = Cr(coords[i,:])
    Psi0[i] = Psir(coords[i,:])
for i in top_nodes:
    C0[i] = Ct(coords[i,:])
    Psi0[i] = Psit(coords[i,:])
for i in bottom_nodes:
    C0[i] = Cb(coords[i,:])
    Psi0[i] = Psib(coords[i,:])
i = 0
C0[i] = Cl(coords[i,:])
Psi0[i] = Psib(coords[i,:])
i = 1
C0[i] = Cr(coords[i,:])
Psi0[i] = Psib(coords[i,:])
i = 2
C0[i] = Cr(coords[i,:])
Psi0[i] = Psit(coords[i,:])
i = 3
C0[i] = Cl(coords[i,:])
Psi0[i] = Psit(coords[i,:])

U0 = np.hstack((Psi0, C0))

#%%
# IVP solution
t_final = 0.21
tspan = [0, t_final]
t_eval = [0, 0.02, 0.05, 0.1, 0.15, 0.21]
sol = solve_ivp(fun, tspan, U0, t_eval=t_eval, method="LSODA")

U = sol.y

#%%
# -- Save solution to file --
sol_data = {
    "coords": coords.tolist(),
    "triangles": faces.tolist(),
    "t_eval": t_eval,
    "U": U.tolist()
}
with open('results/ex6Henry.json', 'w') as file:
    json.dump(sol_data, file, indent=4)
print("\n ============\n Solution saved \n ============")
