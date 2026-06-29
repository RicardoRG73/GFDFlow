import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
# ex7 Elder problem

#%%
# -- import libraries --
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import json

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

from scipy.integrate import solve_ivp

#%%
# -- reading mesh data --
mesh_file = 'examples/legacy/Meshes/mesh7.json'
with open(mesh_file, 'r') as file:
    mesh_data = json.load(file)
    coords = np.array(mesh_data["coords"])
    triangles = np.array(mesh_data["triangles"])
    boundary_nodes = np.array(mesh_data["boundary_nodes"])
    interior_nodes = np.array(mesh_data["interior_nodes"])
    left_nodes = np.array(mesh_data["left_nodes"])
    right_nodes = np.array(mesh_data["right_nodes"])
    top_nodes = np.array(mesh_data["top_nodes"])
    bottom_nodes = np.array(mesh_data["bottom_nodes"])

#%%
# -- discretization --
# Psi discretization
from GFDFlow.GFDM import GFDMI_2D_problem as gfdm

L = np.array([0,0,0,1,0,1])
source = lambda p: 0
k = lambda p: 1

problem = gfdm(coords, triangles, L, source)

problem.material("interior", k, interior_nodes)

problem.dirichlet_boundary("bottom", bottom_nodes, lambda p: 0)
problem.dirichlet_boundary("right", right_nodes, lambda p: 0)
problem.dirichlet_boundary("top", top_nodes, lambda p: 0)
problem.dirichlet_boundary("left", left_nodes, lambda p: 0)

# D2 discretization
D2psi, F2psi = problem.continuous_discretization()

# Dx discretization
problem.L = np.array([0,1,0,0,0,0])
Dxpsi, Fxpsi = problem.continuous_discretization()

# Dy discretization
problem.L = np.array([0,0,1,0,0,0])
Dypsi, Fypsi = problem.continuous_discretization()

# C discretization
L = np.array([0,0,0,1,0,1])
source = lambda p: 0
k = lambda p: 1

C_top = 2

problem = gfdm(coords, triangles, L, source)
problem.material("interior", k, interior_nodes)
problem.dirichlet_boundary("bottom", bottom_nodes, lambda p: 0)
problem.dirichlet_boundary("top", top_nodes, lambda p: C_top)
problem.neumann_boundary("right", k, right_nodes, lambda p: 0)
problem.neumann_boundary("left", k, left_nodes, lambda p: 0)

# D2 discretization
D2c, F2c = problem.continuous_discretization()

# Dx discretization
problem.L = np.array([0,1,0,0,0,0])
Dxc, Fxc = problem.continuous_discretization()

# Dy discretization
problem.L = np.array([0,0,1,0,0,0])
Dyc, Fyc = problem.continuous_discretization()

# -- assemble problem --
Ra = 400

N = coords.shape[0]

zeros_mat = sp.csr_matrix((N, N))
zeros_vec = np.zeros(N)

Dxcpsi = Dxc.copy()
Dxcpsi = sp.lil_matrix(Dxcpsi)
Dxcpsi[boundary_nodes,:] = 0
Fxcpsi = Fxc.copy()
Fxcpsi[boundary_nodes] = 0

Dypsic = Dypsi.copy()
Dypsic = sp.lil_matrix(Dypsic)
Dypsic[boundary_nodes,:] = 0
Fypsic = Fypsi.copy()
Fypsic[boundary_nodes] = 0

Dxpsic = Dxpsi.copy()
Dxpsic = sp.lil_matrix(Dxpsic)
Dxpsic[boundary_nodes,:] = 0
Fxpsic = Fxpsi.copy()
Fxpsic[boundary_nodes] = 0

# Linear
Linear_mat = sp.vstack((
    sp.hstack((D2psi, -Ra*Dxcpsi)),
    sp.hstack((zeros_mat, D2c))
))

Linear_vec = - np.hstack((
    F2psi - Ra*Fxcpsi,
    F2c
))

# Non-Linear
def nonLinear(U):
    term1 = (Dypsic @ U[:N] + Fypsic) * (Dxc @ U[N:] + Fxc)
    term2 = (Dxpsic @ U[:N] + Fxpsic) * (Dyc @ U[N:] + Fyc)
    vec = np.hstack((
        zeros_vec,
        - term1 + term2
    ))
    return vec

def rhs(t,U):
    vec = Linear_mat @ U + Linear_vec
    vec += nonLinear(U)
    return vec

#%%
# -- solve IVP --
tfinal = 1.239
tspan = [0, tfinal]
t_eval = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.5, tfinal])

# initial condition
P0 = zeros_vec.copy()
C0 = zeros_vec.copy()

C0[top_nodes] = C_top

U0 = np.hstack((P0,C0))

# solution
method = "LSODA"
print(f"\n\n Solving IVP with {method} method... \n\n")
sol = solve_ivp(rhs, tspan, U0, t_eval=t_eval, method=method)
U = sol.y
print("Done!")

#%%
# -- Save solution to file --
sol_data = {
    "coords": coords.tolist(),
    "triangles": triangles.tolist(),
    "t_eval": t_eval.tolist(),
    "U": U.tolist()
}
with open('examples/leagcy/results/ex7Elder.json', 'w') as file:
    json.dump(sol_data, file, indent=4)
print("\n ============\n Solution saved \n ============")

