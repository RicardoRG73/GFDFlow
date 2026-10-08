"""Example 06: Coupled Variable-Density Seawater Intrusion (Henry Problem).

This benchmark simulates coupled variable-density groundwater flow and
solute transport in a confined coastal aquifer (the classic Henry problem).
It solves for the stream function :math:`\\psi` and relative salt concentration
:math:`C` using a non-linear coupled IVP formulation.

Governing Equations
-------------------
.. math::
    \\nabla^2 \\psi = -\\frac{1}{a} \\frac{\\partial C}{\\partial x}

.. math::
    \\frac{\\partial C}{\\partial t} = b \\nabla^2 C - \\left(
        \\frac{\\partial \\psi}{\\partial y} \\frac{\\partial C}{\\partial x}
        - \\frac{\\partial \\psi}{\\partial x} \\frac{\\partial C}{\\partial y}
    \\right)

Parameters
----------
* Coupling parameter: :math:`a = 0.2637`
* Dispersion parameter: :math:`b = 0.035`
* Permeability: :math:`k = 1.0`
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d

start_time = time.perf_counter()

# Configure visualization style
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh6.json", "r") as file:
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
normal_vecs = np.array(mesh_data["normal_vecs"])
support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Dimensionless Coefficients
# =============================================================================
a = 0.2637
b = 0.035
k = lambda p: 1.0
source = lambda p: 0.0

# =============================================================================
# 3. Spatial Derivative Operators for Stream Function Psi
# =============================================================================
Psit = lambda p: 1.0
Psib = lambda p: 0.0
Psil = lambda p: 0.0
Psir = lambda p: 0.0

L2 = np.array([0, 0, 0, 1, 0, 1])
problem = gfdmi(coords, faces, normal_vecs, L2, source, M_pinv=M_pinv, support_stencils=support_stencils)
problem.material("0", k, interior_nodes)
problem.dirichlet_boundary("top", top_nodes, Psit)
problem.dirichlet_boundary("bottom", bottom_nodes, Psib)
problem.dirichlet_boundary("esquinas_top", [2, 3], Psit)
problem.dirichlet_boundary("esquinas_bottom", [0, 1], Psib)
problem.neumann_boundary("left", k, left_nodes, Psil)
problem.neumann_boundary("right", k, right_nodes, Psir)

D2psi, F2psi = problem.continuous_discretization()

Lx = np.array([0, 1, 0, 0, 0, 0])
problem.L = Lx
Dxpsi, Fxpsi = problem.continuous_discretization()

Ly = np.array([0, 0, 1, 0, 0, 0])
problem.L = Ly
Dypsi, Fypsi = problem.continuous_discretization()

# =============================================================================
# 4. Spatial Derivative Operators for Concentration C
# =============================================================================
Cl = lambda p: 0.0
Cr = lambda p: 1.0
Ct = lambda p: 0.0
Cb = lambda p: 0.0

problem = gfdmi(coords, faces, normal_vecs, L2, source, M_pinv=M_pinv, support_stencils=support_stencils)
problem.material("0", k, interior_nodes)
problem.dirichlet_boundary("left", left_nodes, Cl)
problem.dirichlet_boundary("right", right_nodes, Cr)
problem.dirichlet_boundary("esquinas_left", [0, 3], Cl)
problem.dirichlet_boundary("esquinas_right", [1, 2], Cr)
problem.neumann_boundary("top", k, top_nodes, Ct)
problem.neumann_boundary("bottom", k, bottom_nodes, Cb)

D2c, F2c = problem.continuous_discretization()

problem.L = Lx
Dxc, Fxc = problem.continuous_discretization()

problem.L = Ly
Dyc, Fyc = problem.continuous_discretization()

# =============================================================================
# 5. Coupled System Matrix Assembly
# =============================================================================
Dxcpsi = sp.lil_matrix(Dxc.copy())
Fxcpsi = Fxc.copy()
Dxcpsi[boundary_nodes, :] = 0
Fxcpsi[boundary_nodes] = 0

Dypsic = sp.lil_matrix(Dypsi.copy())
Fypsic = Fypsi.copy()
Dypsic[boundary_nodes, :] = 0
Fypsic[boundary_nodes] = 0

Dxpsic = sp.lil_matrix(Dxpsi.copy())
Fxpsic = Fxpsi.copy()
Dxpsic[boundary_nodes, :] = 0
Fxpsic[boundary_nodes] = 0

N = coords.shape[0]
print("N = ", N)

A = sp.vstack([
    sp.hstack([D2psi, -1.0 / a * Dxcpsi]),
    sp.hstack([np.zeros((N, N)), D2c]),
])

F = np.hstack([
    -F2psi + 1.0 / a * Fxcpsi,
    -F2c,
])


def B(U: np.ndarray) -> np.ndarray:
    """Evaluate non-linear convective coupling between stream function and concentration.

    Parameters
    ----------
    U : np.ndarray
        State vector of length 2*N containing [Psi, C].

    Returns
    -------
    np.ndarray
        Non-linear advective vector of length 2*N.
    """
    term1 = (Dypsic @ U[:N] - Fypsic) * (Dxc @ U[N:] - Fxc)
    term2 = (Dxpsic @ U[:N] - Fxpsic) * (Dyc @ U[N:] - Fyc)
    vec2 = -1.0 / b * (term1 - term2)
    vec1 = np.zeros(N)
    return np.hstack([vec1, vec2])


fun = lambda t, U: A @ U + F + B(U)

# Initialize state vector
C0 = np.zeros(N)
Psi0 = np.zeros(N)
for i in left_nodes:
    C0[i] = Cl(coords[i, :])
    Psi0[i] = Psil(coords[i, :])
for i in right_nodes:
    C0[i] = Cr(coords[i, :])
    Psi0[i] = Psir(coords[i, :])
for i in top_nodes:
    C0[i] = Ct(coords[i, :])
    Psi0[i] = Psit(coords[i, :])
for i in bottom_nodes:
    C0[i] = Cb(coords[i, :])
    Psi0[i] = Psib(coords[i, :])

for i, cl_val, ps_val in [(0, Cl, Psib), (1, Cr, Psib), (2, Cr, Psit), (3, Cl, Psit)]:
    C0[i] = cl_val(coords[i, :])
    Psi0[i] = ps_val(coords[i, :])

U0 = np.hstack([Psi0, C0])

# =============================================================================
# 6. Initial Value Problem Integration (LSODA)
# =============================================================================
t_final = 0.21
tspan = [0, t_final]
t_eval = [0, 0.02, 0.05, 0.1, 0.15, 0.21]
method = "LSODA"
print(f"\n\n Solving IVP with {method} method\n\n")

sol = solve_ivp(fun, tspan, U0, t_eval=t_eval, method=method)
U = sol.y

# Save solution to file
sol_data = {
    "coords": coords.tolist(),
    "triangles": faces.tolist(),
    "t_eval": t_eval,
    "U": U.tolist(),
}
with open("examples/legacy/results/ex6Henry.json", "w") as file:
    json.dump(sol_data, file, indent=4)
print("\n ============\n Solution saved \n ============")

# =============================================================================
# 7. Visualization and Timing
# =============================================================================
plot_solution_2d(
    coords,
    U[N:, -1],
    triangles=faces,
    levels=20,
    cmap="inferno",
    colorbar_label="Concentration C",
    title=f"Henry Problem - Final Concentration C (t={t_final})",
    savepath="examples/legacy/figures/ex6/contourf.png",
)
plt.show()

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")