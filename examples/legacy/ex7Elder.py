"""Example 07: Buoyancy-Driven Convection and Salt Fingering (Elder Problem).

This benchmark simulates transient buoyancy-driven natural convection in a
porous medium (the classic Elder problem). Brine infiltration from a central
top boundary segment triggers Rayleigh-like fingering instabilities that descend
through the domain.

Governing Equations
-------------------
.. math::
    \\nabla^2 \\psi = -\\operatorname{Ra} \\frac{\\partial C}{\\partial x}

.. math::
    \\frac{\\partial C}{\\partial t} + \\mathbf{v} \\cdot \\nabla C = \\nabla^2 C

Parameters
----------
* Rayleigh number: :math:`\\operatorname{Ra} = 400`
* Top boundary source concentration: :math:`C_{\\text{top}} = 2.0`
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdm
from GFDFlow.visualization import plot_solution_2d

start_time = time.perf_counter()

# Configure visualization style
plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
mesh_file = "examples/legacy/meshes/mesh7.json"
with open(mesh_file, "r") as file:
    mesh_data = json.load(file)

coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])
boundary_nodes = np.array(mesh_data["boundary_nodes"])
interior_nodes = np.array(mesh_data["interior_nodes"])
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
top_right_nodes = np.array(mesh_data["top_right_nodes"])
top_middle_nodes = np.array(mesh_data["top_middle_nodes"])
top_left_nodes = np.array(mesh_data["top_left_nodes"])
bottom_nodes = np.array(mesh_data["bottom_nodes"])
normal_vecs = np.array(mesh_data["normal_vecs"])
support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# =============================================================================
# 2. Spatial Derivative Operators for Stream Function Psi
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
source = lambda p: 0.0
k = lambda p: 1.0

problem = gfdm(
    coords, triangles, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils
)
problem.material("interior", k, interior_nodes)
problem.dirichlet_boundary("bottom", bottom_nodes, lambda p: 0.0)
problem.dirichlet_boundary("right", right_nodes, lambda p: 0.0)
problem.dirichlet_boundary("topright", top_right_nodes, lambda p: 0.0)
problem.dirichlet_boundary("topmiddle", top_middle_nodes, lambda p: 0.0)
problem.dirichlet_boundary("topleft", top_left_nodes, lambda p: 0.0)
problem.dirichlet_boundary("left", left_nodes, lambda p: 0.0)

D2psi, F2psi = problem.continuous_discretization()

problem.L = np.array([0, 1, 0, 0, 0, 0])
Dxpsi, Fxpsi = problem.continuous_discretization()

problem.L = np.array([0, 0, 1, 0, 0, 0])
Dypsi, Fypsi = problem.continuous_discretization()

# =============================================================================
# 3. Spatial Derivative Operators for Concentration C
# =============================================================================
C_top = 2.0

problem = gfdm(coords, triangles, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)
problem.material("interior", k, interior_nodes)
problem.dirichlet_boundary("bottom", bottom_nodes, lambda p: 0.0)
problem.dirichlet_boundary("topmiddle", top_middle_nodes, lambda p: C_top)
problem.neumann_boundary("right", k, right_nodes, lambda p: 0.0)
problem.neumann_boundary("left", k, left_nodes, lambda p: 0.0)
problem.neumann_boundary("topright", k, top_right_nodes, lambda p: 0.0)
problem.neumann_boundary("topleft", k, top_left_nodes, lambda p: 0.0)

D2c, F2c = problem.continuous_discretization()

problem.L = np.array([0, 1, 0, 0, 0, 0])
Dxc, Fxc = problem.continuous_discretization()

problem.L = np.array([0, 0, 1, 0, 0, 0])
Dyc, Fyc = problem.continuous_discretization()

# =============================================================================
# 4. Assemble Coupled Non-Linear System
# =============================================================================
Ra = 400
N = coords.shape[0]

zeros_mat = sp.csr_matrix((N, N))
zeros_vec = np.zeros(N)

Dxcpsi = sp.lil_matrix(Dxc.copy())
Dxcpsi[boundary_nodes, :] = 0
Fxcpsi = Fxc.copy()
Fxcpsi[boundary_nodes] = 0

Dypsic = sp.lil_matrix(Dypsi.copy())
Dypsic[boundary_nodes, :] = 0
Fypsic = Fypsi.copy()
Fypsic[boundary_nodes] = 0

Dxpsic = sp.lil_matrix(Dxpsi.copy())
Dxpsic[boundary_nodes, :] = 0
Fxpsic = Fxpsi.copy()
Fxpsic[boundary_nodes] = 0

Linear_mat = sp.vstack([
    sp.hstack([D2psi, -Ra * Dxcpsi]),
    sp.hstack([zeros_mat, D2c]),
])

Linear_vec = -np.hstack([
    F2psi - Ra * Fxcpsi,
    F2c,
])


def nonLinear(U: np.ndarray) -> np.ndarray:
    """Evaluate non-linear convective transport coupling.

    Parameters
    ----------
    U : np.ndarray
        State vector of length 2*N containing [Psi, C].

    Returns
    -------
    np.ndarray
        Non-linear advective acceleration vector of length 2*N.
    """
    term1 = (Dypsic @ U[:N] + Fypsic) * (Dxc @ U[N:] + Fxc)
    term2 = (Dxpsic @ U[:N] + Fxpsic) * (Dyc @ U[N:] + Fyc)
    return np.hstack([zeros_vec, -term1 + term2])


def rhs(t: float, U: np.ndarray) -> np.ndarray:
    """Evaluate full right-hand side of the coupled ODE system.

    Parameters
    ----------
    t : float
        Current simulation time.
    U : np.ndarray
        State vector of length 2*N containing [Psi, C].

    Returns
    -------
    np.ndarray
        Time derivative dU/dt.
    """
    vec = Linear_mat @ U + Linear_vec
    vec += nonLinear(U)
    return vec


# =============================================================================
# 5. Solve Initial Value Problem (LSODA)
# =============================================================================
tfinal = 1.239
tspan = [0, tfinal]
t_eval = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.5, tfinal])

P0 = zeros_vec.copy()
C0 = zeros_vec.copy()
C0[top_middle_nodes] = C_top
U0 = np.hstack([P0, C0])

method = "LSODA"
print(f"\n\n Solving IVP with {method} method\n\n")
sol = solve_ivp(rhs, tspan, U0, t_eval=t_eval, method=method)
U = sol.y
print("Done!")

# Save solution to file
sol_data = {
    "coords": coords.tolist(),
    "triangles": triangles.tolist(),
    "t_eval": t_eval.tolist(),
    "U": U.tolist(),
}
with open("examples/legacy/results/ex7Elder.json", "w") as file:
    json.dump(sol_data, file, indent=4)
print("\n ============\n Solution saved \n ============")

# =============================================================================
# 6. Visualization and Timing
# =============================================================================
plot_solution_2d(
    coords,
    U[N:, -1],
    triangles=triangles,
    levels=20,
    cmap="inferno",
    colorbar_label="Concentration C",
    title=f"Elder Problem - Final Concentration C (t={tfinal})",
    savepath="examples/legacy/figures/ex7/contourf.png",
)
plt.show()

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")