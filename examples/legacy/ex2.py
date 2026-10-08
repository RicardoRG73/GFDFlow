"""Example 02: Laplace Equation with Sinusoidal Interface and Analytical Verification.

This benchmark solves a 2D Poisson equation with a curved sinusoidal interface
and prescribed jumps in potential :math:`u` and normal flux :math:`\\partial u / \\partial n`,
based on the formulation by Siraj-ul-Islam and Masood Ahmad.

Governing Equation
------------------
.. math::
    \\nabla^2 u = f(x, y) = 4

Analytical Solution
-------------------
.. math::
    u(x, y) = \\begin{cases}
    \\sin(\\pi x) \\sin(\\pi y), & (x, y) \\in \\Omega^+ \\\\
    \\sin(\\pi x)(\\sin(\\pi y) - e^{\\pi y}), & (x, y) \\in \\Omega^-
    \\end{cases}

Interface Definition
--------------------
.. math::
    x(y) = 0.5 + 0.1 \\sin(2\\pi y), \\quad y \\in [0, 1]
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import (
    plot_normal_vectors,
    plot_solution_2d,
    plot_solution_comparison_3d,
)

# Configure visualization style
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# =============================================================================
# 1. Load Mesh Data from File
# =============================================================================
with open("examples/legacy/meshes/mesh2.json", "r") as file:
    loaded_data = json.load(file)

coords = np.array(loaded_data["coords"])
triangles = np.array(loaded_data["triangles"])
omega_minus_nodes = np.array(loaded_data["omega_minus_nodes"])
omega_plus_nodes = np.array(loaded_data["omega_plus_nodes"])
dirichlet_nodes = np.array(loaded_data["dirichlet_nodes"])
interface_left_nodes = np.array(loaded_data["interface_left_nodes"])
interface_right_nodes = np.array(loaded_data["interface_right_nodes"])

support_stencils = {int(k): np.array(v) for k, v in loaded_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in loaded_data["M_pinv"].items()}

# =============================================================================
# 2. Problem Parameters and Boundary Definitions
# =============================================================================
L = np.array([0, 0, 0, 1, 0, 1])
permeability_left = lambda p: 1.0
permeability_right = lambda p: 1.0
source = lambda p: 4.0


def dirichlet_condition(p: np.ndarray) -> float:
    """Evaluate Dirichlet boundary condition from the exact solution.

    Parameters
    ----------
    p : np.ndarray
        Spatial coordinates [x, y].

    Returns
    -------
    float
        Analytical Dirichlet potential value.
    """
    if p[0] < 0.5:
        return float(np.sin(np.pi * p[0]) * np.sin(np.pi * p[1]))
    else:
        return float(np.sin(np.pi * p[0]) * (np.sin(np.pi * p[1]) - np.exp(np.pi * p[1])))


def compute_normal_vecs(b: np.ndarray) -> np.ndarray:
    """Compute unit normal vectors along the sinusoidal interface.

    Parameters
    ----------
    b : np.ndarray
        Array of node indices on the interface curve.

    Returns
    -------
    np.ndarray
        Normalized 2D normal vectors with shape (len(b), 2).
    """
    normals = np.empty((b.shape[0], 2))
    normals[:, 0] = 1.0
    normals[:, 1] = -0.628 * np.cos(6.28 * coords[b, 1])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norms


normal_vecs_left_interface = compute_normal_vecs(interface_left_nodes)
normal_vecs_right_interface = compute_normal_vecs(interface_right_nodes)

normal_vecs = np.zeros((coords.shape[0], 2))
normal_vecs[interface_left_nodes, :] = normal_vecs_left_interface
normal_vecs[interface_right_nodes, :] = normal_vecs_right_interface


def beta(p: np.ndarray) -> float:
    """Evaluate analytical flux jump along the sinusoidal interface.

    Parameters
    ----------
    p : np.ndarray
        Point coordinates [x, y] on the interface.

    Returns
    -------
    float
        Flux jump condition value.
    """
    i = np.argmin(
        np.sqrt(
            (coords[interface_left_nodes, 0] - p[0]) ** 2
            + (coords[interface_left_nodes, 1] - p[1]) ** 2
        )
    )
    val = np.pi * (
        np.cos(np.pi * p[0]) * np.exp(np.pi * p[1]) * normal_vecs[i, 0]
        + np.sin(np.pi * p[0]) * np.exp(np.pi * p[1]) * normal_vecs[i, 1]
    )
    return float(val)


# Solution jump condition: u|_{left} - u|_{right} = alpha
alpha = lambda p: -np.sin(np.pi * p[0]) * np.exp(np.pi * p[1])

# =============================================================================
# 3. GFDM Problem Initialization and Interface Setup
# =============================================================================
problem = gfdmi(
    coords,
    triangles,
    normal_vecs,
    L,
    source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

problem.material("material_left", permeability_left, omega_plus_nodes)
problem.material("material_right", permeability_right, omega_minus_nodes)
problem.dirichlet_boundary("dirichlet", dirichlet_nodes, dirichlet_condition)

problem.interface(
    "interface0",
    permeability_left,
    permeability_right,
    interface_left_nodes,
    interface_right_nodes,
    beta,
    alpha,
    omega_plus_nodes,
    omega_minus_nodes,
)

# =============================================================================
# 4. System Assembly and Solution
# =============================================================================
K, F = problem.discontinuous_discretization()
U = sp.linalg.spsolve(K, F)

# =============================================================================
# 5. Exact Solution and Error Verification
# =============================================================================
def exact(p: np.ndarray) -> float:
    """Evaluate closed-form exact solution at point p.

    Parameters
    ----------
    p : np.ndarray
        Spatial coordinate [x, y].

    Returns
    -------
    float
        Exact potential value.
    """
    if p[0] <= 0.5 + 0.1 * np.sin(6.28 * p[1]):
        return float(np.sin(np.pi * p[0]) * np.sin(np.pi * p[1]))
    else:
        return float(np.sin(np.pi * p[0]) * (np.sin(np.pi * p[1]) - np.exp(np.pi * p[1])))


Uex = np.array([exact(pt) for pt in coords])

rmse = np.sqrt(np.mean((Uex - U) ** 2))
norm2 = np.linalg.norm(Uex - U)
norm_inf = np.max(np.abs(Uex - U))

print("N = %d" % coords.shape[0])
print("\n===============")
print(f"RMSE = {rmse:1.4e}")
print("===============")
print(f"Norm 2 = {norm2:1.4e}")
print("===============")
print(f"Norm infinity = {norm_inf:1.4e}")
print("===============")

# =============================================================================
# 6. Visualization
# =============================================================================
plot_solution_2d(
    coords,
    U,
    cmap="inferno",
    levels=11,
    clabel=True,
    savepath="examples/legacy/figures/ex2/contourf.jpg",
)

plot_normal_vectors(
    coords,
    normal_vecs,
    [interface_left_nodes, interface_right_nodes],
    quiver_alpha=0.3,
    savepath="examples/legacy/figures/ex2/normal_vectors.png",
)

plot_solution_comparison_3d(
    coords,
    U,
    Uex,
    num_label="Numerical",
    exact_label="Exact",
    view_init=(20, -50),
    savepath="examples/legacy/figures/ex2/3dplot.jpg",
)

plt.show()