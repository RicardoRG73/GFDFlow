"""Transient Saturated Groundwater Flow in a Five-Layer Dam.

This module solves the 2D transient (non-stationary) saturated potential
seepage problem for the five-layered embankment dam with central clay core and
foundation described in Section 4.2.3 (Table 2) of Chávez-Negrete et al.
(2023/2024), using the Generalized Finite Difference Method (GFDM) via
``GFDMI_2D_problem``.

Governing Equation
------------------
Within each homogeneous material subdomain :math:`\\Omega_i`:

.. math::
    \\frac{\\partial h}{\\partial t} = k_i \\nabla^2 h + f_i \\quad \\text{in } \\Omega_i

where :math:`h(x, y, t) = \\psi + y` is the total hydraulic head (m),
:math:`k_i` is the hydraulic conductivity (m/s), and
:math:`f_i = 0` is a source/sink term (m/s).

Material Properties
-------------------
- **Rockfill shells** (upstream and downstream): :math:`k_{\\text{rock}} = 10^{-4}\\text{ m/s}`
- **Clay core & cutoff wall**: :math:`k_{\\text{clay}} = 10^{-6}\\text{ m/s}`
- **Soil foundation**: :math:`k_{\\text{foundation}} = 10^{-4}\\text{ m/s}`

Boundary & Interface Conditions
-------------------------------
- **Upstream submerged face & left foundation wall**: Dirichlet, :math:`h = 8.0\\text{ m}`
- **Downstream underdrain & ground surface**: Dirichlet, :math:`h = 0.0\\text{ m}`
- **Bedrock base & dry dam slopes/crest**: Homogeneous Neumann, :math:`\\partial h / \\partial n = 0`
- **Internal interfaces** (``interface_a`` to ``interface_f``): Continuous head and
  normal Darcy flux balance (:math:`k_0 \\partial h / \\partial n + k_1 \\partial h / \\partial n = 0`)
- **Triple junctions** (``tj1`` at ``[20, 0]`` and ``tj2`` at ``[30, 0]``):
  Three-wedge angular flux equilibrium across converging material interfaces.

Inputs
------
- ``examples/paper/mesh_data.json`` : Precomputed mesh coordinates, connectivity,
  boundary/interface node groups, outward normals, support stencils, and ``M_pinv``.

Outputs
-------
- ``examples/paper/figures/sat_tr_snapshots.png`` : Six-panel transient evolution
  of total head contours, equipotentials, and phreatic surface (:math:`h = y`).
- ``examples/paper/figures/sat_tr_phreatic_isochrones.png`` : Phreatic surface
  isochrones across selected days compared against the steady-state baseline.
- ``examples/paper/figures/sat_tr_convergence.png`` : Monitoring piezometer
  locations, head time histories, and :math:`L_\\infty`/RMSE relaxation toward
  steady state.
- ``examples/paper/figures/sat_tr.gif`` : Animated GIF of transient head and
  phreatic surface evolution from :math:`t = 0` to :math:`1000` days.

References
----------
- Chávez-Negrete, C., Domínguez-Mota, F. J., & Román-Gutiérrez, R. (2023/2024).
  "Interface formulation for generalized finite difference method for solving
  groundwater flow." *Computers and Geotechnics*, Vol. 166, 105990.
  DOI: 10.1016/j.compgeo.2023.105990 (indexed under 2023 in some databases
  and published in Vol. 166 in 2024).
"""
# ==============================================================================
# 1. Imports and Plot Configuration
# ==============================================================================
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

from GFDFlow.GFDM import GFDMI_2D_problem
from GFDFlow.visualization import plot_phreatic_surface

# Configure Matplotlib publication style
plt.style.use(["seaborn-v0_8-paper"])
plt.rcParams["figure.autolayout"] = True

# ==============================================================================
# 2. Load Precomputed Mesh Data and GFDM Stencils
# ==============================================================================
mesh_path = "examples/paper/mesh_data.json"
with open(mesh_path, "r") as f:
    mesh_data = json.load(f)

coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])
normal_vecs = np.array(mesh_data["normal_vecs"])

# Outer boundary node groups
left_nodes = np.array(mesh_data["left_nodes"])
right_nodes = np.array(mesh_data["right_nodes"])
bottom_nodes = np.array(mesh_data["bottom_nodes"])
top_nodes = np.array(mesh_data["top_nodes"])

# Internal two-material interface node groups
ia_nodes = np.array(mesh_data["interface_a_nodes"])
ib_nodes = np.array(mesh_data["interface_b_nodes"])
ic_nodes = np.array(mesh_data["interface_c_nodes"])
id_nodes = np.array(mesh_data["interface_d_nodes"])
ie_nodes = np.array(mesh_data["interface_e_nodes"])
if_nodes = np.array(mesh_data["interface_f_nodes"])

# Strictly interior material subdomain node groups
rock_nodes = np.array(mesh_data["rock_nodes"])
clay_nodes = np.array(mesh_data["clay_nodes"])
foundation_nodes = np.array(mesh_data["foundation_nodes"])

# Triple-junction singularity node indices at (20, 0) and (30, 0)
tj1 = int(mesh_data["tj1"][0])
tj2 = int(mesh_data["tj2"][0])

# Convert JSON string keys back to integer node indices
support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# ==============================================================================
# 3. Problem Parameters and GFDM System Assembly
# ==============================================================================
# Saturated hydraulic conductivities from Table 2 of Chavez et al. (2023)
k_rock = lambda p: 1e-4         # Rockfill shells: 0.0001 m/s
k_clay = lambda p: 1e-6         # Clay core & cutoff: 1e-06 m/s
k_foundation = lambda p: 1e-4   # Soil foundation: 0.0001 m/s

# Zero volumetric source term
h_source = lambda p: 0.0

# Boundary and interface condition callables
h_left = lambda p: 8.0          # Upstream reservoir level (m)
h_right = lambda p: 0.0         # Downstream drain head (m)
h_neumann_zero = lambda p: 0.0  # Impermeable / zero-flux boundary
h_interface_source = lambda p: 0.0

# Second-order PDE operator coefficients [A, B, C, D, E, F] -> u_xx + u_yy = 0
L = np.array([0, 0, 0, 1, 0, 1])

# Initialize GFDM problem with precomputed stencils and pseudo-inverses
problem = GFDMI_2D_problem(
    coords,
    triangles,
    normal_vecs,
    L,
    h_source,
    M_pinv=M_pinv,
    support_stencils=support_stencils,
)

# Assign material conductivities to interior subdomain nodes
problem.material("rock", k_rock, rock_nodes)
problem.material("clay", k_clay, clay_nodes)
problem.material("foundation", k_foundation, foundation_nodes)

# Assign outer Dirichlet and homogeneous Neumann boundary conditions
problem.dirichlet_boundary("left", left_nodes, h_left)
problem.dirichlet_boundary("right", right_nodes, h_right)
problem.neumann_boundary("bottom", k_foundation, bottom_nodes, h_neumann_zero)
problem.neumann_boundary("top", k_rock, top_nodes, h_neumann_zero)

# Assign internal material interfaces with continuous normal flux balance
# Note: On interface_e and interface_f, outward normal_vecs point from
# foundation/rock into clay, so m0 is foundation/rock and m1 is clay.
problem.interface("interface_a", k_rock, k_foundation, ia_nodes, None, h_interface_source, None, rock_nodes, foundation_nodes)
problem.interface("interface_b", k_rock, k_foundation, ib_nodes, None, h_interface_source, None, rock_nodes, foundation_nodes)
problem.interface("interface_c", k_clay, k_rock, ic_nodes, None, h_interface_source, None, clay_nodes, rock_nodes)
problem.interface("interface_d", k_clay, k_foundation, id_nodes, None, h_interface_source, None, clay_nodes, foundation_nodes)
problem.interface("interface_e", k_foundation, k_clay, ie_nodes, None, h_interface_source, None, foundation_nodes, clay_nodes)
problem.interface("interface_f", k_rock, k_clay, if_nodes, None, h_interface_source, None, rock_nodes, clay_nodes)

# Assign triple-junction multi-wedge angular flux intersections
# Node 1 at (20, 0): interfaces A, D, C meeting rockfill, clay, and foundation
problem.intersection("inters_1", tj1, "interface_a", "interface_d", "foundation", h_interface_source)
problem.intersection("inters_2", tj1, "interface_d", "interface_c", "clay", h_interface_source)
problem.intersection("inters_3", tj1, "interface_c", "interface_a", "rock", h_interface_source)

# Node 2 at (30, 0): interfaces B, F, E meeting rockfill, clay, and foundation
problem.intersection("inters_4", tj2, "interface_b", "interface_f", "rock", h_interface_source)
problem.intersection("inters_5", tj2, "interface_e", "interface_f", "clay", h_interface_source)
problem.intersection("inters_6", tj2, "interface_e", "interface_b", "foundation", h_interface_source)

# Assemble global sparse system K * U = F and compute steady-state baseline via LU
K, F = problem.continuous_discretization()
U = spsolve(K, F)

# ==============================================================================
# 4. Transient Time Integration (Method of Lines + BDF)
# ==============================================================================
# Initial condition: unpressurized interior (0 m), reservoir head on left boundary
U0 = np.zeros(coords.shape[0], dtype=np.float64)
U0[left_nodes] = [h_left(coords[i]) for i in left_nodes]

# Zero out Dirichlet rows in K_tr and F_tr so du/dt = 0 keeps boundary heads fixed
dirichlet_nodes = np.union1d(left_nodes, right_nodes)

K_tr = K.tolil()
K_tr[dirichlet_nodes, :] = 0.0
K_tr = K_tr.tocsc()  # CSC format accelerates sparse LU inside BDF

F_tr = F.copy()
F_tr[dirichlet_nodes] = 0.0

# Semi-discrete ODE system: du/dt = K_tr @ u - F_tr (with normalized S_s = 1.0)
ode_rhs = lambda t, u: K_tr @ u - F_tr

# Integrate over [0, 1000 days] using stiff implicit BDF with exact sparse Jacobian
t_span = (0.0, 1000.0 * 86400.0)
U_tr = solve_ivp(
    ode_rhs,
    t_span,
    U0,
    method="BDF",
    jac=lambda t, u: K_tr,
    dense_output=True,  # Enables continuous evaluation U_tr.sol(t) at any snapshot time
    rtol=1e-6,
    atol=1e-8,
)
print(f"Transient solve_ivp finished in {U_tr.t.size} steps; L_inf diff to steady = {np.max(np.abs(U_tr.y[:, -1] - U)):.3e} m")

# ==============================================================================
# 5. Visualization and Post-Processing
# ==============================================================================
# A. Different time snapshots
# ---------------------------
snapshot_days = [0.01, 0.1, 1.0, 5.0, 100.0, 1000.0]
levels = np.linspace(0.0, 8.0, 21)

fig_snap, axes_snap = plt.subplots(3, 2, figsize=(10, 5), sharex=True, sharey=True, layout="constrained")
for ax, day in zip(axes_snap.ravel(), snapshot_days):
    u_snap = U_tr.sol(day * 86400.0)
    tcf = ax.tricontourf(coords[:, 0], coords[:, 1], triangles, u_snap, levels=levels, cmap="inferno")
    ax.tricontour(coords[:, 0], coords[:, 1], triangles, u_snap, levels=levels, colors="k", linewidths=0.4, alpha=0.55)
    plot_phreatic_surface(
        ax,
        coords,
        u_snap,
        triangles=triangles,
        color="#E41A1C",
        linestyle="--",
        linewidths=1.4,
        label="Phreatic surface ($h = y$)",
    )
    ax.set_title(f"$t = {day:g}$ day{'s' if day != 1.0 else ''}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")

for ax in axes_snap[-1, :]:
    ax.set_xlabel("Distance (m)")
for ax in axes_snap[:, 0]:
    ax.set_ylabel("Elevation (m)")

cbar_snap = fig_snap.colorbar(tcf, ax=axes_snap, shrink=0.85, pad=0.02)
cbar_snap.set_label("Total head $h$ (m)")
fig_snap.suptitle("Transient Saturated Flow Evolution in Five-Layer Dam")
fig_snap.savefig("examples/paper/figures/sat_tr_snapshots.png", dpi=300, bbox_inches="tight")

# B. Isochrones of phreatic surface vs steady phreatic surface
# ------------------------------------------------------------
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

fig_iso, ax_iso = plt.subplots(figsize=(10, 3.8), layout="constrained")
ax_iso.triplot(coords[:, 0], coords[:, 1], triangles, color="0.75", linewidth=0.5, zorder=1)

# Plot internal material interfaces as dashed lines
interface_segments = [
    ([3.0, 20.0], [0.0, 0.0]),    # interface_a
    ([30.0, 40.0], [0.0, 0.0]),   # interface_b
    ([23.0, 20.0], [10.0, 0.0]),  # interface_c
    ([20.0, 24.0], [0.0, -4.0]),  # interface_d
    ([30.0, 26.0], [0.0, -4.0]),  # interface_e
    ([27.0, 30.0], [10.0, 0.0]),  # interface_f
]
for xs, ys in interface_segments:
    ax_iso.plot(xs, ys, linestyle="--", linewidth=1.8, color="gray", alpha=0.5, zorder=2)

# isochrones
iso_colors = [to_hex(c) for c in plt.cm.plasma(np.linspace(0.1, 0.85, len(snapshot_days)))]
iso_handles = []
for day, col in zip(snapshot_days, iso_colors):
    u_snap = U_tr.sol(day * 86400.0)
    plot_phreatic_surface(
        ax_iso,
        coords,
        u_snap,
        triangles=triangles,
        color=col,
        linestyle="-",
        linewidths=1.8,
        label=None,
    )
    iso_handles.append(Line2D([0], [0], color=col, linewidth=1.8, linestyle="-", label=f"$t = {day:g}$ d"))

plot_phreatic_surface(
    ax_iso,
    coords,
    U,
    triangles=triangles,
    color="k",
    linestyle="--",
    linewidths=2.2,
    label=None,
)
iso_handles.append(Line2D([0], [0], color="k", linewidth=2.2, linestyle="--", label="Steady state"))

ax_iso.set_xlabel("Distance (m)")
ax_iso.set_ylabel("Elevation (m)")
ax_iso.set_title("Phreatic Surface Isochrones ($h = y$) vs. Steady State")
ax_iso.legend(handles=iso_handles, loc="upper right", ncol=2, frameon=True)
ax_iso.grid(True, linestyle="--", alpha=0.4)
ax_iso.axis("equal")
fig_iso.savefig("examples/paper/figures/sat_tr_phreatic_isochrones.png", dpi=300, bbox_inches="tight")

# C. Convergence
# --------------
t_days = U_tr.t / 86400.0
linf_err = np.max(np.abs(U_tr.y - U[:, None]), axis=0)
rmse_err = np.sqrt(np.mean((U_tr.y - U[:, None]) ** 2, axis=0))

probe_targets = {
    "Upstream rockfill (12, 3)": np.array([12.0, 3.0]),
    "Upstream core face (21.5, 5)": np.array([21.5, 5.0]),
    "Clay core center (25, 3)": np.array([25.0, 3.0]),
    "Core cutoff base (25, -2)": np.array([25.0, -2.0]),
    "Downstream core face (28.5, 5)": np.array([28.5, 5.0]),
    "Downstream foundation (35, -2)": np.array([35.0, -2.0]),
}

fig_conv, (ax_mesh, ax_p, ax_err) = plt.subplots(1, 3, figsize=(11, 4.2), layout="constrained")

ax_mesh.triplot(coords[:, 0], coords[:, 1], triangles, color="0.75", linewidth=0.5, zorder=1)
ax_mesh.axis("equal")
ax_mesh.set_title("(a) Mesh with Piezometer Locations")

colors = ["#FF7F00", "#FFFF33", "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]
for (label, pt), color in zip(probe_targets.items(), colors):
    ax_mesh.scatter(pt[0], pt[1], s=18, color=color)
    idx = int(np.argmin(np.linalg.norm(coords - pt, axis=1)))
    ax_p.plot(t_days, U_tr.y[idx, :], linewidth=1.7, color=color, label=label)

ax_p.set_xlabel("Time (days)")
ax_p.set_ylabel("Total head $h$ (m)")
ax_p.set_title("(b) Piezometer Head Evolution")
ax_p.grid(True, linestyle="--", alpha=0.4)
ax_p.legend(loc="best", fontsize=7.5, frameon=True)

ax_err.semilogy(t_days, linf_err, color="#E41A1C", linewidth=1.9, label=r"$\|u(t) - u_{\mathrm{steady}}\|_\infty$")
ax_err.semilogy(t_days, rmse_err, color="#377EB8", linestyle="--", linewidth=1.9, label=r"$\mathrm{RMSE}(u(t) - u_{\mathrm{steady}})$")
ax_err.set_xlabel("Time (days)")
ax_err.set_ylabel("Error vs. steady state (m)")
ax_err.set_title("(c) Asymptotic Convergence to Steady State")
ax_err.grid(True, which="both", linestyle="--", alpha=0.4)
ax_err.legend(loc="upper right", frameon=True)

fig_conv.suptitle("Transient Saturated Flow: Piezometer Histories & Steady-State Convergence")
fig_conv.savefig("examples/paper/figures/sat_tr_convergence.png", dpi=300, bbox_inches="tight")

# D. snapshots GIF into `sat_tr.gif`
# ----------------------------------
from matplotlib.animation import FuncAnimation, PillowWriter

gif_days = np.concatenate(([0.0], np.geomspace(0.05, 1000.0, 59)))
fig_gif, ax_gif = plt.subplots(figsize=(10, 3.6), layout="constrained")

# Initialize colorbar once from a dummy contourf at t = 0
tcf0 = ax_gif.tricontourf(coords[:, 0], coords[:, 1], triangles, U0, levels=levels, cmap="inferno")
cbar_gif = fig_gif.colorbar(tcf0, ax=ax_gif, shrink=0.9, pad=0.02)
cbar_gif.set_label("Total head $h$ (m)")


def update_frame(frame_idx: int) -> None:
    """Render a single time frame of the transient head and phreatic surface."""
    day = gif_days[frame_idx]
    u_frame = U0 if day == 0.0 else U_tr.sol(day * 86400.0)
    ax_gif.clear()
    ax_gif.tricontourf(coords[:, 0], coords[:, 1], triangles, u_frame, levels=levels, cmap="inferno")
    ax_gif.tricontour(coords[:, 0], coords[:, 1], triangles, u_frame, levels=levels, colors="k", linewidths=0.4, alpha=0.55)
    for xs, ys in interface_segments:
        ax_gif.plot(xs, ys, linestyle="--", linewidth=1.6, color="gray", alpha=0.5)
    plot_phreatic_surface(
        ax_gif,
        coords,
        u_frame,
        triangles=triangles,
        color="#E41A1C",
        linestyle="--",
        linewidths=1.6,
        label=None,
    )
    ax_gif.set_xlabel("Distance (m)")
    ax_gif.set_ylabel("Elevation (m)")
    ax_gif.set_title(f"Transient Saturated Flow in Five-Layer Dam ($t = {day:.2f}$ days)")
    ax_gif.grid(True, linestyle="--", alpha=0.4)
    ax_gif.set_aspect("equal")


ani = FuncAnimation(fig_gif, update_frame, frames=len(gif_days), interval=100)
ani.save("examples/paper/figures/sat_tr.gif", writer=PillowWriter(fps=10), dpi=150)
print("Saved animation: examples/paper/figures/sat_tr.gif")

plt.show()