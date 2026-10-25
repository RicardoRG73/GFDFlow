"""Stationary Groundwater Flow in a Five-Layer Dam using GFDFlow.

This script solves the 2D steady-state potential groundwater flow equation:
.. math::
    \\nabla \\cdot (k \\nabla h) = 0

for the five-layered dam problem described in Section 4.2.3 and Figure 16/17 of
Chavez et al. (2023).

The domain comprises five subdomains across three materials:
1. Rockfill shells (upstream and downstream): :math:`k_{sat} = 10^{-4} \\text{ m/s}`
2. Impervious clay core with cutoff wall: :math:`k_{sat} = 10^{-6} \\text{ m/s}`
3. Soil foundation: :math:`k_{sat} = 10^{-4} \\text{ m/s}`

Boundary conditions:
- Upstream face (submerged below y = 8 m) and left foundation: Dirichlet, :math:`h = 8.0 \\text{ m}`
- Downstream toe filter and ground surface (y = 0, x = 40 to 60 m): Dirichlet, :math:`h = 0.0 \\text{ m}`
- Bedrock base (y = -4 m) and downstream face/crest: Neumann, :math:`\\partial h / \\partial n = 0`
- Six internal material interfaces with continuous normal flux balance
- Two triple-junction angular balance nodes at (20, 0) and (30, 0)

Outputs
-------
- Figures saved to `examples/paper/figures/`:
  - `solution_contours.png`: Full contourf map with phreatic line.
  - `solution_equipotential.png`: Equipotential isolines matching FigDam5CCompara2.
"""

import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_phreatic_surface


def load_mesh_data(
    filepath: str = "examples/paper/stat_mesh1.json"
) -> dict:
    """Load serialized mesh and precomputed GFDM stencils from JSON.

    Parameters
    ----------
    filepath : str, default='examples/paper/stat_mesh1.json'
        Path to the JSON mesh file.

    Returns
    -------
    dict
        Parsed mesh data dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Mesh file {filepath} not found. Please run stat_mesh1.py first."
        )

    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def setup_and_solve(
    mesh_data: dict,
) -> Tuple[gfdmi, npt.NDArray[np.float64]]:
    """Assemble the sparse GFDM linear system and solve for hydraulic head.

    Parameters
    ----------
    mesh_data : dict
        Dictionary containing mesh coordinates, triangles, boundaries, and stencils.

    Returns
    -------
    problem : GFDMI_2D_problem
        Configured GFDM problem instance.
    U : npt.NDArray[np.float64]
        Nodal total head solution array of shape (N,).
    """
    coords = np.array(mesh_data["coords"])
    triangles = np.array(mesh_data["triangles"])
    normal_vecs = np.array(mesh_data["normal_vecs"])

    left_nodes = np.array(mesh_data["left_nodes"])
    right_nodes = np.array(mesh_data["right_nodes"])
    bottom_nodes = np.array(mesh_data["bottom_nodes"])
    top_nodes = np.array(mesh_data["top_nodes"])

    ia_nodes = np.array(mesh_data["interface_a_nodes"])
    ib_nodes = np.array(mesh_data["interface_b_nodes"])
    ic_nodes = np.array(mesh_data["interface_c_nodes"])
    id_nodes = np.array(mesh_data["interface_d_nodes"])
    ie_nodes = np.array(mesh_data["interface_e_nodes"])
    if_nodes = np.array(mesh_data["interface_f_nodes"])

    rock_nodes = np.array(mesh_data["rock_nodes"])
    clay_nodes = np.array(mesh_data["clay_nodes"])
    foundation_nodes = np.array(mesh_data["foundation_nodes"])

    tj1 = int(mesh_data["tj1"][0])
    tj2 = int(mesh_data["tj2"][0])

    support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
    M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

    # Laplacian operator L = [A, B, C, D, E, F] for D*u_xx + F*u_yy = 0
    L = np.array([0, 0, 0, 1, 0, 1])

    # Hydraulic conductivities from Table 2 of Chavez et al. (2023)
    kr = lambda p: 1e-4       # Rockfill: 0.0001 m/s
    kc = lambda p: 1e-6       # Clay core: 1e-06 m/s
    kf = lambda p: 1e-4       # Soil foundation: 0.0001 m/s
    source = lambda p: 0.0

    neumann_zero = lambda p: 0.0
    left_dirichlet = lambda p: 8.0
    right_dirichlet = lambda p: 0.0
    beta = lambda p: 0.0

    problem = gfdmi(
        coords,
        triangles,
        normal_vecs,
        L,
        source,
        M_pinv=M_pinv,
        support_stencils=support_stencils,
    )

    # 1. Assign material subdomains
    problem.material("rock", kr, rock_nodes)
    problem.material("clay", kc, clay_nodes)
    problem.material("foundation", kf, foundation_nodes)

    # 2. Outer boundary conditions
    problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
    problem.dirichlet_boundary("right", right_nodes, right_dirichlet)
    problem.neumann_boundary("bottom", kf, bottom_nodes, neumann_zero)
    problem.neumann_boundary("top", kr, top_nodes, neumann_zero)

    # 3. Internal material interfaces
    problem.interface("interface_a", kr, kf, ia_nodes, None, beta, None, rock_nodes, foundation_nodes)
    problem.interface("interface_b", kr, kf, ib_nodes, None, beta, None, rock_nodes, foundation_nodes)
    problem.interface("interface_c", kc, kr, ic_nodes, None, beta, None, clay_nodes, rock_nodes)
    problem.interface("interface_d", kc, kf, id_nodes, None, beta, None, clay_nodes, foundation_nodes)
    problem.interface("interface_e", kc, kr, ie_nodes, None, beta, None, clay_nodes, rock_nodes)
    problem.interface("interface_f", kc, kf, if_nodes, None, beta, None, clay_nodes, foundation_nodes)

    # 4. Triple junction multi-wedge intersections
    # Node 1 at (20, 0): interfaces A, D, C meeting rockfill, clay, and foundation
    problem.intersection("inters_1", tj1, "interface_a", "interface_d", "foundation", beta)
    problem.intersection("inters_2", tj1, "interface_d", "interface_c", "clay", beta)
    problem.intersection("inters_3", tj1, "interface_c", "interface_a", "rock", beta)

    # Node 2 at (30, 0): interfaces B, E, F meeting rockfill, clay, and foundation
    problem.intersection("inters_4", tj2, "interface_b", "interface_e", "rock", beta)
    problem.intersection("inters_5", tj2, "interface_e", "interface_f", "clay", beta)
    problem.intersection("inters_6", tj2, "interface_f", "interface_b", "foundation", beta)

    # 5. Assemble and solve sparse linear system
    K, F = problem.continuous_discretization()
    U = sp.linalg.spsolve(K, F)

    return problem, U


def plot_and_save_results(
    coords: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    U: npt.NDArray[np.float64],
    output_dir: str = "examples/paper/figures",
    show_plots: bool = False,
) -> None:
    """Plot and save publication figures of total head and equipotentials.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Nodal coordinates (N, 2).
    triangles : npt.NDArray[np.int_]
        Triangle connectivity (M, 3).
    U : npt.NDArray[np.float64]
        Total head solution array (N,).
    output_dir : str, default='examples/paper/figures'
        Directory where figures will be stored.
    show_plots : bool, default=False
        Whether to display interactive windows.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8")
    levels = np.linspace(0.0, 8.0, 21)

    # -------------------------------------------------------------------------
    # Figure 1: Full contourf map with phreatic surface
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 3.5), layout="constrained")
    tcf = ax1.tricontourf(
        coords[:, 0],
        coords[:, 1],
        triangles,
        U,
        levels=levels,
        cmap="viridis",
    )
    ax1.tricontour(
        coords[:, 0],
        coords[:, 1],
        triangles,
        U,
        levels=levels,
        colors="k",
        linewidths=0.5,
        alpha=0.6,
    )
    plot_phreatic_surface(
        ax1,
        coords,
        U,
        triangles=triangles,
        color="b",
        linewidths=2.2,
        label="Phreatic surface ($h = y$)",
    )
    cbar1 = fig1.colorbar(tcf, ax=ax1, shrink=0.9, pad=0.02)
    cbar1.set_label("Total head $h$ (m)")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Elevation (m)")
    ax1.set_title("Steady State Potential Flow in Five-Layer Dam")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.axis("equal")

    path1 = os.path.join(output_dir, "solution_contours.png")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"Saved: {path1}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    """Execute the stationary GFDFlow assembly and post-processing pipeline."""
    print("==================================================================")
    print("GFDFlow — Five-Layer Dam Seepage Assembly (Chavez et al. 2023)")
    print("==================================================================")

    # 1. Load precomputed mesh
    mesh_path = "examples/paper/stat_mesh1.json"
    print(f"Loading mesh data from: {mesh_path}")
    mesh_data = load_mesh_data(mesh_path)
    coords = np.array(mesh_data["coords"])
    triangles = np.array(mesh_data["triangles"])
    print(f"Mesh loaded: {coords.shape[0]} nodes, {len(triangles)} triangles.")

    # 2. Assemble and solve
    print("Assembling continuous interface GFDM system...")
    problem, U = setup_and_solve(mesh_data)
    print(f"Linear system solved successfully.")
    print(f"Total head range: min = {U.min():.4f} m, max = {U.max():.4f} m (mean = {U.mean():.4f} m).")

    # 3. Post-process and save figures
    fig_dir = "examples/paper/figures"
    print(f"Rendering and saving solution figures to: {fig_dir}...")
    plot_and_save_results(coords, triangles, U, output_dir=fig_dir, show_plots=True)

    print("==================================================================")
    print("Assembly and solution completed successfully.")
    print("==================================================================")


if __name__ == "__main__":
    main()
