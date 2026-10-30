"""Transient Groundwater Potential Flow in a Five-Layer Dam (Crank-Nicolson).

This module solves the 2D transient (non-stationary) potential groundwater flow
equation:

.. math::
    S_s \\frac{\\partial u}{\\partial t} = \\nabla \\cdot (k \\nabla u)

with normalized specific storage :math:`S_s = 1` for the five-layered embankment
dam with central clay core and foundation described in Section 4.2.3 (Table 2,
Figures 16 and 17) of Chavez et al. (2023), whose stationary counterpart
(:math:`\\nabla \\cdot (k \\nabla u) = 0`) is implemented in
``examples/paper/st_potential1.py``.

Mathematical & Numerical Formulation
------------------------------------
Following Part II (Section 9) of ``02-DOCS/wiki/theory.md``:

1. **Method-of-Lines GFDM Semi-Discretization**:
   Spatial discretization via ``GFDMI_2D_problem.continuous_discretization()``
   partitions the :math:`N` mesh nodes into free nodes :math:`\\mathcal{F}`
   (interior subdomains, ghost-eliminated Neumann boundaries, six two-layer
   flux-continuous interfaces, and two three-wedge triple junctions) and
   Dirichlet boundary nodes :math:`\\mathcal{D}`:

   .. math::
       S_s \\frac{d\\boldsymbol{u}_{\\mathcal{F}}}{dt}
       = \\mathbf{K}_{\\mathcal{F}\\cdot} \\boldsymbol{u}(t) - \\boldsymbol{F}_{\\mathcal{F}},
       \\qquad \\boldsymbol{u}_{\\mathcal{D}}(t) = \\boldsymbol{u}_D

2. **Crank-Nicolson Scheme (:math:`\\theta = 1/2`) with Rannacher Startup**:
   Advancing from :math:`t^n` to :math:`t^{n+1} = t^n + \\Delta t` solves the
   sparse linear system :math:`\\mathbf{A}_{\\mathrm{LHS}} \\boldsymbol{u}^{n+1} = \\boldsymbol{b}^n`
   using ``scipy.sparse.linalg.spsolve``, where on free rows :math:`i \\in \\mathcal{F}`:

   .. math::
       \\left( S_s \\mathbf{I} - \\frac{\\Delta t}{2} \\mathbf{K} \\right)_{i,:} \\boldsymbol{u}^{n+1}
       = \\left( S_s \\mathbf{I} + \\frac{\\Delta t}{2} \\mathbf{K} \\right)_{i,:} \\boldsymbol{u}^n
       - \\Delta t \\, F_i

   and on Dirichlet rows :math:`i \\in \\mathcal{D}`, :math:`u_i^{n+1} = u_D(\\boldsymbol{x}_i)`.
   To suppress high-wavenumber ringing triggered by the impulsive upstream head
   jump at :math:`t = 0^+` (:math:`u_0 = 0 \\to 8\\text{ m}`) while preserving global
   :math:`\\mathcal{O}(\\Delta t^2)` accuracy, four Rannacher Backward-Euler
   (:math:`\\theta = 1`) half-steps are applied at startup (Rannacher, 1984).

Outputs
-------
- Figures saved to ``examples/paper/figures/``:
  - ``tr_potential_snapshots.png``: Six-panel transient evolution of total head
    contours, equipotentials, and phreatic surface from :math:`t = 1` day to steady state.
  - ``tr_potential_phreatic_evolution.png``: Progressive migration of the phreatic
    surface (:math:`u = y`) across the upstream shell, clay core, and downstream shell.
  - ``tr_potential_convergence.png``: Piezometric head time series, exponential
    relaxation toward ``st_potential1.py``, and :math:`\\mathcal{O}(\\Delta t^2)`
    temporal order-of-accuracy verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_phreatic_surface

SECONDS_PER_DAY: float = 86_400.0


@dataclass(frozen=True, slots=True)
class TransientDamSystem:
    """Pre-assembled sparse spatial GFDM operators and mesh topology for the dam.

    Parameters
    ----------
    coords : npt.NDArray[np.float64]
        Nodal coordinates of shape ``(N, 2)``.
    triangles : npt.NDArray[np.int_]
        Delaunay/Gmsh triangle connectivity of shape ``(M, 3)``.
    k_mat : scipy.sparse.csr_matrix
        Stationary GFDM operator :math:`\\mathbf{K}` of shape ``(N, N)``.
    f_vec : npt.NDArray[np.float64]
        Stationary right-hand-side vector :math:`\\boldsymbol{F}` of shape ``(N,)``.
    u_steady : npt.NDArray[np.float64]
        Exact stationary hydraulic head :math:`\\boldsymbol{u}^* = \\mathbf{K}^{-1}\\boldsymbol{F}` (m).
    dirichlet_nodes : npt.NDArray[np.int_]
        Sorted array of Dirichlet node indices :math:`\\mathcal{D}`.
    left_nodes : npt.NDArray[np.int_]
        Upstream Dirichlet boundary node indices (:math:`u = 8.0\\text{ m}`).
    right_nodes : npt.NDArray[np.int_]
        Downstream Dirichlet boundary node indices (:math:`u = 0.0\\text{ m}`).
    """

    coords: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.int_]
    k_mat: sp.csr_matrix
    f_vec: npt.NDArray[np.float64]
    u_steady: npt.NDArray[np.float64]
    dirichlet_nodes: npt.NDArray[np.int_]
    left_nodes: npt.NDArray[np.int_]
    right_nodes: npt.NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class CrankNicolsonResult:
    """Container for Crank-Nicolson transient trajectory and diagnostics.

    Parameters
    ----------
    times : npt.NDArray[np.float64]
        Recorded simulation timestamps in seconds, shape ``(K,)``.
    snapshots : dict[float, npt.NDArray[np.float64]]
        Mapping from target snapshot time (s) to nodal total head vector ``(N,)``.
    u_final : npt.NDArray[np.float64]
        Final nodal total head field :math:`\\boldsymbol{u}(t_{\\mathrm{end}})` (m).
    probe_indices : dict[str, int]
        Mapping from piezometer label to closest mesh node index.
    probe_histories : dict[str, npt.NDArray[np.float64]]
        Time series of hydraulic head (m) at each monitoring piezometer, shape ``(K,)``.
    linf_to_steady : npt.NDArray[np.float64]
        Maximum nodal departure :math:`\\|\\boldsymbol{u}(t) - \\boldsymbol{u}^*\\|_\\infty` (m).
    rmse_to_steady : npt.NDArray[np.float64]
        Root-mean-square departure from stationary head :math:`\\boldsymbol{u}^*` (m).
    min_head : float
        Global minimum head observed over all nodes and steps (DMP check).
    max_head : float
        Global maximum head observed over all nodes and steps (DMP check).
    n_solves : int
        Total number of sparse linear solves executed.
    """

    times: npt.NDArray[np.float64]
    snapshots: dict[float, npt.NDArray[np.float64]]
    u_final: npt.NDArray[np.float64]
    probe_indices: dict[str, int]
    probe_histories: dict[str, npt.NDArray[np.float64]]
    linf_to_steady: npt.NDArray[np.float64]
    rmse_to_steady: npt.NDArray[np.float64]
    min_head: float
    max_head: float
    n_solves: int


@dataclass(frozen=True, slots=True)
class TemporalOrderResult:
    """Empirical temporal convergence verification data for Crank-Nicolson.

    Parameters
    ----------
    dt_values : npt.NDArray[np.float64]
        Tested time-step sizes :math:`\\Delta t` (s).
    errors_linf : npt.NDArray[np.float64]
        :math:`L_\\infty` error against fine reference solution at :math:`t = T_{\\mathrm{test}}`.
    errors_l2 : npt.NDArray[np.float64]
        Root-mean-square (:math:`L_2`) error against fine reference solution.
    orders_linf : npt.NDArray[np.float64]
        Pairwise observed convergence orders in :math:`L_\\infty` norm.
    orders_l2 : npt.NDArray[np.float64]
        Pairwise observed convergence orders in :math:`L_2` norm.
    mean_order_l2 : float
        Least-squares log-log slope of :math:`L_2` error versus :math:`\\Delta t`.
    """

    dt_values: npt.NDArray[np.float64]
    errors_linf: npt.NDArray[np.float64]
    errors_l2: npt.NDArray[np.float64]
    orders_linf: npt.NDArray[np.float64]
    orders_l2: npt.NDArray[np.float64]
    mean_order_l2: float


def load_mesh_data(filepath: Path | str = Path("examples/paper/mesh1.json")) -> dict[str, Any]:
    """Load serialized mesh topology and precomputed GFDM stencils from JSON.

    Parameters
    ----------
    filepath : Path or str, default=Path('examples/paper/mesh1.json')
        Path to the JSON mesh file produced by ``examples/paper/mesh1.py``.

    Returns
    -------
    dict[str, Any]
        Parsed mesh dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Mesh file '{path}' not found. Run 'examples/paper/mesh1.py' first."
        )
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def assemble_dam_system(mesh_data: dict[str, Any]) -> TransientDamSystem:
    """Assemble the five-layer dam GFDM spatial operator matching ``st_potential1.py``.

    Parameters
    ----------
    mesh_data : dict[str, Any]
        Dictionary containing mesh coordinates, connectivity, node groups, and stencils.

    Returns
    -------
    TransientDamSystem
        Spatial sparse matrix :math:`\\mathbf{K}`, vector :math:`\\boldsymbol{F}`,
        exact stationary solution :math:`\\boldsymbol{u}^*`, and boundary sets.
    """
    coords = np.asarray(mesh_data["coords"], dtype=np.float64)
    triangles = np.asarray(mesh_data["triangles"], dtype=np.int_)
    normal_vecs = np.asarray(mesh_data["normal_vecs"], dtype=np.float64)

    left_nodes = np.asarray(mesh_data["left_nodes"], dtype=np.int_)
    right_nodes = np.asarray(mesh_data["right_nodes"], dtype=np.int_)
    bottom_nodes = np.asarray(mesh_data["bottom_nodes"], dtype=np.int_)
    top_nodes = np.asarray(mesh_data["top_nodes"], dtype=np.int_)

    ia_nodes = np.asarray(mesh_data["interface_a_nodes"], dtype=np.int_)
    ib_nodes = np.asarray(mesh_data["interface_b_nodes"], dtype=np.int_)
    ic_nodes = np.asarray(mesh_data["interface_c_nodes"], dtype=np.int_)
    id_nodes = np.asarray(mesh_data["interface_d_nodes"], dtype=np.int_)
    ie_nodes = np.asarray(mesh_data["interface_e_nodes"], dtype=np.int_)
    if_nodes = np.asarray(mesh_data["interface_f_nodes"], dtype=np.int_)

    rock_nodes = np.asarray(mesh_data["rock_nodes"], dtype=np.int_)
    clay_nodes = np.asarray(mesh_data["clay_nodes"], dtype=np.int_)
    foundation_nodes = np.asarray(mesh_data["foundation_nodes"], dtype=np.int_)

    tj1 = int(mesh_data["tj1"][0])
    tj2 = int(mesh_data["tj2"][0])

    support_stencils = {
        int(k): np.asarray(v, dtype=np.int_) for k, v in mesh_data["support_stencils"].items()
    }
    m_pinv = {
        int(k): np.asarray(v, dtype=np.float64) for k, v in mesh_data["M_pinv"].items()
    }

    op_l = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    kr = lambda _p: 1e-4
    kc = lambda _p: 1e-6
    kf = lambda _p: 1e-4
    zero_fn = lambda _p: 0.0
    left_bc = lambda _p: 8.0
    right_bc = lambda _p: 0.0

    problem = gfdmi(
        coords,
        triangles,
        normal_vecs,
        op_l,
        zero_fn,
        M_pinv=m_pinv,
        support_stencils=support_stencils,
    )

    problem.material("rock", kr, rock_nodes)
    problem.material("clay", kc, clay_nodes)
    problem.material("foundation", kf, foundation_nodes)

    problem.dirichlet_boundary("left", left_nodes, left_bc)
    problem.dirichlet_boundary("right", right_nodes, right_bc)
    problem.neumann_boundary("bottom", kf, bottom_nodes, zero_fn)
    problem.neumann_boundary("top", kr, top_nodes, zero_fn)

    problem.interface(
        "interface_a", kr, kf, ia_nodes, None, zero_fn, None, rock_nodes, foundation_nodes
    )
    problem.interface(
        "interface_b", kr, kf, ib_nodes, None, zero_fn, None, rock_nodes, foundation_nodes
    )
    problem.interface(
        "interface_c", kc, kr, ic_nodes, None, zero_fn, None, clay_nodes, rock_nodes
    )
    problem.interface(
        "interface_d", kc, kf, id_nodes, None, zero_fn, None, clay_nodes, foundation_nodes
    )
    problem.interface(
        "interface_e", kf, kc, ie_nodes, None, zero_fn, None, foundation_nodes, clay_nodes
    )
    problem.interface(
        "interface_f", kr, kc, if_nodes, None, zero_fn, None, rock_nodes, clay_nodes
    )

    problem.intersection("inters_1", tj1, "interface_a", "interface_d", "foundation", zero_fn)
    problem.intersection("inters_2", tj1, "interface_d", "interface_c", "clay", zero_fn)
    problem.intersection("inters_3", tj1, "interface_c", "interface_a", "rock", zero_fn)

    problem.intersection("inters_4", tj2, "interface_b", "interface_f", "rock", zero_fn)
    problem.intersection("inters_5", tj2, "interface_e", "interface_f", "clay", zero_fn)
    problem.intersection("inters_6", tj2, "interface_e", "interface_b", "foundation", zero_fn)

    k_mat, f_vec = problem.continuous_discretization()
    u_steady = np.asarray(spla.spsolve(k_mat, f_vec), dtype=np.float64)
    dirichlet_nodes = np.union1d(left_nodes, right_nodes)

    return TransientDamSystem(
        coords=coords,
        triangles=triangles,
        k_mat=k_mat.tocsr(),
        f_vec=np.asarray(f_vec, dtype=np.float64),
        u_steady=u_steady,
        dirichlet_nodes=dirichlet_nodes,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
    )


def build_theta_step_matrices(
    k_mat: sp.csr_matrix,
    f_vec: npt.NDArray[np.float64],
    dirichlet_nodes: npt.NDArray[np.int_],
    dt: float,
    s_s: float = 1.0,
    theta: float = 0.5,
) -> tuple[sp.csc_matrix, sp.csr_matrix, npt.NDArray[np.float64]]:
    """Build sparse time-marching matrices for :math:`S_s \\, du/dt = K u - F`.

    Constructs :math:`\\mathbf{A}_{\\mathrm{LHS}}`, :math:`\\mathbf{A}_{\\mathrm{RHS}}`,
    and constant source/boundary offset :math:`\\boldsymbol{c}` such that each step
    solves:

    .. math::
        \\mathbf{A}_{\\mathrm{LHS}} \\boldsymbol{u}^{n+1}
        = \\mathbf{A}_{\\mathrm{RHS}} \\boldsymbol{u}^n + \\boldsymbol{c}

    with :math:`\\theta = 0.5` yielding the Crank-Nicolson scheme and :math:`\\theta = 1.0`
    yielding the :math:`L`-stable Backward Euler scheme used in Rannacher startup.

    Parameters
    ----------
    k_mat : scipy.sparse.csr_matrix
        Global GFDM spatial operator :math:`\\mathbf{K}` of shape ``(N, N)``.
    f_vec : npt.NDArray[np.float64]
        Global right-hand-side vector :math:`\\boldsymbol{F}` of shape ``(N,)``.
    dirichlet_nodes : npt.NDArray[np.int_]
        Indices of Dirichlet boundary nodes :math:`\\mathcal{D}`.
    dt : float
        Time-step size :math:`\\Delta t > 0` in seconds.
    s_s : float, default=1.0
        Specific storage coefficient :math:`S_s` in :math:`S_s \\, du/dt`.
    theta : float, default=0.5
        Implicitness parameter :math:`\\theta \\in [0, 1]` (:math:`0.5` for Crank-Nicolson).

    Returns
    -------
    a_lhs : scipy.sparse.csc_matrix
        Left-hand-side sparse matrix in CSC format ready for ``spsolve``.
    a_rhs : scipy.sparse.csr_matrix
        Right-hand-side sparse matrix multiplying :math:`\\boldsymbol{u}^n`.
    c_offset : npt.NDArray[np.float64]
        Constant source and Dirichlet boundary vector added to RHS each step.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
    if s_s <= 0.0:
        raise ValueError(f"s_s must be strictly positive, got {s_s}")
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must lie in [0, 1], got {theta}")

    n_nodes = k_mat.shape[0]
    dir_ind = np.zeros(n_nodes, dtype=np.float64)
    dir_ind[dirichlet_nodes] = 1.0
    free_ind = 1.0 - dir_ind

    p_dir = sp.diags(dir_ind, format="csr")
    p_free = sp.diags(free_ind, format="csr")
    k_free = p_free @ k_mat
    m_free = s_s * p_free

    a_lhs = (p_dir + m_free - (theta * dt) * k_free).tocsc()
    a_rhs = (m_free + ((1.0 - theta) * dt) * k_free).tocsr()
    c_offset = -dt * (free_ind * f_vec) + dir_ind * f_vec
    return a_lhs, a_rhs, c_offset


def build_initial_condition(system: TransientDamSystem) -> npt.NDArray[np.float64]:
    """Construct the initial hydraulic head vector :math:`\\boldsymbol{u}(\\boldsymbol{x}, 0)`.

    Interior, Neumann, and interface nodes start unpressurized at :math:`u = 0.0\\text{ m}`,
    while Dirichlet boundary nodes satisfy :math:`u_{\\mathrm{left}} = 8.0\\text{ m}` and
    :math:`u_{\\mathrm{right}} = 0.0\\text{ m}` at :math:`t = 0^+`.

    Parameters
    ----------
    system : TransientDamSystem
        Assembled dam system.

    Returns
    -------
    npt.NDArray[np.float64]
        Initial head array ``u0`` of shape ``(N,)``.
    """
    u0 = np.zeros(system.coords.shape[0], dtype=np.float64)
    u0[system.left_nodes] = 8.0
    u0[system.right_nodes] = 0.0
    return u0


def advance_uniform_crank_nicolson(
    system: TransientDamSystem,
    u_init: npt.NDArray[np.float64],
    dt: float,
    n_steps: int,
    s_s: float = 1.0,
    rannacher_half_steps: int = 4,
) -> npt.NDArray[np.float64]:
    """Advance :math:`S_s \\, du/dt = K u - F` over ``n_steps`` of size ``dt`` via Crank-Nicolson.

    Optionally replaces the first ``rannacher_half_steps // 2`` Crank-Nicolson steps
    with ``rannacher_half_steps`` Backward-Euler half-steps (:math:`\\Delta t / 2`)
    following Rannacher (1984) to damp non-smooth boundary transients while preserving
    second-order global accuracy :math:`\\mathcal{O}(\\Delta t^2)`.

    Parameters
    ----------
    system : TransientDamSystem
        Assembled spatial GFDM system.
    u_init : npt.NDArray[np.float64]
        Initial nodal head vector of shape ``(N,)``.
    dt : float
        Crank-Nicolson time step :math:`\\Delta t` (s).
    n_steps : int
        Total number of time steps of size ``dt`` to advance (:math:`T = n_{\\mathrm{steps}} \\Delta t`).
    s_s : float, default=1.0
        Specific storage coefficient :math:`S_s` in :math:`S_s \\, du/dt`.
    rannacher_half_steps : int, default=4
        Even number of Backward-Euler half-steps of size ``dt / 2`` applied at :math:`t = 0`.

    Returns
    -------
    npt.NDArray[np.float64]
        Nodal head vector at :math:`t = n_{\\mathrm{steps}} \\Delta t`.
    """
    if rannacher_half_steps % 2 != 0:
        raise ValueError("rannacher_half_steps must be an even integer.")

    u = u_init.copy()
    cn_steps_replaced = rannacher_half_steps // 2
    if cn_steps_replaced > n_steps:
        raise ValueError("n_steps must be at least rannacher_half_steps // 2.")

    if rannacher_half_steps > 0:
        lhs_be, rhs_be, c_be = build_theta_step_matrices(
            system.k_mat,
            system.f_vec,
            system.dirichlet_nodes,
            dt=0.5 * dt,
            s_s=s_s,
            theta=1.0,
        )
        for _ in range(rannacher_half_steps):
            u = np.asarray(spla.spsolve(lhs_be, rhs_be @ u + c_be), dtype=np.float64)

    remaining_cn_steps = n_steps - cn_steps_replaced
    if remaining_cn_steps > 0:
        lhs_cn, rhs_cn, c_cn = build_theta_step_matrices(
            system.k_mat,
            system.f_vec,
            system.dirichlet_nodes,
            dt=dt,
            s_s=s_s,
            theta=0.5,
        )
        for _ in range(remaining_cn_steps):
            u = np.asarray(spla.spsolve(lhs_cn, rhs_cn @ u + c_cn), dtype=np.float64)

    return u


def run_transient_dam_simulation(
    system: TransientDamSystem,
    s_s: float = 1.0,
    snapshot_days: tuple[float, ...] = (1.0, 5.0, 20.0, 60.0, 180.0, 1000.0),
) -> CrankNicolsonResult:
    """Simulate the full transient seepage evolution from :math:`t = 0` to steady state.

    Uses a two-stage Crank-Nicolson schedule with Rannacher startup:
    - **Stage 1** (:math:`0 \\to 10` days, :math:`\\Delta t_1 = 0.05` day = 4,320 s):
      Resolves the rapid hydraulic head diffusion across the high-permeability
      upstream rockfill shell and foundation (:math:`k = 10^{-4}\\text{ m/s}`).
    - **Stage 2** (:math:`10 \\to 1000` days, :math:`\\Delta t_2 = 1.0` day = 86,400 s):
      Integrates the slow pressure front across the low-permeability clay core
      (:math:`k = 10^{-6}\\text{ m/s}`, characteristic time :math:`\\tau \\approx 65.9` days)
      until converging to the stationary solution of ``st_potential1.py``.

    Parameters
    ----------
    system : TransientDamSystem
        Assembled dam spatial system.
    s_s : float, default=1.0
        Specific storage coefficient :math:`S_s` in :math:`S_s \\, du/dt`.
    snapshot_days : tuple[float, ...], optional
        Target times (in days) at which full spatial head fields are stored.

    Returns
    -------
    CrankNicolsonResult
        Full transient snapshots, piezometer histories, and steady-state convergence metrics.
    """
    u = build_initial_condition(system)
    target_seconds = {round(d * SECONDS_PER_DAY, 6): d for d in snapshot_days}

    # Select monitoring piezometer nodes across the five-layer dam
    probe_targets = {
        "Upstream rockfill (12, 3)": np.array([12.0, 3.0]),
        "Upstream core face (21.5, 5)": np.array([21.5, 5.0]),
        "Clay core center (25, 3)": np.array([25.0, 3.0]),
        "Core cutoff base (25, -2)": np.array([25.0, -2.0]),
        "Downstream core face (28.5, 5)": np.array([28.5, 5.0]),
        "Downstream foundation (35, -2)": np.array([35.0, -2.0]),
    }
    probe_indices: dict[str, int] = {}
    for label, pt in probe_targets.items():
        idx = int(np.argmin(np.linalg.norm(system.coords - pt, axis=1)))
        probe_indices[label] = idx

    times_list: list[float] = [0.0]
    probe_lists: dict[str, list[float]] = {
        label: [float(u[idx])] for label, idx in probe_indices.items()
    }
    diff0 = u - system.u_steady
    linf_list: list[float] = [float(np.max(np.abs(diff0)))]
    rmse_list: list[float] = [float(np.sqrt(np.mean(diff0 ** 2)))]

    snapshots: dict[float, npt.NDArray[np.float64]] = {}
    if 0.0 in target_seconds:
        snapshots[0.0] = u.copy()

    min_head = float(u.min())
    max_head = float(u.max())
    n_solves = 0
    t_curr = 0.0

    # Stage 1: 0 -> 10 days with dt1 = 0.05 day (4,320 s), including 4 Rannacher half-steps
    dt1 = 0.05 * SECONDS_PER_DAY
    n_stage1 = 200  # 200 * 0.05 = 10.0 days
    lhs_be, rhs_be, c_be = build_theta_step_matrices(
        system.k_mat, system.f_vec, system.dirichlet_nodes, dt=0.5 * dt1, s_s=s_s, theta=1.0
    )
    lhs_cn1, rhs_cn1, c_cn1 = build_theta_step_matrices(
        system.k_mat, system.f_vec, system.dirichlet_nodes, dt=dt1, s_s=s_s, theta=0.5
    )

    for step in range(1, n_stage1 + 1):
        if step <= 2:
            # Two half-steps of Backward Euler per replaced CN step (4 total Rannacher half-steps)
            for _ in range(2):
                u = np.asarray(spla.spsolve(lhs_be, rhs_be @ u + c_be), dtype=np.float64)
                n_solves += 1
        else:
            u = np.asarray(spla.spsolve(lhs_cn1, rhs_cn1 @ u + c_cn1), dtype=np.float64)
            n_solves += 1

        t_curr = round(step * dt1, 6)
        min_head = min(min_head, float(u.min()))
        max_head = max(max_head, float(u.max()))

        times_list.append(t_curr)
        for label, idx in probe_indices.items():
            probe_lists[label].append(float(u[idx]))
        diff = u - system.u_steady
        linf_list.append(float(np.max(np.abs(diff))))
        rmse_list.append(float(np.sqrt(np.mean(diff ** 2))))

        if t_curr in target_seconds:
            snapshots[target_seconds[t_curr]] = u.copy()

    # Stage 2: 10 -> max(1000, max(snapshot_days)) days with dt2 = 1.0 day (86,400 s)
    end_days = int(max(1000.0, max(snapshot_days)))
    dt2 = 1.0 * SECONDS_PER_DAY
    n_stage2 = end_days - 10
    lhs_cn2, rhs_cn2, c_cn2 = build_theta_step_matrices(
        system.k_mat, system.f_vec, system.dirichlet_nodes, dt=dt2, s_s=s_s, theta=0.5
    )

    for step in range(1, n_stage2 + 1):
        u = np.asarray(spla.spsolve(lhs_cn2, rhs_cn2 @ u + c_cn2), dtype=np.float64)
        n_solves += 1
        t_curr = round((10 + step) * SECONDS_PER_DAY, 6)
        min_head = min(min_head, float(u.min()))
        max_head = max(max_head, float(u.max()))

        times_list.append(t_curr)
        for label, idx in probe_indices.items():
            probe_lists[label].append(float(u[idx]))
        diff = u - system.u_steady
        linf_list.append(float(np.max(np.abs(diff))))
        rmse_list.append(float(np.sqrt(np.mean(diff ** 2))))

        if t_curr in target_seconds:
            snapshots[target_seconds[t_curr]] = u.copy()

    return CrankNicolsonResult(
        times=np.asarray(times_list, dtype=np.float64),
        snapshots=snapshots,
        u_final=u,
        probe_indices=probe_indices,
        probe_histories={k: np.asarray(v, dtype=np.float64) for k, v in probe_lists.items()},
        linf_to_steady=np.asarray(linf_list, dtype=np.float64),
        rmse_to_steady=np.asarray(rmse_list, dtype=np.float64),
        min_head=min_head,
        max_head=max_head,
        n_solves=n_solves,
    )


def verify_temporal_order(
    system: TransientDamSystem,
    s_s: float = 1.0,
    t_horizon: float = 2.0 * SECONDS_PER_DAY,
    step_counts: tuple[int, ...] = (16, 32, 64, 128),
    ref_steps: int = 1024,
) -> TemporalOrderResult:
    """Verify second-order :math:`\\mathcal{O}(\\Delta t^2)` accuracy of Crank-Nicolson.

    Advances the transient dam PDE over ``[0, t_horizon]`` using uniform step counts
    ``step_counts`` and compares against a fine Crank-Nicolson reference ``ref_steps``.

    Parameters
    ----------
    system : TransientDamSystem
        Assembled spatial GFDM system.
    s_s : float, default=1.0
        Specific storage coefficient :math:`S_s`.
    t_horizon : float, default=172800.0
        Verification time horizon :math:`T_{\\mathrm{test}}` in seconds (default 2 days).
    step_counts : tuple[int, ...], default=(16, 32, 64, 128)
        Coarse-to-fine step counts tested.
    ref_steps : int, default=1024
        Reference step count used as exact semi-discrete benchmark.

    Returns
    -------
    TemporalOrderResult
        Error norms and empirical convergence rates.
    """
    u0 = build_initial_condition(system)
    u_ref = advance_uniform_crank_nicolson(
        system, u0, dt=t_horizon / ref_steps, n_steps=ref_steps, s_s=s_s, rannacher_half_steps=4
    )

    dts: list[float] = []
    err_inf: list[float] = []
    err_l2: list[float] = []

    for m_steps in step_counts:
        dt = t_horizon / m_steps
        u_num = advance_uniform_crank_nicolson(
            system, u0, dt=dt, n_steps=m_steps, s_s=s_s, rannacher_half_steps=4
        )
        diff = u_num - u_ref
        dts.append(dt)
        err_inf.append(float(np.max(np.abs(diff))))
        err_l2.append(float(np.sqrt(np.mean(diff ** 2))))

    dt_arr = np.asarray(dts, dtype=np.float64)
    inf_arr = np.asarray(err_inf, dtype=np.float64)
    l2_arr = np.asarray(err_l2, dtype=np.float64)

    orders_inf = np.log(inf_arr[:-1] / inf_arr[1:]) / np.log(dt_arr[:-1] / dt_arr[1:])
    orders_l2 = np.log(l2_arr[:-1] / l2_arr[1:]) / np.log(dt_arr[:-1] / dt_arr[1:])
    slope_l2 = float(np.polyfit(np.log(dt_arr), np.log(l2_arr), 1)[0])

    return TemporalOrderResult(
        dt_values=dt_arr,
        errors_linf=inf_arr,
        errors_l2=l2_arr,
        orders_linf=orders_inf,
        orders_l2=orders_l2,
        mean_order_l2=slope_l2,
    )


def _draw_dam_interfaces(
    ax: plt.Axes, color: str = "white", lw: float = 0.85, alpha: float = 0.7
) -> None:
    """Overlay internal structural layer boundaries of the five-layer dam."""
    segments = [
        ([3.0, 20.0], [0.0, 0.0]),
        ([30.0, 47.0], [0.0, 0.0]),
        ([23.0, 20.0, 24.0], [10.0, 0.0, -4.0]),
        ([27.0, 30.0, 26.0], [10.0, 0.0, -4.0]),
    ]
    for xs, ys in segments:
        ax.plot(xs, ys, linestyle="--", color=color, linewidth=lw, alpha=alpha)


def _draw_dam_backdrop(ax: plt.Axes) -> None:
    """Draw outer dam-foundation outline and shaded clay core."""
    outer_xy = [
        (3.0, 0.0),
        (23.0, 10.0),
        (27.0, 10.0),
        (47.0, 0.0),
        (60.0, 0.0),
        (60.0, -4.0),
        (3.0, -4.0),
    ]
    clay_xy = [
        (23.0, 10.0),
        (27.0, 10.0),
        (30.0, 0.0),
        (26.0, -4.0),
        (24.0, -4.0),
        (20.0, 0.0),
    ]
    ax.add_patch(Polygon(outer_xy, closed=True, facecolor="#f4f5f8", edgecolor="none", zorder=0))
    ax.add_patch(Polygon(clay_xy, closed=True, facecolor="#e1e5ee", edgecolor="none", zorder=0.5))
    ax.add_patch(
        Polygon(outer_xy, closed=True, facecolor="none", edgecolor="0.2", linewidth=0.95, zorder=4)
    )


def save_transient_figures(
    system: TransientDamSystem,
    cn_result: CrankNicolsonResult,
    order_result: TemporalOrderResult,
    output_dir: Path | str = Path("examples/paper/figures"),
) -> list[Path]:
    """Render and save publication figures for the transient Crank-Nicolson dam problem.

    Parameters
    ----------
    system : TransientDamSystem
        Assembled spatial dam system including stationary baseline ``u_steady``.
    cn_result : CrankNicolsonResult
        Transient Crank-Nicolson snapshots and histories.
    order_result : TemporalOrderResult
        Empirical :math:`\\mathcal{O}(\\Delta t^2)` verification data.
    output_dir : Path or str, default=Path('examples/paper/figures')
        Destination directory for generated PNG images.

    Returns
    -------
    list[Path]
        Saved figure file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    coords = system.coords
    triangles = system.triangles
    x, y = coords[:, 0], coords[:, 1]
    levels_u = np.linspace(0.0, 8.0, 17)
    saved: list[Path] = []

    # =========================================================================
    # Figure 1: 6-Panel Transient Head Contours & Phreatic Surface Evolution
    # =========================================================================
    snap_days = sorted(cn_result.snapshots.keys())
    fig1, axes1 = plt.subplots(3, 2, figsize=(14.0, 7.6), sharex=True, sharey=True, layout="constrained")
    ax_flat = axes1.ravel()
    tcf_last = None

    for idx, day in enumerate(snap_days[:6]):
        ax = ax_flat[idx]
        u_snap = np.clip(cn_result.snapshots[day], 0.0, 8.0)
        tcf_last = ax.tricontourf(x, y, triangles, u_snap, levels=levels_u, cmap="viridis")
        ax.tricontour(
            x, y, triangles, u_snap, levels=levels_u, colors="k", linewidths=0.45, alpha=0.6
        )
        _draw_dam_interfaces(ax, color="w", lw=0.8, alpha=0.7)
        plot_phreatic_surface(
            ax,
            coords,
            u_snap,
            triangles=triangles,
            color="#00d5ff",
            linewidths=1.9,
            label=None,
        )
        err_st = float(np.max(np.abs(cn_result.snapshots[day] - system.u_steady)))
        panel_tag = chr(ord("a") + idx)
        ax.set_title(
            f"({panel_tag}) $t = {day:g}$ d  "
            f"($\\|u - u_{{\\mathrm{{st}}}}\\|_\\infty = {err_st:.2e}$ m)",
            fontsize=10.5,
        )
        ax.set_aspect("equal")
        if idx % 2 == 0:
            ax.set_ylabel("Elevation $y$ (m)")
        if idx >= 4:
            ax.set_xlabel("Distance $x$ (m)")

    if tcf_last is not None:
        cbar = fig1.colorbar(tcf_last, ax=axes1, shrink=0.88, pad=0.015)
        cbar.set_label("Total hydraulic head $u$ (m)")

    fig1.suptitle(
        "Transient Potential Seepage ($S_s \\, \\partial u / \\partial t = \\nabla \\cdot (k \\nabla u)$, "
        "$S_s = 1$) — Crank-Nicolson Evolution",
        fontsize=12.5,
        fontweight="bold",
    )
    p1 = out / "tr_potential_snapshots.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    saved.append(p1)

    # =========================================================================
    # Figure 2: Progressive Migration of Phreatic Surface vs Stationary Baseline
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(11.0, 3.9), layout="constrained")
    _draw_dam_backdrop(ax2)
    _draw_dam_interfaces(ax2, color="0.45", lw=0.9, alpha=0.8)

    cmap_time = plt.get_cmap("plasma")
    colors_time = [cmap_time(v) for v in np.linspace(0.08, 0.88, len(snap_days))]
    legend_handles: list[Line2D] = []

    for day, col in zip(snap_days, colors_time):
        u_snap = np.clip(cn_result.snapshots[day], 0.0, 8.0)
        psi_snap = u_snap - y
        ax2.tricontour(
            x,
            y,
            triangles,
            psi_snap,
            levels=[0.0],
            colors=[col],
            linewidths=1.85,
            linestyles="-",
        )
        legend_handles.append(Line2D([0], [0], color=col, lw=1.85, label=f"$t = {day:g}$ d"))

    ax2.tricontour(
        x,
        y,
        triangles,
        np.clip(system.u_steady, 0.0, 8.0) - y,
        levels=[0.0],
        colors=["#0b6623"],
        linewidths=2.4,
        linestyles="--",
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#0b6623",
            lw=2.4,
            ls="--",
            label="Stationary (st_potential1.py)",
        )
    )

    ax2.set_xlabel("Distance $x$ (m)")
    ax2.set_ylabel("Elevation $y$ (m)")
    ax2.set_title(
        "Transient Migration of Phreatic Surface ($u(x, y, t) = y$) Across the Five-Layer Dam"
    )
    ax2.set_aspect("equal")
    ax2.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
        fontsize=8.8,
        frameon=True,
        framealpha=0.94,
    )

    p2 = out / "tr_potential_phreatic_evolution.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p2)

    # =========================================================================
    # Figure 3: (a) Piezometer Histories & Steady Decay, (b) O(dt^2) Verification
    # =========================================================================
    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(14.5, 4.3), layout="constrained")
    t_days = cn_result.times / SECONDS_PER_DAY

    # Panel (a): Nodal piezometer head evolution
    for label, hist in cn_result.probe_histories.items():
        idx = cn_result.probe_indices[label]
        line = ax3a.plot(t_days, hist, linewidth=1.85, label=label)[0]
        ax3a.axhline(
            float(system.u_steady[idx]),
            color=line.get_color(),
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
    ax3a.set_xlim(0.0, 500.0)
    ax3a.set_ylim(-0.2, 8.3)
    ax3a.set_xlabel("Time $t$ (days)")
    ax3a.set_ylabel("Hydraulic head $u(t)$ (m)")
    ax3a.set_title("(a) Piezometer Evolution (Dotted: Steady Limit)")
    ax3a.legend(
        fontsize=7.5,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.16),
        frameon=True,
        framealpha=0.95,
    )

    # Panel (b): Semilog convergence toward st_potential1.py stationary solution
    ax3b.semilogy(
        t_days,
        np.maximum(cn_result.linf_to_steady, 1e-16),
        color="#1f77b4",
        linewidth=2.1,
        label="$\\|\\boldsymbol{u}(t) - \\boldsymbol{u}_{\\mathrm{st}}\\|_\\infty$",
    )
    ax3b.semilogy(
        t_days,
        np.maximum(cn_result.rmse_to_steady, 1e-16),
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="$\\mathrm{RMSE}(\\boldsymbol{u}(t), \\boldsymbol{u}_{\\mathrm{st}})$",
    )
    ax3b.set_xlabel("Time $t$ (days)")
    ax3b.set_ylabel("Departure from stationary head (m)")
    ax3b.set_title("(b) Exponential Relaxation to Stationary Solution")
    ax3b.legend(fontsize=8.8, frameon=True)

    # Panel (c): Log-log O(dt^2) temporal convergence verification
    dt_h = order_result.dt_values / 3600.0
    ax3c.loglog(
        dt_h,
        order_result.errors_linf,
        "o-",
        color="#2ca02c",
        linewidth=2.0,
        markersize=6,
        label="$L_\\infty$ temporal error",
    )
    ax3c.loglog(
        dt_h,
        order_result.errors_l2,
        "s--",
        color="#9467bd",
        linewidth=2.0,
        markersize=6,
        label=f"$L_2$ error (slope $= {order_result.mean_order_l2:.2f}$)",
    )
    ref_line = order_result.errors_l2[0] * (dt_h / dt_h[0]) ** 2
    ax3c.loglog(
        dt_h,
        ref_line,
        ":",
        color="0.25",
        linewidth=1.6,
        label="Reference $\\mathcal{O}(\\Delta t^2)$",
    )
    ax3c.set_xlabel("Time step $\\Delta t$ (hours)")
    ax3c.set_ylabel("Temporal discretization error at $T = 2$ d (m)")
    ax3c.set_title("(c) Crank-Nicolson $\\mathcal{O}(\\Delta t^2)$ Order Verification")
    ax3c.legend(fontsize=8.5, frameon=True)

    p3 = out / "tr_potential_convergence.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    saved.append(p3)

    return saved


def main() -> None:
    """Execute the Crank-Nicolson transient potential dam seepage pipeline."""
    print("======================================================================")
    print("GFDFlow — Transient Potential Dam Seepage (Crank-Nicolson, S_s = 1)")
    print("======================================================================")

    mesh_path = Path("examples/paper/mesh1.json")
    print(f"1. Loading precomputed mesh and stencils from: {mesh_path}")
    mesh_data = load_mesh_data(mesh_path)
    system = assemble_dam_system(mesh_data)
    print(
        f"   Mesh: {system.coords.shape[0]} nodes, {system.triangles.shape[0]} triangles "
        f"({len(system.dirichlet_nodes)} Dirichlet, "
        f"{system.coords.shape[0] - len(system.dirichlet_nodes)} free)."
    )
    print(
        f"   Stationary target range (`st_potential1.py`): "
        f"[{system.u_steady.min():.4f}, {system.u_steady.max():.4f}] m "
        f"(mean = {system.u_steady.mean():.4f} m)."
    )

    s_s = 1.0
    print(f"2. Integrating S_s * du/dt = div(k grad u) with S_s = {s_s:.1f} via Crank-Nicolson...")
    cn_result = run_transient_dam_simulation(system, s_s=s_s)
    print(f"   Completed {cn_result.n_solves} sparse linear solves (`spsolve`).")
    print(
        f"   Global head bounds across all steps: "
        f"min = {cn_result.min_head:.6f} m, max = {cn_result.max_head:.6f} m."
    )
    print(
        f"   Final departure from `st_potential1.py` at t = {cn_result.times[-1] / SECONDS_PER_DAY:.0f} d: "
        f"L_inf = {cn_result.linf_to_steady[-1]:.3e} m, "
        f"RMSE = {cn_result.rmse_to_steady[-1]:.3e} m."
    )

    print("3. Verifying second-order O(dt^2) temporal accuracy of Crank-Nicolson...")
    order_result = verify_temporal_order(system, s_s=s_s)
    for dt_s, e_inf, e_l2 in zip(
        order_result.dt_values, order_result.errors_linf, order_result.errors_l2
    ):
        print(
            f"   dt = {dt_s:7.1f} s ({dt_s / 3600.0:5.2f} h) -> "
            f"L_inf = {e_inf:.3e} m, L_2 = {e_l2:.3e} m"
        )
    print(
        f"   Observed pairwise L_2 orders: "
        f"{np.array2string(order_result.orders_l2, precision=3)} "
        f"(mean log-log slope = {order_result.mean_order_l2:.3f})."
    )

    fig_dir = Path("examples/paper/figures")
    print(f"4. Rendering publication figures to: {fig_dir}")
    saved_figs = save_transient_figures(system, cn_result, order_result, output_dir=fig_dir)
    for fig_path in saved_figs:
        print(f"   Saved: {fig_path}")

    print("======================================================================")
    print("Transient Crank-Nicolson simulation completed successfully.")
    print("======================================================================")


if __name__ == "__main__":
    main()
