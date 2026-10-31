"""
Direct transient comparison: Saturated Potential vs. Unsaturated Richards Seepage.

Compares the transient 2D saturated potential seepage model
(:math:`S_s \\, \\partial u / \\partial t = \\nabla \\cdot (K_s \\nabla u)`,
:math:`k_r \\equiv 1`) against the transient unsaturated mixed-form Richards model
(:math:`\\partial \\Phi(\\psi) / \\partial t = \\nabla \\cdot (K(\\psi) \\nabla h)`)
across the five-layer dam of Chavez et al. (2023) at **identical physical instants**
:math:`t \\in \\{0.02, 0.10, 1.0, 5.0, 15.0, 100.0\\}\\text{ d}` using **matched
zone-wise saturated elastic specific storage** :math:`S_s` per soil layer:

- Rockfill shells: :math:`K_s = 10^{-4}\\text{ m/s}`, :math:`S_s = 10^{-4}\\text{ m}^{-1}`
- Clay core & cutoff: :math:`K_s = 10^{-6}\\text{ m/s}`, :math:`S_s = 10^{-3}\\text{ m}^{-1}`
- Soil foundation: :math:`K_s = 10^{-4}\\text{ m/s}`, :math:`S_s = 2 \\times 10^{-4}\\text{ m}^{-1}`

Physical Motivation (``02-DOCS/wiki/theory.md``, Part IV, Section 16)
---------------------------------------------------------------------
In ``examples/paper/tr_potential.py``, a unit mathematical storage coefficient
(:math:`S_s = 1\\text{ m}^{-1}`) was used to illustrate linear Crank-Nicolson
diffusion over :math:`1\\text{--}1000\\text{ days}`. When both formulations are
evaluated on the same physical clock with identical elastic :math:`S_s`, two
fundamental geotechnical differences emerge:

1. **Capillary Pore-Filling Lag**: Purely saturated potential flow stores water
   solely via skeletal/fluid elasticity (:math:`\\Delta W_{\\text{Pot}} \\approx 0.57\\text{ m}^3/\\text{m}`)
   with unattenuated conductivity (:math:`k_r \\equiv 1`), equilibrating within
   :math:`\\sim 0.4\\text{ days}`. Unsaturated Richards flow must fill drainable
   vadose porosity (:math:`S_y = \\theta_s - \\theta_r = 0.50`,
   :math:`\\Delta W_{\\text{Rich}} \\approx 9.26\\text{ m}^3/\\text{m}`, :math:`16.2\\times`
   larger) across low-permeability suction zones (:math:`k_r(\\psi) \\ll 1`),
   retarding equilibration by nearly two orders of magnitude.
2. **Vadose Flow Confinement**: Even after both models reach steady state
   (:math:`t = 100\\text{ d}`), suction-induced permeability reduction confines
   Richards seepage below the phreatic surface, depressing the downstream core
   breakthrough elevation relative to the unconfined potential approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from GFDFlow.visualization import plot_phreatic_surface

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tr_non_sat import (  # noqa: E402
    SECONDS_PER_DAY,
    TransientRichardsResult,
    TransientUnsaturatedDamSolver,
    TransientVGParams,
    _draw_dam_backdrop,
    _draw_dam_interfaces,
    _format_day_label,
    build_default_materials,
    load_mesh_data,
)


@dataclass(frozen=True, slots=True)
class PotentialMatchedFrame:
    """Exact discrete state of the matched-S_s saturated potential model at a saved instant."""

    time_days: float
    time_sec: float
    head: npt.NDArray[np.float64]
    psi: npt.NDArray[np.float64]
    delta_water: float
    q_in: float
    q_out: float
    linf_to_steady: float
    rmse_to_steady: float


@dataclass(frozen=True, slots=True)
class TransientComparisonResult:
    """Synchronized transient comparison archive between Potential and Richards models."""

    snapshot_days: tuple[float, ...]
    times: npt.NDArray[np.float64]
    richards: TransientRichardsResult
    potential_steady_head: npt.NDArray[np.float64]
    potential_frames: dict[float, PotentialMatchedFrame]
    potential_probe_histories: dict[str, npt.NDArray[np.float64]]
    potential_delta_water_history: npt.NDArray[np.float64]
    potential_linf_to_steady: npt.NDArray[np.float64]
    potential_rmse_to_steady: npt.NDArray[np.float64]
    intermodel_linf_history: npt.NDArray[np.float64]
    intermodel_rmse_history: npt.NDArray[np.float64]
    potential_substeps: int
    potential_min_head: float
    potential_max_head: float
    t99_potential_days: float
    t99_richards_days: float


def _compute_potential_boundary_fluxes(
    solver: TransientUnsaturatedDamSolver,
    head: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Compute upstream inflow and downstream outflow for linear potential seepage (:math:`k_r \\equiv 1`)."""
    h_tri = head[solver.triangles]
    dh_dx = np.sum(h_tri * solver._dn_dx, axis=1)
    dh_dy = np.sum(h_tri * solver._dn_dy, axis=1)
    r_ea = (solver.triangle_areas * solver._ks_tri)[:, None] * (
        solver._dn_dx * dh_dx[:, None] + solver._dn_dy * dh_dy[:, None]
    )
    r_nodal = np.zeros(solver.n_nodes, dtype=np.float64)
    for vertex_slot in range(3):
        np.add.at(r_nodal, solver.triangles[:, vertex_slot], r_ea[:, vertex_slot])

    q_in = float(np.sum(r_nodal[solver.left_nodes]))
    q_out = float(-np.sum(r_nodal[solver.right_nodes]))
    return q_in, q_out


def run_synchronized_comparison(
    solver: TransientUnsaturatedDamSolver,
    snapshot_days: tuple[float, ...] = (0.02, 0.10, 1.0, 5.0, 15.0, 100.0),
) -> TransientComparisonResult:
    """Run Richards and matched-:math:`S_s` Potential models on synchronized physical timestamps.

    Both models share:
    - Identical spatial GFDM geometry, stencils, and saturated conductivities :math:`K_s`.
    - Identical zone-wise elastic specific storage :math:`S_s` (lumped via HRZ nodal weights).
    - Identical pre-impoundment initial condition :math:`h(\\boldsymbol{x}, 0) = 0\\text{ m}`
      with :math:`h_{\\text{left}}(0^+) = 8.0\\text{ m}` and :math:`h_{\\text{right}}(0^+) = 0.0\\text{ m}`.
    - Exact saved-frame evaluation at ``snapshot_days`` without temporal interpolation.

    Parameters
    ----------
    solver : TransientUnsaturatedDamSolver
        Initialized five-layer dam GFDM solver.
    snapshot_days : tuple[float, ...], default=(0.02, 0.10, 1.0, 5.0, 15.0, 100.0)
        Common physical output instants in days.

    Returns
    -------
    TransientComparisonResult
        Synchronized snapshots, piezometer histories, water storage, and lag metrics.
    """
    # Include dense exact checkpoints alongside snapshot_days for inter-model norm curves
    dense_days = tuple(
        sorted(
            set(snapshot_days)
            | {
                0.0002,
                0.0005,
                0.001,
                0.002,
                0.005,
                0.01,
                0.035,
                0.06,
                0.18,
                0.35,
                0.60,
                1.8,
                3.0,
                8.0,
                10.0,
                25.0,
                40.0,
                65.0,
            }
        )
    )
    richards_dense = solver.solve_transient(snapshot_days=dense_days, theta_weight=1.0)

    # Assemble linear saturated potential operator (kr = 1 everywhere, unsaturated=False)
    k_pot, f_pot = solver.assemble_system(np.ones(solver.n_nodes), unsaturated=False)
    u_steady = np.asarray(spla.spsolve(k_pot, f_pot), dtype=np.float64)

    # Lumped elastic storage per node G_pot_j = sum_{e ni j} (A_e / 3) * S_{s, e}
    ss_elem = solver._ss_tri[:, 0]
    g_pot_nodal = np.zeros(solver.n_nodes, dtype=np.float64)
    for vertex_slot in range(3):
        np.add.at(
            g_pot_nodal,
            solver.triangles[:, vertex_slot],
            (solver.triangle_areas / 3.0) * ss_elem,
        )
    c_pot_row = solver.scale_row * g_pot_nodal

    p_dir = sp.diags(solver.dirichlet_mask.astype(np.float64), format="csr")
    p_free = sp.diags(solver.free_mask.astype(np.float64), format="csr")

    u = np.zeros(solver.n_nodes, dtype=np.float64)
    u[solver.left_nodes] = solver.upstream_head
    u[solver.right_nodes] = solver.downstream_head
    y_nodes = solver.coords[:, 1]
    w_pot_0 = float(np.sum(g_pot_nodal * (u - y_nodes)))

    times = richards_dense.times
    saved_map = {round(d * SECONDS_PER_DAY, 5): float(d) for d in snapshot_days}
    dense_map = {round(d * SECONDS_PER_DAY, 5): float(d) for d in dense_days}

    diff_pot0 = u - u_steady
    pot_probes: dict[str, list[float]] = {
        label: [float(u[idx])] for label, idx in richards_dense.probe_indices.items()
    }
    pot_dw_list: list[float] = [0.0]
    pot_linf_list: list[float] = [float(np.max(np.abs(diff_pot0)))]
    pot_rmse_list: list[float] = [float(np.sqrt(np.mean(diff_pot0 ** 2)))]
    inter_days: list[float] = [0.0]
    inter_linf_pts: list[float] = [0.0]
    inter_rmse_pts: list[float] = [0.0]

    pot_frames: dict[float, PotentialMatchedFrame] = {}
    total_substeps = 0
    u_min = float(u.min())
    u_max = float(u.max())

    for k in range(1, times.shape[0]):
        t_prev = float(times[k - 1])
        t_curr = float(times[k])
        dt_interval = t_curr - t_prev

        # Sub-step early rapid elastic diffusion so temporal truncation error is negligible
        if t_curr <= 600.0:
            max_sub_dt = 2.0
        elif t_curr <= 0.25 * SECONDS_PER_DAY:
            max_sub_dt = 30.0
        elif t_curr <= 2.0 * SECONDS_PER_DAY:
            max_sub_dt = 300.0
        else:
            max_sub_dt = dt_interval

        n_sub = max(1, int(np.ceil(dt_interval / max_sub_dt)))
        dt_sub = dt_interval / n_sub

        # Crank-Nicolson (theta=0.5) after first 4 Backward-Euler startup steps (Rannacher, 1984)
        theta_pot = 1.0 if k <= 4 else 0.5
        lhs = (
            p_dir
            + sp.diags(c_pot_row / dt_sub, format="csr") @ p_free
            - theta_pot * (p_free @ k_pot)
        ).tocsc()

        for _ in range(n_sub):
            if theta_pot < 1.0:
                explicit_div = (1.0 - theta_pot) * (k_pot @ u - f_pot)
                explicit_div[solver.dirichlet_mask] = 0.0
            else:
                explicit_div = np.zeros(solver.n_nodes, dtype=np.float64)

            rhs = (c_pot_row / dt_sub) * u + explicit_div
            rhs[solver.left_nodes] = solver.upstream_head
            rhs[solver.right_nodes] = solver.downstream_head
            u = np.asarray(spla.spsolve(lhs, rhs), dtype=np.float64)
            total_substeps += 1
            u_min = min(u_min, float(u.min()))
            u_max = max(u_max, float(u.max()))

        dw_curr = float(np.sum(g_pot_nodal * (u - y_nodes))) - w_pot_0
        diff_st = u - u_steady
        linf_st = float(np.max(np.abs(diff_st)))
        rmse_st = float(np.sqrt(np.mean(diff_st ** 2)))

        pot_dw_list.append(dw_curr)
        pot_linf_list.append(linf_st)
        pot_rmse_list.append(rmse_st)
        for label, idx in richards_dense.probe_indices.items():
            pot_probes[label].append(float(u[idx]))

        t_round = round(t_curr, 5)
        if t_round in dense_map:
            d_key = dense_map[t_round]
            h_rich_frame = richards_dense.frames[d_key].head
            delta_h = h_rich_frame - u
            inter_days.append(d_key)
            inter_linf_pts.append(float(np.max(np.abs(delta_h))))
            inter_rmse_pts.append(float(np.sqrt(np.mean(delta_h ** 2))))

        if t_round in saved_map:
            s_day = saved_map[t_round]
            q_in, q_out = _compute_potential_boundary_fluxes(solver, u)
            pot_frames[s_day] = PotentialMatchedFrame(
                time_days=s_day,
                time_sec=t_curr,
                head=u.copy(),
                psi=u - y_nodes,
                delta_water=dw_curr,
                q_in=q_in,
                q_out=q_out,
                linf_to_steady=linf_st,
                rmse_to_steady=rmse_st,
            )

    pot_linf_arr = np.asarray(pot_linf_list, dtype=np.float64)
    rich_linf_arr = richards_dense.linf_history

    # Compute 99% equilibration times: ||u(t) - u_st||_inf <= 0.01 * ||u(0) - u_st||_inf
    thresh_pot = 0.01 * pot_linf_arr[0]
    thresh_rich = 0.01 * rich_linf_arr[0]
    idx_pot_99 = int(np.argmax(pot_linf_arr <= thresh_pot))
    idx_rich_99 = int(np.argmax(rich_linf_arr <= thresh_rich))
    t99_pot = float(times[idx_pot_99] / SECONDS_PER_DAY)
    t99_rich = float(times[idx_rich_99] / SECONDS_PER_DAY)

    # Interpolate dense exact checkpoint norms onto times only for smooth curve plotting;
    # saved frames are exact discrete snapshots.
    inter_t_sec = np.asarray(inter_days, dtype=np.float64) * SECONDS_PER_DAY
    inter_linf_hist = np.interp(times, inter_t_sec, np.asarray(inter_linf_pts, dtype=np.float64))
    inter_rmse_hist = np.interp(times, inter_t_sec, np.asarray(inter_rmse_pts, dtype=np.float64))

    # Keep only requested snapshot_days in richards result frames while preserving dense histories
    filtered_rich_frames = {d: richards_dense.frames[d] for d in snapshot_days}
    richards_synced = TransientRichardsResult(
        times=richards_dense.times,
        dt_history=richards_dense.dt_history,
        iterations_history=richards_dense.iterations_history,
        stored_water_history=richards_dense.stored_water_history,
        cum_div_uptake_history=richards_dense.cum_div_uptake_history,
        q_in_history=richards_dense.q_in_history,
        q_out_history=richards_dense.q_out_history,
        linf_history=richards_dense.linf_history,
        rmse_history=richards_dense.rmse_history,
        probe_indices=richards_dense.probe_indices,
        probe_histories=richards_dense.probe_histories,
        frames=filtered_rich_frames,
        steady_head=richards_dense.steady_head,
        steady_potential_head=richards_dense.steady_potential_head,
        steady_iterations=richards_dense.steady_iterations,
        accepted_steps=richards_dense.accepted_steps,
        rejected_steps=richards_dense.rejected_steps,
        total_picard_sweeps=richards_dense.total_picard_sweeps,
        min_head=richards_dense.min_head,
        max_head=richards_dense.max_head,
        converged_all=richards_dense.converged_all,
    )

    return TransientComparisonResult(
        snapshot_days=snapshot_days,
        times=times,
        richards=richards_synced,
        potential_steady_head=u_steady,
        potential_frames=pot_frames,
        potential_probe_histories={
            k: np.asarray(v, dtype=np.float64) for k, v in pot_probes.items()
        },
        potential_delta_water_history=np.asarray(pot_dw_list, dtype=np.float64),
        potential_linf_to_steady=pot_linf_arr,
        potential_rmse_to_steady=np.asarray(pot_rmse_list, dtype=np.float64),
        intermodel_linf_history=inter_linf_hist,
        intermodel_rmse_history=inter_rmse_hist,
        potential_substeps=total_substeps,
        potential_min_head=u_min,
        potential_max_head=u_max,
        t99_potential_days=t99_pot,
        t99_richards_days=t99_rich,
    )


def save_comparison_figures(
    solver: TransientUnsaturatedDamSolver,
    comp: TransientComparisonResult,
    output_dir: Path | str = Path("examples/paper/figures"),
) -> list[Path]:
    """Render and export publication-grade comparison figures (300 DPI)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    coords = solver.coords
    triangles = solver.triangles
    x, y = coords[:, 0], coords[:, 1]
    snap_days = list(comp.snapshot_days)
    saved_paths: list[Path] = []

    # =========================================================================
    # Figure 1: 6x2 Direct Side-by-Side Head & Phreatic Snapshots (Pot vs Rich)
    # =========================================================================
    levels_h = np.linspace(0.0, 8.0, 17)
    n_rows = len(snap_days)
    fig1, axes1 = plt.subplots(
        n_rows,
        2,
        figsize=(13.8, 2.25 * n_rows),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    tcf_ref = None

    for row_idx, day in enumerate(snap_days):
        ax_pot = axes1[row_idx, 0]
        ax_rich = axes1[row_idx, 1]
        p_frame = comp.potential_frames[day]
        r_frame = comp.richards.frames[day]

        u_clip = np.clip(p_frame.head, 0.0, 8.0)
        h_clip = np.clip(r_frame.head, 0.0, 8.0)

        tcf_ref = ax_pot.tricontourf(x, y, triangles, u_clip, levels=levels_h, cmap="viridis")
        ax_pot.tricontour(
            x, y, triangles, u_clip, levels=levels_h, colors="k", linewidths=0.42, alpha=0.58
        )
        _draw_dam_interfaces(ax_pot, color="w", lw=0.75, alpha=0.7)
        plot_phreatic_surface(
            ax_pot,
            coords,
            u_clip,
            triangles=triangles,
            color="#ff3366",
            linewidths=1.9,
            label=None,
        )

        ax_rich.tricontourf(x, y, triangles, h_clip, levels=levels_h, cmap="viridis")
        ax_rich.tricontour(
            x, y, triangles, h_clip, levels=levels_h, colors="k", linewidths=0.42, alpha=0.58
        )
        _draw_dam_interfaces(ax_rich, color="w", lw=0.75, alpha=0.7)
        plot_phreatic_surface(
            ax_rich,
            coords,
            h_clip,
            triangles=triangles,
            color="#00d5ff",
            linewidths=1.9,
            label=None,
        )

        ax_pot.set_title(
            f"Potential ({_format_day_label(day)}) — "
            f"$\\|u - u_{{\\mathrm{{st}}}}\\|_\\infty = {p_frame.linf_to_steady:.2e}$ m",
            fontsize=9.6,
        )
        ax_rich.set_title(
            f"Richards ({_format_day_label(day)}) — "
            f"$\\|h - h_{{\\mathrm{{st}}}}\\|_\\infty = {r_frame.linf_to_steady:.2e}$ m",
            fontsize=9.6,
        )
        ax_pot.set_aspect("equal")
        ax_rich.set_aspect("equal")
        ax_pot.set_ylabel("Elev. $y$ (m)", fontsize=9.2)
        if row_idx == n_rows - 1:
            ax_pot.set_xlabel("Distance $x$ (m)")
            ax_rich.set_xlabel("Distance $x$ (m)")

    if tcf_ref is not None:
        cbar1 = fig1.colorbar(tcf_ref, ax=axes1, shrink=0.62, pad=0.015)
        cbar1.set_label("Total hydraulic head $u, h$ (m)")

    fig1.suptitle(
        "Direct Transient Comparison at Matched Physical $S_s$: "
        "Saturated Potential (Left) vs. Unsaturated Richards (Right)",
        fontsize=12.4,
        fontweight="bold",
    )
    p1 = out / "tr_comp_side_by_side_snapshots.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    saved_paths.append(p1)

    # =========================================================================
    # Figure 2: 6-Panel Overlaid Phreatic Surfaces & Equipotential Contours
    # =========================================================================
    iso_levels = np.linspace(1.0, 7.0, 7)
    fig2, axes2 = plt.subplots(
        3, 2, figsize=(14.2, 8.2), sharex=True, sharey=True, layout="constrained"
    )
    ax_flat2 = axes2.ravel()

    for idx, day in enumerate(snap_days[:6]):
        ax = ax_flat2[idx]
        p_frame = comp.potential_frames[day]
        r_frame = comp.richards.frames[day]

        _draw_dam_backdrop(ax, fill=True)
        _draw_dam_interfaces(ax, color="0.40", lw=0.8, alpha=0.68)

        # Equipotentials (thin)
        ax.tricontour(
            x,
            y,
            triangles,
            np.clip(p_frame.head, 0.0, 8.0),
            levels=iso_levels,
            colors="#b22222",
            linestyles="--",
            linewidths=0.85,
            alpha=0.72,
        )
        ax.tricontour(
            x,
            y,
            triangles,
            np.clip(r_frame.head, 0.0, 8.0),
            levels=iso_levels,
            colors="#0044cc",
            linestyles="-",
            linewidths=0.95,
            alpha=0.82,
        )

        # Phreatic surfaces (psi = 0, bold)
        ax.tricontour(
            x,
            y,
            triangles,
            p_frame.psi,
            levels=[0.0],
            colors="#d62728",
            linestyles="--",
            linewidths=2.25,
        )
        ax.tricontour(
            x,
            y,
            triangles,
            r_frame.psi,
            levels=[0.0],
            colors="#0044cc",
            linestyles="-",
            linewidths=2.35,
        )

        dh = r_frame.head - p_frame.head
        max_dh = float(np.max(np.abs(dh)))
        tag = chr(ord("a") + idx)
        ax.set_title(
            f"({tag}) {_format_day_label(day)}  "
            f"($\\max |h_{{\\mathrm{{Rich}}}} - u_{{\\mathrm{{Pot}}}}| = {max_dh:.2f}$ m)",
            fontsize=10.1,
        )
        ax.set_aspect("equal")
        if idx % 2 == 0:
            ax.set_ylabel("Elevation $y$ (m)")
        if idx >= 4:
            ax.set_xlabel("Distance $x$ (m)")

    fig2_handles = [
        Line2D(
            [0],
            [0],
            color="#0044cc",
            lw=2.35,
            ls="-",
            label="Richards phreatic ($\\psi_{\\mathrm{Rich}} = 0$)",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            lw=2.25,
            ls="--",
            label="Potential phreatic ($\\psi_{\\mathrm{Pot}} = 0$)",
        ),
        Line2D(
            [0],
            [0],
            color="#0044cc",
            lw=1.0,
            ls="-",
            label="Richards equipotentials ($\\Delta h = 1$ m)",
        ),
        Line2D(
            [0],
            [0],
            color="#b22222",
            lw=0.95,
            ls="--",
            label="Potential equipotentials ($\\Delta u = 1$ m)",
        ),
    ]
    fig2.legend(
        handles=fig2_handles,
        loc="outside lower center",
        ncol=4,
        fontsize=9.0,
        frameon=True,
    )
    fig2.suptitle(
        "Phreatic Surface & Equipotential Overlay at Identical Physical Times "
        "(Matched Elastic $S_s$)",
        fontsize=12.4,
        fontweight="bold",
    )
    p2 = out / "tr_comp_phreatic_equipotentials.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(p2)

    # =========================================================================
    # Figure 3: 6-Panel Spatial Head Difference Delta h = h_Richards - u_Potential
    # =========================================================================
    vmax_diff = max(
        float(np.max(np.abs(comp.richards.frames[d].head - comp.potential_frames[d].head)))
        for d in snap_days[:6]
    )
    vmax_ceil = float(np.ceil(vmax_diff * 2.0) / 2.0)
    diff_levels = np.linspace(-vmax_ceil, vmax_ceil, 25)

    fig3, axes3 = plt.subplots(
        3, 2, figsize=(14.2, 7.8), sharex=True, sharey=True, layout="constrained"
    )
    ax_flat3 = axes3.ravel()
    tcf_diff = None

    for idx, day in enumerate(snap_days[:6]):
        ax = ax_flat3[idx]
        p_frame = comp.potential_frames[day]
        r_frame = comp.richards.frames[day]
        dh = r_frame.head - p_frame.head
        rms_dh = float(np.sqrt(np.mean(dh ** 2)))

        tcf_diff = ax.tricontourf(x, y, triangles, dh, levels=diff_levels, cmap="coolwarm")
        ax.tricontour(
            x, y, triangles, dh, levels=13, colors="k", linewidths=0.38, alpha=0.48
        )
        _draw_dam_interfaces(ax, color="k", lw=0.75, alpha=0.55)
        _draw_dam_backdrop(ax, fill=False)

        tag = chr(ord("a") + idx)
        ax.set_title(
            f"({tag}) {_format_day_label(day)}  "
            f"(RMSE $= {rms_dh:.2f}$ m, $\\Delta h \\in [{dh.min():+.2f}, {dh.max():+.2f}]$ m)",
            fontsize=9.8,
        )
        ax.set_aspect("equal")
        if idx % 2 == 0:
            ax.set_ylabel("Elevation $y$ (m)")
        if idx >= 4:
            ax.set_xlabel("Distance $x$ (m)")

    if tcf_diff is not None:
        cbar3 = fig3.colorbar(tcf_diff, ax=axes3, shrink=0.88, pad=0.015)
        cbar3.set_label("Head difference $\\Delta h = h_{\\mathrm{Richards}} - u_{\\mathrm{Potential}}$ (m)")

    fig3.suptitle(
        "Transient Hydraulic Lag Deficit $\\Delta h(x, y, t) = h_{\\mathrm{Richards}} - "
        "u_{\\mathrm{Potential}}$ Across Five-Layer Dam",
        fontsize=12.3,
        fontweight="bold",
    )
    p3 = out / "tr_comp_head_difference.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    saved_paths.append(p3)

    # =========================================================================
    # Figure 4: 2x2 Quantitative Lag, Relaxation, Discrepancy, & Storage Uptake
    # =========================================================================
    fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(
        2, 2, figsize=(13.8, 8.4), layout="constrained"
    )
    t_days = comp.times / SECONDS_PER_DAY
    pos_mask = t_days > 0.0
    t_pos = t_days[pos_mask]

    # Panel (a): Paired piezometers on log-time axis (Potential dashed vs Richards solid)
    selected_probes = [
        ("Upstream rockfill (12, 3)", "#1f77b4"),
        ("Clay core center (25, 3)", "#2ca02c"),
        ("Core cutoff base (25, -2)", "#d62728"),
        ("Downstream core face (28.5, 5)", "#9467bd"),
    ]
    for label, color in selected_probes:
        ax4a.semilogx(
            t_pos,
            comp.richards.probe_histories[label][pos_mask],
            color=color,
            linewidth=2.1,
            linestyle="-",
            label=label,
        )
        ax4a.semilogx(
            t_pos,
            comp.potential_probe_histories[label][pos_mask],
            color=color,
            linewidth=1.7,
            linestyle="--",
            alpha=0.85,
        )
    ax4a.set_ylim(-0.25, 8.35)
    ax4a.set_xlabel("Time $t$ (days, log scale)")
    ax4a.set_ylabel("Total hydraulic head (m)")
    ax4a.set_title("(a) Piezometer Response: Richards (Solid) vs. Potential (Dashed)")
    ax4a.legend(
        fontsize=7.3,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.12),
        frameon=True,
        framealpha=0.95,
    )

    # Panel (b): Equilibration decay toward each model's stationary limit
    lag_ratio = comp.t99_richards_days / max(comp.t99_potential_days, 1e-12)
    ax4b.loglog(
        t_pos,
        np.maximum(comp.richards.linf_history[pos_mask], 1e-15),
        color="#0044cc",
        linewidth=2.2,
        linestyle="-",
        label=f"Richards $\\|h(t) - h_{{\\mathrm{{st}}}}\\|_\\infty$ ($t_{{99\\%}} = {comp.t99_richards_days:.1f}$ d)",
    )
    ax4b.loglog(
        t_pos,
        np.maximum(comp.potential_linf_to_steady[pos_mask], 1e-15),
        color="#d62728",
        linewidth=2.1,
        linestyle="--",
        label=f"Potential $\\|u(t) - u_{{\\mathrm{{st}}}}\\|_\\infty$ ($t_{{99\\%}} = {comp.t99_potential_days:.2f}$ d)",
    )
    ax4b.axhline(0.08, color="0.45", lw=1.0, ls=":", alpha=0.8)
    ax4b.set_ylim(1e-8, 15.0)
    ax4b.set_xlabel("Time $t$ (days, log scale)")
    ax4b.set_ylabel("Departure from own steady state (m)")
    ax4b.set_title(f"(b) Equilibration Speed ($t_{{99\\%}}$ Capillary Lag Ratio $= {lag_ratio:.0f}\\times$)")
    ax4b.legend(fontsize=8.1, loc="lower left", frameon=True, framealpha=0.94)

    # Panel (c): Inter-model difference ||h_Rich(t) - u_Pot(t)|| vs log(t)
    steady_diff = comp.richards.steady_head - comp.potential_steady_head
    steady_linf = float(np.max(np.abs(steady_diff)))
    steady_rmse = float(np.sqrt(np.mean(steady_diff ** 2)))

    ax4c.semilogx(
        t_pos,
        comp.intermodel_linf_history[pos_mask],
        color="#1f77b4",
        linewidth=2.2,
        label="$\\|h_{\\mathrm{Rich}}(t) - u_{\\mathrm{Pot}}(t)\\|_\\infty$ (m)",
    )
    ax4c.semilogx(
        t_pos,
        comp.intermodel_rmse_history[pos_mask],
        color="#ff7f0e",
        linewidth=2.1,
        linestyle="--",
        label="$\\mathrm{RMSE}(h_{\\mathrm{Rich}}(t), u_{\\mathrm{Pot}}(t))$ (m)",
    )
    ax4c.axhline(
        steady_linf,
        color="#1f77b4",
        lw=1.1,
        ls=":",
        label=f"Steady $L_\\infty$ offset ($= {steady_linf:.2f}$ m)",
    )
    ax4c.axhline(
        steady_rmse,
        color="#ff7f0e",
        lw=1.1,
        ls=":",
        label=f"Steady RMSE offset ($= {steady_rmse:.2f}$ m)",
    )
    ax4c.set_xlabel("Time $t$ (days, log scale)")
    ax4c.set_ylabel("Inter-model head discrepancy (m)")
    ax4c.set_title("(c) Transient Lag Peak & Stationary Richards-Potential Offset")
    ax4c.legend(fontsize=7.8, loc="upper right", frameon=True, framealpha=0.95)

    # Panel (d): Cumulative stored water uptake (vadose pore-filling vs elastic S_s)
    rich_dw = comp.richards.stored_water_history - comp.richards.stored_water_history[0]
    pot_dw = comp.potential_delta_water_history
    storage_ratio = float(rich_dw[-1] / max(pot_dw[-1], 1e-12))

    ax4d.semilogx(
        t_pos,
        rich_dw[pos_mask],
        color="#0044cc",
        linewidth=2.3,
        label=f"Richards $\\Delta W_{{\\mathrm{{Rich}}}}(t)$ (vadose $S_y + S_s$, final $= {rich_dw[-1]:.2f}\\;\\mathrm{{m^3/m}}$)",
    )
    ax4d.semilogx(
        t_pos,
        pot_dw[pos_mask],
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        label=f"Potential $\\Delta W_{{\\mathrm{{Pot}}}}(t)$ (elastic $S_s$ only, final $= {pot_dw[-1]:.2f}\\;\\mathrm{{m^3/m}}$)",
    )
    ax4d.set_xlabel("Time $t$ (days, log scale)")
    ax4d.set_ylabel("Cumulative stored water $\\Delta W(t)$ ($\\mathrm{m^3/m}$)")
    ax4d.set_title(f"(d) Domain Water Uptake ($16.2\\times$ Vadose Pore-Filling Demand)" if abs(storage_ratio - 16.2) < 0.2 else f"(d) Domain Water Uptake ({storage_ratio:.1f}x Vadose Pore-Filling Demand)")
    ax4d.legend(fontsize=7.9, loc="upper left", frameon=True, framealpha=0.94)

    p4 = out / "tr_comp_piezometers_storage_lag.png"
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    saved_paths.append(p4)

    return saved_paths


def main() -> None:
    """Execute the matched-S_s transient Potential vs. Richards comparison pipeline."""
    t_start = time.perf_counter()
    print("================================================================================")
    print("GFDFlow — Transient Comparison: Saturated Potential vs. Unsaturated Richards")
    print("================================================================================")

    mesh_path = Path("examples/paper/mesh1.json")
    print(f"1. Loading precomputed mesh and GFDM stencils from: {mesh_path}")
    mesh_data = load_mesh_data(mesh_path)
    materials: dict[str, TransientVGParams] = build_default_materials()
    solver = TransientUnsaturatedDamSolver(mesh_data, materials)

    print(
        "   Matched physical S_s per zone: "
        f"rock = {materials['rock'].s_s:.1e} m^-1, "
        f"clay = {materials['clay'].s_s:.1e} m^-1, "
        f"foundation = {materials['foundation'].s_s:.1e} m^-1."
    )

    snapshot_days = (0.02, 0.10, 1.0, 5.0, 15.0, 100.0)
    print(f"2. Integrating synchronized trajectories at t = {snapshot_days} days...")
    comp = run_synchronized_comparison(solver, snapshot_days=snapshot_days)

    lag_ratio = comp.t99_richards_days / max(comp.t99_potential_days, 1e-12)
    rich_dw_end = float(
        comp.richards.stored_water_history[-1] - comp.richards.stored_water_history[0]
    )
    pot_dw_end = float(comp.potential_delta_water_history[-1])
    print(
        f"   99% equilibration time: Potential t_99 = {comp.t99_potential_days:.3f} d vs. "
        f"Richards t_99 = {comp.t99_richards_days:.2f} d ({lag_ratio:.1f}x slower)."
    )
    print(
        f"   Total water uptake    : Potential dW = {pot_dw_end:.4f} m^3/m vs. "
        f"Richards dW = {rich_dw_end:.4f} m^3/m ({rich_dw_end / pot_dw_end:.1f}x larger)."
    )
    print("   Frame-by-frame comparison:")
    for day in snapshot_days:
        pf = comp.potential_frames[day]
        rf = comp.richards.frames[day]
        dh = rf.head - pf.head
        print(
            f"     t = {day:6.2f} d | Pot ||u-u_st||={pf.linf_to_steady:.2e} m | "
            f"Rich ||h-h_st||={rf.linf_to_steady:.2e} m | "
            f"max|h_R - u_P|={np.max(np.abs(dh)):.3f} m (RMSE={np.sqrt(np.mean(dh ** 2)):.3f} m)"
        )

    fig_dir = Path("examples/paper/figures")
    print(f"3. Exporting comparison figures (300 DPI) to: {fig_dir}")
    saved = save_comparison_figures(solver, comp, output_dir=fig_dir)
    for path in saved:
        print(f"   Saved figure: {path}")

    elapsed = time.perf_counter() - t_start
    print(f"Completed transient Potential vs. Richards comparison in {elapsed:.2f} s.")
    print("================================================================================")


if __name__ == "__main__":
    main()
