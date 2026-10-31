"""
Transient 2D unsaturated seepage in a five-layer embankment dam via GFDM.

Solves the non-stationary (transient) mixed-form Richards equation on the
five-layer dam-foundation system of Chavez et al. (2023) using the Generalized
Finite Difference Method (GFDM) coupled with the mass-conservative sub-element
storage, hybrid chord-tangent Picard linearization, and adaptive time-stepping
formulation documented in ``02-DOCS/wiki/theory.md`` (Part IV, Section 16;
Celia et al., 1990; XSLOPE Documentation, 2026b):

.. math::

    \\frac{\\partial \\Phi(\\psi)}{\\partial t}
    = \\nabla \\cdot \\left( K(\\psi) \\nabla h \\right),
    \\qquad \\psi = h - y,
    \\qquad K(\\psi) = K_s \\, k_r(\\psi),

where :math:`h(x, y, t)` is total hydraulic head (m), :math:`y` is elevation (m),
and :math:`\\Phi(\\psi)` is the zone-separated water storage density relative to
full saturation:

.. math::

    \\Phi(\\psi) = \\begin{cases}
        S_s \\, \\psi, & \\psi \\ge 0 \\quad (\\text{saturated elastic storage}), \\\\
        \\left(\\theta(\\psi) - \\theta_s\\right) + S_{\\text{floor}} \\, \\psi,
        & \\psi < 0 \\quad (\\text{vadose capillary drainage + floor}).
    \\end{cases}

Key Numerical Features
----------------------
1. **Vectorized Multi-Subdomain GFDM Spatial Operator**: Reproduces the exact
   interior logarithmic-gradient stars, ghost-eliminated impermeable Neumann
   boundaries, six flux-continuous two-layer interfaces, four boundary-interface
   corners, and two triple junctions of ``examples/paper/st_non_sat.py`` using
   batched tensor contractions.
2. **7-Point Sub-Element Quadrature & Chord-Capacity Splitting**: Integrates
   stored water :math:`W_e(t)` over all 2,778 ``tri3`` elements via degree-5
   Dunavant quadrature and partitions element water uptake :math:`\\Delta W_e`
   across vertices via strictly positive chord capacities :math:`G_{ea}`,
   preserving :math:`\\sum_j \\Delta W_j \\equiv \\sum_e \\Delta W_e` to machine
   precision while eliminating node-by-node phreatic snapping.
3. **Conservative Mixed Theta-Method & Envelope Linearization**: Uses Backward
   Euler (:math:`\\theta = 1.0`, :math:`L`-stable geotechnical default) with
   exact :math:`\\Delta W` on the right-hand side, diagonal Jacobian envelope
   :math:`\\tilde{C}_j = \\max(C_{\\text{chord}, j}, C_{\\text{tangent}, j})`,
   per-node sign-reversal damping, and a minimum two-sweep acceptance rule.
4. **Adaptive Time-Stepping & Saved-Frame Clamping**: Combines Backward-Euler
   local truncation error control, free-node RMS head-change limiting (excluding
   Dirichlet reservoir jumps), iteration-based growth/shrinkage, and exact
   clamping onto scheduled output instants.
5. **Transient Post-Processing**: Evaluates separate upstream inflow
   :math:`Q_{\\text{in}}(t)` and downstream outflow :math:`Q_{\\text{out}}(t)`,
   cumulative mixed-form mass-balance closure, and divergent Darcy velocity
   fields :math:`\\mathbf{v} = -K(\\psi)\\nabla h` (where
   :math:`\\nabla \\cdot \\mathbf{v} = -S(\\psi)\\partial h/\\partial t \\neq 0`
   precludes stream-function flow nets prior to steady state).

References
----------
- Celia, M. A., Bouloutas, E. T., & Zarba, R. L. (1990). A general
  mass-conservative numerical solution for the unsaturated flow equation.
  *Water Resources Research*, 26(7), 1483-1496.
- Chavez-Negrete, C., Dominguez-Mota, F. J., & Santana-Quintero, J. A. (2023).
  Finite differences applied to steady-state seepage in multi-layered earth dams.
- XSLOPE Documentation (2026). *Transient Seepage Theory*
  (``https://xslope.org/en/latest/seep/transient/``; summarized in
  ``02-DOCS/wiki/theory.md``, Part IV, Section 16).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from GFDFlow.visualization import plot_phreatic_surface

GAMMA_WATER_KPA_PER_M: float = 9.81
SECONDS_PER_DAY: float = 86_400.0


@dataclass(frozen=True, slots=True)
class TransientVGParams:
    """Van Genuchten-Mualem hydraulic and transient storage parameters.

    Combines the steady retention/permeability parameters of Chavez et al. (2023)
    with the zone-separated saturated specific storage :math:`S_s` and unsaturated
    residual storage floor :math:`S_{\\text{floor}}` of XSLOPE (``theory.md``,
    Section 16.3).

    Parameters
    ----------
    name : str
        Human-readable material zone identifier.
    k_sat : float
        Saturated isotropic hydraulic conductivity :math:`K_s` (m/s).
    theta_s : float
        Saturated volumetric water content :math:`\\theta_s` (dimensionless).
    theta_r : float
        Residual volumetric water content :math:`\\theta_r` (dimensionless).
    alpha_kpa : float
        Inverse air-entry parameter :math:`\\alpha` in :math:`\\text{kPa}^{-1}`.
    n : float
        Pore-size distribution exponent (:math:`n > 1`).
    s_s : float
        Saturated elastic specific storage :math:`S_s` (:math:`\\text{m}^{-1}`),
        active strictly for :math:`\\psi \\ge 0`.
    ell : float, default=0.5
        Mualem pore-connectivity exponent :math:`\\ell`.
    kr_min : float, default=1e-8
        Lower numerical bound on relative conductivity :math:`k_r(\\psi)`.
    floor_ratio : float, default=1e-4
        Ratio of unsaturated residual capacity floor :math:`S_{\\text{floor}}`
        to the analytical peak moisture capacity :math:`C_{\\max}`.
    """

    name: str
    k_sat: float
    theta_s: float
    theta_r: float
    alpha_kpa: float
    n: float
    s_s: float
    ell: float = 0.5
    kr_min: float = 1e-8
    floor_ratio: float = 1e-4

    @property
    def alpha_m(self) -> float:
        """Return van Genuchten :math:`\\alpha` in :math:`\\text{m}^{-1}`."""
        return self.alpha_kpa * GAMMA_WATER_KPA_PER_M

    @property
    def m(self) -> float:
        """Return Mualem exponent :math:`m = 1 - 1/n`."""
        return 1.0 - 1.0 / self.n

    @property
    def s_y(self) -> float:
        """Return drainable porosity (specific yield) :math:`S_y = \\theta_s - \\theta_r`."""
        return self.theta_s - self.theta_r

    @property
    def c_max(self) -> float:
        """Return analytical peak moisture capacity :math:`C_{\\max} = \\max_{\\psi < 0} d\\theta/d\\psi`."""
        # Exact inflection point of van Genuchten SWRC satisfies (alpha * |psi_peak|)^n = m
        u_peak = self.m
        num = self.alpha_m * self.m * self.n * self.s_y * (u_peak ** (1.0 - 1.0 / self.n))
        den = (1.0 + u_peak) ** (self.m + 1.0)
        return float(num / den)

    @property
    def s_floor(self) -> float:
        """Return unsaturated residual storage capacity floor :math:`S_{\\text{floor}}` (:math:`\\text{m}^{-1}`)."""
        return self.floor_ratio * self.c_max


@dataclass(frozen=True, slots=True)
class SavedFrame:
    """Exact discrete state archived at a scheduled transient output instant."""

    time_days: float
    time_sec: float
    head: npt.NDArray[np.float64]
    psi: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    kr: npt.NDArray[np.float64]
    darcy_vel: npt.NDArray[np.float64]
    stored_water: float
    delta_water: float
    q_in: float
    q_out: float
    linf_to_steady: float
    rmse_to_steady: float
    mass_balance_rel_error: float


@dataclass(frozen=True, slots=True)
class TransientRichardsResult:
    """Complete simulation record for the transient unsaturated five-layer dam."""

    times: npt.NDArray[np.float64]
    dt_history: npt.NDArray[np.float64]
    iterations_history: npt.NDArray[np.int_]
    stored_water_history: npt.NDArray[np.float64]
    cum_div_uptake_history: npt.NDArray[np.float64]
    q_in_history: npt.NDArray[np.float64]
    q_out_history: npt.NDArray[np.float64]
    linf_history: npt.NDArray[np.float64]
    rmse_history: npt.NDArray[np.float64]
    probe_indices: dict[str, int]
    probe_histories: dict[str, npt.NDArray[np.float64]]
    frames: dict[float, SavedFrame]
    steady_head: npt.NDArray[np.float64]
    steady_potential_head: npt.NDArray[np.float64]
    steady_iterations: int
    accepted_steps: int
    rejected_steps: int
    total_picard_sweeps: int
    min_head: float
    max_head: float
    converged_all: bool


@dataclass(frozen=True, slots=True)
class _GhostStencilOp:
    """Precomputed geometric GFDM operators for a boundary, interface, or wedge star."""

    center: int
    support: npt.NDArray[np.int_]
    normal: npt.NDArray[np.float64]
    k_sat: float
    mp_aug: npt.NDArray[np.float64]
    mp_lin: npt.NDArray[np.float64]
    flux_scaled: bool


@dataclass(frozen=True, slots=True)
class _InteriorBatch:
    """Vectorized batch of interior GFDM stars sharing identical support size."""

    centers: npt.NDArray[np.int_]
    ks_arr: npt.NDArray[np.float64]
    stencils: npt.NDArray[np.int_]
    mp_arr: npt.NDArray[np.float64]
    rows_flat: npt.NDArray[np.int_]
    cols_flat: npt.NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class _GhostBatch:
    """Vectorized batch of ghost-eliminated GFDM stars sharing identical support size."""

    centers: npt.NDArray[np.int_]
    ks_arr: npt.NDArray[np.float64]
    flux_scaled: npt.NDArray[np.bool_]
    supports: npt.NDArray[np.int_]
    mp_aug: npt.NDArray[np.float64]
    mp_lin: npt.NDArray[np.float64]
    gn_unit: npt.NDArray[np.float64]
    g_lap: npt.NDArray[np.float64]
    rows_flat: npt.NDArray[np.int_]
    cols_flat: npt.NDArray[np.int_]


def effective_saturation(
    psi: npt.ArrayLike,
    params: TransientVGParams,
) -> npt.NDArray[np.float64]:
    """Evaluate normalized van Genuchten effective saturation :math:`S_e(\\psi)`.

    Parameters
    ----------
    psi : npt.ArrayLike
        Pressure head :math:`\\psi = h - y` in meters (:math:`\\psi < 0` in vadose zone).
    params : TransientVGParams
        Material hydraulic parameters.

    Returns
    -------
    npt.NDArray[np.float64]
        Effective saturation :math:`S_e \\in (0, 1]`.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    se = np.ones_like(psi_arr)
    unsat = psi_arr < 0.0
    if np.any(unsat):
        se[unsat] = (1.0 + (params.alpha_m * (-psi_arr[unsat])) ** params.n) ** (-params.m)
    return se


def volumetric_water_content(
    psi: npt.ArrayLike,
    params: TransientVGParams,
) -> npt.NDArray[np.float64]:
    """Evaluate volumetric water content :math:`\\theta(\\psi)` (:math:`\\text{m}^3/\\text{m}^3`)."""
    se = effective_saturation(psi, params)
    return params.theta_r + params.s_y * se


def specific_moisture_capacity(
    psi: npt.ArrayLike,
    params: TransientVGParams,
) -> npt.NDArray[np.float64]:
    """Evaluate analytical retention curve slope :math:`C(\\psi) = d\\theta / d\\psi` (:math:`\\text{m}^{-1}`)."""
    psi_arr = np.asarray(psi, dtype=np.float64)
    cap = np.zeros_like(psi_arr)
    unsat = psi_arr < 0.0
    if np.any(unsat):
        ap = params.alpha_m * (-psi_arr[unsat])
        num = params.alpha_m * params.m * params.n * params.s_y * (ap ** (params.n - 1.0))
        den = (1.0 + ap ** params.n) ** (params.m + 1.0)
        cap[unsat] = num / den
    return cap


def relative_conductivity(
    psi: npt.ArrayLike,
    params: TransientVGParams,
) -> npt.NDArray[np.float64]:
    """Evaluate van Genuchten-Mualem relative hydraulic conductivity :math:`k_r(\\psi)`."""
    psi_arr = np.asarray(psi, dtype=np.float64)
    kr = np.ones_like(psi_arr)
    unsat = psi_arr < 0.0
    if np.any(unsat):
        se = effective_saturation(psi_arr[unsat], params)
        bracket = 1.0 - (1.0 - se ** (1.0 / params.m)) ** params.m
        kr_raw = (se ** params.ell) * (bracket ** 2)
        kr[unsat] = np.maximum(kr_raw, params.kr_min)
    return kr


def load_mesh_data(filepath: Path | str = Path("examples/paper/mesh1.json")) -> dict[str, Any]:
    """Load serialized mesh topology and precomputed GFDM stencils from JSON."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Mesh file '{path}' not found. Run 'examples/paper/mesh1.py' first."
        )
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _outward_wedge_normal(
    center: int,
    support: npt.NDArray[np.int_],
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute unit outward bisector normal pointing away from a material wedge."""
    vecs = coords[support] - coords[center]
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-10
    unit_vecs = vecs[valid] / norms[valid, None]
    outward = -np.mean(unit_vecs, axis=0)
    return outward / np.linalg.norm(outward)


def _build_ghost_op(
    center: int,
    support: npt.NDArray[np.int_],
    normal: npt.NDArray[np.float64],
    k_sat: float,
    coords: npt.NDArray[np.float64],
    flux_scaled: bool,
) -> _GhostStencilOp:
    """Precompute augmented 2nd-order and linear 1st-order pseudo-inverses for a star."""
    dx = coords[support, 0] - coords[center, 0]
    dy = coords[support, 1] - coords[center, 1]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    nonzero = dists > 1e-12
    mean_h = float(np.mean(dists[nonzero])) if np.any(nonzero) else 0.5

    gx, gy = normal * mean_h
    aug_dx = np.insert(dx, 0, gx)
    aug_dy = np.insert(dy, 0, gy)

    m_aug = np.vstack(
        (np.ones_like(aug_dx), aug_dx, aug_dy, aug_dx ** 2, aug_dx * aug_dy, aug_dy ** 2)
    )
    m_lin = np.vstack((np.ones_like(dx), dx, dy))
    return _GhostStencilOp(
        center=center,
        support=support,
        normal=normal,
        k_sat=k_sat,
        mp_aug=np.linalg.pinv(m_aug),
        mp_lin=np.linalg.pinv(m_lin),
        flux_scaled=flux_scaled,
    )


def _dunavant_7pt_rule() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return degree-5 symmetric 7-point Gauss quadrature on a reference triangle.

    Returns
    -------
    n_quad : npt.NDArray[np.float64]
        Strictly positive barycentric shape-function matrix of shape ``(7, 3)``.
    w_hat : npt.NDArray[np.float64]
        Normalized positive quadrature weights of shape ``(7,)`` summing to ``1.0``.
    """
    sqrt15 = float(np.sqrt(15.0))
    a1 = (6.0 - sqrt15) / 21.0
    b1 = (9.0 + 2.0 * sqrt15) / 21.0
    a2 = (6.0 + sqrt15) / 21.0
    b2 = (9.0 - 2.0 * sqrt15) / 21.0
    w1 = (155.0 - sqrt15) / 1200.0
    w2 = (155.0 + sqrt15) / 1200.0

    n_quad = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [b1, a1, a1],
            [a1, b1, a1],
            [a1, a1, b1],
            [b2, a2, a2],
            [a2, b2, a2],
            [a2, a2, b2],
        ],
        dtype=np.float64,
    )
    w_hat = np.array([9.0 / 40.0, w1, w1, w1, w2, w2, w2], dtype=np.float64)
    return n_quad, w_hat


class TransientUnsaturatedDamSolver:
    """Sparse GFDM solver for transient unsaturated seepage in the five-layer dam.

    Combines vectorized strong-form GFDM spatial discretization of the multi-layer
    dam (matching ``examples/paper/st_non_sat.py``) with XSLOPE/Celia conservative
    7-point sub-element storage integration, hybrid chord-tangent Picard iteration,
    and adaptive time-step control.

    Parameters
    ----------
    mesh_data : dict[str, Any]
        Parsed dictionary from ``examples/paper/mesh1.json``.
    materials : dict[str, TransientVGParams]
        Mapping ``{'rock', 'clay', 'foundation'}`` to hydraulic/storage parameters.
    upstream_head : float, default=8.0
        Dirichlet total head on the submerged upstream face and left wall (m).
    downstream_head : float, default=0.0
        Dirichlet total head on the horizontal underdrain and downstream ground (m).
    """

    def __init__(
        self,
        mesh_data: dict[str, Any],
        materials: dict[str, TransientVGParams],
        upstream_head: float = 8.0,
        downstream_head: float = 0.0,
    ) -> None:
        self.coords: npt.NDArray[np.float64] = np.asarray(mesh_data["coords"], dtype=np.float64)
        self.triangles: npt.NDArray[np.int_] = np.asarray(mesh_data["triangles"], dtype=np.int_)
        self.normals: npt.NDArray[np.float64] = np.asarray(
            mesh_data["normal_vecs"], dtype=np.float64
        )
        self.n_nodes: int = int(self.coords.shape[0])
        self.n_triangles: int = int(self.triangles.shape[0])
        self.materials = materials
        self.upstream_head = upstream_head
        self.downstream_head = downstream_head

        self.stencils: dict[int, npt.NDArray[np.int_]] = {
            int(k): np.asarray(v, dtype=np.int_) for k, v in mesh_data["support_stencils"].items()
        }
        self.m_pinv: dict[int, npt.NDArray[np.float64]] = {
            int(k): np.asarray(v, dtype=np.float64) for k, v in mesh_data["M_pinv"].items()
        }

        self.rock_nodes = np.asarray(mesh_data["rock_nodes"], dtype=np.int_)
        self.clay_nodes = np.asarray(mesh_data["clay_nodes"], dtype=np.int_)
        self.foundation_nodes = np.asarray(mesh_data["foundation_nodes"], dtype=np.int_)
        self.left_nodes = np.asarray(mesh_data["left_nodes"], dtype=np.int_)
        self.right_nodes = np.asarray(mesh_data["right_nodes"], dtype=np.int_)
        self.dirichlet_nodes = np.concatenate((self.left_nodes, self.right_nodes))

        self.dirichlet_mask = np.zeros(self.n_nodes, dtype=np.bool_)
        self.dirichlet_mask[self.dirichlet_nodes] = True
        self.free_mask = ~self.dirichlet_mask

        raw_bottom = np.asarray(mesh_data["bottom_nodes"], dtype=np.int_)
        raw_top = np.asarray(mesh_data["top_nodes"], dtype=np.int_)
        self.bottom_nodes = np.setdiff1d(raw_bottom, [9, 10])
        self.top_nodes = np.setdiff1d(raw_top, [5, 6])

        ia = np.asarray(mesh_data["interface_a_nodes"], dtype=np.int_)
        ib = np.asarray(mesh_data["interface_b_nodes"], dtype=np.int_)
        ic_base = np.asarray(mesh_data["interface_c_nodes"], dtype=np.int_)
        id_base = np.asarray(mesh_data["interface_d_nodes"], dtype=np.int_)
        ie_base = np.asarray(mesh_data["interface_e_nodes"], dtype=np.int_)
        if_base = np.asarray(mesh_data["interface_f_nodes"], dtype=np.int_)

        ic = np.append(ic_base, 6)
        id_ = np.append(id_base, 9)
        ie = np.append(ie_base, 10)
        if_ = np.append(if_base, 5)

        tj1 = int(mesh_data["tj1"][0])
        tj2 = int(mesh_data["tj2"][0])

        normals_fixed = self.normals.copy()
        normals_fixed[6] = self.normals[ic_base[0]]
        normals_fixed[9] = self.normals[id_base[0]]
        normals_fixed[10] = self.normals[ie_base[0]]
        normals_fixed[5] = self.normals[if_base[0]]
        normals_fixed[11] = np.array([1.0, -1.0]) / np.sqrt(2.0)

        ks_r = materials["rock"].k_sat
        ks_c = materials["clay"].k_sat
        ks_f = materials["foundation"].k_sat

        self.nodal_ksat = np.full(self.n_nodes, ks_r, dtype=np.float64)
        self.nodal_ksat[self.foundation_nodes] = ks_f
        self.nodal_ksat[self.clay_nodes] = ks_c
        clay_crest = self.top_nodes[
            (self.coords[self.top_nodes, 0] >= 23.0) & (self.coords[self.top_nodes, 0] <= 27.0)
        ]
        clay_base = self.bottom_nodes[
            (self.coords[self.bottom_nodes, 0] >= 24.0)
            & (self.coords[self.bottom_nodes, 0] <= 26.0)
        ]
        self.nodal_ksat[clay_crest] = ks_c
        self.nodal_ksat[clay_base] = ks_c

        ops: list[_GhostStencilOp] = []

        # 1. Outer impermeable Neumann boundaries (dh/dn = 0)
        for b_nodes, default_ks in ((self.bottom_nodes, ks_f), (self.top_nodes, ks_r)):
            for i in b_nodes:
                ni = normals_fixed[int(i)]
                is_clay_cap = (
                    23.0 < self.coords[i, 0] < 27.0 and abs(self.coords[i, 1]) > 3.9
                )
                ks_i = ks_c if is_clay_cap else default_ks
                ops.append(
                    _build_ghost_op(
                        int(i), self.stencils[int(i)], ni, ks_i, self.coords, flux_scaled=False
                    )
                )

        # 2. Six two-layer interfaces + four interface-Neumann corner vertices
        interf_specs = [
            ("ia", ks_r, ks_f, ia, self.rock_nodes, self.foundation_nodes),
            ("ib", ks_r, ks_f, ib, self.rock_nodes, self.foundation_nodes),
            ("ic", ks_c, ks_r, ic, self.clay_nodes, self.rock_nodes),
            ("id", ks_c, ks_f, id_, self.clay_nodes, self.foundation_nodes),
            ("ie", ks_f, ks_c, ie, self.foundation_nodes, self.clay_nodes),
            ("if", ks_r, ks_c, if_, self.rock_nodes, self.clay_nodes),
        ]
        interf_map = {name: nodes for name, _, _, nodes, _, _ in interf_specs}

        for _, ks0, ks1, b_nodes, m0, m1 in interf_specs:
            for i in b_nodes:
                ni = normals_fixed[int(i)]
                for side_ks, other_m, sgn in ((ks0, m1, 1.0), (ks1, m0, -1.0)):
                    other_all = other_m
                    if i in (5, 6):
                        other_b = (
                            self.top_nodes[
                                (self.coords[self.top_nodes, 0] < 23.0)
                                | (self.coords[self.top_nodes, 0] > 27.0)
                            ]
                            if side_ks == ks_c
                            else self.top_nodes[
                                (self.coords[self.top_nodes, 0] >= 23.0)
                                & (self.coords[self.top_nodes, 0] <= 27.0)
                            ]
                        )
                        other_all = np.union1d(other_m, other_b)
                    elif i in (9, 10):
                        other_b = (
                            self.bottom_nodes[
                                (self.coords[self.bottom_nodes, 0] < 24.0)
                                | (self.coords[self.bottom_nodes, 0] > 26.0)
                            ]
                            if side_ks == ks_c
                            else self.bottom_nodes[
                                (self.coords[self.bottom_nodes, 0] >= 24.0)
                                & (self.coords[self.bottom_nodes, 0] <= 26.0)
                            ]
                        )
                        other_all = np.union1d(other_m, other_b)

                    sub_stencil = np.setdiff1d(self.stencils[int(i)], other_all)
                    ops.append(
                        _build_ghost_op(
                            int(i),
                            sub_stencil,
                            sgn * ni,
                            side_ks,
                            self.coords,
                            flux_scaled=True,
                        )
                    )

        # 3. Triple junctions at (20, 0) and (30, 0)
        tj_specs = [
            (tj1, "ia", "id", self.foundation_nodes, ks_f),
            (tj1, "id", "ic", self.clay_nodes, ks_c),
            (tj1, "ic", "ia", self.rock_nodes, ks_r),
            (tj2, "ib", "if", self.rock_nodes, ks_r),
            (tj2, "ie", "if", self.clay_nodes, ks_c),
            (tj2, "ie", "ib", self.foundation_nodes, ks_f),
        ]
        for cnode, b_left, b_right, m_nodes, ks_w in tj_specs:
            allowed = np.union1d(
                np.union1d(interf_map[b_left], interf_map[b_right]),
                np.union1d(m_nodes, [cnode]),
            )
            sub_stencil = np.intersect1d(self.stencils[cnode], allowed)
            nw = _outward_wedge_normal(cnode, sub_stencil, self.coords)
            ops.append(
                _build_ghost_op(cnode, sub_stencil, nw, ks_w, self.coords, flux_scaled=True)
            )

        self._ghost_ops = ops
        self._build_vectorized_batches()
        self._build_subelement_quadrature()

    def _build_vectorized_batches(self) -> None:
        """Group GFDM stars by support size for fast vectorized sparse assembly."""
        int_groups: dict[int, list[tuple[int, float, npt.NDArray[np.int_], npt.NDArray[np.float64]]]] = (
            defaultdict(list)
        )
        interior_sets = (
            (self.materials["rock"].k_sat, self.rock_nodes),
            (self.materials["clay"].k_sat, self.clay_nodes),
            (self.materials["foundation"].k_sat, self.foundation_nodes),
        )
        for ks, nodes in interior_sets:
            for i in nodes:
                idx = int(i)
                st = self.stencils[idx]
                int_groups[int(st.shape[0])].append((idx, ks, st, self.m_pinv[idx]))

        int_batches: list[_InteriorBatch] = []
        for q_len, items in int_groups.items():
            centers = np.array([it[0] for it in items], dtype=np.int_)
            ks_arr = np.array([it[1] for it in items], dtype=np.float64)
            st_mat = np.stack([it[2] for it in items], axis=0)
            mp_arr = np.stack([it[3] for it in items], axis=0)
            int_batches.append(
                _InteriorBatch(
                    centers=centers,
                    ks_arr=ks_arr,
                    stencils=st_mat,
                    mp_arr=mp_arr,
                    rows_flat=np.repeat(centers, q_len),
                    cols_flat=st_mat.ravel(),
                )
            )
        self._int_batches = int_batches

        ghost_groups: dict[int, list[_GhostStencilOp]] = defaultdict(list)
        for op in self._ghost_ops:
            ghost_groups[int(op.support.shape[0])].append(op)

        ghost_batches: list[_GhostBatch] = []
        for q_len, ops_list in ghost_groups.items():
            centers = np.array([op.center for op in ops_list], dtype=np.int_)
            ks_arr = np.array([op.k_sat for op in ops_list], dtype=np.float64)
            fscaled = np.array([op.flux_scaled for op in ops_list], dtype=np.bool_)
            sup_mat = np.stack([op.support for op in ops_list], axis=0)
            mp_aug = np.stack([op.mp_aug for op in ops_list], axis=0)
            mp_lin = np.stack([op.mp_lin for op in ops_list], axis=0)
            normals = np.stack([op.normal for op in ops_list], axis=0)
            gn_unit = mp_aug[:, :, 1] * normals[:, 0:1] + mp_aug[:, :, 2] * normals[:, 1:2]
            g_lap = 2.0 * (mp_aug[:, :, 3] + mp_aug[:, :, 5])
            ghost_batches.append(
                _GhostBatch(
                    centers=centers,
                    ks_arr=ks_arr,
                    flux_scaled=fscaled,
                    supports=sup_mat,
                    mp_aug=mp_aug,
                    mp_lin=mp_lin,
                    gn_unit=gn_unit,
                    g_lap=g_lap,
                    rows_flat=np.repeat(centers, q_len),
                    cols_flat=sup_mat.ravel(),
                )
            )
        self._ghost_batches = ghost_batches

        self._all_rows = np.concatenate(
            [b.rows_flat for b in self._int_batches]
            + [b.rows_flat for b in self._ghost_batches]
            + [self.dirichlet_nodes]
        )
        self._all_cols = np.concatenate(
            [b.cols_flat for b in self._int_batches]
            + [b.cols_flat for b in self._ghost_batches]
            + [self.dirichlet_nodes]
        )
        dir_zero_mask = np.isin(self._all_rows, self.dirichlet_nodes)
        dir_zero_mask[-self.dirichlet_nodes.shape[0] :] = False
        self._dir_zero_mask = dir_zero_mask

        rhs_template = np.zeros(self.n_nodes, dtype=np.float64)
        rhs_template[self.left_nodes] = self.upstream_head
        rhs_template[self.right_nodes] = self.downstream_head
        self._rhs_template = rhs_template

    def _build_subelement_quadrature(self) -> None:
        """Precompute triangle geometry, 7-point Gauss weights, and HRZ row scaling."""
        tri_xy = self.coords[self.triangles]
        centroids = tri_xy.mean(axis=1)
        xc, yc = centroids[:, 0], centroids[:, 1]

        x_clay_left = np.where(yc >= 0.0, 20.0 + 0.3 * yc, 20.0 - 1.0 * yc)
        x_clay_right = np.where(yc >= 0.0, 30.0 - 0.3 * yc, 30.0 + 1.0 * yc)
        is_clay = (xc >= x_clay_left - 1e-9) & (xc <= x_clay_right + 1e-9)
        is_found = (~is_clay) & (yc < 0.0)

        # Material index per triangle: 0 = rock, 1 = clay, 2 = foundation
        tri_mat = np.zeros(self.n_triangles, dtype=np.int_)
        tri_mat[is_clay] = 1
        tri_mat[is_found] = 2
        self.triangle_material_id = tri_mat

        x1, y1 = tri_xy[:, 0, 0], tri_xy[:, 0, 1]
        x2, y2 = tri_xy[:, 1, 0], tri_xy[:, 1, 1]
        x3, y3 = tri_xy[:, 2, 0], tri_xy[:, 2, 1]
        det2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        self.triangle_areas = 0.5 * np.abs(det2)
        self._dn_dx = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / det2[:, None]
        self._dn_dy = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / det2[:, None]

        self._n_quad, self._w_hat = _dunavant_7pt_rule()
        self._w_eq = self.triangle_areas[:, None] * self._w_hat[None, :]

        # HRZ / row-summed lumped nodal areas m_j = sum_{e ni j} A_e / 3
        m_nodal = np.zeros(self.n_nodes, dtype=np.float64)
        for vertex_slot in range(3):
            np.add.at(m_nodal, self.triangles[:, vertex_slot], self.triangle_areas / 3.0)
        self.m_nodal = m_nodal

        # Effective geometric half-width on flux-scaled interface & triple-junction rows
        sigma_row = np.ones(self.n_nodes, dtype=np.float64)
        interf_accum = np.zeros(self.n_nodes, dtype=np.float64)
        for op in self._ghost_ops:
            if op.flux_scaled:
                g0 = op.mp_aug @ np.array([0.0, 0.0, 0.0, 2.0, 0.0, 2.0])
                gn0 = op.mp_aug @ np.array([0.0, op.normal[0], op.normal[1], 0.0, 0.0, 0.0])
                interf_accum[op.center] += abs(float(gn0[0] / g0[0]))
        has_interf = interf_accum > 0.0
        sigma_row[has_interf] = interf_accum[has_interf]
        self.sigma_row = sigma_row
        self.scale_row = sigma_row / m_nodal

        self._mat_order = ("rock", "clay", "foundation")
        ss_vals = np.array([self.materials[m].s_s for m in self._mat_order], dtype=np.float64)
        ks_vals = np.array([self.materials[m].k_sat for m in self._mat_order], dtype=np.float64)
        self._ss_tri = ss_vals[tri_mat][:, None]
        self._ks_tri = ks_vals[tri_mat]

    def evaluate_nodal_kr(self, head: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate relative hydraulic conductivity :math:`k_r(\\psi)` at all mesh nodes."""
        psi = head - self.coords[:, 1]
        kr = relative_conductivity(psi, self.materials["rock"])
        if (
            self.materials["clay"].alpha_kpa != self.materials["rock"].alpha_kpa
            or self.materials["clay"].n != self.materials["rock"].n
        ):
            kr[self.clay_nodes] = relative_conductivity(
                psi[self.clay_nodes], self.materials["clay"]
            )
        if (
            self.materials["foundation"].alpha_kpa != self.materials["rock"].alpha_kpa
            or self.materials["foundation"].n != self.materials["rock"].n
        ):
            kr[self.foundation_nodes] = relative_conductivity(
                psi[self.foundation_nodes], self.materials["foundation"]
            )
        return kr

    def evaluate_nodal_theta(self, head: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate volumetric water content :math:`\\theta(\\psi)` at all mesh nodes."""
        psi = head - self.coords[:, 1]
        theta = volumetric_water_content(psi, self.materials["rock"])
        theta[self.clay_nodes] = volumetric_water_content(
            psi[self.clay_nodes], self.materials["clay"]
        )
        theta[self.foundation_nodes] = volumetric_water_content(
            psi[self.foundation_nodes], self.materials["foundation"]
        )
        return theta

    def assemble_system(
        self,
        kr_nodal: npt.NDArray[np.float64],
        unsaturated: bool = True,
    ) -> tuple[sp.csr_matrix, npt.NDArray[np.float64]]:
        """Assemble global sparse GFDM system :math:`\\mathbf{K}(h)\\mathbf{h} = \\mathbf{F}`.

        Vectorized equivalent of ``UnsaturatedDamGFDMSolver.assemble_system`` in
        ``examples/paper/st_non_sat.py``, reproducing its matrix entries to machine
        precision (:math:`\\sim 10^{-19}`).
        """
        ln_kr = np.log(kr_nodal) if unsaturated else np.zeros(self.n_nodes, dtype=np.float64)
        val_parts: list[npt.NDArray[np.float64]] = []

        for batch in self._int_batches:
            if unsaturated:
                lnk_s = ln_kr[batch.stencils]
                dlnk_dx = np.einsum("bq,bq->b", lnk_s, batch.mp_arr[:, :, 1])
                dlnk_dy = np.einsum("bq,bq->b", lnk_s, batch.mp_arr[:, :, 2])
                k_i = batch.ks_arr * kr_nodal[batch.centers]
            else:
                dlnk_dx = np.zeros_like(batch.ks_arr)
                dlnk_dy = np.zeros_like(batch.ks_arr)
                k_i = batch.ks_arr
            weights = k_i[:, None] * (
                batch.mp_arr[:, :, 1] * dlnk_dx[:, None]
                + batch.mp_arr[:, :, 2] * dlnk_dy[:, None]
                + 2.0 * (batch.mp_arr[:, :, 3] + batch.mp_arr[:, :, 5])
            )
            val_parts.append(weights.ravel())

        for gbatch in self._ghost_batches:
            if unsaturated:
                lnk_s = ln_kr[gbatch.supports]
                dlnk_dx = np.einsum("bq,bq->b", lnk_s, gbatch.mp_lin[:, :, 1])
                dlnk_dy = np.einsum("bq,bq->b", lnk_s, gbatch.mp_lin[:, :, 2])
                k_i = gbatch.ks_arr * kr_nodal[gbatch.centers]
            else:
                dlnk_dx = np.zeros_like(gbatch.ks_arr)
                dlnk_dy = np.zeros_like(gbatch.ks_arr)
                k_i = gbatch.ks_arr
            g_full = k_i[:, None] * (
                gbatch.mp_aug[:, :, 1] * dlnk_dx[:, None]
                + gbatch.mp_aug[:, :, 2] * dlnk_dy[:, None]
                + gbatch.g_lap
            )
            gn_full = k_i[:, None] * gbatch.gn_unit
            gg = g_full[:, 0] / gn_full[:, 0]
            weights = np.where(
                gbatch.flux_scaled[:, None],
                g_full[:, 1:] / gg[:, None] - gn_full[:, 1:],
                g_full[:, 1:] - gg[:, None] * gn_full[:, 1:],
            )
            val_parts.append(weights.ravel())

        val_parts.append(np.ones(self.dirichlet_nodes.shape[0], dtype=np.float64))
        vals = np.concatenate(val_parts)
        vals[self._dir_zero_mask] = 0.0

        k_mat = sp.coo_matrix(
            (vals, (self._all_rows, self._all_cols)),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()
        return k_mat, self._rhs_template.copy()

    def evaluate_subelement_storage(
        self,
        h_curr: npt.NDArray[np.float64],
        h_old: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Integrate sub-element water storage and lumped chord/tangent capacities.

        Implements the 7-point Gauss quadrature chord-capacity splitting of
        ``02-DOCS/wiki/theory.md`` (Section 16.2-16.3), guaranteeing exact global
        mesh water conservation :math:`\\sum_j \\Delta W_j \\equiv \\sum_e \\Delta W_e`
        and diagonal (lumped) storage stiffness.

        Returns
        -------
        dw_nodal : npt.NDArray[np.float64]
            Exact nodal water volume uptake :math:`\\Delta W_j` (:math:`\\text{m}^3/\\text{m}`).
        g_chord_nodal : npt.NDArray[np.float64]
            Integrated nodal chord capacity :math:`G_{\\text{chord}, j}` (:math:`\\text{m}^2/\\text{m}`).
        g_tan_nodal : npt.NDArray[np.float64]
            Integrated nodal tangent capacity :math:`G_{\\text{tangent}, j}` (:math:`\\text{m}^2/\\text{m}`).
        w_mesh_total : float
            Total water volume :math:`\\sum_e W_e` held across the mesh at ``h_curr``
            (:math:`\\text{m}^3/\\text{m}`).
        """
        y_nodes = self.coords[:, 1]
        psi_new = h_curr - y_nodes
        psi_old = h_old - y_nodes
        dpsi_nodal = psi_new - psi_old

        psi_q_new = psi_new[self.triangles] @ self._n_quad.T
        psi_q_old = psi_old[self.triangles] @ self._n_quad.T
        dpsi_q = psi_q_new - psi_q_old

        phi_q_new = np.empty_like(psi_q_new)
        phi_q_old = np.empty_like(psi_q_old)
        s_tan_q = np.empty_like(psi_q_new)
        theta_sat_tri = np.empty((self.n_triangles, 1), dtype=np.float64)

        same_swrc = (
            self.materials["rock"].alpha_kpa == self.materials["clay"].alpha_kpa
            and self.materials["rock"].alpha_kpa == self.materials["foundation"].alpha_kpa
            and self.materials["rock"].n == self.materials["clay"].n
            and self.materials["rock"].n == self.materials["foundation"].n
            and self.materials["rock"].theta_s == self.materials["clay"].theta_s
            and self.materials["rock"].theta_s == self.materials["foundation"].theta_s
            and self.materials["rock"].theta_r == self.materials["clay"].theta_r
            and self.materials["rock"].theta_r == self.materials["foundation"].theta_r
        )

        if same_swrc:
            vg = self.materials["rock"]
            s_floor = vg.s_floor
            theta_sat_tri.fill(vg.theta_s)
            phi_q_new = np.where(
                psi_q_new >= 0.0,
                self._ss_tri * psi_q_new,
                (volumetric_water_content(psi_q_new, vg) - vg.theta_s) + s_floor * psi_q_new,
            )
            phi_q_old = np.where(
                psi_q_old >= 0.0,
                self._ss_tri * psi_q_old,
                (volumetric_water_content(psi_q_old, vg) - vg.theta_s) + s_floor * psi_q_old,
            )
            s_tan_q = np.where(
                psi_q_new >= 0.0,
                self._ss_tri,
                specific_moisture_capacity(psi_q_new, vg) + s_floor,
            )
        else:
            for mat_id, mat_name in enumerate(self._mat_order):
                mask = self.triangle_material_id == mat_id
                if not np.any(mask):
                    continue
                vg = self.materials[mat_name]
                s_floor = vg.s_floor
                theta_sat_tri[mask, 0] = vg.theta_s
                pn = psi_q_new[mask]
                po = psi_q_old[mask]
                phi_q_new[mask] = np.where(
                    pn >= 0.0,
                    vg.s_s * pn,
                    (volumetric_water_content(pn, vg) - vg.theta_s) + s_floor * pn,
                )
                phi_q_old[mask] = np.where(
                    po >= 0.0,
                    vg.s_s * po,
                    (volumetric_water_content(po, vg) - vg.theta_s) + s_floor * po,
                )
                s_tan_q[mask] = np.where(
                    pn >= 0.0,
                    vg.s_s,
                    specific_moisture_capacity(pn, vg) + s_floor,
                )

        big_dq = np.abs(dpsi_q) > 1e-12
        c_chord_q = np.where(
            big_dq,
            (phi_q_new - phi_q_old) / np.where(big_dq, dpsi_q, 1.0),
            s_tan_q,
        )

        g_chord_ea = (self._w_eq * c_chord_q) @ self._n_quad
        g_tan_ea = (self._w_eq * s_tan_q) @ self._n_quad

        g_chord_nodal = np.zeros(self.n_nodes, dtype=np.float64)
        g_tan_nodal = np.zeros(self.n_nodes, dtype=np.float64)
        for vertex_slot in range(3):
            np.add.at(g_chord_nodal, self.triangles[:, vertex_slot], g_chord_ea[:, vertex_slot])
            np.add.at(g_tan_nodal, self.triangles[:, vertex_slot], g_tan_ea[:, vertex_slot])

        dw_nodal = g_chord_nodal * dpsi_nodal
        w_mesh_total = float(np.sum(self._w_eq * (theta_sat_tri + phi_q_new)))
        return dw_nodal, g_chord_nodal, g_tan_nodal, w_mesh_total

    def compute_boundary_fluxes(self, head: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Compute independent upstream inflow :math:`Q_{\\text{in}}` and downstream outflow :math:`Q_{\\text{out}}`.

        Evaluates 7-point Gauss-quadrature Green's identity reaction fluxes across
        the submerged upstream boundary (``left_nodes``) and horizontal downstream
        underdrain/ground boundary (``right_nodes``), capturing flux contributions
        from both sides of the internal toe drain (:math:`x \\in [40, 47], y = 0`).
        """
        psi = head - self.coords[:, 1]
        psi_q = psi[self.triangles] @ self._n_quad.T
        kr_q = relative_conductivity(psi_q, self.materials["rock"])
        for mat_id, mat_name in enumerate(self._mat_order):
            if mat_name == "rock":
                continue
            vg = self.materials[mat_name]
            if vg.alpha_kpa != self.materials["rock"].alpha_kpa or vg.n != self.materials["rock"].n:
                mask = self.triangle_material_id == mat_id
                kr_q[mask] = relative_conductivity(psi_q[mask], vg)

        k_bar = self._ks_tri * (kr_q @ self._w_hat)
        h_tri = head[self.triangles]
        dh_dx = np.sum(h_tri * self._dn_dx, axis=1)
        dh_dy = np.sum(h_tri * self._dn_dy, axis=1)
        r_ea = (self.triangle_areas * k_bar)[:, None] * (
            self._dn_dx * dh_dx[:, None] + self._dn_dy * dh_dy[:, None]
        )
        r_nodal = np.zeros(self.n_nodes, dtype=np.float64)
        for vertex_slot in range(3):
            np.add.at(r_nodal, self.triangles[:, vertex_slot], r_ea[:, vertex_slot])

        q_in = float(np.sum(r_nodal[self.left_nodes]))
        q_out = float(-np.sum(r_nodal[self.right_nodes]))
        return q_in, q_out

    def compute_darcy_velocity(self, head: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute nodal Darcy velocity vectors :math:`\\mathbf{v} = -K(\\psi)\\nabla h` (m/s).

        Recomputes :math:`\\nabla h` directly from vectorized GFDM interior and
        one-sided boundary/interface stars (with area-weighted element recovery on
        Dirichlet nodes) so divergent transient seepage (:math:`\\nabla \\cdot \\mathbf{v} \\neq 0`)
        can be inspected without invoking steady stream-function assumptions (Section 16.7).
        """
        kr = self.evaluate_nodal_kr(head)
        vel = np.zeros((self.n_nodes, 2), dtype=np.float64)

        # 1. Pure interior subdomain nodes via vectorized 2nd-order GFDM stars
        for batch in self._int_batches:
            h_s = head[batch.stencils]
            dh_dx = np.einsum("bq,bq->b", h_s, batch.mp_arr[:, :, 1])
            dh_dy = np.einsum("bq,bq->b", h_s, batch.mp_arr[:, :, 2])
            k_i = batch.ks_arr * kr[batch.centers]
            vel[batch.centers, 0] = -k_i * dh_dx
            vel[batch.centers, 1] = -k_i * dh_dy

        # 2. Neumann, interface, corner, and triple-junction nodes via one-sided GFDM stars
        wedge_vel_sum = np.zeros((self.n_nodes, 2), dtype=np.float64)
        wedge_counts = np.zeros(self.n_nodes, dtype=np.int_)
        for gbatch in self._ghost_batches:
            h_s = head[gbatch.supports]
            dh_dx = np.einsum("bq,bq->b", h_s, gbatch.mp_lin[:, :, 1])
            dh_dy = np.einsum("bq,bq->b", h_s, gbatch.mp_lin[:, :, 2])
            k_i = gbatch.ks_arr * kr[gbatch.centers]
            np.add.at(wedge_vel_sum[:, 0], gbatch.centers, -k_i * dh_dx)
            np.add.at(wedge_vel_sum[:, 1], gbatch.centers, -k_i * dh_dy)
            np.add.at(wedge_counts, gbatch.centers, 1)

        has_wedge = wedge_counts > 0
        vel[has_wedge] = wedge_vel_sum[has_wedge] / wedge_counts[has_wedge, None]

        # 3. Dirichlet boundary nodes via area-weighted 7-point quadrature element velocity
        psi = head - self.coords[:, 1]
        psi_q = psi[self.triangles] @ self._n_quad.T
        kr_q = relative_conductivity(psi_q, self.materials["rock"])
        k_bar = self._ks_tri * (kr_q @ self._w_hat)
        h_tri = head[self.triangles]
        vx_e = -k_bar * np.sum(h_tri * self._dn_dx, axis=1)
        vy_e = -k_bar * np.sum(h_tri * self._dn_dy, axis=1)
        dir_vel_num = np.zeros((self.n_nodes, 2), dtype=np.float64)
        for vertex_slot in range(3):
            v_idx = self.triangles[:, vertex_slot]
            w_a = self.triangle_areas / 3.0
            np.add.at(dir_vel_num[:, 0], v_idx, w_a * vx_e)
            np.add.at(dir_vel_num[:, 1], v_idx, w_a * vy_e)
        vel[self.dirichlet_mask] = (
            dir_vel_num[self.dirichlet_mask] / self.m_nodal[self.dirichlet_mask, None]
        )
        return vel

    def solve_steady_baseline(
        self,
        tol_head: float = 1e-8,
        tol_residual: float = 1e-11,
        max_iter: int = 120,
        initial_omega: float = 0.8,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Solve the stationary Richards baseline (``st_non_sat.py``) for verification."""
        k_pot, f_pot = self.assemble_system(np.ones(self.n_nodes), unsaturated=False)
        h_potential = np.asarray(spla.spsolve(k_pot, f_pot), dtype=np.float64)
        h = h_potential.copy()

        y_span = float(self.coords[:, 1].max() - self.coords[:, 1].min())
        bc_span = abs(self.upstream_head - self.downstream_head)
        h_scale = max(y_span, bc_span)

        ladder = [initial_omega, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        rung = 0
        best_diff = float("inf")
        stalled_sweeps = 0
        record_low_streak = 0
        step_up_threshold = 10
        it_used = 0

        for it in range(1, max_iter + 1):
            it_used = it
            kr_m = self.evaluate_nodal_kr(h)
            k_mat, rhs = self.assemble_system(kr_m, unsaturated=True)
            h_star = np.asarray(spla.spsolve(k_mat, rhs), dtype=np.float64)
            raw_diff = float(np.max(np.abs(h_star - h)))

            if raw_diff < best_diff * (1.0 - 1e-12):
                best_diff = raw_diff
                stalled_sweeps = 0
                record_low_streak += 1
                if record_low_streak >= step_up_threshold and rung > 0:
                    rung -= 1
                    record_low_streak = 0
                    step_up_threshold *= 2
            else:
                stalled_sweeps += 1
                record_low_streak = 0
                if stalled_sweeps >= 5 and rung < len(ladder) - 1:
                    rung += 1
                    stalled_sweeps = 0

            omega = ladder[rung]
            h_next = h + omega * (h_star - h)
            step_diff = float(np.max(np.abs(h_next - h)))

            kr_next = self.evaluate_nodal_kr(h_next)
            k_next, rhs_next = self.assemble_system(kr_next, unsaturated=True)
            post_res = float(np.max(np.abs(k_next @ h_next - rhs_next)))
            h = h_next

            if (
                raw_diff <= tol_head * h_scale
                and post_res <= tol_residual
                and step_diff <= tol_head * h_scale
            ):
                break

        return h, h_potential, it_used

    def solve_transient(
        self,
        snapshot_days: tuple[float, ...] = (0.02, 0.10, 1.0, 5.0, 15.0, 100.0),
        theta_weight: float = 1.0,
        dt_init: float = 10.0,
        dt_min: float = 0.5,
        dt_max: float = 4.0 * SECONDS_PER_DAY,
        picard_tol: float = 1e-6,
        max_picard_iter: int = 35,
        time_error_frac: float = 3.5e-3,
        max_rms_dh_frac: float = 0.05,
    ) -> TransientRichardsResult:
        """Integrate the transient mixed Richards system from :math:`t = 0` to steady state.

        Starts from pre-impoundment hydrostatic equilibrium :math:`h(\\boldsymbol{x}, 0) = 0\\text{ m}`
        with upstream reservoir head :math:`h_{\\text{left}}(0^+) = 8.0\\text{ m}` and
        downstream drain head :math:`h_{\\text{right}}(0^+) = 0.0\\text{ m}`.
        """
        if not 0.5 <= theta_weight <= 1.0:
            raise ValueError("theta_weight must lie in [0.5, 1.0] for unconditional stability.")

        h_steady, h_potential, steady_iters = self.solve_steady_baseline()

        y_span = float(self.coords[:, 1].max() - self.coords[:, 1].min())
        bc_span = abs(self.upstream_head - self.downstream_head)
        h_scale = max(y_span, bc_span)

        # Initial condition: h(x, 0) = 0 m inside domain, Dirichlet values imposed on boundaries
        h = np.zeros(self.n_nodes, dtype=np.float64)
        h[self.left_nodes] = self.upstream_head
        h[self.right_nodes] = self.downstream_head
        _, _, _, w0 = self.evaluate_subelement_storage(h, h)
        q_in0, q_out0 = self.compute_boundary_fluxes(h)

        # Monitoring piezometers matching examples/paper/tr_potential.py
        probe_targets = {
            "Upstream rockfill (12, 3)": np.array([12.0, 3.0]),
            "Upstream core face (21.5, 5)": np.array([21.5, 5.0]),
            "Clay core center (25, 3)": np.array([25.0, 3.0]),
            "Core cutoff base (25, -2)": np.array([25.0, -2.0]),
            "Downstream core face (28.5, 5)": np.array([28.5, 5.0]),
            "Downstream foundation (35, -2)": np.array([35.0, -2.0]),
        }
        probe_indices: dict[str, int] = {
            label: int(np.argmin(np.linalg.norm(self.coords - pt, axis=1)))
            for label, pt in probe_targets.items()
        }

        saved_schedule = sorted(set(float(d) for d in snapshot_days if d > 0.0))
        t_end = saved_schedule[-1] * SECONDS_PER_DAY

        p_dir = sp.diags(self.dirichlet_mask.astype(np.float64), format="csr")
        p_free = sp.diags(self.free_mask.astype(np.float64), format="csr")

        times_list: list[float] = [0.0]
        dt_list: list[float] = [0.0]
        iters_list: list[int] = [0]
        w_list: list[float] = [w0]
        cum_div_list: list[float] = [0.0]
        q_in_list: list[float] = [q_in0]
        q_out_list: list[float] = [q_out0]

        diff0 = h - h_steady
        linf_list: list[float] = [float(np.max(np.abs(diff0)))]
        rmse_list: list[float] = [float(np.sqrt(np.mean(diff0 ** 2)))]
        probe_lists: dict[str, list[float]] = {
            label: [float(h[idx])] for label, idx in probe_indices.items()
        }
        frames: dict[float, SavedFrame] = {}

        t = 0.0
        dt = dt_init
        dt_prev: float | None = None
        h_prev: npt.NDArray[np.float64] | None = None
        save_idx = 0
        accepted_steps = 0
        rejected_steps = 0
        total_sweeps = 0
        cum_div_uptake = 0.0
        min_head = float(h.min())
        max_head = float(h.max())
        converged_all = True

        while t < t_end - 1e-9:
            next_save_sec = (
                saved_schedule[save_idx] * SECONDS_PER_DAY
                if save_idx < len(saved_schedule)
                else t_end
            )
            clamped = dt >= (next_save_sec - t - 1e-9)
            dt_step = min(dt, next_save_sec - t, t_end - t)

            # Explicit spatial flux contribution if theta_weight < 1.0 (e.g. Crank-Nicolson)
            if theta_weight < 1.0:
                kr_old = self.evaluate_nodal_kr(h)
                k_old, f_old = self.assemble_system(kr_old, unsaturated=True)
                div_old = k_old @ h - f_old
            else:
                div_old = np.zeros(self.n_nodes, dtype=np.float64)

            h_m = h.copy()
            prev_dh = np.zeros(self.n_nodes, dtype=np.float64)
            damp = np.ones(self.n_nodes, dtype=np.float64)
            step_converged = False
            sweeps_used = 0
            h_accepted = h_m

            for m in range(1, max_picard_iter + 1):
                sweeps_used = m
                kr_m = self.evaluate_nodal_kr(h_m)
                k_mat, f_vec = self.assemble_system(kr_m, unsaturated=True)
                dw_nodal, g_chord, g_tan, _ = self.evaluate_subelement_storage(h_m, h)

                # Envelope diagonal storage Jacobian: max(C_chord, C_tangent) (XSLOPE Section 16.4)
                c_env = self.scale_row * np.maximum(g_chord, g_tan)
                div_m = k_mat @ h_m - f_vec
                spatial_theta = theta_weight * div_m + (1.0 - theta_weight) * div_old

                res_free = self.scale_row * (dw_nodal / dt_step) - spatial_theta
                res_free[self.dirichlet_mask] = 0.0

                lhs = (
                    p_dir
                    + sp.diags(c_env / dt_step, format="csr") @ p_free
                    - theta_weight * (p_free @ k_mat)
                ).tocsc()
                rhs = -res_free
                rhs[self.left_nodes] = self.upstream_head - h_m[self.left_nodes]
                rhs[self.right_nodes] = self.downstream_head - h_m[self.right_nodes]

                dh_raw = np.asarray(spla.spsolve(lhs, rhs), dtype=np.float64)
                max_raw = float(np.max(np.abs(dh_raw[self.free_mask])))

                # Per-node sign-reversal damping across sharp wetting fronts
                if m > 1:
                    reversed_sign = (dh_raw * prev_dh < 0.0) & (
                        np.abs(dh_raw) >= 0.5 * np.abs(prev_dh)
                    )
                    damp = np.where(
                        reversed_sign,
                        np.maximum(0.125, 0.5 * damp),
                        np.minimum(1.0, 1.25 * damp),
                    )
                dh = damp * dh_raw
                prev_dh = dh_raw

                # Minimum two-sweep rule: never accept at m = 1 without verification sweep
                if m >= 2 and max_raw <= picard_tol * h_scale:
                    h_accepted = h_m + dh_raw
                    step_converged = True
                    break

                h_m = h_m + dh

            if not step_converged:
                if dt_step > dt_min * (1.0 + 1e-9):
                    dt = max(dt_min, 0.5 * dt_step)
                    rejected_steps += 1
                    continue
                # Floor recovery at dt_min (XSLOPE Section 16.5)
                h_accepted = h_m
                converged_all = False

            # Controller 2: Free-node RMS head-change limiter (Dirichlet nodes excluded)
            dh_free = h_accepted[self.free_mask] - h[self.free_mask]
            rms_dh = float(np.sqrt(np.mean(dh_free ** 2)))
            if rms_dh > max_rms_dh_frac * h_scale and dt_step > dt_min * (1.0 + 1e-9):
                dt = max(dt_min, 0.5 * dt_step)
                rejected_steps += 1
                continue

            # Controller 1: Backward-Euler local truncation error estimator
            if dt_prev is not None and h_prev is not None and dt_prev > 0.2 * dt_step:
                rate_curr = (h_accepted[self.free_mask] - h[self.free_mask]) / dt_step
                rate_prev = (h[self.free_mask] - h_prev[self.free_mask]) / dt_prev
                err_rms = (dt_step ** 2 / (dt_step + dt_prev)) * float(
                    np.sqrt(np.mean((rate_curr - rate_prev) ** 2))
                )
                tol_t = time_error_frac * h_scale
                if err_rms > 2.0 * tol_t and dt_step > dt_min * (1.0 + 1e-9):
                    dt = max(dt_min, 0.5 * dt_step)
                    rejected_steps += 1
                    continue
                fac_err = 0.9 * float(np.sqrt(tol_t / max(err_rms, 1e-16)))
            else:
                fac_err = 1.6

            # Controller 3: Convergence-based multiplicative step adjustment
            fac_iter = 1.45 if sweeps_used <= 8 else (1.10 if sweeps_used <= 16 else 0.75)
            grow = min(1.65, max(0.5, min(fac_err, fac_iter)))

            # Archive exact conservative uptake & fluxes for accepted step
            kr_acc = self.evaluate_nodal_kr(h_accepted)
            k_acc, f_acc = self.assemble_system(kr_acc, unsaturated=True)
            dw_acc, _, _, w_acc = self.evaluate_subelement_storage(h_accepted, h)
            div_acc = theta_weight * (k_acc @ h_accepted - f_acc) + (1.0 - theta_weight) * div_old
            div_flux_free = (
                float(np.sum(div_acc[self.free_mask] / self.scale_row[self.free_mask])) * dt_step
            )
            cum_div_uptake += div_flux_free + float(np.sum(dw_acc[self.dirichlet_mask]))
            q_in_acc, q_out_acc = self.compute_boundary_fluxes(h_accepted)

            h_prev = h.copy()
            dt_prev = dt_step
            h = h_accepted
            t += dt_step
            accepted_steps += 1
            total_sweeps += sweeps_used

            min_head = min(min_head, float(h.min()))
            max_head = max(max_head, float(h.max()))

            diff_st = h - h_steady
            linf_curr = float(np.max(np.abs(diff_st)))
            rmse_curr = float(np.sqrt(np.mean(diff_st ** 2)))

            times_list.append(t)
            dt_list.append(dt_step)
            iters_list.append(sweeps_used)
            w_list.append(w_acc)
            cum_div_list.append(cum_div_uptake)
            q_in_list.append(q_in_acc)
            q_out_list.append(q_out_acc)
            linf_list.append(linf_curr)
            rmse_list.append(rmse_curr)
            for label, idx in probe_indices.items():
                probe_lists[label].append(float(h[idx]))

            if not clamped:
                dt = min(dt_step * grow, dt_max)
            else:
                dt = min(max(dt, dt_step * grow), dt_max)

            # Controller 4: Exact saved-frame capture (never interpolated between steps)
            if save_idx < len(saved_schedule) and abs(t - next_save_sec) <= 1e-5:
                day_key = saved_schedule[save_idx]
                dw_tot = w_acc - w0
                mb_rel_err = abs(dw_tot - cum_div_uptake) / max(abs(dw_tot), 1e-12)
                frames[day_key] = SavedFrame(
                    time_days=day_key,
                    time_sec=t,
                    head=h.copy(),
                    psi=h - self.coords[:, 1],
                    theta=self.evaluate_nodal_theta(h),
                    kr=kr_acc,
                    darcy_vel=self.compute_darcy_velocity(h),
                    stored_water=w_acc,
                    delta_water=dw_tot,
                    q_in=q_in_acc,
                    q_out=q_out_acc,
                    linf_to_steady=linf_curr,
                    rmse_to_steady=rmse_curr,
                    mass_balance_rel_error=mb_rel_err,
                )
                save_idx += 1

        return TransientRichardsResult(
            times=np.asarray(times_list, dtype=np.float64),
            dt_history=np.asarray(dt_list, dtype=np.float64),
            iterations_history=np.asarray(iters_list, dtype=np.int_),
            stored_water_history=np.asarray(w_list, dtype=np.float64),
            cum_div_uptake_history=np.asarray(cum_div_list, dtype=np.float64),
            q_in_history=np.asarray(q_in_list, dtype=np.float64),
            q_out_history=np.asarray(q_out_list, dtype=np.float64),
            linf_history=np.asarray(linf_list, dtype=np.float64),
            rmse_history=np.asarray(rmse_list, dtype=np.float64),
            probe_indices=probe_indices,
            probe_histories={k: np.asarray(v, dtype=np.float64) for k, v in probe_lists.items()},
            frames=frames,
            steady_head=h_steady,
            steady_potential_head=h_potential,
            steady_iterations=steady_iters,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            total_picard_sweeps=total_sweeps,
            min_head=min_head,
            max_head=max_head,
            converged_all=converged_all,
        )


def _draw_dam_interfaces(
    ax: plt.Axes,
    color: str = "white",
    lw: float = 0.85,
    alpha: float = 0.7,
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


def _draw_dam_backdrop(ax: plt.Axes, fill: bool = True) -> None:
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
    if fill:
        ax.add_patch(Polygon(outer_xy, closed=True, facecolor="#f4f5f8", edgecolor="none", zorder=0))
        ax.add_patch(Polygon(clay_xy, closed=True, facecolor="#e1e5ee", edgecolor="none", zorder=0.5))
    ax.add_patch(
        Polygon(outer_xy, closed=True, facecolor="none", edgecolor="0.2", linewidth=0.95, zorder=4)
    )


def _select_quiver_nodes(
    coords: npt.NDArray[np.float64],
    dx_bin: float = 3.2,
    dy_bin: float = 1.6,
) -> npt.NDArray[np.int_]:
    """Select one interior-representative mesh node per spatial grid cell for clean quiver arrows."""
    chosen: dict[tuple[int, int], tuple[int, float]] = {}
    for idx in range(coords.shape[0]):
        xi, yi = float(coords[idx, 0]), float(coords[idx, 1])
        # Keep margin from all outer perimeter edges so arrow tips remain strictly inside domain
        if yi < -3.55 or yi > 9.35 or xi < 4.2 or xi > 58.6:
            continue
        if xi > 46.5 and yi > -0.40:
            continue
        if xi < 23.0 and yi > 0.5 * (xi - 3.0) - 0.55:
            continue
        if 27.0 < xi <= 47.0 and yi > 0.5 * (47.0 - xi) - 0.55:
            continue
        bx = int(np.floor(xi / dx_bin))
        by = int(np.floor((yi + 4.0) / dy_bin))
        cx = (bx + 0.5) * dx_bin
        cy = -4.0 + (by + 0.5) * dy_bin
        dist2 = (xi - cx) ** 2 + (yi - cy) ** 2
        if (bx, by) not in chosen or dist2 < chosen[(bx, by)][1]:
            chosen[(bx, by)] = (idx, dist2)
    return np.array([val[0] for val in chosen.values()], dtype=np.int_)


def _format_day_label(day: float) -> str:
    """Format a snapshot timestamp concisely in days (and hours when < 1 day)."""
    if day < 1.0:
        return f"$t = {day:g}$ d (${day * 24.0:.2g}$ h)"
    return f"$t = {day:g}$ d"


def save_transient_unsaturated_figures(
    solver: TransientUnsaturatedDamSolver,
    result: TransientRichardsResult,
    output_dir: Path | str = Path("examples/paper/figures"),
) -> list[Path]:
    """Render and export publication-grade figures for transient unsaturated seepage."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    coords = solver.coords
    triangles = solver.triangles
    x, y = coords[:, 0], coords[:, 1]
    snap_days = sorted(result.frames.keys())
    saved_paths: list[Path] = []

    # =========================================================================
    # Figure 1: 6-Panel Transient Total Head Contours & Phreatic Surface
    # =========================================================================
    levels_h = np.linspace(0.0, 8.0, 17)
    fig1, axes1 = plt.subplots(
        3, 2, figsize=(14.2, 7.8), sharex=True, sharey=True, layout="constrained"
    )
    ax_flat1 = axes1.ravel()
    tcf_last = None

    for idx, day in enumerate(snap_days[:6]):
        ax = ax_flat1[idx]
        frame = result.frames[day]
        h_clip = np.clip(frame.head, 0.0, 8.0)
        tcf_last = ax.tricontourf(x, y, triangles, h_clip, levels=levels_h, cmap="viridis")
        ax.tricontour(
            x, y, triangles, h_clip, levels=levels_h, colors="k", linewidths=0.45, alpha=0.6
        )
        _draw_dam_interfaces(ax, color="w", lw=0.8, alpha=0.72)
        plot_phreatic_surface(
            ax,
            coords,
            h_clip,
            triangles=triangles,
            color="#00d5ff",
            linewidths=1.95,
            label=None,
        )
        tag = chr(ord("a") + idx)
        ax.set_title(
            f"({tag}) {_format_day_label(day)}  "
            f"($\\|h - h_{{\\mathrm{{st}}}}\\|_\\infty = {frame.linf_to_steady:.2e}$ m)",
            fontsize=10.2,
        )
        ax.set_aspect("equal")
        if idx % 2 == 0:
            ax.set_ylabel("Elevation $y$ (m)")
        if idx >= 4:
            ax.set_xlabel("Distance $x$ (m)")

    if tcf_last is not None:
        cbar1 = fig1.colorbar(tcf_last, ax=axes1, shrink=0.88, pad=0.015)
        cbar1.set_label("Total hydraulic head $h(x, y, t)$ (m)")

    fig1.suptitle(
        "Transient Unsaturated Seepage ($\\partial \\Phi(\\psi)/\\partial t = "
        "\\nabla \\cdot (K(\\psi)\\nabla h)$) — Total Head & Phreatic Line",
        fontsize=12.4,
        fontweight="bold",
    )
    p1 = out / "tr_non_sat_snapshots.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    saved_paths.append(p1)

    # =========================================================================
    # Figure 2: 6-Panel Moisture Content theta(psi) & Divergent Darcy Vectors
    # =========================================================================
    quiver_idx = _select_quiver_nodes(coords, dx_bin=3.3, dy_bin=1.65)
    theta_min = float(min(f.theta.min() for f in result.frames.values()))
    v_ref = max(
        float(np.max(np.linalg.norm(f.darcy_vel[quiver_idx], axis=1)))
        for f in result.frames.values()
    )

    fig2, axes2 = plt.subplots(
        3, 2, figsize=(14.2, 7.8), sharex=True, sharey=True, layout="constrained"
    )
    ax_flat2 = axes2.ravel()
    tpc_last = None

    for idx, day in enumerate(snap_days[:6]):
        ax = ax_flat2[idx]
        frame = result.frames[day]
        tpc_last = ax.tripcolor(
            x,
            y,
            triangles,
            frame.theta,
            shading="gouraud",
            cmap="YlGnBu",
            vmin=theta_min,
            vmax=0.50,
        )
        ax.tricontour(
            x, y, triangles, frame.psi, levels=[0.0], colors="#002b80", linewidths=1.8
        )
        _draw_dam_interfaces(ax, color="0.25", lw=0.75, alpha=0.6)
        _draw_dam_backdrop(ax, fill=False)

        vel_sub = frame.darcy_vel[quiver_idx]
        speed = np.linalg.norm(vel_sub, axis=1)
        active = speed > 1e-10
        if np.any(active):
            q_nodes = quiver_idx[active]
            v_act = vel_sub[active]
            sp_act = speed[active]
            # Mild power-law scaling makes multi-decade rockfill/core/vadose vectors legible
            scaled_mag = (sp_act / max(v_ref, 1e-12)) ** 0.22
            ux = (v_act[:, 0] / sp_act) * scaled_mag
            uy = (v_act[:, 1] / sp_act) * scaled_mag
            ax.quiver(
                coords[q_nodes, 0],
                coords[q_nodes, 1],
                ux,
                uy,
                color="#ff5500",
                alpha=0.92,
                angles="xy",
                scale_units="xy",
                scale=0.36,
                width=0.0030,
                headwidth=3.7,
                headlength=4.3,
                zorder=5,
            )

        tag = chr(ord("a") + idx)
        ax.set_title(
            f"({tag}) {_format_day_label(day)}  "
            f"($\\Delta W = {frame.delta_water:.2f}\\;\\mathrm{{m^3/m}}$)",
            fontsize=10.2,
        )
        ax.set_aspect("equal")
        if idx % 2 == 0:
            ax.set_ylabel("Elevation $y$ (m)")
        if idx >= 4:
            ax.set_xlabel("Distance $x$ (m)")

    if tpc_last is not None:
        cbar2 = fig2.colorbar(tpc_last, ax=axes2, shrink=0.88, pad=0.015)
        cbar2.set_label("Volumetric water content $\\theta(\\psi)$ ($\\mathrm{m^3/m^3}$)")

    fig2.suptitle(
        "Transient Moisture Content $\\theta(\\psi)$ & Divergent Darcy Velocity Vectors "
        "$\\mathbf{v} = -K(\\psi)\\nabla h$ ($\\nabla \\cdot \\mathbf{v} \\neq 0$)",
        fontsize=12.2,
        fontweight="bold",
    )
    p2 = out / "tr_non_sat_moisture_velocity.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(p2)

    # =========================================================================
    # Figure 3: Progressive Migration of Unsaturated Phreatic Surface
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(11.2, 4.0), layout="constrained")
    _draw_dam_backdrop(ax3, fill=True)
    _draw_dam_interfaces(ax3, color="0.45", lw=0.9, alpha=0.8)

    cmap_time = plt.get_cmap("plasma")
    colors_time = [cmap_time(v) for v in np.linspace(0.08, 0.86, len(snap_days))]
    legend_handles: list[Line2D] = [
        Line2D([0], [0], color="0.45", lw=1.5, ls=":", label="Initial $t = 0$ ($y = 0$ m)"),
    ]
    ax3.plot([3.0, 60.0], [0.0, 0.0], color="0.45", lw=1.5, ls=":", zorder=2.5)

    for day, col in zip(snap_days, colors_time):
        psi_snap = result.frames[day].psi
        ax3.tricontour(
            x,
            y,
            triangles,
            psi_snap,
            levels=[0.0],
            colors=[col],
            linewidths=1.9,
            linestyles="-",
        )
        legend_handles.append(Line2D([0], [0], color=col, lw=1.9, label=_format_day_label(day)))

    ax3.tricontour(
        x,
        y,
        triangles,
        result.steady_head - y,
        levels=[0.0],
        colors=["#0b6623"],
        linewidths=2.4,
        linestyles="--",
    )
    legend_handles.append(
        Line2D([0], [0], color="#0b6623", lw=2.4, ls="--", label="Stationary (st_non_sat.py)")
    )

    ax3.set_xlim(3.0, 62.2)
    ax3.set_xlabel("Distance $x$ (m)")
    ax3.set_ylabel("Elevation $y$ (m)")
    ax3.set_title(
        "Transient Migration of Unsaturated Phreatic Surface ($\\psi(x, y, t) = 0$) "
        "Across the Five-Layer Dam"
    )
    ax3.set_aspect("equal")
    ax3.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
        fontsize=7.9,
        columnspacing=1.0,
        handlelength=1.8,
        frameon=True,
        framealpha=0.95,
    )

    p3 = out / "tr_non_sat_phreatic_evolution.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    saved_paths.append(p3)

    # =========================================================================
    # Figure 4: 2x2 Diagnostics (Piezometers, Relaxation, Fluxes, Water Budget)
    # =========================================================================
    fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(
        2, 2, figsize=(13.6, 8.2), layout="constrained"
    )
    t_days = result.times / SECONDS_PER_DAY

    # Panel (a): Monitoring piezometer histories across shell, clay core, and foundation
    for label, hist in result.probe_histories.items():
        idx = result.probe_indices[label]
        line = ax4a.plot(t_days, hist, linewidth=1.85, label=label)[0]
        ax4a.axhline(
            float(result.steady_head[idx]),
            color=line.get_color(),
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
    ax4a.set_xlim(0.0, 40.0)
    ax4a.set_ylim(-0.25, 8.35)
    ax4a.set_xlabel("Time $t$ (days)")
    ax4a.set_ylabel("Total hydraulic head $h(t)$ (m)")
    ax4a.set_title("(a) Piezometer Evolution Across Zones (Dotted: Steady Limit)")
    ax4a.legend(
        fontsize=7.5,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.16),
        frameon=True,
        framealpha=0.95,
    )

    # Panel (b): Asymptotic relaxation toward st_non_sat.py & adaptive dt growth
    ax4b.semilogy(
        t_days,
        np.maximum(result.linf_history, 1e-16),
        color="#1f77b4",
        linewidth=2.1,
        label="$\\|\\mathbf{h}(t) - \\mathbf{h}_{\\mathrm{st}}\\|_\\infty$ (m)",
    )
    ax4b.semilogy(
        t_days,
        np.maximum(result.rmse_history, 1e-16),
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="$\\mathrm{RMSE}(\\mathbf{h}(t), \\mathbf{h}_{\\mathrm{st}})$ (m)",
    )
    ax4b.semilogy(
        t_days[1:],
        result.dt_history[1:] / SECONDS_PER_DAY,
        color="#2ca02c",
        linewidth=1.6,
        linestyle="-.",
        alpha=0.85,
        label="Adaptive step $\\Delta t(t)$ (days)",
    )
    ax4b.set_xlabel("Time $t$ (days)")
    ax4b.set_ylabel("Error norm (m) / step size (days)")
    ax4b.set_title("(b) Asymptotic Relaxation to Steady State & Adaptive $\\Delta t$")
    ax4b.legend(fontsize=8.3, loc="lower left", frameon=True, framealpha=0.94)

    # Panel (c): Separate upstream inflow Q_in(t) vs downstream outflow Q_out(t)
    pos_mask = t_days > 0.0
    ax4c.loglog(
        t_days[pos_mask],
        np.maximum(result.q_in_history[pos_mask], 1e-12),
        color="#0055cc",
        linewidth=2.1,
        label="Upstream reservoir inflow $Q_{\\mathrm{in}}(t)$",
    )
    ax4c.loglog(
        t_days[pos_mask],
        np.maximum(result.q_out_history[pos_mask], 1e-12),
        color="#cc3300",
        linewidth=2.1,
        linestyle="--",
        label="Downstream drain outflow $Q_{\\mathrm{out}}(t)$",
    )
    ax4c.set_xlabel("Time $t$ (days, log scale)")
    ax4c.set_ylabel("Boundary flow rate ($\\mathrm{m^3 \\cdot s^{-1} \\cdot m^{-1}}$)")
    ax4c.set_title("(c) Boundary Hydrographs: Reservoir Absorption vs. Delayed Toe Seepage")
    ax4c.legend(fontsize=8.3, loc="lower left", frameon=True, framealpha=0.94)

    # Panel (d): Cumulative stored water W(t) - W(0) vs discrete mixed-form uptake
    delta_w_hist = result.stored_water_history - result.stored_water_history[0]
    final_mb_pct = (
        abs(delta_w_hist[-1] - result.cum_div_uptake_history[-1])
        / max(abs(delta_w_hist[-1]), 1e-12)
        * 100.0
    )
    ax4d.semilogx(
        t_days[pos_mask],
        delta_w_hist[pos_mask],
        color="#1f77b4",
        linewidth=2.3,
        label="7-pt quadrature domain storage $\\Delta W_{\\mathrm{mesh}}(t)$",
    )
    ax4d.semilogx(
        t_days[pos_mask],
        result.cum_div_uptake_history[pos_mask],
        color="#ff7f0e",
        linewidth=1.9,
        linestyle="--",
        label=f"Cumulative mixed-form flux uptake (closure err $= {final_mb_pct:.4f}\\%$)",
    )
    ax4d.set_xlabel("Time $t$ (days, log scale)")
    ax4d.set_ylabel("Cumulative stored water $\\Delta W$ ($\\mathrm{m^3/m}$)")
    ax4d.set_title("(d) Celia Mixed-Form Mass-Conservation Ledger")
    ax4d.legend(fontsize=8.1, loc="lower right", frameon=True, framealpha=0.94)

    p4 = out / "tr_non_sat_convergence_budget.png"
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    saved_paths.append(p4)

    return saved_paths


def build_default_materials() -> dict[str, TransientVGParams]:
    """Build Chavez et al. (2023) van Genuchten parameters with XSLOPE storage properties."""
    return {
        "rock": TransientVGParams(
            name="Rockfill Shell",
            k_sat=1e-4,
            theta_s=0.50,
            theta_r=0.0,
            alpha_kpa=1.0 / 50.0,
            n=1.4,
            s_s=1e-4,
        ),
        "clay": TransientVGParams(
            name="Clay Core",
            k_sat=1e-6,
            theta_s=0.50,
            theta_r=0.0,
            alpha_kpa=1.0 / 50.0,
            n=1.4,
            s_s=1e-3,
        ),
        "foundation": TransientVGParams(
            name="Soil Foundation",
            k_sat=1e-4,
            theta_s=0.50,
            theta_r=0.0,
            alpha_kpa=1.0 / 50.0,
            n=1.4,
            s_s=2e-4,
        ),
    }


def main() -> None:
    """Execute the transient unsaturated GFDM seepage simulation and export figures."""
    t_start = time.perf_counter()
    print("==============================================================================")
    print("GFDFlow — Transient Unsaturated Dam Seepage (Mixed Richards + Adaptive GFDM)")
    print("==============================================================================")

    mesh_path = Path("examples/paper/mesh1.json")
    print(f"1. Loading precomputed mesh and GFDM stencils from: {mesh_path}")
    mesh_data = load_mesh_data(mesh_path)
    materials = build_default_materials()
    solver = TransientUnsaturatedDamSolver(mesh_data, materials)

    vg_ref = materials["rock"]
    print(
        f"   Mesh: {solver.n_nodes} nodes, {solver.n_triangles} tri3 elements, "
        f"area = {solver.triangle_areas.sum():.2f} m^2."
    )
    print(
        f"   Storage parameters: S_y = {vg_ref.s_y:.2f}, C_max = {vg_ref.c_max:.4e} m^-1, "
        f"S_floor = {vg_ref.s_floor:.4e} m^-1."
    )

    snapshot_days = (0.02, 0.10, 1.0, 5.0, 15.0, 100.0)
    print("2. Integrating mixed-form Richards equation (Backward Euler, 7-pt quadrature)...")
    result = solver.solve_transient(snapshot_days=snapshot_days, theta_weight=1.0)

    print(
        f"   Steady baseline (`st_non_sat.py`): converged in {result.steady_iterations} sweeps, "
        f"h_st in [{result.steady_head.min():.6f}, {result.steady_head.max():.6f}] m."
    )
    print(
        f"   Transient march: {result.accepted_steps} accepted steps "
        f"({result.rejected_steps} rejected), {result.total_picard_sweeps} total Picard sweeps "
        f"(avg {result.total_picard_sweeps / max(result.accepted_steps, 1):.1f}/step)."
    )
    print(
        f"   Global head bounds across all steps: "
        f"min = {result.min_head:.6f} m, max = {result.max_head:.6f} m."
    )
    print("   Saved-frame summary:")
    for day in sorted(result.frames.keys()):
        fr = result.frames[day]
        print(
            f"     t = {day:6.2f} d | ||h - h_st||_inf = {fr.linf_to_steady:.3e} m | "
            f"dW = {fr.delta_water:6.4f} m^3/m | Q_in = {fr.q_in:.2e} | "
            f"Q_out = {fr.q_out:.2e} | MB_err = {fr.mass_balance_rel_error * 100.0:.5f}%"
        )

    fig_dir = Path("examples/paper/figures")
    print(f"3. Exporting publication figures (300 DPI) to: {fig_dir}")
    saved_paths = save_transient_unsaturated_figures(solver, result, output_dir=fig_dir)
    for fig_path in saved_paths:
        print(f"   Saved figure: {fig_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Completed transient unsaturated analysis in {elapsed:.2f} s.")
    print("==============================================================================")


if __name__ == "__main__":
    main()
