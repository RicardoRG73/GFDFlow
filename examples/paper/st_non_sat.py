"""Stationary Unsaturated Groundwater Flow (Richards Equation) in a Five-Layer Dam.

This module solves the 2D steady-state unsaturated groundwater seepage problem
(stationary Richards equation):

.. math::
    \\nabla \\cdot \\left( K(\\psi) \\nabla h \\right) = 0, \\quad \\psi = h - y

for the five-layered embankment dam with central clay core and foundation
described in Section 4.2.3 (Table 2, Figures 16 and 17) of Chavez et al. (2023),
following the constitutive and nonlinear linearization theory documented in
``02-DOCS/wiki/non_saturated_flow.md``.

Physical & Numerical Formulation
--------------------------------
1. **Constitutive Closure (van Genuchten, 1980; Mualem, 1976)**:
   - Effective saturation :math:`S_e(\\psi) = [1 + (\\alpha_m |\\psi|)^n]^{-m}` for
     :math:`\\psi < 0` (:math:`S_e = 1` for :math:`\\psi \\ge 0`), with
     :math:`m = 1 - 1/n` and :math:`\\alpha_m = \\alpha_{\\text{kPa}} \\gamma_w`.
   - Relative hydraulic conductivity
     :math:`k_r(\\psi) = \\max\\!\\left(S_e^{\\ell}[1 - (1 - S_e^{1/m})^m]^2,\\, k_{r,\\min}\\right)`
     with Mualem pore-connectivity :math:`\\ell = 0.5` and numerical guard
     :math:`k_{r,\\min} = 10^{-8}`.
2. **Strong-Form GFDM Discretization with Interface Flux Continuity**:
   - Within each subdomain, expanding :math:`\\nabla \\cdot (K(\\psi) \\nabla h) = 0`
     via the logarithmic conductivity gradient yields
     :math:`K(\\psi)\\left(\\nabla^2 h + \\nabla(\\ln k_r) \\cdot \\nabla h\\right) = 0`,
     preserving discrete maximum principle positivity across sharp capillary transitions.
   - Ghost-node elimination enforces zero normal flux (:math:`\\partial h / \\partial n = 0`)
     at impermeable and exposed boundaries, and continuous outward normal Darcy flux
     (:math:`\\sum_s -K_s(\\psi)\\,\\partial h / \\partial n_s = 0`) along the six internal
     material interfaces, four interface-boundary corners, and two triple junctions.
3. **Picard Fixed-Point Iteration with Adaptive Under-Relaxation**:
   - Lags :math:`K(\\psi^m)` at iteration :math:`m` and solves the sparse linear system
     :math:`\\mathbf{K}(h^m)\\,\\mathbf{h}^{m+1,\\star} = \\mathbf{F}` via ``spsolve``.
   - Controls damping :math:`\\omega_m` via the discrete relaxation ladder
     :math:`\\Omega = \\{1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01\\}` and verifies both scaled
     :math:`L_\\infty` head increment and unlagged nonlinear residual closure.

Outputs
-------
- Figures saved to ``examples/paper/figures/``:
  - ``non_sat_head_contours.png``: Total head contours, equipotentials, and phreatic line.
  - ``non_sat_hydraulic_fields.png``: Pressure head :math:`\\psi`, water content :math:`\\theta`,
    and log-conductivity :math:`\\log_{10} K(\\psi)`.
  - ``non_sat_constitutive_picard.png``: Van Genuchten curves and Picard convergence history.
  - ``non_sat_vs_potential_comparison.png``: Saturated potential vs unsaturated Richards comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from GFDFlow.visualization import plot_phreatic_surface

GAMMA_W: float = 9.81  # Unit weight of water [kN/m^3 == kPa/m]


@dataclass(frozen=True, slots=True)
class VanGenuchtenParams:
    """Soil hydraulic parameters for the van Genuchten–Mualem model.

    Parameters
    ----------
    name : str
        Material identifier.
    k_sat : float
        Saturated hydraulic conductivity :math:`K_s` in m/s.
    theta_s : float, default=0.50
        Saturated volumetric water content :math:`\\theta_s` (dimensionless).
    theta_r : float, default=0.0
        Residual volumetric water content :math:`\\theta_r` (dimensionless).
    alpha_kpa : float, default=0.02
        Inverse air-entry suction parameter :math:`\\alpha` in :math:`\\text{kPa}^{-1}`
        (Chavez et al., 2023, Table 2: :math:`1/50\\text{ kPa}^{-1}`).
    n : float, default=1.4
        Pore-size distribution exponent :math:`n > 1` (dimensionless).
    ell : float, default=0.5
        Mualem pore-connectivity exponent :math:`\\ell` (dimensionless).
    kr_min : float, default=1e-8
        Numerical lower floor :math:`k_{r,\\min}` preventing singular dry stencils.
    gamma_w : float, default=9.81
        Unit weight of water :math:`\\gamma_w` in :math:`\\text{kN/m}^3` used to convert
        :math:`\\alpha` from :math:`\\text{kPa}^{-1}` to :math:`\\text{m}^{-1}`.
    """

    name: str
    k_sat: float
    theta_s: float = 0.50
    theta_r: float = 0.0
    alpha_kpa: float = 1.0 / 50.0
    n: float = 1.4
    ell: float = 0.5
    kr_min: float = 1e-8
    gamma_w: float = GAMMA_W

    @property
    def m(self) -> float:
        """Return Mualem exponent :math:`m = 1 - 1/n`."""
        return 1.0 - 1.0 / self.n

    @property
    def alpha_m(self) -> float:
        """Return inverse air-entry parameter :math:`\\alpha_m` in :math:`\\text{m}^{-1}`."""
        return self.alpha_kpa * self.gamma_w


@dataclass(frozen=True, slots=True)
class PicardResult:
    """Container for Picard nonlinear solver solution and convergence diagnostics.

    Parameters
    ----------
    head : npt.NDArray[np.float64]
        Converged nodal total hydraulic head :math:`h` (m), shape ``(N,)``.
    head_potential : npt.NDArray[np.float64]
        Baseline saturated potential total head :math:`h_{\\text{sat}}` (m), shape ``(N,)``.
    psi : npt.NDArray[np.float64]
        Converged nodal pressure head :math:`\\psi = h - y` (m), shape ``(N,)``.
    theta : npt.NDArray[np.float64]
        Converged nodal volumetric water content :math:`\\theta(\\psi)`, shape ``(N,)``.
    kr : npt.NDArray[np.float64]
        Converged nodal relative hydraulic conductivity :math:`k_r(\\psi)`, shape ``(N,)``.
    k_eff : npt.NDArray[np.float64]
        Representative nodal effective conductivity :math:`K(\\psi)` (m/s), shape ``(N,)``.
    iterations : int
        Number of Picard sweeps executed.
    converged : bool
        Whether hybrid convergence criteria were met within ``max_iter``.
    head_diff_history : list[float]
        Maximum nodal head update :math:`\\|h^{m+1} - h^m\\|_\\infty` per sweep.
    residual_history : list[float]
        Unlagged nonlinear residual :math:`\\|\\mathbf{K}(h^m)\\mathbf{h}^m - \\mathbf{F}\\|_\\infty` per sweep.
    omega_history : list[float]
        Under-relaxation factor :math:`\\omega_m` applied at each sweep.
    """

    head: npt.NDArray[np.float64]
    head_potential: npt.NDArray[np.float64]
    psi: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    kr: npt.NDArray[np.float64]
    k_eff: npt.NDArray[np.float64]
    iterations: int
    converged: bool
    head_diff_history: list[float]
    residual_history: list[float]
    omega_history: list[float]


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


def effective_saturation(
    psi: npt.ArrayLike,
    params: VanGenuchtenParams,
) -> npt.NDArray[np.float64]:
    """Evaluate normalized effective saturation :math:`S_e(\\psi)`.

    Parameters
    ----------
    psi : npt.ArrayLike
        Pressure head :math:`\\psi = h - y` in meters (negative in unsaturated zone).
    params : VanGenuchtenParams
        Van Genuchten–Mualem soil parameters.

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
    params: VanGenuchtenParams,
) -> npt.NDArray[np.float64]:
    """Evaluate volumetric water content :math:`\\theta(\\psi)`.

    Parameters
    ----------
    psi : npt.ArrayLike
        Pressure head :math:`\\psi = h - y` in meters.
    params : VanGenuchtenParams
        Van Genuchten–Mualem soil parameters.

    Returns
    -------
    npt.NDArray[np.float64]
        Volumetric water content :math:`\\theta \\in [\\theta_r, \\theta_s]`.
    """
    se = effective_saturation(psi, params)
    return params.theta_r + (params.theta_s - params.theta_r) * se


def specific_moisture_capacity(
    psi: npt.ArrayLike,
    params: VanGenuchtenParams,
) -> npt.NDArray[np.float64]:
    """Evaluate specific moisture capacity :math:`C(\\psi) = d\\theta / d\\psi`.

    Parameters
    ----------
    psi : npt.ArrayLike
        Pressure head :math:`\\psi = h - y` in meters.
    params : VanGenuchtenParams
        Van Genuchten–Mualem soil parameters.

    Returns
    -------
    npt.NDArray[np.float64]
        Specific moisture capacity :math:`C(\\psi)` in :math:`\\text{m}^{-1}`.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    cap = np.zeros_like(psi_arr)
    unsat = psi_arr < 0.0
    if np.any(unsat):
        ap = params.alpha_m * (-psi_arr[unsat])
        num = (
            params.alpha_m
            * params.m
            * params.n
            * (params.theta_s - params.theta_r)
            * (ap ** (params.n - 1.0))
        )
        den = (1.0 + ap ** params.n) ** (params.m + 1.0)
        cap[unsat] = num / den
    return cap


def relative_conductivity(
    psi: npt.ArrayLike,
    params: VanGenuchtenParams,
) -> npt.NDArray[np.float64]:
    """Evaluate van Genuchten–Mualem relative hydraulic conductivity :math:`k_r(\\psi)`.

    Applies the numerical floor :math:`k_{r,\\min} = 10^{-8}` in high-suction zones
    to prevent ill-conditioned or singular GFDM star matrices.

    Parameters
    ----------
    psi : npt.ArrayLike
        Pressure head :math:`\\psi = h - y` in meters.
    params : VanGenuchtenParams
        Van Genuchten–Mualem soil parameters.

    Returns
    -------
    npt.NDArray[np.float64]
        Dimensionless relative permeability :math:`k_r \\in [k_{r,\\min}, 1]`.
    """
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
    """Load serialized mesh topology and precomputed GFDM stencils from JSON.

    Parameters
    ----------
    filepath : Path or str, default=Path('examples/paper/mesh1.json')
        Path to the JSON mesh file generated by ``mesh1.py``.

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


class UnsaturatedDamGFDMSolver:
    """Sparse GFDM solver for steady-state Richards seepage in the five-layer dam.

    Precomputes all geometric star pseudo-inverses once at initialization so each
    Picard iteration only evaluates nodal constitutive properties and assembles the
    sparse matrix :math:`\\mathbf{K}(h^m)` in :math:`\\mathcal{O}(N)` operations.

    Parameters
    ----------
    mesh_data : dict[str, Any]
        Parsed mesh dictionary from ``examples/paper/mesh1.json``.
    materials : dict[str, VanGenuchtenParams]
        Mapping of ``{'rock', 'clay', 'foundation'}`` to van Genuchten parameters.
    upstream_head : float, default=8.0
        Dirichlet total head on the submerged upstream face and left wall (m).
    downstream_head : float, default=0.0
        Dirichlet total head on the downstream horizontal drain/surface (m).
    """

    def __init__(
        self,
        mesh_data: dict[str, Any],
        materials: dict[str, VanGenuchtenParams],
        upstream_head: float = 8.0,
        downstream_head: float = 0.0,
    ) -> None:
        self.coords: npt.NDArray[np.float64] = np.asarray(mesh_data["coords"], dtype=np.float64)
        self.triangles: npt.NDArray[np.int_] = np.asarray(mesh_data["triangles"], dtype=np.int_)
        self.normals: npt.NDArray[np.float64] = np.asarray(
            mesh_data["normal_vecs"], dtype=np.float64
        )
        self.n_nodes: int = self.coords.shape[0]
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

        # Separate interface-Neumann corner vertices (5, 6 at crest y=10; 9, 10 at base y=-4)
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

        # Representative nodal K_sat map for post-processing visualization
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

        # Precompute ghost/flux stencil operators for Neumann, interfaces, and triple junctions
        ops: list[_GhostStencilOp] = []

        # 1. Outer Neumann boundaries (dh/dn = 0)
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

        # 2. Six two-material interfaces (including crest/base interface endpoints 5, 6, 9, 10)
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

    def evaluate_nodal_kr(self, head: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate relative hydraulic conductivity :math:`k_r(\\psi)` at all mesh nodes.

        Parameters
        ----------
        head : npt.NDArray[np.float64]
            Nodal total hydraulic head array of shape ``(N,)``.

        Returns
        -------
        npt.NDArray[np.float64]
            Nodal relative permeability :math:`k_r(h - y)` of shape ``(N,)``.
        """
        psi = head - self.coords[:, 1]
        kr = relative_conductivity(psi, self.materials["rock"])
        if (
            self.materials["clay"].alpha_kpa != self.materials["rock"].alpha_kpa
            or self.materials["clay"].n != self.materials["rock"].n
        ):
            kr[self.clay_nodes] = relative_conductivity(psi[self.clay_nodes], self.materials["clay"])
        if (
            self.materials["foundation"].alpha_kpa != self.materials["rock"].alpha_kpa
            or self.materials["foundation"].n != self.materials["rock"].n
        ):
            kr[self.foundation_nodes] = relative_conductivity(
                psi[self.foundation_nodes], self.materials["foundation"]
            )
        return kr

    def assemble_system(
        self,
        kr_nodal: npt.NDArray[np.float64],
        unsaturated: bool = True,
    ) -> tuple[sp.csr_matrix, npt.NDArray[np.float64]]:
        """Assemble global sparse GFDM system :math:`\\mathbf{K}\\mathbf{h} = \\mathbf{F}`.

        Parameters
        ----------
        kr_nodal : npt.NDArray[np.float64]
            Frozen nodal relative permeability field :math:`k_r(h^m - y)` of shape ``(N,)``.
        unsaturated : bool, default=True
            If ``True``, includes the :math:`\\nabla(\\ln k_r) \\cdot \\nabla h` advection-like
            capillary gradient term of the strong-form Richards equation; if ``False``,
            assembles the saturated potential operator (:math:`k_r \\equiv 1`).

        Returns
        -------
        K : scipy.sparse.csr_matrix
            Global sparse stiffness/conductance matrix of shape ``(N, N)``.
        F : npt.NDArray[np.float64]
            Global right-hand-side vector of shape ``(N,)``.
        """
        ln_kr = np.log(kr_nodal) if unsaturated else np.zeros(self.n_nodes, dtype=np.float64)

        row_chunks: list[npt.NDArray[np.int_]] = []
        col_chunks: list[npt.NDArray[np.int_]] = []
        val_chunks: list[npt.NDArray[np.float64]] = []
        rhs = np.zeros(self.n_nodes, dtype=np.float64)

        # 1. Interior material nodes
        interior_groups = (
            (self.materials["rock"].k_sat, self.rock_nodes),
            (self.materials["clay"].k_sat, self.clay_nodes),
            (self.materials["foundation"].k_sat, self.foundation_nodes),
        )
        for ks, nodes in interior_groups:
            for i in nodes:
                idx = int(i)
                stencil = self.stencils[idx]
                mp = self.m_pinv[idx]
                k_i = ks * (kr_nodal[idx] if unsaturated else 1.0)
                if unsaturated:
                    dlnk_dx = float(ln_kr[stencil] @ mp[:, 1])
                    dlnk_dy = float(ln_kr[stencil] @ mp[:, 2])
                else:
                    dlnk_dx, dlnk_dy = 0.0, 0.0

                weights = k_i * (mp @ np.array([0.0, dlnk_dx, dlnk_dy, 2.0, 0.0, 2.0]))
                row_chunks.append(np.full(stencil.shape[0], idx, dtype=np.int_))
                col_chunks.append(stencil)
                val_chunks.append(weights)

        # 2. Ghost-eliminated Neumann boundaries, interfaces, corners, and triple junctions
        for op in self._ghost_ops:
            k_i = op.k_sat * (kr_nodal[op.center] if unsaturated else 1.0)
            if unsaturated:
                dlnk_dx = float(ln_kr[op.support] @ op.mp_lin[:, 1])
                dlnk_dy = float(ln_kr[op.support] @ op.mp_lin[:, 2])
            else:
                dlnk_dx, dlnk_dy = 0.0, 0.0

            g_full = k_i * (op.mp_aug @ np.array([0.0, dlnk_dx, dlnk_dy, 2.0, 0.0, 2.0]))
            gn_full = k_i * (
                op.mp_aug @ np.array([0.0, op.normal[0], op.normal[1], 0.0, 0.0, 0.0])
            )
            gg = g_full[0] / gn_full[0]

            if op.flux_scaled:
                # Divide by gg so each material side contributes outward normal Darcy flux -q_n
                weights = g_full[1:] / gg - gn_full[1:]
            else:
                weights = g_full[1:] - gg * gn_full[1:]

            row_chunks.append(np.full(op.support.shape[0], op.center, dtype=np.int_))
            col_chunks.append(op.support)
            val_chunks.append(weights)

        # 3. Dirichlet boundary conditions (overwrite rows via solitary diagonal 1)
        dir_nodes = np.concatenate((self.left_nodes, self.right_nodes))
        row_chunks.append(dir_nodes)
        col_chunks.append(dir_nodes)
        val_chunks.append(np.ones(dir_nodes.shape[0], dtype=np.float64))

        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(val_chunks)

        # Zero out any pre-existing entries on Dirichlet rows before setting identity
        dir_mask = np.isin(rows, dir_nodes)
        dir_mask[-dir_nodes.shape[0] :] = False
        vals[dir_mask] = 0.0

        rhs[self.left_nodes] = self.upstream_head
        rhs[self.right_nodes] = self.downstream_head

        k_mat = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_nodes, self.n_nodes)).tocsr()
        k_mat.eliminate_zeros()
        return k_mat, rhs

    def solve_picard(
        self,
        tol_head: float = 1e-6,
        tol_residual: float = 1e-9,
        max_iter: int = 100,
        initial_omega: float = 0.8,
    ) -> PicardResult:
        """Solve the steady-state Richards equation via Picard iteration.

        Implements the adaptive under-relaxation ladder and hybrid head/residual
        convergence controls described in Sections 4.1, 5.1, and 5.2 of
        ``02-DOCS/wiki/non_saturated_flow.md``.

        Parameters
        ----------
        tol_head : float, default=1e-6
            Dimensionless relative tolerance for scaled :math:`L_\\infty` head increment
            (:math:`\\|h^{m+1} - h^m\\|_\\infty \\le \\text{tol}_h \\cdot H_{\\text{scale}}`).
        tol_residual : float, default=1e-9
            Absolute tolerance on unlagged nonlinear nodal PDE/flux residual.
        max_iter : int, default=100
            Maximum allowed Picard sweeps.
        initial_omega : float, default=0.8
            Starting relaxation factor :math:`\\omega_0 \\in (0, 1]`.

        Returns
        -------
        PicardResult
            Converged solution fields and iteration history.
        """
        # Step 0: Solve saturated potential problem (kr = 1) as physical initial state
        k_sat_mat, f_sat = self.assemble_system(np.ones(self.n_nodes), unsaturated=False)
        h_potential = np.asarray(spla.spsolve(k_sat_mat, f_sat), dtype=np.float64)
        h = h_potential.copy()

        y_span = float(self.coords[:, 1].max() - self.coords[:, 1].min())
        bc_span = abs(self.upstream_head - self.downstream_head)
        h_scale = max(y_span, bc_span)

        # Adaptive under-relaxation ladder (non_saturated_flow.md Section 5.1)
        ladder = [initial_omega, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        rung = 0
        best_diff = float("inf")
        stalled_sweeps = 0
        record_low_streak = 0
        step_up_threshold = 10

        head_diff_history: list[float] = []
        residual_history: list[float] = []
        omega_history: list[float] = []
        converged = False
        it_used = 0

        for it in range(1, max_iter + 1):
            it_used = it
            kr_m = self.evaluate_nodal_kr(h)
            k_mat, rhs = self.assemble_system(kr_m, unsaturated=True)

            # Unlagged nonlinear residual ||K(h^m) h^m - F||_inf before update
            nl_res = float(np.max(np.abs(k_mat @ h - rhs)))

            h_star = np.asarray(spla.spsolve(k_mat, rhs), dtype=np.float64)
            raw_diff = float(np.max(np.abs(h_star - h)))

            # Dynamic ladder adjustment
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

            # Evaluate post-update nonlinear residual
            kr_next = self.evaluate_nodal_kr(h_next)
            k_next, rhs_next = self.assemble_system(kr_next, unsaturated=True)
            post_res = float(np.max(np.abs(k_next @ h_next - rhs_next)))

            head_diff_history.append(raw_diff)
            residual_history.append(post_res if post_res < nl_res else nl_res)
            omega_history.append(omega)
            h = h_next

            if raw_diff <= tol_head * h_scale and post_res <= tol_residual and step_diff <= tol_head * h_scale:
                converged = True
                break

        psi = h - self.coords[:, 1]
        kr_final = self.evaluate_nodal_kr(h)
        theta_final = volumetric_water_content(psi, self.materials["rock"])
        theta_final[self.clay_nodes] = volumetric_water_content(
            psi[self.clay_nodes], self.materials["clay"]
        )
        theta_final[self.foundation_nodes] = volumetric_water_content(
            psi[self.foundation_nodes], self.materials["foundation"]
        )
        k_eff = self.nodal_ksat * kr_final

        return PicardResult(
            head=h,
            head_potential=h_potential,
            psi=psi,
            theta=theta_final,
            kr=kr_final,
            k_eff=k_eff,
            iterations=it_used,
            converged=converged,
            head_diff_history=head_diff_history,
            residual_history=residual_history,
            omega_history=omega_history,
        )


def _draw_dam_interfaces(ax: plt.Axes, color: str = "white", lw: float = 0.9, alpha: float = 0.7) -> None:
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
    """Draw the outer dam-foundation perimeter and optional subtle material tint."""
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
        ax.add_patch(
            Polygon(outer_xy, closed=True, facecolor="#f3f4f7", edgecolor="none", zorder=0)
        )
        ax.add_patch(
            Polygon(clay_xy, closed=True, facecolor="#e1e4ec", edgecolor="none", zorder=0.5)
        )
    ax.add_patch(
        Polygon(outer_xy, closed=True, facecolor="none", edgecolor="0.2", linewidth=0.95, zorder=4)
    )


def save_unsaturated_figures(
    solver: UnsaturatedDamGFDMSolver,
    result: PicardResult,
    output_dir: Path | str = Path("examples/paper/figures"),
) -> list[Path]:
    """Render and export publication-grade figures for the unsaturated dam analysis.

    Parameters
    ----------
    solver : UnsaturatedDamGFDMSolver
        Configured GFDM dam solver holding coordinates, connectivity, and parameters.
    result : PicardResult
        Converged Picard solution and iteration metrics.
    output_dir : Path or str, default=Path('examples/paper/figures')
        Destination folder for saved PNG figures.

    Returns
    -------
    list[Path]
        Paths of all generated figure files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    coords = solver.coords
    triangles = solver.triangles
    x, y = coords[:, 0], coords[:, 1]
    saved_paths: list[Path] = []

    # =========================================================================
    # Figure 1: Unsaturated Total Head Contours & Phreatic Surface
    # =========================================================================
    levels_h = np.linspace(0.0, 8.0, 17)
    fig1, ax1 = plt.subplots(figsize=(10.5, 3.6), layout="constrained")
    tcf1 = ax1.tricontourf(x, y, triangles, result.head, levels=levels_h, cmap="viridis")
    ax1.tricontour(
        x, y, triangles, result.head, levels=levels_h, colors="k", linewidths=0.55, alpha=0.7
    )
    _draw_dam_interfaces(ax1, color="w", lw=0.85, alpha=0.65)
    plot_phreatic_surface(
        ax1,
        coords,
        result.head,
        triangles=triangles,
        color="#0055ff",
        linewidths=2.3,
        label=None,
    )
    cbar1 = fig1.colorbar(tcf1, ax=ax1, shrink=0.92, pad=0.015)
    cbar1.set_label("Total head $h$ (m)")
    ax1.set_xlabel("Distance $x$ (m)")
    ax1.set_ylabel("Elevation $y$ (m)")
    ax1.set_title(
        "Steady-State Unsaturated Seepage (Richards Equation — Van Genuchten & Picard)"
    )
    ax1.set_aspect("equal")
    ax1.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#0055ff",
                lw=2.3,
                label="Phreatic surface ($\\psi = 0 \\Leftrightarrow h = y$)",
            )
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.92,
    )

    p1 = out / "non_sat_head_contours.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    saved_paths.append(p1)

    # =========================================================================
    # Figure 2: Unsaturated Physical Fields (Pressure Head, Moisture, log10 K)
    # =========================================================================
    fig2, axes2 = plt.subplots(3, 1, figsize=(10.5, 8.8), sharex=True, layout="constrained")

    # Panel (a): Pressure head psi = h - y
    psi_levels = np.linspace(
        float(np.floor(result.psi.min())), float(np.ceil(result.psi.max())), 25
    )
    tcf_a = axes2[0].tricontourf(x, y, triangles, result.psi, levels=psi_levels, cmap="RdYlBu")
    axes2[0].tricontour(
        x, y, triangles, result.psi, levels=[0.0], colors="navy", linewidths=2.0
    )
    _draw_dam_interfaces(axes2[0], color="k", lw=0.7, alpha=0.45)
    cb_a = fig2.colorbar(tcf_a, ax=axes2[0], shrink=0.92, pad=0.015)
    cb_a.set_label("$\\psi = h - y$ (m)")
    axes2[0].set_ylabel("Elevation $y$ (m)")
    axes2[0].set_title(
        "(a) Pore-Water Pressure Head $\\psi$ (Suction $\\psi < 0$ Above Phreatic Line)"
    )
    axes2[0].set_aspect("equal")

    # Panel (b): Volumetric water content theta(psi)
    theta_min = float(result.theta.min())
    tpc_b = axes2[1].tripcolor(
        x, y, triangles, result.theta, shading="gouraud", cmap="YlGnBu", vmin=theta_min, vmax=0.50
    )
    axes2[1].tricontour(
        x,
        y,
        triangles,
        result.theta,
        levels=np.linspace(theta_min + 0.005, 0.495, 9),
        colors="k",
        linewidths=0.45,
        alpha=0.5,
    )
    axes2[1].tricontour(
        x, y, triangles, result.psi, levels=[0.0], colors="navy", linewidths=1.9
    )
    _draw_dam_interfaces(axes2[1], color="w", lw=0.75, alpha=0.65)
    cb_b = fig2.colorbar(tpc_b, ax=axes2[1], shrink=0.92, pad=0.015)
    cb_b.set_label("$\\theta(\\psi)$ ($\\text{m}^3/\\text{m}^3$)")
    axes2[1].set_ylabel("Elevation $y$ (m)")
    axes2[1].set_title("(b) Volumetric Water Content $\\theta(\\psi)$ (Van Genuchten SWRC)")
    axes2[1].set_aspect("equal")

    # Panel (c): Element-wise Log10 effective hydraulic conductivity K(psi)
    centroids = coords[triangles].mean(axis=1)
    clay_poly = MplPath(
        [[23.0, 10.0], [27.0, 10.0], [30.0, 0.0], [26.0, -4.0], [24.0, -4.0], [20.0, 0.0]]
    )
    is_clay_elem = clay_poly.contains_points(centroids)
    ksat_elem = np.where(
        is_clay_elem,
        solver.materials["clay"].k_sat,
        np.where(
            centroids[:, 1] < 0.0,
            solver.materials["foundation"].k_sat,
            solver.materials["rock"].k_sat,
        ),
    )
    psi_elem = result.head[triangles].mean(axis=1) - centroids[:, 1]
    kr_elem = relative_conductivity(psi_elem, solver.materials["rock"])
    log_k_elem = np.log10(ksat_elem * kr_elem)

    tpc_c = axes2[2].tripcolor(
        x,
        y,
        triangles,
        facecolors=log_k_elem,
        shading="flat",
        cmap="plasma",
        vmin=float(log_k_elem.min()),
        vmax=-4.0,
    )
    axes2[2].tricontour(
        x, y, triangles, result.psi, levels=[0.0], colors="cyan", linewidths=1.9
    )
    _draw_dam_interfaces(axes2[2], color="w", lw=0.75, alpha=0.7)
    cb_c = fig2.colorbar(tpc_c, ax=axes2[2], shrink=0.92, pad=0.015)
    cb_c.set_label("$\\log_{10} K(\\psi)$ ($\\text{m/s}$)")
    axes2[2].set_xlabel("Distance $x$ (m)")
    axes2[2].set_ylabel("Elevation $y$ (m)")
    axes2[2].set_title(
        "(c) Effective Hydraulic Conductivity $\\log_{10} K(\\psi)$ Across Layers & Vadose Zone"
    )
    axes2[2].set_aspect("equal")

    p2 = out / "non_sat_hydraulic_fields.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(p2)

    # =========================================================================
    # Figure 3: Van Genuchten Constitutive Curves & Picard Convergence History
    # =========================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")

    vg_ref = solver.materials["rock"]
    suction_m = np.logspace(-2, 2, 300)
    psi_curve = -suction_m
    se_curve = effective_saturation(psi_curve, vg_ref)
    kr_curve = relative_conductivity(psi_curve, vg_ref)
    max_dam_suction = float(max(0.01, -result.psi.min()))

    ax3a.semilogx(
        suction_m,
        se_curve,
        color="#1f77b4",
        linewidth=2.2,
        label="Effective saturation $S_e(|\\psi|)$",
    )
    ax3a.semilogx(
        suction_m,
        kr_curve,
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        label="Relative conductivity $k_r(|\\psi|)$",
    )
    ax3a.axvspan(
        0.01,
        max_dam_suction,
        color="#ffdd88",
        alpha=0.28,
        label=f"Dam suction range ($|\\psi| \\leq {max_dam_suction:.2f}$ m)",
    )
    ax3a.set_xlabel("Matric suction head $|\\psi|$ (m)")
    ax3a.set_ylabel("$S_e(|\\psi|),\\; k_r(|\\psi|)$ (dimensionless)")
    ax3a.set_ylim(-0.03, 1.05)
    ax3a.set_title(
        f"(a) Van Genuchten–Mualem ($\\alpha = 1/50\\,\\text{{kPa}}^{{-1}},\\; n = {vg_ref.n}$)"
    )
    ax3a.legend(loc="lower left", frameon=True)

    iters = np.arange(1, len(result.head_diff_history) + 1)
    ax3b.semilogy(
        iters,
        result.head_diff_history,
        marker="o",
        markersize=3.8,
        linewidth=1.8,
        color="#1f77b4",
        label="$\\|h^{m+1,\\star} - h^m\\|_\\infty$ (m)",
    )
    ax3b.semilogy(
        iters,
        result.residual_history,
        marker="s",
        markersize=3.5,
        linewidth=1.6,
        linestyle="--",
        color="#2ca02c",
        label="$\\|\\mathbf{K}(h^m)\\mathbf{h}^m - \\mathbf{F}\\|_\\infty$",
    )
    ax3b.set_xlabel("Picard iteration $m$")
    ax3b.set_ylabel("Convergence norm (log scale)")
    ax3b.set_title(f"(b) Picard Convergence ({result.iterations} Iterations, $\\omega_0 = {result.omega_history[0]}$)")
    ax3b.legend(loc="upper right", frameon=True)

    p3 = out / "non_sat_constitutive_picard.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    saved_paths.append(p3)

    # =========================================================================
    # Figure 4: Direct Comparison — (a) Phreatic Lines, (b) Equipotentials, (c) Delta h
    # =========================================================================
    fig4, (ax4a, ax4b, ax4c) = plt.subplots(
        3, 1, figsize=(10.5, 9.2), sharex=True, layout="constrained"
    )

    # Panel (a): Phreatic surface comparison (Richards vs Potential)
    _draw_dam_backdrop(ax4a, fill=True)
    _draw_dam_interfaces(ax4a, color="0.35", lw=0.8, alpha=0.65)
    ax4a.tricontour(
        x,
        y,
        triangles,
        result.psi,
        levels=[0.0],
        colors="#0044cc",
        linestyles="-",
        linewidths=2.4,
    )
    ax4a.tricontour(
        x,
        y,
        triangles,
        result.head_potential - y,
        levels=[0.0],
        colors="crimson",
        linestyles="--",
        linewidths=2.2,
    )
    ax4a.legend(
        handles=[
            Line2D([0], [0], color="#0044cc", lw=2.4, ls="-", label="Richards phreatic line (Unsaturated)"),
            Line2D([0], [0], color="crimson", lw=2.2, ls="--", label="Potential phreatic line (Saturated)"),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=8.8,
    )
    ax4a.set_ylabel("Elevation $y$ (m)")
    ax4a.set_title("(a) Phreatic Surface Comparison ($\\psi = 0 \\Leftrightarrow h = y$): Richards vs Potential")
    ax4a.set_aspect("equal")

    # Panel (b): Equipotential curves comparison (Unsaturated vs Saturated)
    iso_levels = np.linspace(0.5, 7.5, 15)
    _draw_dam_backdrop(ax4b, fill=True)
    _draw_dam_interfaces(ax4b, color="0.35", lw=0.8, alpha=0.65)
    ax4b.tricontour(
        x,
        y,
        triangles,
        result.head_potential,
        levels=iso_levels,
        colors="k",
        linestyles="--",
        linewidths=1.0,
    )
    ax4b.tricontour(
        x,
        y,
        triangles,
        result.head,
        levels=iso_levels,
        colors="#0044cc",
        linestyles="-",
        linewidths=1.15,
    )
    ax4b.legend(
        handles=[
            Line2D([0], [0], color="#0044cc", lw=1.4, ls="-", label="Unsaturated equipotentials (Richards)"),
            Line2D([0], [0], color="k", lw=1.15, ls="--", label="Saturated equipotentials (Potential)"),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=8.8,
    )
    ax4b.set_ylabel("Elevation $y$ (m)")
    ax4b.set_title("(b) Equipotential Curves ($\\Delta h = 0.5$ m): Unsaturated vs Saturated")
    ax4b.set_aspect("equal")

    # Panel (c): Hydraulic head difference Delta h
    delta_h = result.head - result.head_potential
    vmax = float(max(abs(delta_h.min()), abs(delta_h.max())))
    diff_levels = np.linspace(-vmax, vmax, 23)
    tcf_d = ax4c.tricontourf(x, y, triangles, delta_h, levels=diff_levels, cmap="coolwarm")
    ax4c.tricontour(x, y, triangles, delta_h, levels=11, colors="k", linewidths=0.4, alpha=0.5)
    _draw_dam_interfaces(ax4c, color="k", lw=0.75, alpha=0.55)
    _draw_dam_backdrop(ax4c, fill=False)
    cb_d = fig4.colorbar(tcf_d, ax=ax4c, shrink=0.92, pad=0.015)
    cb_d.set_label("$\\Delta h = h_{\\text{Richards}} - h_{\\text{Potential}}$ (m)")
    ax4c.set_xlabel("Distance $x$ (m)")
    ax4c.set_ylabel("Elevation $y$ (m)")
    ax4c.set_title(
        f"(c) Hydraulic Head Difference $\\Delta h$ (RMS $\\Delta h = {np.sqrt(np.mean(delta_h ** 2)):.4f}$ m, "
        f"Max $|\\Delta h| = {vmax:.4f}$ m)"
    )
    ax4c.set_aspect("equal")

    p4 = out / "non_sat_vs_potential_comparison.png"
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    saved_paths.append(p4)

    return saved_paths


def main() -> None:
    """Execute the steady-state unsaturated GFDM seepage simulation and export figures."""
    print("==========================================================================")
    print("GFDFlow — Unsaturated Seepage in Five-Layer Dam (Richards + Van Genuchten)")
    print("==========================================================================")

    mesh_path = Path("examples/paper/mesh1.json")
    print(f"Loading precomputed mesh and stencils from: {mesh_path}")
    mesh_data = load_mesh_data(mesh_path)

    # Chavez et al. (2023), Table 2 soil parameters
    materials = {
        "rock": VanGenuchtenParams(
            name="Rockfill", k_sat=1e-4, theta_s=0.50, theta_r=0.0, alpha_kpa=1.0 / 50.0, n=1.4
        ),
        "clay": VanGenuchtenParams(
            name="Clay Core", k_sat=1e-6, theta_s=0.50, theta_r=0.0, alpha_kpa=1.0 / 50.0, n=1.4
        ),
        "foundation": VanGenuchtenParams(
            name="Soil Foundation",
            k_sat=1e-4,
            theta_s=0.50,
            theta_r=0.0,
            alpha_kpa=1.0 / 50.0,
            n=1.4,
        ),
    }

    solver = UnsaturatedDamGFDMSolver(mesh_data, materials)
    print(
        f"Initialized GFDM solver: {solver.n_nodes} nodes, {len(solver.triangles)} triangles."
    )

    print("Running Picard fixed-point iteration with adaptive under-relaxation...")
    result = solver.solve_picard(tol_head=1e-6, tol_residual=1e-9, max_iter=100, initial_omega=0.8)

    status_str = "CONVERGED" if result.converged else "MAX_ITER REACHED"
    print(f"Solver status: {status_str} in {result.iterations} iterations.")
    print(
        f"Total head h       : min = {result.head.min():.6f} m, "
        f"max = {result.head.max():.6f} m, mean = {result.head.mean():.4f} m"
    )
    print(
        f"Pressure head psi  : min = {result.psi.min():.4f} m, "
        f"max = {result.psi.max():.4f} m"
    )
    print(
        f"Water content theta: min = {result.theta.min():.4f}, "
        f"max = {result.theta.max():.4f}"
    )
    print(
        f"Relative perm. kr  : min = {result.kr.min():.4e}, "
        f"max = {result.kr.max():.4e}"
    )
    delta_h = result.head - result.head_potential
    print(
        f"Richards vs Potent.: RMS diff = {np.sqrt(np.mean(delta_h ** 2)):.4f} m, "
        f"max |diff| = {np.max(np.abs(delta_h)):.4f} m"
    )

    fig_dir = Path("examples/paper/figures")
    saved = save_unsaturated_figures(solver, result, output_dir=fig_dir)
    for path in saved:
        print(f"Saved figure: {path}")

    print("==========================================================================")
    print("Unsaturated Richards analysis completed successfully.")
    print("==========================================================================")


if __name__ == "__main__":
    main()
