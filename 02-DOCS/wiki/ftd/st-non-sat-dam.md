---
author: Ricardo
branch: paper
status: in-progress
---
# Unsaturated Seepage Solver for Five-Layer Dam (`st_non_sat.py`) and Potential Fix (`st_potential1.py`)

## Intent
Analyze the steady-state saturated potential flow implementation in `examples/paper/st_potential1.py`, fix its stale `stat_mesh1.json` path and swapped `interface_e`/`interface_f` subdomain assignments, and implement `examples/paper/st_non_sat.py` to solve the 2D steady-state unsaturated Richards equation on the same five-layer dam (Chavez et al., 2023, Section 4.2.3, Table 2) using the Van Genuchten–Mualem constitutive model and Picard fixed-point iteration with adaptive under-relaxation documented in `02-DOCS/wiki/non_saturated_flow.md`.

## Scope
- In scope:
  - Technical and numerical audit of `examples/paper/st_potential1.py`.
  - Fix mesh path (`examples/paper/mesh1.json`), `interface_e`/`interface_f` material orientation, and `tj2` wedge assignments in `examples/paper/st_potential1.py`.
  - Create `examples/paper/st_non_sat.py` implementing:
    - Van Genuchten (1980) – Mualem (1976) retention ($S_e, \theta, C$) and relative conductivity $k_r(\psi)$ with $k_{r,\min} = 10^{-8}$ numerical floor.
    - Strong-form GFDM assembly of $\nabla \cdot (K(\psi) \nabla h) = 0$ via logarithmic conductivity gradients, ghost-node Neumann boundaries, flux-continuous two-layer interfaces, interface-boundary corners, and three-wedge triple junctions using `scipy.sparse` matrices.
    - Picard fixed-point solver with the dynamic under-relaxation ladder and hybrid head/residual convergence check from `02-DOCS/wiki/non_saturated_flow.md`.
  - Generate and verify publication figures in `examples/paper/figures/`.
- Out of scope:
  - Modifying core package modules under `src/GFDFlow/`.
  - Modifying `examples/paper/mesh1.py` or `examples/paper/mesh1.json`.
  - Transient time-stepping simulations.

## Checklist
- [x] Fix `mesh1.json` path and `interface_e`/`interface_f` & `tj2` subdomain mapping in `examples/paper/st_potential1.py` (proof: `st_potential1.py` runs with exit code 0 and updates `solution_contours.png`). — Rationale: `normal_vecs` in `mesh1.json` point from `foundation`/`rock` into `clay` on `interface_e` (`y in [-4, 0]`) and `interface_f` (`y in [0, 10]`).
- [x] Implement `examples/paper/st_non_sat.py` with NumPy docstrings, type annotations, Van Genuchten–Mualem constitutive functions, sparse GFDM Richards assembly, and Picard iteration with adaptive relaxation (proof: syntax/type check + clean execution in `gfdflow` Python 3.11 env). — Rationale: expanding $\nabla \cdot (K(\psi)\nabla h) = K(\psi)(\nabla^2 h + \nabla(\ln k_r)\cdot\nabla h) = 0$ and scaling interface ghost stencils by `Gg` preserves the Discrete Maximum Principle ($0 \le h \le 8$ m) and exact outward Darcy flux continuity across contrasting mesh sizes.
- [x] Generate and visually verify unsaturated seepage figures in `examples/paper/figures/` (`non_sat_head_contours.png`, `non_sat_hydraulic_fields.png`, `non_sat_constitutive_picard.png`, `non_sat_vs_potential_comparison.png`) (proof: files exist and visual inspection confirms clean layout and physical bounds $0 \le h \le 8$ m).
- [x] Run existing test suite (`pytest`) in `gfdflow` environment to ensure zero regressions (proof: `pytest` exit code 0).
- [x] Split `non_sat_vs_potential_comparison.png` into 3 vertically stacked panels: `(a)` phreatic surface comparison (Richards vs Potential), `(b)` equipotential curves comparison (Unsaturated vs Saturated), and `(c)` hydraulic head difference $\Delta h$ (proof: `st_non_sat.py` runs with exit code 0 and visual inspection of `non_sat_vs_potential_comparison.png` confirms 3 clean panels `(a)`, `(b)`, `(c)`).
- [x] Restore the original contour count (`levels=11` isolines and `23` fill levels) in panel `(c)` of `non_sat_vs_potential_comparison.png` (proof: `st_non_sat.py` runs with exit code 0 and visual inspection confirms clean original contour density in `(c)`).

## Evidence
- `MPLBACKEND=Agg python examples/paper/st_potential1.py` → Exit code `0`; `1490 nodes, 2778 triangles`; `Total head range: min = 0.0000 m, max = 8.0000 m (mean = 3.7435 m)`; saved `examples/paper/figures/solution_contours.png`.
- `MPLBACKEND=Agg python examples/paper/st_non_sat.py` → Exit code `0` (zero warnings); Picard `CONVERGED in 19 iterations` (`tol_head = 1e-6`, `tol_residual = 1e-9`, `omega_0 = 0.8`); `h in [0.000000, 8.000000] m` (`mean = 3.8040 m`), `psi in [-8.8682, 12.0000] m`, `theta in [0.3595, 0.5000]`, `kr in [8.9224e-03, 1.0000]`; comparison vs saturated potential: `RMS diff = 0.5866 m`, `max |diff| = 2.9912 m`.
- Visual inspection of all 5 PNGs in `examples/paper/figures/` (`solution_contours.png`, `non_sat_head_contours.png`, `non_sat_hydraulic_fields.png`, `non_sat_constitutive_picard.png`, `non_sat_vs_potential_comparison.png`) confirmed clean legends, full saturated/vadose color coverage without contour clipping, and convex equipotential arching in the clay core matching Chavez et al. (2023, Figures 16–17).
- Updated `save_unsaturated_figures` in `examples/paper/st_non_sat.py` and re-ran `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python examples/paper/st_non_sat.py` (exit code `0`): visual inspection of `examples/paper/figures/non_sat_vs_potential_comparison.png` confirmed 3 aligned subfigures `(a)` phreatic lines, `(b)` equipotential curves ($\Delta h = 0.5$ m), and `(c)` $\Delta h$ map restored to the clean original contour density (`23` fill levels and `levels=11` isolines).
- `MPLBACKEND=Agg python -m pytest -q tests/test_gfdmi.py tests/test_visualization.py` → `13 passed in 0.49s` (exit code `0`), plus constitutive unit invariants verified (`alpha_m = 0.1962 m^-1`, `m = 2/7`, `S_e(0) = 1.0`, `k_r(-1e6) = 1e-8`).

## Next
Review generated figures and commit `st_potential1.py`, `st_non_sat.py`, and `examples/paper/figures/*.png` on branch `paper` (or add an exit-face seepage boundary sweep if needed for the manuscript).
