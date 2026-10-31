---
author: RicardoRG73
branch: paper
status: in-progress
---
# Transient Saturated (Potential) vs. Unsaturated (Richards) Dam Seepage Comparison (`tr_comparison.py`)

## Intent
Create `examples/paper/tr_comparison.py` to perform a rigorous, apples-to-apples transient comparison between the saturated potential seepage model ($S_s \partial u / \partial t = \nabla \cdot (K_s \nabla u)$, $k_r \equiv 1$) and the unsaturated mixed Richards model ($\partial \Phi(\psi) / \partial t = \nabla \cdot (K(\psi) \nabla h)$) at identical physical instants $t \in \{0.02, 0.10, 1.0, 5.0, 15.0, 100.0\}\text{ d}$ using matched physical elastic specific storage $S_s$ per soil layer (`rock`: $10^{-4}\text{ m}^{-1}$, `clay`: $10^{-3}\text{ m}^{-1}$, `foundation`: $2 \times 10^{-4}\text{ m}^{-1}$).

## Scope
- In scope:
  - Implement `examples/paper/tr_comparison.py` with Python 3.11+ strict typing and NumPy docstrings:
    - Assemble and march the transient saturated potential model using the exact same HRZ/sub-element lumped elastic storage $S_{s, e}$ per material zone (`rock`, `clay`, `foundation`), same initial condition $u(\boldsymbol{x}, 0) = 0\text{ m}$ ($u_{\text{left}}(0^+) = 8.0\text{ m}, u_{\text{right}}(0^+) = 0.0\text{ m}$), and exact saved-frame clamping at $t \in \{0.02, 0.10, 1.0, 5.0, 15.0, 100.0\}\text{ d}$.
    - Run `TransientUnsaturatedDamSolver` at the identical timestamps and physical $S_s$ parameters.
    - Quantify and plot:
      1. `tr_comp_phreatic_equipotentials.png`: 6-panel (`3 x 2`) evolution at $t = 0.02, 0.10, 1.0, 5.0, 15.0, 100.0\text{ d}$ overlaying saturated potential vs. unsaturated Richards phreatic surfaces ($\psi = 0$) and equipotential contours ($\Delta h = 1.0\text{ m}$), highlighting the dramatic hydraulic lag of the unsaturated wetting front.
      2. `tr_comp_head_difference.png`: 6-panel (`3 x 2`) spatial head difference $\Delta h(x, y, t) = h_{\text{Richards}}(x, y, t) - u_{\text{Potential}}(x, y, t)$ across all 6 instants, capturing the transient capillary lag deficit ($\Delta h < 0$, up to $\sim -7.5\text{ m}$) and its relaxation to the steady-state Richards–Potential difference of `st_non_sat.py`.
      3. `tr_comp_side_by_side_snapshots.png`: Direct visual comparison matrix (`6 rows x 2 columns` or `3 instants x 2 columns` + full `6 x 2` paired head contour & phreatic evolution) directly contrasting the saturated potential head field against the unsaturated Richards head field at identical physical times.
      4. `tr_comp_piezometers_storage_lag.png`: `2 x 2` quantitative comparison of (a) paired piezometer histories $u_{\text{Pot}}(t)$ vs. $h_{\text{Rich}}(t)$ on log-time axis, (b) relaxation to respective steady states $\|u(t) - u_{\text{st}}\|_\infty$ vs. $\|h(t) - h_{\text{st}}\|_\infty$ (quantifying equilibration time ratio), (c) transient RMS & max $|\Delta h(t)| = |h_{\text{Rich}}(t) - u_{\text{Pot}}(t)|$, and (d) cumulative stored water $\Delta W_{\text{Pot}}(t)$ (pure elastic $S_s$) vs. $\Delta W_{\text{Rich}}(t)$ (vadose pore-filling $S_y + S_s$).
  - Visually inspect all generated PNGs in `examples/paper/figures/` and verify clean execution and unit tests.
- Out of scope:
  - Modifying `src/GFDFlow/`, `examples/paper/st_non_sat.py`, `examples/paper/tr_potential.py`, or `examples/paper/mesh1.json`.

## Checklist
- [x] Implement `examples/paper/tr_comparison.py` comparing transient saturated potential vs. unsaturated Richards seepage at identical physical instants $t \in \{0.02, 0.10, 1.0, 5.0, 15.0, 100.0\}\text{ d}$ with matched zone-wise physical $S_s$ (proof: exits with code `0`, both models converge to their respective steady solutions `st_potential1.py` / `st_non_sat.py` with $L_\infty < 10^{-6}\text{ m}$).
- [x] Export and visually inspect comparison figures in `examples/paper/figures/` (`tr_comp_side_by_side_snapshots.png`, `tr_comp_phreatic_equipotentials.png`, `tr_comp_head_difference.png`, `tr_comp_piezometers_storage_lag.png`) (proof: visual PNG inspection shows zero overlapping text/legends and clear physical contrast).
- [x] Verify clean workspace state and pass `pytest` unit checks (proof: `pytest` passes with exit code `0` and `src/GFDFlow/` has zero modified files).

## Evidence
- **Synchronized solver & physical lag metrics (`examples/paper/tr_comparison.py`)**:
  - Command `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /home/ricardo/miniconda3/envs/gfdflow/bin/python -u examples/paper/tr_comparison.py` exited with code `0` (`24.55 s`).
  - Matched zone-wise physical $S_s$: `rock = 1.0e-4 m^-1`, `clay = 1.0e-3 m^-1`, `foundation = 2.0e-4 m^-1`.
  - 99% equilibration times: Saturated Potential $t_{99\%} = 0.257\text{ d}$ ($6.17\text{ h}$) vs. Unsaturated Richards $t_{99\%} = 33.11\text{ d}$ (**$128.6\times$ slower** due to capillary pore-filling across low-$k_r(\psi)$ suction zones).
  - Domain water storage uptake: Potential $\Delta W_{\text{Pot}} = 0.5705\text{ m}^3/\text{m}$ (elastic $S_s$ only) vs. Richards $\Delta W_{\text{Rich}} = 9.2590\text{ m}^3/\text{m}$ (**$16.2\times$ larger** vadose pore-filling demand).
  - Asymptotic verification at $t = 100\text{ d}$: $\|u_{\text{Pot}}(100\text{ d}) - u_{\text{st}}\|_\infty = 4.75 \times 10^{-12}\text{ m}$, $\|h_{\text{Rich}}(100\text{ d}) - h_{\text{st}}\|_\infty = 2.53 \times 10^{-7}\text{ m}$; stationary Richards–Potential offset $\max |h_{\text{Rich}} - u_{\text{Pot}}| = 2.991\text{ m}$, $\text{RMSE} = 0.587\text{ m}$ (matching `st_non_sat.py`).
- **Visual inspection (`examples/paper/figures/`)**:
  - Visually inspected all 4 comparison PNGs (`tr_comp_side_by_side_snapshots.png`, `tr_comp_phreatic_equipotentials.png`, `tr_comp_head_difference.png`, `tr_comp_piezometers_storage_lag.png`); verified zero legend/title overlaps and crisp 300-DPI rendering.
- **Workspace & regression checks**:
  - `pytest -q tests/test_gfdmi.py tests/test_visualization.py` -> `13 passed in 0.55s` (and full `pytest -q` suite `26 passed in 248.24s`).
  - `git status --short` confirms zero modified files in `src/GFDFlow/`, `examples/paper/st_non_sat.py`, `examples/paper/tr_potential.py`, or `examples/paper/mesh1.json`.

## Next
Commit `examples/paper/tr_non_sat.py`, `examples/paper/tr_comparison.py`, and generated figures on branch `paper` when requested.
