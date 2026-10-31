---
author: RicardoRG73
branch: paper
status: in-progress
---
# Transient Unsaturated Seepage Solver for Five-Layer Dam (`tr_non_sat.py`)

## Intent
Implement `examples/paper/tr_non_sat.py` to solve the 2D transient (non-stationary) unsaturated Richards seepage problem on the five-layer dam of Chavez et al. (2023)—the transient counterpart of `examples/paper/st_non_sat.py`—grounded directly in the XSLOPE / Celia et al. (1990) transient seepage theory documented in Part IV (Section 16) of `02-DOCS/wiki/theory.md`.

## Scope
- In scope:
  - Implement `examples/paper/tr_non_sat.py` with Python 3.11+ strict type annotations, NumPy docstrings, and `scipy.sparse` + `scipy.sparse.linalg.spsolve`:
    - Mixed mass-conservative continuity equation $\frac{\partial \Phi(\psi)}{\partial t} = \nabla \cdot (K(\psi)\nabla h)$ with zone-separated saturated elastic storage $S_s$ ($\psi \ge 0$), van Genuchten drainage capacity $C(\psi) = d\theta/d\psi$ ($\psi < 0$), and unsaturated residual capacity floor $S_{\text{floor}} = 10^{-4} C_{\max}$ integrated consistently into $\Phi(\psi)$.
    - Vectorized strong-form GFDM spatial assembly matching `st_non_sat.py` across interior subdomains, ghost-eliminated Neumann boundaries, six flux-continuous two-layer interfaces, four interface-boundary corners, and two triple junctions.
    - Nonlinear $\theta$-method time stepping (default Backward Euler $\theta = 1.0$) with diagonal HRZ/control-volume lumped storage, envelope Jacobian linearization $\tilde{C}_i = \max(C_{\text{chord}, i}, C_{\text{tangent}, i})$, exact conservative RHS residual, and per-node sign-reversal damping.
    - Adaptive time-stepping controller combining Backward-Euler local truncation error estimation (`time_error_frac`), free-node head-change limiting (excluding Dirichlet jumps), convergence-based step growth/halving, and exact saved-frame schedule clamping.
    - Post-processing of 7-point triangle quadrature domain water storage $W(t) = \sum_e W_e(t)$, separate boundary inflow/outflow rates ($Q_{\text{in}}(t), Q_{\text{out}}(t)$), cumulative mass-balance ledger, Darcy velocity vectors $\mathbf{v} = -K(\psi)\nabla h$ on transient frames (where $\nabla \cdot \mathbf{v} \neq 0$ precludes stream-function flow nets), and asymptotic convergence verification against `st_non_sat.py`.
  - Generate and visually inspect publication-grade transient unsaturated figures in `examples/paper/figures/`.
  - Run `pytest` suite to ensure zero regressions.
- Out of scope:
  - Modifying core package files under `src/GFDFlow/`.
  - Modifying `examples/paper/mesh1.json` or `examples/paper/st_non_sat.py`.

## Checklist
- [x] Implement `examples/paper/tr_non_sat.py` with vectorized GFDM Richards assembly, mixed mass-conservative storage, envelope Picard linearization, adaptive step controller, Darcy velocity post-processing, and mass-balance ledger (proof: clean execution in `gfdflow` Python 3.11 environment with exit code `0`, physical head bounds $0 \le h \le 8$ m at all saved frames, and steady-state convergence $\|h(t_{\text{end}}) - h_{\text{st\_non\_sat}}\|_\infty < 10^{-5}$ m).
- [x] Export and visually inspect publication figures in `examples/paper/figures/` (`tr_non_sat_snapshots.png`, `tr_non_sat_moisture_velocity.png`, `tr_non_sat_phreatic_evolution.png`, `tr_non_sat_convergence_budget.png`) (proof: visual PNG inspection confirms clean layout, non-overlapping legends/titles, and physically consistent wetting front and Darcy vectors).
- [x] Run `pytest` regression suite in `gfdflow` environment (proof: `pytest` exits with code `0`).

## Evidence
- **Solver & steady-state verification (`examples/paper/tr_non_sat.py`)**:
  - Command `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /home/ricardo/miniconda3/envs/gfdflow/bin/python -u examples/paper/tr_non_sat.py` exited with code `0` in `20.22 s`.
  - Spatial operator parity against `examples/paper/st_non_sat.py`: $\max |K_{\text{tr}} - K_{\text{st}}| = 2.168 \times 10^{-19}$; steady baseline head parity $\max |h_{\text{st,tr}} - h_{\text{st,non\_sat}}| = 2.398 \times 10^{-14}\text{ m}$ (`27` Picard sweeps).
  - Adaptive Backward-Euler ($\theta = 1.0$) trajectory over $t \in [0, 100]\text{ d}$: `261` accepted steps (`11` rejected), `3076` total Picard sweeps (`11.8` avg/step).
  - Final departure from `st_non_sat.py` at $t = 100\text{ d}$: $\|h(100\text{ d}) - h_{\text{st}}\|_\infty = 2.795 \times 10^{-7}\text{ m}$, total 7-point quadrature stored water uptake $\Delta W = 9.2590\text{ m}^3/\text{m}$ ($W_0 = 219.9832 \to W_{100\text{ d}} = 229.2422\text{ m}^3/\text{m}$), cumulative Celia mixed-form mass-balance closure error $= 0.00006\%$.
- **Visual inspection (`examples/paper/figures/`)**:
  - Inspected `tr_non_sat_snapshots.png`, `tr_non_sat_moisture_velocity.png`, `tr_non_sat_phreatic_evolution.png`, and `tr_non_sat_convergence_budget.png` directly via image viewer; verified clean 300-DPI typography, non-overlapping legends, clear vermilion-orange Darcy vectors inside the perimeter, and exact phreatic convergence onto `st_non_sat.py`.
- **Unit & regression tests (`pytest`)**:
  - `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /home/ricardo/miniconda3/envs/gfdflow/bin/python -m pytest -q` -> `26 passed in 248.24s (0:04:08)` (exit code `0`).

## Next
Commit `examples/paper/tr_non_sat.py` and generated figures on branch `paper` when requested.
