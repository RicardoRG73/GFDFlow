---
author: RicardoRG73
branch: paper
status: in-progress
---
# Crank-Nicolson Theory, Unified GFDFlow Theory Monograph, and Transient Potential Dam Seepage (`tr_potential.py`)

## Intent
Investigate and document the mathematical theory of the Crank-Nicolson temporal discretization scheme in `02-DOCS/wiki/theory.md`, consolidate `02-DOCS/wiki/non_saturated_flow.md` into `02-DOCS/wiki/theory.md` with a complete section index at the top while removing the redundant `non_saturated_flow.md` file, and implement `examples/paper/tr_potential.py` to solve the 2D transient groundwater potential flow problem ($S_s \frac{\partial u}{\partial t} = \nabla \cdot (k \nabla u)$ with $S_s = 1$) for the five-layer dam of Chavez et al. (2023) using Crank-Nicolson until converging to the stationary baseline in `examples/paper/st_potential1.py`.

## Scope
- In scope:
  - Add a complete clickable Table of Contents (section index) at the start of `02-DOCS/wiki/theory.md`.
  - Write a rigorous mathematical section on the Crank-Nicolson method in `02-DOCS/wiki/theory.md` (Method-of-Lines GFDM semi-discretization, generalized $\theta$-family, $\mathcal{O}(\Delta t^2 + \bar{h}^2)$ Taylor truncation error, spectral/von Neumann $A$-stability vs. lack of $L$-stability, positivity condition, Rannacher Backward-Euler startup smoothing, and sparse `GFDFlow` implementation).
  - Merge the full contents of `02-DOCS/wiki/non_saturated_flow.md` into `02-DOCS/wiki/theory.md` without loss of equations, tables, diagrams, or references, and delete `02-DOCS/wiki/non_saturated_flow.md`.
  - Implement `examples/paper/tr_potential.py` using `scipy.sparse` and `scipy.sparse.linalg.spsolve`, NumPy docstrings, strict type annotations, explicit temporal storage term $S_s \frac{du}{dt}$ with $S_s = 1.0$, initial state $u(x, y, 0) = 0\text{ m}$ (with compatible Dirichlet boundary values), Crank-Nicolson time marching, verification of convergence to `st_potential1.py` and $\mathcal{O}(\Delta t^2)$ temporal accuracy, and publication figures in `examples/paper/figures/`.
- Out of scope:
  - Modifying core library modules under `src/GFDFlow/`.
  - Regenerating `examples/paper/mesh1.json`.
  - Transient unsaturated Richards time-stepping (`tr_non_sat.py`).

## Checklist
- [x] Research and write Crank-Nicolson theory, merge `02-DOCS/wiki/non_saturated_flow.md` into `02-DOCS/wiki/theory.md`, and add a complete section index at the top of `02-DOCS/wiki/theory.md` (proof: anchor verification script confirms zero broken ToC links and complete section coverage). — Rationale: organized into Part I (GFDM Foundations, Sections 1–8), Part II (Crank-Nicolson Parabolic Theory, Section 9), Part III (Unsaturated Seepage, Sections 10–17), and unified References (Section 18).
- [x] Delete `02-DOCS/wiki/non_saturated_flow.md` after verifying lossless merge into `02-DOCS/wiki/theory.md` (proof: `test ! -e 02-DOCS/wiki/non_saturated_flow.md` exits 0 and `ls 02-DOCS/wiki/` confirms clean directory).
- [x] Implement `examples/paper/tr_potential.py` solving transient potential seepage via Crank-Nicolson (`scipy.sparse` + `spsolve`) and verify convergence to `st_potential1.py` plus $\mathcal{O}(\Delta t^2)$ temporal order (proof: execution in `gfdflow` Python 3.11 env exits 0, physical bounds $0 \le u \le 8$ m hold at all steps, $\|u(t_{\text{end}}) - u_{\text{steady}}\|_\infty < 10^{-5}$ m, and empirical temporal order $\approx 2.0$). — Rationale: implemented $S_s \frac{du}{dt} = \mathbf{K}\boldsymbol{u} - \boldsymbol{F}$ with $S_s = 1.0$, algebraic enforcement of Dirichlet rows on $\mathcal{D}$, and 4 Rannacher Backward-Euler half-steps at $t = 0^+$.
- [x] Generate and visually inspect transient figures (`tr_potential_snapshots.png`, `tr_potential_phreatic_evolution.png`, `tr_potential_convergence.png`) and run `pytest` regression suite (proof: PNG inspection confirms clean rendering and `pytest` exits 0).

## Evidence
- **Unified Theory Monograph & Section Index (`02-DOCS/wiki/theory.md`)**:
  - Anchor verification script output: `Total lines: 1120, ToC links checked: 47, broken anchors: 0` (exit code `0`).
- **Directory Cleanliness (`02-DOCS/wiki/non_saturated_flow.md` deleted)**:
  - `test ! -e 02-DOCS/wiki/non_saturated_flow.md && ls -la 02-DOCS/wiki/` exited `0`; `git status --short` shows `D 02-DOCS/wiki/non_saturated_flow.md` and `M 02-DOCS/wiki/theory.md`.
- **Crank-Nicolson Transient Dam Solver (`examples/paper/tr_potential.py`)**:
  - Executed `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /home/ricardo/miniconda3/envs/gfdflow/bin/python examples/paper/tr_potential.py` (exit code `0`, 1,192 sparse solves via `scipy.sparse.linalg.spsolve`).
  - Discrete Maximum Principle bounds across all time steps: `min = -0.000000 m, max = 8.000000 m`.
  - Steady-state convergence against `st_potential1.py` at $t = 1000\text{ d}$: `L_inf = 1.788e-06 m`, `RMSE = 4.688e-07 m`.
  - Empirical Crank-Nicolson temporal convergence order at $T = 2\text{ d}$ ($\Delta t \in \{10800, 5400, 2700, 1350\}\text{ s}$ vs. $N_{\text{ref}} = 1024$): pairwise $L_2$ orders `[2.004, 2.005, 2.017]`, least-squares log-log slope `2.008`.
- **Visual Figure Verification & Regression Tests**:
  - Visually inspected `examples/paper/figures/tr_potential_snapshots.png`, `examples/paper/figures/tr_potential_phreatic_evolution.png`, and `examples/paper/figures/tr_potential_convergence.png` (300 DPI, zero overlapping labels or legend collisions).
  - Full test suite `MPLBACKEND=Agg /home/ricardo/miniconda3/envs/gfdflow/bin/python -m pytest -q`: `26 passed in 228.46s (0:03:48)` (exit code `0`).

## Next
Review generated transient figures and commit changes on branch `paper` when ready.
