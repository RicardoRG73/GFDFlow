---
author: RicardoRG73
branch: paper
status: in-progress
---
# Transient Seepage Theory and Zoned Earth Dam Benchmark in `theory.md`

## Intent
Expand `02-DOCS/wiki/theory.md` with the complete mathematical and algorithmic theory of Transient Seepage from XSLOPE (`https://xslope.org/en/latest/seep/transient/`), complement it with the benchmark example *2. Earth Dam with Core* (`https://xslope.org/en/latest/seep/samples/#2-earth-dam-with-core`) including its transient reservoir-drawdown extension, and store and embed the two official XSLOPE figures (`earth_dam1_problem_sketch.png` and `earth_dam1_solution.png`).

## Scope
- In scope:
  - Store `earth_dam1_problem_sketch.png` and `earth_dam1_solution.png` locally under `02-DOCS/wiki/images/` for self-contained offline and Markdown rendering, citing their canonical XSLOPE URLs.
  - Add **Part IV — Transient Unsaturated and Unconfined Seepage (XSLOPE Reference Formulation and Benchmark)** to `02-DOCS/wiki/theory.md`:
    - **Section 16**: Transient seepage formulation (mixed vs. head continuity equation, sub-element water integration and nodal chord-capacity splitting, specific storage $S_s$, specific yield $S_y$, zone-separated moisture capacity $S(\psi)$ with residual dry-zone storage floor, HRZ lumped storage, nonlinear $\theta$-method time stepping with $\max(C_{\text{chord}}, C_{\text{tangent}})$ linearization, adaptive time-error and head-change step control, `tseep` time-series semantics, three-state hysteretic `reservoir` boundary and quadratic exit-face cycling fallback, divergence of transient velocity field and absence of flow nets, cumulative mass-balance ledger, and `stability_time` / `stage_1`–`stage_2` rapid-drawdown coupling).
    - **Section 17**: Benchmark case study *Earth Dam with Core* (`xslope_earth_dam1.xlsx`, Sample #2) with exact vertex coordinates, anisotropic shell/core conductivities, linear-front vs. van Genuchten steady flowrate comparison, embedded XSLOPE problem sketch and solution figures, and direct connection to transient 45-day reservoir drawdown dynamics ($v_{\text{drain}} \sim k_2 / S_y$).
  - Update the Table of Contents, Comparative Methodological Matrices (Section 18), Synthesis and Implementation Recommendations for `GFDFlow` (Section 19), and References (Section 20) in `02-DOCS/wiki/theory.md`.
- Out of scope:
  - Modifying Python source files under `src/GFDFlow/` or `examples/`.

## Checklist
- [x] Save `earth_dam1_problem_sketch.png` and `earth_dam1_solution.png` into `02-DOCS/wiki/images/` and verify PNG integrity and dimensions (proof: Python PNG header inspection and visual check confirm `1700x678` / `62975` bytes and `2134x630` / `379554` bytes).
- [x] Add Part IV (Section 16: Transient Seepage Theory; Section 17: Zoned Earth Dam with Core Benchmark & Transient Extension) and update ToC, comparative matrices, synthesis recommendations, and references in `02-DOCS/wiki/theory.md` (proof: verification script confirms 59/59 valid ToC anchors, 0 missing numbered sections, 2/2 valid embedded images, 28/28 intact cross-repo `theory.md#...` anchors, and 0 unbalanced LaTeX delimiters). — Rationale: structured as Part IV (Sections 16–17) followed by Part V (Sections 18–20) so all existing links to Sections 1–15 remain untouched.

## Evidence
- **XSLOPE Benchmark Figures (`02-DOCS/wiki/images/`)**:
  - `earth_dam1_problem_sketch.png`: `PNG_OK= True 1700x678 bytes= 62975` (visually inspected via `read_file`).
  - `earth_dam1_solution.png`: `PNG_OK= True 2134x630 bytes= 379554` (visually inspected via `read_file`).
- **Monograph Integrity Verification (`02-DOCS/wiki/theory.md`)**:
  - `Total lines: 1403`
  - `ToC links checked: 59, broken_toc: 0 []`
  - `Numbered headings (## X. / ### X.Y.): 59, missing_from_toc: 0 []`
  - `Images checked: [('images/earth_dam1_problem_sketch.png', True, 62975), ('images/earth_dam1_solution.png', True, 379554)]`
  - `External theory.md#anchor links checked: 28, broken: 0 []`
  - `Display $$ delimiter count (even expected): 202, odd inline $ lines: []` (`ALL CHECKS PASSED`, exit code `0`).

## Next
Review the added Part IV sections in `02-DOCS/wiki/theory.md` and commit the documentation and images on branch `paper` (or implement a transient unsaturated `GFDFlow` benchmark script for the zoned dam).
