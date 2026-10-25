---
type: plan
title: Plan — Five-Layer Dam Seepage (Chavez 2023)
description: The structure-level implementation plan for reproducing the five-layer dam example from Chavez et al. (2023).
tags: [sdd, plan, paper, mesh, gfdm]
timestamp: 2026-10-23T11:20:00Z
topic: sdd
slug: paper-dam-5c
status: approved
---

# Plan — Five-Layer Dam Seepage (Chavez 2023)

> Spec: [../specs/paper-dam-5c.md](../specs/paper-dam-5c.md) · Constitution: [../constitution.md](../constitution.md) · Status: approved
> Last updated: 2026-10-23

## 0. Global Constraints

- **Python Floor:** Python 3.11 (`gfdflow` conda environment).
- **Matrix Operations:** Exclusively sparse matrices via `scipy.sparse` and `scipy.sparse.linalg.spsolve`.
- **Documentation Standard:** NumPy docstring style for all modules, classes, and helper functions.
- **Figures Output Directory:** All output artifacts and figures must be saved under `examples/paper/figures/`.
- **Precomputed Topology:** GFDM support stencils and $M_{pinv}$ matrices must be precomputed during meshing and exported to `examples/paper/stat_mesh1.json`.
- **Authorship:** Human authorship; no AI attribution comments.

## 1. Context & Constraints

- **Paper Reference:** Section 4.2.3 of Chavez et al. (2023), Figures 16 and 17, Table 2.
- **Physical Geometry Dimensions:**
  - Dam height: 10 m; crest width: 4 m (from $x=23$ to $x=27$ m).
  - Upstream slope: 2:1 ($dx/dy = 2.0$), from $(3, 0)$ to $(23, 10)$. Reservoir water level at $y=8$ m ($x=19$ m).
  - Downstream slope: 2:1, from $(27, 10)$ to $(47, 0)$. Downstream toe filter/drain: $x=40$ to $47$ m on foundation surface ($y=0$).
  - Foundation: Depth $4$ m ($y=-4$ to $0$), horizontal span $x=3$ to $60$ m.
  - Clay core: Symmetrical trapezoid tapering downwards into foundation cutoff:
    - Crest: $x=23$ to $27$ at $y=10$ m.
    - Surface level: $x=20$ to $30$ at $y=0$ m (triple junctions).
    - Cutoff base: $x=24$ to $26$ at $y=-4$ m.
- **Subdomains / Materials:**
  - Material 1: Rockfill ($k_{sat} = 10^{-4}$ m/s).
  - Material 2: Clay core ($k_{sat} = 10^{-6}$ m/s).
  - Material 3: Soil Foundation ($k_{sat} = 10^{-4}$ m/s).
- **Target Node Count:** ~1503 nodes (obtained via CalfEM GmshMesh with `el_size_factor` $\approx 0.655$).

## 2. Architecture

```text
[ Chavez 2023 Paper ]
        |
        v
[ stat_mesh1.py ] ---> (Geometry & CalfEM Mesh) ---> [ stat_mesh1.json ]
        |                                                     |
        +------------> [ figures/*.png ]                      |
                                                              v
                                                    [ stat_gfdflow.py ]
                                                              |
                                                              v
                                                    [ Solution Plot & RMSE ]
```

- **`stat_mesh1.py`** (internal): Constructs geometry, runs CalfEM Gmsh mesher, extracts boundaries, isolates triple junctions, computes unit normals, builds star stencils & $M_{pinv}$, exports to JSON, and generates diagnostic plots.
- **`stat_mesh1.json`** (internal artifact): Serialized data contract containing coordinates, element connectivities, boundary groupings, outward normals, stencils, and precomputed matrices.
- **`stat_gfdflow.py`** (internal): Loads `stat_mesh1.json`, configures materials and boundary conditions, applies continuous interface flux balance and triple junction balance, assembles global sparse system $KU=F$, solves for total head $U$, and renders comparison contours against FEM.

**Top architectural decision:** Decouple mesh generation and stencil precomputation from the PDE solver. This guarantees deterministic, reproducible mesh inputs and lightning-fast solution iteration for stationary and transient solvers.

## 3. Interfaces & Contracts

### `stat_mesh1.json` Data Schema
```text
stat_mesh1.json:
  - coords: list[[x: float, y: float]] (shape N x 2)
  - triangles: list[[n1: int, n2: int, n3: int]] (shape M x 3)
  - normal_vecs: list[[nx: float, ny: float]] (shape N x 2)
  - left_nodes: list[int]
  - right_nodes: list[int]
  - bottom_nodes: list[int]
  - top_nodes: list[int]
  - interface_a_nodes: list[int] (upstream rockfill-foundation interface)
  - interface_b_nodes: list[int] (downstream rockfill-foundation interface)
  - interface_c_nodes: list[int] (upstream core-rockfill interface)
  - interface_d_nodes: list[int] (upstream core-foundation interface)
  - interface_e_nodes: list[int] (downstream core-foundation interface)
  - interface_f_nodes: list[int] (downstream core-rockfill interface)
  - rock_nodes: list[int]
  - clay_nodes: list[int]
  - foundation_nodes: list[int]
  - support_stencils: dict[node_id_str, list[int]]
  - M_pinv: dict[node_id_str, list[list[float]]]
```

### Precomputation Invariants
- Triple junctions (node indices at `[20, 0]` and `[30, 0]`) are cleanly removed from regular interface lists and handled via intersection balance.
- Normal vectors at internal interfaces point from material 1 to material 2; triple junction normals are initialized horizontally `[-1.0, 0.0]`.

## 4. Data Model & Flow

1. **Geometry definition**: 13 key points, 16 splines, and 5 surfaces.
2. **Meshing**: CalfEM 2D Gmsh triangular element generation.
3. **Partitioning**: Boundary DOF extraction via markers; set operations to eliminate duplicate corner nodes.
4. **Topology generation**: Star support nodes via `get_support_nodes_2D`, pseudo-inverses via `compute_M_matrix` / `compute_M_matrix_neumann`.
5. **Persistence**: Structured JSON export and high-resolution matplotlib rendering in `examples/paper/figures/`.

## 5. Testing Strategy

| Acceptance Criterion | Level | Asserts | Fakes / Mocks |
| --- | --- | --- | --- |
| Node count ~1500 | Unit test in script | `1450 <= N <= 1550` | None |
| JSON validity & integrity | Contract check | Keys present, shapes match $N$, no NaNs | None |
| Normal vectors computation | Unit test | Unit norm for boundary nodes | None |
| PDE assembly & range | Integration (`stat_gfdflow.py`) | $0.0 \le U \le 8.0$ | None |
| Physical continuity | Visual check | Smooth isolines across foundation/rock interface | FEM comparison |

## 6. Sequencing & Dependencies

1. **Step 1 (Completed)**: Implement `examples/paper/stat_mesh1.py` with full NumPy documentation and visualization logic.
2. **Step 2 (Completed)**: Run `stat_mesh1.py` to generate `examples/paper/stat_mesh1.json` and diagnostic figures (`geometry.png`, `mesh.png`, `nodes.png`, `normal_vectors.png`) in `examples/paper/figures/`.
3. **Step 3 (Completed)**: Implement `examples/paper/stat_gfdflow.py` for PDE assembly, linear solution, and comparison plot generation (`solution_contours.png`, `solution_equipotential.png`). All steps verified.

## 7. Risks & Open Decisions

### Risks
| Risk | Trigger | Impact | Mitigation |
| --- | --- | --- | --- |
| Gmsh version drift | Different gmsh library minor versions yielding different node counts | Node count differs slightly from 1503 | Calibrated `el_size_factor = 0.655` yielding 1490 nodes (within 1% of 1503). |
| Corner node misclassification | Spline endpoints overlapping between Neumann and Dirichlet | Wrong BC applied to corner | Strict priority hierarchy: Dirichlet overrides Neumann at boundary intersections. |

### Open Decisions
- None. All geometric coordinates, material conductivities, and boundary condition assignments were resolved directly from `main_elsevier.tex`, `FigMallaPresa5C.png`, and `FigDam5CCompara2.png`.
