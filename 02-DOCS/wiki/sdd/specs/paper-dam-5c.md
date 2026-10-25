---
type: spec
title: Spec — Five-Layer Dam Seepage (Chavez 2023)
description: Specification to reproduce the five-layered dam stationary groundwater flow example from Chavez et al. (2023).
tags: [sdd, spec, paper, mesh, gfdm]
timestamp: 2026-10-23T11:20:00Z
topic: sdd
slug: paper-dam-5c
status: approved
---

# Spec — Five-Layer Dam Seepage (Chavez 2023)

## 1. Intent & Scope

### Intent
Reproduce the multi-material groundwater flow simulation of the five-layered embankment dam with foundation and clay core presented in Chavez et al. (2023) (*A generalized finite differences method for groundwater flow modeling in layered domains*, Computers and Geotechnics, Section 4.2.3, Fig. 16 & Fig. 17).

### Scope
- **Phase 1: Mesh Generation (`stat_mesh1.py`)**:
  - Define geometry matching the exact dimensions from the paper: dam height 10 m, crest width 4 m (x = 23 to 27 m), 2:1 slopes, foundation depth 4 m (y = -4 to 0 m), horizontal foundation extent x = 3 to 60 m, core cutoff base at y = -4 m (x = 24 to 26 m).
  - Unstructured triangular mesh of approximately 1500 nodes (paper target: ~1503 nodes).
  - Accurate identification and separation of boundary condition nodes (Dirichlet, Neumann, Interface segments, Triple Junctions).
  - Precomputation of outward unit normal vectors, GFDM support stencils, and star matrices ($M$ pseudo-inverses $M_{pinv}$).
  - Export serialized mesh and stencil topology to `examples/paper/stat_mesh1.json`.
  - Save diagnostic plots to `examples/paper/figures/` (`geometry.png`, `mesh.png`, `nodes.png`, `normal_vectors.png`).
- **Phase 2: PDE Assembly & Solution (`stat_gfdflow.py`)** (Planned for next step):
  - Assemble steady-state 2D potential flow equation ($\nabla \cdot (k \nabla h) = 0$).
  - Assign soil hydraulic conductivities from Table 2: Rockfill $k_{sat} = 10^{-4}$ m/s, Clay core $k_{sat} = 10^{-6}$ m/s, Foundation $k_{sat} = 10^{-4}$ m/s.
  - Apply boundary conditions: Dirichlet upstream $h = 8$ m, Dirichlet downstream $h = 0$ m, Neumann $\partial h / \partial n = 0$ at impermeable bedrock base, downstream face, and crest.
  - Continuous flux balance at the 6 internal interface curves and the 2 triple junctions.
  - Solve system with `scipy.sparse.linalg.spsolve`.
  - Plot equipotential lines and phreatic surface matching `FigDam5CCompara2.png` and save to `examples/paper/figures/`.

## 2. Acceptance Criteria

1. **Geometry & Topology Fidelity**: Control points match the published geometry (slopes 2:1, crest x: [23, 27], y: 10, foundation y: [-4, 0], x: [3, 60], core cutoff at y = -4: [24, 26]).
2. **Node Resolution**: Mesh generated with ~1500 nodes (tolerance ±30 nodes).
3. **Data Serialization**: Mesh script writes valid JSON containing coords, triangles, boundary node groups, normal vectors, support stencils, and $M_{pinv}$ matrices.
4. **Diagnostic Visuals**: Visualizations generated and saved to `examples/paper/figures/` without display dependencies.
5. **Stack Compliance**: Uses Python 3.11 environment (`gfdflow`), NumPy docstring style, and SciPy sparse compatibility.
