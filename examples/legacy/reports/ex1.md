# Example 01: 2D Poisson Equation with Material Interface and Solution Jump

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), local Taylor series expansions, support stars, and discontinuous interface flux formulations, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Model steady-state 2D potential flow across two adjacent subdomains with differing permeabilities and a prescribed jump discontinuity in the scalar potential across the vertical interface.
- **Domain Type:** Rectangular composite domain partitioned into two subdomains (`material0` and `material1`) by a vertical interface.
- **Governing Physics:** 2D Poisson equation with constant internal source generation ($f = -1$) and material-dependent permeability ($k_0 = 1.0$, $k_1 = 0.1$).
- **Key Validation Points:** Accurate coupling of discontinuous interface conditions ($\alpha = 0.5$, $\beta = 0$) using `problem.interface` and assembly via `discontinuous_discretization()`.
- **Associated Code:** [`../ex1.py`](../ex1.py)
- **Mesh / Input Data:** [`../meshes/mesh1.json`](../meshes/mesh1.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
According to the general GFDFlow differential operator $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = A u + B u_x + C u_y + D u_{xx} + E u_{xy} + F u_{yy} = f(x, y)$$

Configured parameters:
- **Coefficient Vector $\mathbf{L}$:** $[A, B, C, 2D, E, 2F]^T = [0, 0, 0, 1, 0, 1]^T$
- **Resulting PDE:** $\nabla^2 u = u_{xx} + u_{yy} = -1$
- **Source Term:** Uniform source $f(x, y) = -1$
- **Material Permeabilities:**
  - Subdomain 0 (left): $k_0 = 1.0$
  - Subdomain 1 (right): $k_1 = 0.1$

### 2.2 Boundary and Interface Conditions
The boundary and interface conditions are assigned as follows (see [Theory §6](../../../02-DOCS/wiki/theory.md#6-implementation-of-boundary-conditions) and [Theory §7](../../../02-DOCS/wiki/theory.md#7-interface-flux-balance-for-layered-media)):

| Identifier | Type | Condition | GFDFlow Function | Physical Meaning |
|---|---|---|---|---|
| `left` | Dirichlet | $u(x, y) = 1 - y^2$ | `problem.dirichlet_boundary` | Parabolic potential inflow profile |
| `right` | Dirichlet | $u(x, y) = 1$ | `problem.dirichlet_boundary` | Constant potential outflow |
| `top_left`, `top_right` | Neumann | $\frac{\partial u}{\partial n} = 0$ | `problem.neumann_boundary` | Impermeable upper boundary |
| `bottom_left`, `bottom_right` | Neumann | $\frac{\partial u}{\partial n} = 0$ | `problem.neumann_boundary` | Impermeable lower boundary |
| `interface` | Interface | Jump: $u_0 - u_1 = 0.5$<br>Flux balance: $k_0 \frac{\partial u_0}{\partial n} - k_1 \frac{\partial u_1}{\partial n} = 0$ | `problem.interface` | Continuous flux with step jump $\alpha = 0.5$ |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Definition
The composite domain is bounded within planar rectangular boundaries, divided vertically into two subdomains with independent node groupings for `material0` and `material1`.

### 3.2 Nodal Point Cloud & Interface Sets
The mesh loaded from `mesh1.json` categorizes nodes into:
- Left and right exterior Dirichlet boundaries (`left_nodes`, `right_nodes`)
- Top and bottom Neumann boundaries (`top_left_nodes`, `top_right_nodes`, `bottom_left_nodes`, `bottom_right_nodes`)
- Duplicate paired interface nodes (`left_interface_nodes`, `right_interface_nodes`) to support discontinuous function values across the boundary.

---

## 4. GFDFlow Pipeline & Implementation

The implementation in [`../ex1.py`](../ex1.py) configures the discontinuous interface problem:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

problem = gfdmi(coords, triangles, normal_vecs, L, source)

problem.material('material0', permeability_mat0, interior_material_0_nodes)
problem.material('material1', permeability_mat1, interior_material_1_nodes)

problem.neumann_boundary('bottom_left', permeability_mat0, bottom_left_nodes, bottom_condition)
problem.neumann_boundary('top_left', permeability_mat0, top_left_nodes, top_condition)
problem.neumann_boundary('top_right', permeability_mat1, top_right_nodes, top_condition)
problem.neumann_boundary('bottom_right', permeability_mat1, bottom_right_nodes, bottom_condition)

problem.dirichlet_boundary('left', left_nodes, left_condition)
problem.dirichlet_boundary('right', right_nodes, right_condition)

problem.interface(
    'interface',
    permeability_mat0, permeability_mat1,
    left_interface_nodes, right_interface_nodes,
    flux_difference, solution_difference,
    interior_material_0_nodes, interior_material_1_nodes
)

K, F = problem.discontinuous_discretization()
U = sp.linalg.spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The contour plot displays the resulting scalar field across both materials:

![Solution Contour Map](../figures/ex1/contourf.jpg)

**Key Observations:**
- **Step Discontinuity:** A clear contour line concentration and sharp potential drop of exactly $\Delta u = 0.5$ occurs at the interface location.
- **Flux Continuity Across Contrasting Permeability:** Despite the 10:1 permeability contrast ($k_0 = 1.0$ vs $k_1 = 0.1$), the normal flux conservation condition is verified.
- **Impermeable Boundaries:** Top and bottom boundaries exhibit orthogonal contour intersections, satisfying $\partial u / \partial n = 0$.

### 5.2 3D Surface Visualization
The 3D surface plot depicts the discrete potential step at the material interface:

![3D Surface Plot](../figures/ex1/3dplot.jpg)

---

## 6. Verification and Conclusions

- **Discontinuous Formulation Fidelity:** The double-node interface coupling in `problem.interface` accurately reproduces prescribed step jumps $\alpha$ without smoothing artifacts.
- **Stability:** The sparse linear solver solves the discontinuous block system without ill-conditioning.
- **Conclusion:** Example 01 confirms GFDFlow's capability to model discontinuous potential fields across layered heterogeneous media.
