# Example 05: 2D Porous Media Flow with Curved Circular Inclusions

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), curved interface approximations, and continuous flux jump formulations, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Simulate 2D groundwater flow through a composite porous medium featuring both vertical layered boundaries and circular internal inclusions with differing conductivities.
- **Domain Type:** Rectangular multi-region domain containing a vertical material interface and two circular inclusions (one material inclusion and one internal Dirichlet boundary).
- **Governing Physics:** Steady-state Poisson/Laplace equation with discontinuous piecewise constant permeability ($k_{\text{sand}} = 1.0$, $k_{\text{rock}} = 0.3$, $k_{\text{circ}} = 0.1$).
- **Key Validation Points:** Accurate flux preservation along curved circular interface boundaries and enforcement of non-homogeneous Dirichlet conditions along a circular boundary cavity.
- **Associated Code:** [`../ex5.py`](../ex5.py)
- **Mesh / Input Data:** [`../meshes/mesh5.json`](../meshes/mesh5.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
Under the general operator formulation $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = \nabla^2 u = u_{xx} + u_{yy} = 0$$

- **Operator Vector:** $\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$
- **Conductivity Properties:**
  - Sand (left half): $k_{\text{sand}} = 1.0$
  - Rock (right half): $k_{\text{rock}} = 0.3$
  - Circular rock inclusion: $k_{\text{circ}} = 0.1$
- **Source Term:** $f(x, y) = 0$

### 2.2 Boundary & Interface Conditions
Boundaries and internal interfaces (see [Theory §6](../../../02-DOCS/wiki/theory.md#6-implementation-of-boundary-conditions) and [Theory §7](../../../02-DOCS/wiki/theory.md#7-interface-flux-balance-for-layered-media)):

| Identifier | Type | Condition | Description |
|---|---|---|---|
| `izq` (left) | Dirichlet | $u = 1.0$ | Constant inlet potential |
| `der` (right) | Dirichlet | $u = 0.0$ | Constant outlet potential |
| `right_circle` | Dirichlet | $u(x, y) = 0.75 - x/3$ | Linear Dirichlet boundary on internal circular boundary |
| `left_top`, `right_top` | Neumann | $\frac{\partial u}{\partial n} = 0$ | Impermeable upper boundary |
| `left_bottom`, `right_bottom` | Neumann | $\frac{\partial u}{\partial n} = 0$ | Impermeable lower boundary |
| `interf` (vertical) | Interface | Flux balance $\beta = 0$ | Transition between sand and rock |
| `interf_circle` | Interface | Flux balance $\beta = 0$ | Circular interface of the embedded low-permeability inclusion |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout
The domain is divided into a left region (sand) and a right region (rock). The sand region encloses a circular low-permeability obstacle ($k = 0.1$), and the rock region features an internal circular Dirichlet boundary.

### 3.2 Nodal Discretization
From `mesh5.json`:
- Boundary node groupings for straight and circular perimeters
- Interface node sets for both vertical and circular boundaries
- Support stencils and pseudo-inverse moment matrices precomputed for all mesh nodes.

---

## 4. GFDFlow Pipeline & Implementation

The assembly in [`../ex5.py`](../ex5.py) combines exterior boundaries with circular interfaces:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from scipy.sparse.linalg import spsolve

problem = gfdmi(coords, faces, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material("sand", k_sand, left_half_nodes)
problem.material("rock", k_rock, right_half_nodes)
problem.material("rock_circ", k_rock_circ, left_circ_mat_nodes)

problem.neumann_boundary("left_top", k_sand, left_top_nodes, neumann_cond)
problem.neumann_boundary("left_bottom", k_sand, left_bottom_nodes, neumann_cond)
problem.neumann_boundary("right_top", k_rock, right_top_nodes, neumann_cond)
problem.neumann_boundary("right_bottom", k_rock, right_bottom_nodes, neumann_cond)

problem.dirichlet_boundary("izq", left_nodes, left_dirichlet)
problem.dirichlet_boundary("der", right_nodes, right_dirichlet)
problem.dirichlet_boundary("right_circle", right_circ_nodes, lambda p: 0.75 - p[0]/3)

problem.interface("interf", k_sand, k_rock, interface_nodes, None, beta, None, left_half_nodes, right_half_nodes)
problem.interface("interf_circle", k_sand, k_rock_circ, left_circ_nodes, None, beta, None, left_half_nodes, left_circ_mat_nodes)

K, F = problem.continuous_discretization()
U = spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The contour map demonstrates how flow diverts around low-permeability inclusions:

![Solution Contour Map](../figures/ex5/contourf.png)

**Key Observations:**
- **Flow Deflection:** Equipotential lines wrap smoothly around the circular low-permeability inclusion ($k=0.1$), indicating that flow is redirected around the hydraulic obstacle.
- **Continuity Across Interfaces:** The scalar potential field $U$ remains continuous across both planar and circular material boundaries while satisfying the normal flux balance.

### 5.2 3D Surface Visualization
The 3D potential surface illustrates the spatial gradient variations between high-permeability and low-permeability zones:

![3D Surface Plot](../figures/ex5/3dplot.png)

---

## 6. Verification and Conclusions

- **Circular Interface Fidelity:** GFDM provides smooth approximations along curved internal interfaces without requiring fitted conforming boundary rings.
- **Conclusion:** Demonstrates the applicability of GFDFlow to heterogeneous aquifers with embedded impermeable or low-permeability geological inclusions.
