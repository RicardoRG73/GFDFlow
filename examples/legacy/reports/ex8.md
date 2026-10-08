# Example 08: 2D Confined Seepage Flow with Mixed Boundary Conditions

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), ghost-node Neumann formulations, and global system assembly, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Simulate steady-state 2D confined seepage through a permeable formation subjected to differential head boundaries and impermeable confining boundaries.
- **Domain Type:** 2D channel representing a confined aquifer segment.
- **Governing Physics:** Laplace potential seepage equation ($\nabla^2 u = 0$) with uniform permeability $k = 0.5$.
- **Key Validation Points:** Linear head loss verification between inlet and outlet Dirichlet boundaries and zero-flux enforcement along confining boundaries via the ghost-node technique.
- **Associated Code:** [`../ex8.py`](../ex8.py)
- **Mesh / Input Data:** [`../meshes/mesh8.json`](../meshes/mesh8.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
Under the general operator form $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = \nabla^2 u = u_{xx} + u_{yy} = 0$$

- **Operator Vector:** $\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$
- **Permeability:** $k(p) = 0.5$
- **Source Term:** $f(x, y) = 0$

### 2.2 Boundary Conditions
Boundaries are classified into (see [Theory §6](../../../02-DOCS/wiki/theory.md#6-implementation-of-boundary-conditions)):

| Boundary | Type | Value / Condition | Description |
|---|---|---|---|
| `left` | Dirichlet | $u = 50.0$ | Upstream head boundary |
| `right` | Dirichlet | $u = 30.0$ | Downstream discharge boundary |
| `neumann` | Neumann | $\frac{\partial u}{\partial n} = 0$ | Impermeable top and bottom confining units |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout & Point Cloud
Discretized with an unstructured mesh from `mesh8.json`, defining interior nodes, left/right Dirichlet nodes, and confining Neumann nodes with unit normal vectors $\mathbf{n}$.

### 3.2 Support Stars
Each interior and boundary node is assigned a support star with $q \ge 5$ spatial neighbors, computing the local moment matrix and its Moore-Penrose pseudo-inverse $M^\dagger$.

---

## 4. GFDFlow Pipeline & Implementation

The execution in [`../ex8.py`](../ex8.py) runs the standard linear assembly pipeline:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

problem = gfdmi(coords, faces, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material("interior", k, interior_nodes)
problem.dirichlet_boundary("left", left_nodes, lambda p: 50)
problem.dirichlet_boundary("right", right_nodes, lambda p: 30)
problem.neumann_boundary("neumann", k, neumann_nodes, neumann_condition)

K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Hydraulic Head Distribution (2D Contour)
The contour plot displays the resulting potential gradient:

![Seepage Head Contour Map](../figures/ex8/contourf.png)

**Key Observations:**
- **Uniform Gradient:** Potential contours transition smoothly and uniformly from $h = 50$ down to $h = 30$, matching theoretical 1D confined Darcy flow expectations.
- **Orthogonal Equipotentials:** Contours meet the upper and lower boundaries at 90-degree angles, verifying zero normal discharge through the confining walls.
- **High Efficiency:** System assembly and solution complete in under $0.3$ seconds.

---

## 6. Verification and Conclusions

- **Basic Validation:** Validates the basic confined seepage workflow in GFDFlow with high computational speed and zero-flux boundary fidelity.
