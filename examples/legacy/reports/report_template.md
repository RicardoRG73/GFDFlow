# [Example ID]: [Problem Title]

> **Reference Documentation:** For mathematical derivations, Taylor series expansions, support star selection, and boundary/interface formulations, see the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md). Do not repeat general numerical theory in this report; focus on case-specific parameters, discretization, and results.

---

## 1. Executive Summary

- **Objective:** [Concise statement of what this benchmark or example simulates and validates.]
- **Domain Type:** [e.g., 2D irregular bounded domain, layered media, rectangular channel, etc.]
- **Governing Physics:** [e.g., Steady-state Poisson equation, Darcy groundwater flow, advection-diffusion.]
- **Key Validation Points:** [e.g., Mixed Dirichlet-Neumann boundary adherence, ghost-node flux condition, material interface continuity.]
- **Associated Code:** [`../exX.py`](../exX.py)
- **Mesh / Input Data:** [`../meshes/meshX.json`](../meshes/meshX.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
GFDFlow models linear second-order differential operators in the general form $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = A u + B u_x + C u_y + D u_{xx} + E u_{xy} + F u_{yy} = f(x, y)$$

For this specific case, the parameters and source term are defined as:

- **Coefficient Vector $\mathbf{L}$:** $[A, B, C, 2D, E, 2F]^T = [A_0, B_0, C_0, 2D_0, E_0, 2F_0]^T$
- **Resulting PDE:** [e.g., $\nabla^2 u = u_{xx} + u_{yy} = f(x, y)$]
- **Source Term $f(x, y)$:** [e.g., Constant $-2$, spatial function $f(x,y)$, or zero]
- **Material Parameter(s):** [e.g., Isotropic permeability $k(p) = 1.0$, hydraulic conductivity, or layer-specific properties]

### 2.2 Boundary Conditions
Boundary conditions are applied to designated node sets (see [Theory §6](../../../02-DOCS/wiki/theory.md#6-implementation-of-boundary-conditions)):

| Boundary ID / Segment | Type | Mathematical Condition | GFDFlow Function | Physical Meaning |
|---|---|---|---|---|
| `[e.g., left]` | Dirichlet | $u = g(x, y)$ | `problem.dirichlet_boundary` | [Prescribed potential / head] |
| `[e.g., right]` | Neumann | $\frac{\partial u}{\partial n} = \mathbf{n} \cdot \nabla u = q(x, y)$ | `problem.neumann_boundary` | [Impermeable flux / prescribed inflow] |
| `[e.g., interface]` | Interface | Flux balance (if layered) | `problem.interface_condition` | [Discontinuous material transition] |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Definition
- **Geometry Coordinates / Primitives:** [Description of control points, bounding curves, or CAD/Gmsh geometry.]
- **Domain Sketch / Schematics:**

![Domain Geometry](../figures/exX/geometry.png)

### 3.2 Nodal Point Cloud & Mesh
- **Discretization Tool:** [e.g., `calfem-python`, `Gmsh`, structured grid generator]
- **Discretization Parameters:** [e.g., Characteristic element size $\delta$, total node count $N$, triangle count $T$]
- **Support Stencils:** Generated using Delaunay stars with minimum support nodes $q \ge 5$ (see [Theory §5](../../../02-DOCS/wiki/theory.md#5-domain-discretization-and-support-node-selection-stars)).

![Nodal Cloud & Mesh](../figures/exX/mesh.png)

### 3.3 Boundary Classification & Normal Vectors
- **Boundary Node Partition:** [List or summarize node groups: interior, boundary subsets]
- **Neumann Normal Vectors:** Outward unit normal vectors $\mathbf{n} = (n_x, n_y)$ computed on derivative boundaries to configure ghost-node constraints (see [Theory §6.2](../../../02-DOCS/wiki/theory.md#62-neumann-boundary-conditions-and-ghost-node-formulation)).

![Boundary Subsets](../figures/exX/boundaries.png)
![Normal Vectors](../figures/exX/normal_vectors.png)

---

## 4. GFDFlow Pipeline & Implementation

Summary of script workflow in `exX.py`:

1. **Data Ingestion:** Load nodal coordinates, connectivity, boundary subsets, and precomputed stencils (`support_stencils`, `M_pinv`) from JSON.
2. **Problem Initialization:** Instantiate `GFDMI_2D_problem(coords, triangles, normal_vectors, L, source, support_stencils, M_pinv)`.
3. **Material & Boundary Assignment:** Assign material zones and boundary operators (`dirichlet_boundary`, `neumann_boundary`, `interface_condition`).
4. **Assembly:** Call `problem.continuous_discretization()` to assemble sparse system $K \mathbf{U} = \mathbf{F}$ (see [Theory §8](../../../02-DOCS/wiki/theory.md#8-global-system-assembly-and-solution)).
5. **Linear Solution:** Solve with `scipy.sparse.linalg.spsolve(K, F)`.

```python
# Code snippet highlighting key problem setup:
# (Keep snippet concise, focusing on physics and boundary setup)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The solution field $\mathbf{U}(x, y)$ is evaluated across the domain:

![Solution Contour Map](../figures/exX/contourf.png)

**Key Observations:**
- **Boundary Verification:** [Confirm whether Dirichlet values match prescribed boundary curves/constants.]
- **Flux / Normal Gradient Behavior:** [Verify if contours arrive perpendicular to zero-flux Neumann boundaries.]
- **Field Curvature & Extrema:** [Relate field curvature to source term $f(x, y)$ or material transitions.]

### 5.2 3D Surface Visualization
The 3D potential surface illustrates spatial gradients and smoothness:

![3D Surface Plot](../figures/exX/3dplot.png)

---

## 6. Verification and Conclusions

- **Verification Summary:** [Assess convergence, boundary compliance, and consistency against analytical or physical benchmarks.]
- **Computational Performance:** [Matrix dimensions, sparsity, solve time if relevant.]
- **Key Takeaways:** [Summary of GFDM performance on this geometry and boundary configuration.]
