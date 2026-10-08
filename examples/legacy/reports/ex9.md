# Example 09: Multi-Layered Domain with Three Materials and Central Vertex Intersection

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), multi-material interfaces, and intersection node balancing, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Solve the 2D Poisson equation in a square domain partitioned into three distinct angular material sectors intersecting at the origin $(0, 0)$.
- **Domain Type:** Square domain $[-1, 1] \times [-1, 1]$ divided into three wedge-shaped subdomains with high permeability contrast ($5 : 500 : 100$).
- **Governing Physics:** Poisson equation with intense source generation ($f = -100$) and sector-dependent permeabilities ($k_0 = 5$, $k_1 = 500$, $k_2 = 100$).
- **Key Validation Points:** Treatment of three interfaces converging at a single central intersection node using `problem.intersection`, and continuous potential transmission.
- **Associated Code:** [`../ex9multilayer.py`](../ex9multilayer.py)
- **Mesh / Input Data:** [`../meshes/mesh9.json`](../meshes/mesh9.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
Under the general operator form $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = \nabla^2 u = u_{xx} + u_{yy} = -100$$

- **Operator Vector:** $\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$
- **Source Term:** Uniform constant $f(x, y) = -100$
- **Material Permeabilities:**
  - Material 0: $k_0 = 5.0$
  - Material 1: $k_1 = 500.0$
  - Material 2: $k_2 = 100.0$

### 2.2 Boundary & Interface Conditions
- **Outer Perimeter:** Dirichlet condition $u(x, y) = e^{x + y}$ on all exterior edges.
- **Radial Interfaces:** Three interfaces connecting at central node $(0,0)$ enforce continuous potential and normal flux conservation ($\beta = 0$).
- **Central Intersection Node:** Multi-material angular balance configured via `problem.intersection`.

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout & Interfaces
The square domain $[-1, 1] \times [-1, 1]$ is split into three sectors radiating from the central coordinate $(0, 0)$:

![Domain Geometry](../figures/ex9/geometry.png)
![Triangular Mesh](../figures/ex9/mesh.png)

### 3.2 Nodal Point Cloud & Normal Vectors
Nodal subsets classify the three material zones, three interface segments, and exterior boundaries:

![Node Sets Classification](../figures/ex9/nodes.png)
![Interface Normal Vectors](../figures/ex9/normal_vectors.png)

---

## 4. GFDFlow Pipeline & Implementation

The implementation in [`../ex9multilayer.py`](../ex9multilayer.py) configures the sectors and central vertex:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d
import scipy.sparse as sp

problem = gfdmi(coords, faces, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material("mat0", k0, mat0_nodes)
problem.material("mat1", k1, mat1_nodes)
problem.material("mat2", k2, mat2_nodes)

problem.dirichlet_boundary("dirichlet", dirichlet_nodes, fd)

problem.interface("interf0", k0, k1, interf0_nodes, None, fi0, None, mat0_nodes, mat1_nodes)
problem.interface("interf1", k1, k2, interf1_nodes, None, fi1, None, mat1_nodes, mat2_nodes)
problem.interface("interf2", k2, k0, interf2_nodes, None, fi2, None, mat2_nodes, mat0_nodes)

problem.intersection("intersec0", 5, "interf0", "interf1", "mat1", fi_intersc)
problem.intersection("intersec1", 5, "interf1", "interf2", "mat2", fi_intersc)
problem.intersection("intersec2", 5, "interf2", "interf0", "mat0", fi_intersc)

K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The contour distribution reflects the sharp contrast between material sectors:

![Solution Contour Map](../figures/ex9/contourf.png)

**Key Observations:**
- **Permeability Contrast Effect:** Material 0 ($k_0 = 5$, low permeability) accumulates steep potential gradients, whereas Material 1 ($k_1 = 500$, high permeability) exhibits flat, highly dispersed potential contours.
- **Central Node Stability:** The intersection formulation at center node 5 smoothly couples all three sectors without numerical artifacts or flux leakage.

### 5.2 3D Surface Visualization
The 3D surface plot depicts the distinct slope variations across the three material sectors:

![3D Surface Plot](../figures/ex9/3dplot.png)

---

## 6. Verification and Conclusions

- **Multi-Sector Accuracy:** Verifies the capability of GFDFlow to resolve complex piecewise-constant material domains with intersecting internal interfaces.
- **Convergence:** Demonstrates smooth continuity of $u$ across high permeability ratios (up to 100:1) with central junction stability.
