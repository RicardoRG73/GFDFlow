# Example 00: 2D Poisson Equation on an Irregular Curved Domain

> **Reference Documentation:** For the mathematical foundations of the Generalized Finite Difference Method (GFDM), local Taylor series expansions, support star selection, and ghost-node boundary formulations, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Compute the numerical solution of a two-dimensional Poisson equation on an irregular domain with mixed Dirichlet and Neumann conditions.
- **Domain Type:** 2D bounded domain with planar boundaries and a curved circular arc on the right boundary.
- **Governing Physics:** Steady-state Poisson equation with uniform negative source generation ($f = -2$).
- **Key Validation Points:** Accurate enforcement of non-homogeneous linear Dirichlet boundaries and homogeneous Neumann flux on a circular arc using the ghost-node technique on unstructured nodes.
- **Associated Code:** [`../ex0.py`](../ex0.py)
- **Mesh / Input Data:** [`../meshes/mesh0.json`](../meshes/mesh0.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
In accordance with the general GFDFlow differential operator $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = A u + B u_x + C u_y + D u_{xx} + E u_{xy} + F u_{yy} = f(x, y)$$

The parameters for this benchmark problem are configured as:

- **Coefficient Vector $\mathbf{L}$:** $[A, B, C, 2D, E, 2F]^T = [0, 0, 0, 1, 0, 1]^T$
- **Resulting PDE:** 2D Poisson equation
  $$\nabla^2 u = u_{xx} + u_{yy} = -2$$
- **Source Term:** Uniform constant source $f(x, y) = -2$
- **Permeability:** Isotropic permeability $k(p) = 1.0$

### 2.2 Boundary Conditions
The domain boundary $\partial \Omega$ is decomposed into four segments (see [Theory §6](../../../02-DOCS/wiki/theory.md#6-implementation-of-boundary-conditions)):

| Boundary ID / Segment | Type | Mathematical Condition | GFDFlow Function | Physical Meaning |
|---|---|---|---|---|
| `left` ($x = 0$) | Dirichlet | $u(0, y) = 0$ | `problem.dirichlet_boundary` | Zero potential baseline |
| `bottom` ($y = 0$) | Dirichlet | $u(x, 0) = 0.5 x$ | `problem.dirichlet_boundary` | Linear potential gradient |
| `top` ($y = 1$) | Dirichlet | $u(x, 1) = x$ | `problem.dirichlet_boundary` | Linear potential gradient |
| `right` (circular arc) | Neumann | $\frac{\partial u}{\partial n} = \mathbf{n} \cdot \nabla u = 0$ | `problem.neumann_boundary` | Impermeable boundary (zero flux) |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Definition
The physical domain $\Omega$ is defined by five control points:
- $P_0 = (0, 0)$
- $P_1 = (1, 0)$
- $P_2 = (2, 0)$
- $P_3 = (1, 1)$
- $P_4 = (0, 1)$

The right boundary is formed by a circular arc centered at $(1, 0)$ with radius $R = 1$ connecting $P_2 = (2, 0)$ to $P_3 = (1, 1)$. The remaining boundaries are straight segments.

![Domain Geometry](../figures/ex0/geometry.png)

### 3.2 Nodal Point Cloud & Mesh
The unstructured triangular mesh was generated using `calfem-python` and `Gmsh` with an element size factor $\delta = 0.08$. Support stars for each node were selected with a minimum of $q \ge 5$ support nodes to guarantee full rank in the least-squares system (see [Theory §5](../../../02-DOCS/wiki/theory.md#5-domain-discretization-and-support-node-selection-stars)).

![Unstructured Triangular Mesh](../figures/ex0/mesh.png)

### 3.3 Boundary Classification & Normal Vectors
Nodal sets are partitioned into:
- **Left Nodes:** Dirichlet condition ($u = 0$)
- **Bottom Nodes:** Dirichlet condition ($u = 0.5x$)
- **Top Nodes:** Dirichlet condition ($u = x$)
- **Right Nodes:** Neumann condition ($\partial u / \partial n = 0$)
- **Interior Nodes:** Governing Poisson equation ($\nabla^2 u = -2$)

![Boundary Node Classification](../figures/ex0/boundaries.png)

For the curved Neumann right boundary, outward unit normal vectors $\mathbf{n} = (n_x, n_y)$ were computed geometrically to enforce directional flux constraints via the ghost-node formulation (see [Theory §6.2](../../../02-DOCS/wiki/theory.md#62-neumann-boundary-conditions-and-ghost-node-formulation)).

![Normal Vectors on Neumann Boundary](../figures/ex0/normal_vectors.png)

---

## 4. GFDFlow Pipeline & Implementation

The numerical solution is assembled and solved using the pipeline in [`../ex0.py`](../ex0.py):

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

# 1. Initialize GFDM problem with geometry and precomputed stencils
problem = gfdmi(coords, triangles, normal_vectors, L, source, support_stencils, M_pinv)

# 2. Assign material and boundary conditions
problem.material('0', permeability, interior_nodes)
problem.neumann_boundary('right', permeability, right_nodes, right_condition)
problem.dirichlet_boundary('left', left_nodes, left_condition)
problem.dirichlet_boundary('top', top_nodes, top_condition)
problem.dirichlet_boundary('bottom', bottom_nodes, bottom_condition)

# 3. Assemble sparse system KU = F (see Theory §8)
K, F = problem.continuous_discretization()

# 4. Solve sparse linear system
U = sp.linalg.spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The solution field $U(x, y)$ transitions smoothly across the domain, reflecting the interaction between the Dirichlet boundaries and the uniform internal source term $f(x, y) = -2$.

![Solution Contour Map](../figures/ex0/contourf.png)

**Key Observations:**
- **Dirichlet Adherence:**
  - On the left boundary ($x = 0$), $U = 0$ is strictly satisfied.
  - On the bottom boundary ($y = 0$), the potential increases linearly from $0$ at $x = 0$ to $1.0$ at $x = 2$.
  - On the top boundary ($y = 1$), the potential increases linearly from $0$ at $x = 0$ to $1.0$ at $x = 1$.
- **Neumann Flux Condition:** Equipotential contour lines meet the curved circular arc orthogonally, verifying that $\partial u / \partial n = 0$ is accurately captured without spurious distortion.
- **Internal Curvature:** The negative source term creates internal downward curvature consistent with Poisson physics.

### 5.2 3D Surface Visualization
The 3D surface plot depicts the continuous potential landscape across the irregular domain:

![3D Surface Plot](../figures/ex0/3dplot.png)

---

## 6. Verification and Conclusions

- **Boundary Enforcement:** Both linear Dirichlet profiles and the directional Neumann condition on the curved boundary are enforced with high numerical fidelity.
- **Ghost-Node Performance:** The ghost-node formulation eliminates the need for body-fitted structured exterior grids, providing stable derivative boundary enforcement on unstructured point clouds.
- **Conclusion:** Example 00 validates the core GFDM discretization workflow in **GFDFlow** for irregular 2D domains with curved geometries and mixed boundary conditions.
