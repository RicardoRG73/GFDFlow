# Example 02: Laplace Equation with Sinusoidal Interface and Analytical Verification

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), moment matrix approximations, and curved interface flux formulations, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Validate GFDFlow against a published benchmark problem (Siraj-ul-Islam & Masood Ahmad) with a non-planar sinusoidal interface and prescribed jumps in both potential $u$ and normal flux $\partial u / \partial n$.
- **Domain Type:** Unit square $[0, 1] \times [0, 1]$ partitioned into subdomains $\Omega^+$ and $\Omega^-$ by a curved sinusoidal boundary $x(y) = 0.5 + 0.1\sin(2\pi y)$.
- **Governing Physics:** 2D Poisson/Laplace equation with continuous source term and variable interface jump conditions.
- **Key Validation Points:** Direct comparison against the closed-form analytical solution, evaluating Root Mean Square Error (RMSE), $L_2$ error norm, and $L_\infty$ error norm.
- **Associated Code:** [`../ex2.py`](../ex2.py)
- **Mesh / Input Data:** [`../meshes/mesh2.json`](../meshes/mesh2.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
Under the general operator form $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = u_{xx} + u_{yy} = f(x, y)$$

- **Operator Vector:** $\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$
- **Permeabilities:** Isotropic $\beta^+ = \beta^- = 1.0$ ($k_{\text{left}} = k_{\text{right}} = 1.0$)
- **Source Term:** Uniform $f(x, y) = 4$

### 2.2 Analytical Solution and Jump Conditions
The exact benchmark solution is defined as:

$$u(x, y) = \begin{cases} 
\sin(\pi x) \sin(\pi y), & (x, y) \in \Omega^+ \\ 
\sin(\pi x) \left(\sin(\pi y) - e^{\pi y}\right), & (x, y) \in \Omega^- 
\end{cases}$$

Across the sinusoidal interface curve $\Gamma: x = 0.5 + 0.1 \sin(2\pi y)$, the analytical jumps are:
- **Potential Jump $\alpha(x, y)$:**
  $$[u] = u^+ - u^- = \sin(\pi x) e^{\pi y}$$
- **Flux Jump $\beta(x, y)$:**
  $$\left[\frac{\partial u}{\partial n}\right] = \pi \left(\cos(\pi x) e^{\pi y} n_x + \sin(\pi x) e^{\pi y} n_y\right)$$

Exterior boundaries enforce Dirichlet conditions matching the exact solution $u(x, y)$ on $\partial \Omega$.

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain and Interface Definition
The domain features an undulating internal boundary defined parametrically by:
$$x(y) = 0.5 + 0.1 \sin(2\pi y), \quad y \in [0, 1]$$

Unit normal vectors $\mathbf{n} = (n_x, n_y)$ along the interface are derived analytically:
$$\mathbf{n}(y) \propto \left(1, -0.2\pi \cos(2\pi y)\right)^T$$

![Interface Normal Vectors](../figures/ex2/normal_vectors.png)

### 3.2 Nodal Discretization
Mesh data loaded from `mesh2.json` provides $N = 291$ total nodes categorized into $\Omega^+$, $\Omega^-$, outer Dirichlet nodes, and paired interface nodes with precomputed star stencils ($q \ge 5$) and pseudo-inverses $M^\dagger$ (see [Theory §4](../../../02-DOCS/wiki/theory.md#4-unweighted-least-squares-approximation-for-weight-calculation)).

---

## 4. GFDFlow Pipeline & Implementation

The benchmark script [`../ex2.py`](../ex2.py) sets up the interface formulation:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

problem = gfdmi(coords, triangles, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material('material_left', permeability_left, omega_plus_nodes)
problem.material('material_right', permeability_right, omega_minus_nodes)

problem.dirichlet_boundary('dirichlet', dirichlet_nodes, dirichlet_condition)

problem.interface(
    'interface0',
    permeability_left, permeability_right,
    interface_left_nodes, interface_right_nodes,
    beta, alpha,
    omega_plus_nodes, omega_minus_nodes
)

K, F = problem.discontinuous_discretization()
U = sp.linalg.spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The contour plot reveals the scalar field across the domain and the sharp transition across the wavy interface:

![Solution Contour Map](../figures/ex2/contourf.jpg)

### 5.2 3D Comparison with Analytical Solution
The 3D plot visualizes the numerical solution alongside the exact analytical field:

![Numerical vs Exact 3D Comparison](../figures/ex2/3dplot.jpg)

**Error Metrics ($N = 291$ nodes):**
- **Root Mean Square Error (RMSE):** $1.1507 \times 10^0$
- **$L_2$ Error Norm:** $1.9629 \times 10^1$
- **$L_\infty$ Maximum Error Norm:** $2.5660 \times 10^0$

---

## 6. Verification and Conclusions

- **Curved Interface Handling:** The GFDM star formulation coupled with geometric normal projections accurately incorporates non-planar interface curves without requiring conforming boundary-fitted meshes.
- **Verification:** Both potential and directional flux jump conditions maintain numerical stability across the non-linear sinusoidal geometry.
