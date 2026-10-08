# Example 03: Stationary Groundwater Flow and Transient Diffusion with Phreatic Line Tracking

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), semi-discrete time-stepping formulations, and layered interface balance laws, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Simulate both steady-state seepage and transient diffusion in a layered geological domain (rock-clay-rock), including phreatic surface tracking and continuous interface flux balance.
- **Domain Type:** 2D layered permeable medium composed of rock and clay strata.
- **Governing Physics:**
  - *Phase 1:* Stationary Laplace groundwater flow ($\nabla^2 u = 0$).
  - *Phase 2:* Time-dependent parabolic diffusion equation ($\frac{\partial u}{\partial t} = \nabla^2 u$).
- **Key Validation Points:** Continuous interface flux transmission across rock/clay boundaries, extraction of the zero-pressure phreatic line, and time-integration via `scipy.integrate.solve_ivp`.
- **Associated Code:** [`../ex3.py`](../ex3.py)
- **Mesh / Input Data:** [`../meshes/mesh3.json`](../meshes/mesh3.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
GFDFlow discretizes the spatial operator $L u$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = \nabla^2 u = u_{xx} + u_{yy}$$

- **Operator Vector:** $\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$
- **Conductivities:** Rock $k_r = 1.0$, Clay $k_c = 0.1$
- **Source Term:** $f(x, y) = 0$

### 2.2 Boundary & Interface Conditions
Layer and boundary specifications (see [Theory §6](../../../02-DOCS/wiki/theory.md#6-implementation-of-boundary-conditions) and [Theory §7](../../../02-DOCS/wiki/theory.md#7-interface-flux-balance-for-layered-media)):

| Identifier | Type | Condition | Description |
|---|---|---|---|
| `left` | Dirichlet | $u = 8.0$ | Upstream reservoir hydraulic head |
| `right` | Dirichlet | $u = 0.0$ | Downstream outlet hydraulic head |
| `top`, `bottom` | Neumann | $\frac{\partial u}{\partial n} = 0$ | Impermeable bedrock / top boundary |
| `left_interface`, `right_interface` | Interface | Continuous flux $\beta = 0$ | Rock-clay material transitions |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout
The domain represents a horizontal channel stratified into rock sections enclosing an intermediate low-permeability clay layer.

### 3.2 Nodal Discretization & Stencils
Mesh data from `mesh3.json` specifies:
- Rock interior nodes (`rock_nodes`) and clay interior nodes (`clay_nodes`)
- Left and right interface boundaries (`left_interface_nodes`, `right_interface_nodes`)
- Precomputed support stencils and pseudo-inverse moment matrices $M^\dagger$.

---

## 4. GFDFlow Pipeline & Implementation

The workflow in [`../ex3.py`](../ex3.py) integrates both steady-state assembly and transient ODE solution:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
from scipy.integrate import solve_ivp
import scipy.sparse as sp

# 1. Assembling stationary problem
problem = gfdmi(coords, triangles, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)
problem.material("rock", kr, rock_nodes)
problem.material("clay", kc, clay_nodes)
problem.neumann_boundary("bottom", kr, bottom_nodes, neumann_cond)
problem.neumann_boundary("top", kr, top_nodes, neumann_cond)
problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)
problem.interface("left_interface", kr, kc, left_interface_nodes, None, beta, None, rock_nodes, clay_nodes)
problem.interface("right_interface", kc, kr, right_interface_nodes, None, beta, None, clay_nodes, rock_nodes)

K, F = problem.continuous_discretization()
U_steady = sp.linalg.spsolve(K, F)

# 2. Transient diffusion integration: du/dt = KU - F
fun = lambda t, U: K @ U - F
sol = solve_ivp(fun, [0, 80], U0)
```

---

## 5. Numerical Results and Discussion

### 5.1 Stationary Solution & Phreatic Line
The stationary hydraulic head field displays the steep gradient across the low-permeability clay zone:

![Stationary 2D Contour & Phreatic Surface](../figures/ex3/contourf_stationary.png)
![Stationary 3D Surface](../figures/ex3/3dplot_stationary.png)

The phreatic line ($u = y$) delineates the saturation boundary across the stratigraphic sequence.

### 5.2 Transient Diffusion Evolution
The diffusion process progresses from the initial state $U_0$ towards the steady-state equilibrium:

![Initial Condition U0](../figures/ex3/3dplot_u0.png)
![Transient Diffusion Time Snapshots](../figures/ex3/diffusion_steps.png)
![Solution at Final Time t=80](../figures/ex3/3dplot.png)

**Matrix Conditioning:**
- System matrix condition number: $\text{cond}(K) = 4.841 \times 10^3$, indicating a well-conditioned sparse linear operator for time integration.

---

## 6. Verification and Conclusions

- **Coupled Stratigraphy:** The continuous interface formulation maintains smooth physical flux transfer across sharp 10-fold conductivity contrasts.
- **Dynamic Stability:** The spatial GFDM discretization matrix $K$ seamlessly serves as the semi-discrete operator for standard ODE integrators (`solve_ivp`).
