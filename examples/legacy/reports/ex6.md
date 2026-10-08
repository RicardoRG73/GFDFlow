# Example 06: Coupled Variable-Density Seawater Intrusion (Henry Problem)

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), spatial derivative operators, and coupled system discretizations, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Simulate the classic Henry problem benchmark for variable-density groundwater flow and coupled solute transport (seawater intrusion in a coastal confined aquifer).
- **Domain Type:** Rectangular cross-section ($2 \times 1$) representing a coastal confined aquifer.
- **Governing Physics:** Non-linear coupled system consisting of a Poisson flow equation for the stream function $\psi$ coupled with an advection-dispersion transport equation for salt concentration $C$:
  $$\nabla^2 \psi = -\frac{1}{a} \frac{\partial C}{\partial x}$$
  $$\frac{\partial C}{\partial t} = b \nabla^2 C - \left(\frac{\partial \psi}{\partial y} \frac{\partial C}{\partial x} - \frac{\partial \psi}{\partial x} \frac{\partial C}{\partial y}\right)$$
- **Key Validation Points:** Accurate assembly of coupled block-differential operators ($\mathbf{D}_x, \mathbf{D}_y, \mathbf{D}^2$) in GFDM and time-stepping of the stiff advection-dispersion IVP using the LSODA integrator.
- **Associated Code:** [`../ex6Henry.py`](../ex6Henry.py)
- **Mesh / Input Data:** [`../meshes/mesh6.json`](../meshes/mesh6.json)
- **Computed Results:** [`../results/ex6Henry.json`](../results/ex6Henry.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Coupled Equations
The dimensionless Henry problem parameters are:
- Density coupling parameter: $a = 0.2637$
- Dispersion parameter: $b = 0.035$
- Permeability: $k = 1.0$

The coupled system is formulated in state vector $\mathbf{U} = [\boldsymbol{\psi}, \mathbf{C}]^T$ of dimension $2N = 5496$ (for $N = 2748$ spatial nodes).

### 2.2 Boundary Conditions

| Boundary | Stream Function $\psi$ | Concentration $C$ | Physical Meaning |
|---|---|---|---|
| **Left ($x=0$)** | $\psi = y$ (constant freshwater inflow $Q=1$) | $C = 0$ | Fresh inland groundwater recharge |
| **Right ($x=2$)** | Hydrostatic coastal exit condition | $C = 1$ | Seawater boundary (salt wedge) |
| **Bottom ($y=0$)** | $\psi = 0$ (streamline $\partial \psi / \partial n = 0$) | $\frac{\partial C}{\partial y} = 0$ | Impermeable bedrock base |
| **Top ($y=1$)** | $\psi = 1$ (streamline $\partial \psi / \partial n = 0$) | $\frac{\partial C}{\partial y} = 0$ | Confining impermeable upper unit |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout
The computational domain spans $[0, 2] \times [0, 1]$. Mesh data loaded from `mesh6.json` defines $N = 2748$ nodes and associated triangular elements.

### 3.2 Discrete Operator Matrices
GFDFlow generates discrete differential matrices:
- First-order spatial derivatives: $\mathbf{D}_x \approx \frac{\partial}{\partial x}$, $\mathbf{D}_y \approx \frac{\partial}{\partial y}$
- Second-order Laplacian: $\mathbf{D}^2 \approx \nabla^2$
- Ghost-node projection for Neumann conditions along top and bottom boundaries.

---

## 4. GFDFlow Pipeline & Implementation

The implementation in [`../ex6Henry.py`](../ex6Henry.py) builds the linear and non-linear block system:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdm
from scipy.integrate import solve_ivp
import scipy.sparse as sp

# Discrete differential operators for psi and C
problem = gfdm(coords, faces, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)
# ... [boundary condition configuration]

D2psi, F2psi = problem.continuous_discretization()

# Build coupled system matrix A and non-linear convective term B(U)
A = sp.vstack([
    sp.hstack([D2psi, -1/a * Dxcpsi]),
    sp.hstack([np.zeros((N, N)), D2c])
])

fun = lambda t, U: A @ U + F + B(U)

# Integrate stiff system using LSODA
sol = solve_ivp(fun, [0, 0.21], U0, t_eval=[0, 0.02, 0.05, 0.1, 0.15, 0.21], method="LSODA")
```

---

## 5. Numerical Results and Discussion

### 5.1 Final Concentration Field ($t = 0.21$)
The contour plot displays the steady-state saltwater wedge developing along the bottom of the aquifer:

![Henry Problem Final Concentration](../figures/ex6/contourf.png)

**Key Observations:**
- **Salt Wedge Toe:** Dense seawater advances inland along the aquifer base ($y=0$), establishing the characteristic curved saltwater toe.
- **Freshwater Transition:** Inland freshwater discharge displaces salt upwards toward the top-right exit, forming a stable dispersive mixing zone.
- **Coupled Stability:** The multi-operator GFDM formulation preserves stability across the non-linear convection coupling throughout the LSODA integration.

---

## 6. Verification and Conclusions

- **Multi-Physics Coupling:** Demonstrates that GFDFlow operators can be combined into coupled flow and transport systems without modification to the core package.
- **Benchmark Validity:** The resulting concentration profile matches the classic semi-analytical and numerical solutions for the Henry benchmark.
