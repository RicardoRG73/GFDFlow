# Example 07: Buoyancy-Driven Convective Flow (Elder Problem)

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), first and second-order differential operators, and coupled variable-density systems, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Simulate the Elder benchmark problem for natural buoyancy-driven convection in porous media, characterized by fingering instabilities triggered by high-density brine infiltration at the top boundary.
- **Domain Type:** Rectangular 2D porous domain representing an initially quiescent aquifer.
- **Governing Physics:** Coupled Darcy flow and non-linear advection-dispersion transport under non-uniform fluid density, resulting in Rayleigh-like convective cellular circulation.
- **Key Validation Points:** Formation, growth, and downward propagation of dense salinity plumes ("salt fingers") integrated up to $t_{\text{final}} = 1.239$ with LSODA.
- **Associated Code:** [`../ex7Elder.py`](../ex7Elder.py)
- **Mesh / Input Data:** [`../meshes/mesh7.json`](../meshes/mesh7.json)
- **Computed Results:** [`../results/ex7Elder.json`](../results/ex7Elder.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Equations
The dimensionless governing system couples the stream function $\psi$ and concentration $C$:

$$\nabla^2 \psi = -\operatorname{Ra} \frac{\partial C}{\partial x}$$
$$\frac{\partial C}{\partial t} + \mathbf{v} \cdot \nabla C = \nabla^2 C$$

where $\operatorname{Ra}$ is the Rayleigh number governing convective instability, and Darcy velocities are $\mathbf{v} = (\psi_y, -\psi_x)$.

### 2.2 Boundary Conditions
- **Bottom and Lateral Boundaries:** Impermeable streamlines ($\psi = 0$) and zero solute flux ($\partial C / \partial n = 0$).
- **Top Boundary:** Impermeable stream boundary ($\psi = 0$) partitioned into three segments:
  - Top Left / Top Right: Zero solute concentration ($C = 0$).
  - Top Center (`top_middle_nodes`): Continuous high-density salt source ($C = 1.0$).

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout
The rectangular domain is discretized with an unstructured point cloud of $N = 1326$ nodes (loaded from `mesh7.json`), with refined nodal resolution beneath the central infiltration zone.

### 3.2 Discrete Differential Operators
Discrete GFDM operators $\mathbf{D}_x, \mathbf{D}_y, \mathbf{D}^2$ are constructed with precomputed Delaunay star stencils and pseudo-inverses $M^\dagger$.

---

## 4. GFDFlow Pipeline & Implementation

The implementation in [`../ex7Elder.py`](../ex7Elder.py) structures the non-linear convective ODE system:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdm
from scipy.integrate import solve_ivp
import scipy.sparse as sp

problem = gfdm(coords, triangles, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)
# ... [boundary condition definitions]

D2psi, F2psi = problem.continuous_discretization()

# Non-linear ODE RHS function
def rhs(t, U):
    # Evaluates coupled stream function and advective transport
    return fun(t, U)

# Integrate up to t = 1.239 using LSODA
t_eval = [0, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.5, 1.239]
sol = solve_ivp(rhs, [0, 1.239], U0, t_eval=t_eval, method="LSODA")
```

---

## 5. Numerical Results and Discussion

### 5.1 Final Concentration Field ($t = 1.239$)
The contour map exhibits the classic symmetric descending convective salt fingers:

![Elder Problem Final Concentration Field](../figures/ex7/contourf.png)

**Key Observations:**
- **Plume Symmetry & Descent:** Dense fluid originating from the central top section descends vertically and bifurcates into symmetric convection cells.
- **Upwelling Recirculation:** Displaced freshwater circulates upwards along the lateral boundaries, maintaining convective equilibrium.
- **Robust Integration:** High-order LSODA time-stepping captures the non-linear fingering dynamics without spurious numerical oscillations.

---

## 6. Verification and Conclusions

- **Convective Instability Validation:** GFDFlow reliably simulates highly non-linear, density-driven flow instabilities in unstructured 2D domains.
- **Benchmark Consistency:** Finger morphology and descent timing conform to published Elder benchmark literature.
