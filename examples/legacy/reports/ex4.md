# Example 04: Multi-Material Flow with Interface Triple Junctions

> **Reference Documentation:** For theoretical derivations of the Generalized Finite Difference Method (GFDM), multi-material interface balances, and singular intersection node formulations, refer to the [GFDM Theoretical Foundations](../../../02-DOCS/wiki/theory.md).

---

## 1. Executive Summary

- **Objective:** Simulate 2D potential seepage through a complex geological domain with three distinct materials meeting at multiple triple-junction interface intersection nodes.
- **Domain Type:** Multi-material heterogeneous domain partitioned into rock, clay, and mixed lithologies.
- **Governing Physics:** Steady-state Laplace groundwater potential flow ($\nabla^2 u = 0$) with material conductivities ($k_r = 1.0$, $k_c = 0.1$, $k_m = 0.5$).
- **Key Validation Points:** Treatment of interface triple-junction points using `problem.intersection` to enforce directional flux balance where three materials converge at a single vertex.
- **Associated Code:** [`../ex4.py`](../ex4.py)
- **Mesh / Input Data:** [`../meshes/mesh4.json`](../meshes/mesh4.json)

---

## 2. Problem Formulation & Boundary Conditions

### 2.1 Governing Differential Operator
Under the general operator formulation $L u = f$ (see [Theory §2](../../../02-DOCS/wiki/theory.md#2-the-general-second-order-linear-differential-operator)):

$$L u = \nabla^2 u = u_{xx} + u_{yy} = 0$$

- **Operator Vector:** $\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$
- **Conductivities:**
  - Rock: $k_r = 1.0$
  - Clay: $k_c = 0.1$
  - Mixed: $k_m = 0.5$
- **Source Term:** $f(x, y) = 0$

### 2.2 Boundary & Interface Network
The system features a network of 6 interface segments and 2 triple intersection nodes (see [Theory §7](../../../02-DOCS/wiki/theory.md#7-interface-flux-balance-for-layered-media)):

| Identifier | Type | Condition | Description |
|---|---|---|---|
| `left` | Dirichlet | $u = 8.0$ | High-pressure reservoir boundary |
| `right` | Dirichlet | $u = 0.0$ | Low-pressure drainage boundary |
| `top`, `bottom` | Neumann | $\frac{\partial u}{\partial n} = 0$ | No-flow impermeable boundaries |
| `interface_a` to `interface_f` | Interface | Continuous flux ($\beta = 0$) | Internal boundaries dividing rock, clay, and mixed zones |
| `inters_1` to `inters_6` | Intersection | Multi-material flux balance | Multi-wedge angular flux balance at nodes 1 and 2 |

---

## 3. Geometry and Spatial Discretization

### 3.1 Domain Layout & Multiple Interfaces
The domain is partitioned into polygonal geological blocks:
- **Center Node 1:** Intersection of `interface_a`, `interface_d`, and `interface_c` between mixed, clay, and rock domains.
- **Center Node 2:** Intersection of `interface_b`, `interface_e`, and `interface_f` between rock, clay, and mixed domains.

### 3.2 Nodal Discretization
From `mesh4.json`:
- Interior sets for each lithology: `rock_nodes`, `clay_nodes`, `mixed_nodes`
- 6 interface nodal sets: `interface_a_nodes` through `interface_f_nodes`
- Precomputed support stencils and pseudo-inverses $M^\dagger$.

---

## 4. GFDFlow Pipeline & Implementation

The setup in [`../ex4.py`](../ex4.py) configures the intersection operators:

```python
from GFDFlow.GFDM import GFDMI_2D_problem as gfdmi
import scipy.sparse as sp

problem = gfdmi(coords, triangles, normal_vecs, L, source, M_pinv=M_pinv, support_stencils=support_stencils)

problem.material("rock", kr, rock_nodes)
problem.material("clay", kc, clay_nodes)
problem.material("mixed", km, mixed_nodes)

problem.dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.dirichlet_boundary("right", right_nodes, right_dirichlet)
problem.neumann_boundary("bottom", kr, bottom_nodes, neumann_zero)
problem.neumann_boundary("top", kr, top_nodes, neumann_zero)

# Configure 6 interface segments
problem.interface("interface_a", kr, km, interface_a_nodes, None, beta, None, rock_nodes, mixed_nodes)
# ... [interfaces b through f]

# Configure triple-junction intersections at central nodes
problem.intersection("inters_1", center_node_1, "interface_a", "interface_d", "mixed", beta)
problem.intersection("inters_2", center_node_1, "interface_d", "interface_c", "clay", beta)
problem.intersection("inters_3", center_node_1, "interface_c", "interface_a", "rock", beta)

problem.intersection("inters_4", center_node_2, "interface_b", "interface_e", "rock", beta)
problem.intersection("inters_5", center_node_2, "interface_e", "interface_f", "clay", beta)
problem.intersection("inters_6", center_node_2, "interface_f", "interface_b", "mixed", beta)

K, F = problem.continuous_discretization()
U = sp.linalg.spsolve(K, F)
```

---

## 5. Numerical Results and Discussion

### 5.1 Potential Field Distribution (2D Contour)
The scalar hydraulic head field transitions across the multi-material domain:

![Solution Contour Map](../figures/ex4/contourf.png)

**Key Observations:**
- **Refraction of Flownet:** Streamlines and potential contours bend characteristically as they cross conductivity boundaries ($1.0 \to 0.5 \to 0.1$).
- **Singular Intersection Stability:** No numerical instability or singular behavior occurs at the multi-junction nodes 1 and 2, proving the robustness of the wedge-based intersection weighting.
- **Physical Consistency:** High hydraulic gradients develop inside the low-permeability clay wedge, shielding downstream low-head regions.

---

## 6. Verification and Conclusions

- **Multi-Wedge Treatment:** GFDFlow's `intersection` method properly assigns compound equilibrium equations at multi-material vertices.
- **Practical Application:** Demonstrates applicability to complex geotechnical geology featuring faulted, layered, and lenticular rock-soil configurations.
