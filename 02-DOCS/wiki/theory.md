# GFDM - Theory

Mathematical Foundations and Boundary Implementations of the Generalized Finite Difference Method (GFDM) in GFDFlow

---

## 1. Introduction to GFDM and the GFDFlow Framework

The Generalized Finite Difference Method (GFDM) is a robust numerical technique implemented in the `GFDFlow` Python package to solve complex two-dimensional transport phenomena. As an object-oriented framework, `GFDFlow` discretizes second-order differential operators, converting partial differential equations (PDEs) into solvable systems of linear algebraic equations.

GFDM offers significant computational advantages over traditional Finite Element Methods (FEM) and Finite Difference Methods (FDM):

* **Flexibility in Node Distribution:** Unlike standard FDM, which is constrained to rectangular grids, GFDM operates on arbitrary node distributions, including non-rectangular structured and unstructured grids.
* **Meshless "Point-Set" Nature:** GFDM eliminates the need for formal mesh connectivity, managing irregular geometries by utilizing local "stars" of support nodes.
* **Computational Efficiency:** The method provides high-order accuracy with lower computational overhead than FEM, particularly in geosciences and engineering applications involving irregular domains.

---

## 2. The General Second-Order Linear Differential Operator

`GFDFlow` is designed to solve the general second-order linear differential operator $L$. The governing problem is formulated as $Lu = f$, where $u$ represents the unknown function (e.g., hydraulic head $h$ or concentration $C$) and $f$ is the source term. In a two-dimensional domain $D$, the operator $L$ is expanded as:

$$Lu = A u + B \frac{\partial u}{\partial x} + C \frac{\partial u}{\partial y} + D \frac{\partial^2 u}{\partial x^2} + E \frac{\partial^2 u}{\partial x \partial y} + F \frac{\partial^2 u}{\partial y^2} = f$$

Alternatively expressed in subscript notation as:

$$Lu = A u + B u_x + C u_y + D u_{xx} + E u_{xy} + F u_{yy} = f$$

where the coefficients and terms are defined as:

* $A, B, C, D, E, F$: Spatial functions $f(x, y)$ representing physical properties like permeability, dispersion, or reaction rates.
* $u_x, u_y$: First-order partial derivatives ($\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}$).
* $u_{xx}, u_{xy}, u_{yy}$: Second-order partial derivatives ($\frac{\partial^2 u}{\partial x^2}, \frac{\partial^2 u}{\partial x \partial y}, \frac{\partial^2 u}{\partial y^2}$).
* $f$: The source value, often provided as a function of spatial coordinates $\boldsymbol{p} = [x, y]$.

---

## 3. Taylor Series Expansion and Local Approximation

The core logic of GFDM involves approximating the operator $L$ at a central node $\boldsymbol{p}_0$ through the function values at a local set of support nodes, known as a "star".

1. **Selection of Support Nodes:** For a central node $\boldsymbol{p}_0$, a set of $q$ support nodes $\boldsymbol{p}_i$ ($i = 1, \dots, q$) is selected.
2. **Local Approximation:** The local operator $L_0$ is defined as a linear combination of function values $u_i = u(\boldsymbol{p}_i)$:

$$L_0 u = \sum_{i=0}^q \Gamma_i u(\boldsymbol{p}_i)$$

3. **Consistency Requirement:** The weights $\Gamma_i$ must be chosen such that the local truncation error $\tau = Lu - L_0 u$ approaches zero.
4. **Taylor Expansion:** By expanding $u(\boldsymbol{p}_i)$ around $\boldsymbol{p}_0$ using a second-order Taylor polynomial and substituting into the consistency condition, a linear system of equations is derived to solve for the weights.

---

## 4. Unweighted Least-Squares Approximation for Weight Calculation

To calculate the weights $\Gamma_i$, `GFDFlow` solves an unweighted least-squares problem. This avoids the parameter calibration required by weighted versions while maintaining stability on irregular grids.

### 4.1. Block Matrix System Construction

The weights for the support nodes $(\Gamma_1, \dots, \Gamma_q)^T$ are solved via the system:

$$M \Gamma = L$$

where $M$ is a $5 \times q$ matrix of spatial increments:

$$M = \begin{pmatrix} 
\Delta x_1 & \dots & \Delta x_q \\ 
\Delta y_1 & \dots & \Delta y_q \\ 
(\Delta x_1)^2 & \dots & (\Delta x_q)^2 \\ 
\Delta x_1 \Delta y_1 & \dots & \Delta x_q \Delta y_q \\ 
(\Delta y_1)^2 & \dots & (\Delta y_q)^2 
\end{pmatrix}$$

The vector $L$ contains the operator coefficients at $\boldsymbol{p}_0$, specifically ordered as $L = [B, C, 2D, E, 2F]^T$ to correspond to the Taylor derivatives:

| Matrix $M$ Term | Derivative Component | Vector $L$ Coefficient |
|---|---|---|
| $\Delta x_i$ | $u_x$ | $B$ |
| $\Delta y_i$ | $u_y$ | $C$ |
| $(\Delta x_i)^2$ | $u_{xx}$ | $2D$ |
| $\Delta x_i \Delta y_i$ | $u_{xy}$ | $E$ |
| $(\Delta y_i)^2$ | $u_{yy}$ | $2F$ |

*(Note: $\Delta x_i = x_i - x_0$ and $\Delta y_i = y_i - y_0$.)*

### 4.2. Solving the Weights

When $q > 5$, the system is overdetermined. The kernel of $M$ is non-trivial, creating a linear manifold of potential solutions. `GFDFlow` selects the optimal weights by solving the normal equations:

$$M^T M \Gamma = M^T L$$

This is solved efficiently using matrix factorizations. Finally, the weight for the central node $\Gamma_0$ is determined by the consistency requirement for the zeroth-order term:

$$\Gamma_0 = A - \sum_{i=1}^q \Gamma_i$$

where $A$ is the coefficient of the function $u$ in the governing operator.

---

## 5. Domain Discretization and Support Node Selection (Stars)

`GFDFlow` discretizes the domain $D$ into a point cloud. The `get_support_nodes` method identifies the "star" for each node using the following logic:

* **Initial Selection:** Nodes are initially selected if they share a triangle (via Delaunay triangulation) with the central node.
* **Minimum Node Requirement:** A default `min_support_nodes` of 5 is required to ensure the matrix $M$ has full row rank.
* **Iteration Logic:** If the criteria are not met, the algorithm performs a maximum of two iterations (`max_iter=2`).
  * In iterations 0 and 1, the search area expands to include "neighbors of neighbors", increasing the search size exponentially.
* **Symmetry:** The algorithm seeks a symmetric distribution around $\boldsymbol{p}_0$ to maintain the self-adjoint properties of physical operators.

---

## 6. Implementation of Boundary Conditions

### 6.1. Dirichlet Boundary Conditions

Dirichlet conditions fix $u$ to known values at the boundary. These are defined as a function of coordinates $\boldsymbol{p} = [x, y]$ and are directly incorporated into the global system, replacing the governing equation for those specific nodes.

### 6.2. Neumann Boundary Conditions and Ghost Node Formulation

Neumann conditions specify the normal derivative $\frac{\partial u}{\partial n}$ at the boundary. `GFDFlow` utilizes a ghost node formulation to maintain accuracy without relying on one-sided derivatives.

* **Ghost Node ($\boldsymbol{p}_g$):** A virtual node is placed outside the domain such that the boundary node $\boldsymbol{p}_0$ is the midpoint between $\boldsymbol{p}_g$ and an interior node $\boldsymbol{p}_c$ (the neighbor in the negative direction of the outer normal vector).
* **Algebraic Derivation:** The normal derivative at $\boldsymbol{p}_0$ is approximated using $\boldsymbol{p}_g$ and the interior support nodes. This expression is solved algebraically for $u(\boldsymbol{p}_g)$.
* **Substitution:** The value $u(\boldsymbol{p}_g)$ is substituted back into the governing equation for the boundary node $\boldsymbol{p}_0$. This eliminates the virtual unknown, resulting in a discrete operator that consistently enforces the boundary physics.

---

## 7. Interface Flux Balance for Layered Media

In layered media with discontinuous properties (e.g., varying saturated permeability $k_{\mathrm{sat}}$), `GFDFlow` enforces a flux balance at the interface. The balance requires that the sum of flux leaving one material equals the flux entering the next.

* **Interface Balance Equation:** For $n_l$ layers, the governing balance at interface nodes is:

$$\sum_{i=1}^{n_l} \left( k_{xi} \frac{\partial u}{\partial x} l_{xi} + k_{yi} \frac{\partial u}{\partial y} l_{yi} \right) = 0$$

  where $l_{x}, l_{y}$ are the direction cosines of the outward normal.
* **Global Integration:** Critically, this interface balance equation replaces the governing PDE row in the global matrix $K$ for all nodes identified in the interface node sets.
* **Parameters:**
  * `k_left` / `k_right` (`permeability_left_mat` / `permeability_right_mat`): Property functions for each layer.
  * `beta` (`flux_difference_beta`): Handles external sources at the interface.
  * `alpha` (`solution_difference_alpha`): Manages jumps in the solution for discontinuous interfaces (omitted if the solution is continuous).

---

## 8. Global System Assembly and Solution

The transition from local approximations to the global solution $K \boldsymbol{U} = \boldsymbol{F}$ (or $H \boldsymbol{h} + \boldsymbol{f} = \mathbf{0}$) constitutes the final phase of the numerical workflow.

* **Assembly:** Local weights $\Gamma_i$ for each node are mapped to their corresponding global indices in a sparse matrix.
* **Sparse Handling:** `GFDFlow` leverages SciPy and NumPy for high-performance sparse matrix operations, ensuring efficiency even for large-scale problems.
* **Workflow Summary:**
  1. **Input definition:** Define coordinates, connectivity (triangles), and PDE coefficients.
  2. **Problem initialization:** Instantiate the core problem class (`GFDMI_2D_problem`).
  3. **Geometric processing:** Select support nodes and define boundary normal vectors.
  4. **Boundary and material:** Assign permeabilities, boundary conditions, and interface conditions.
  5. **Numerical discretization:** Convert the differential operator into a linear system ($K \boldsymbol{U} = \boldsymbol{F}$).
  6. **Solution and visualization:** Solve the linear system using `scipy.sparse.linalg.spsolve` and visualize results via Matplotlib.
