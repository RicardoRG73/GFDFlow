# 2D Poisson Equation Solution

## 1. Executive Summary

This report documents the numerical solution of a two-dimensional Poisson equation defined over an arbitrary bounded domain with a curved boundary. The spatial discretization is performed using the **Generalized Finite Difference Method (GFDM)** as implemented in the **GFDFlow** framework. The domain features mixed boundary conditions, including non-homogeneous Dirichlet conditions along three boundary segments and a homogeneous Neumann flux condition on a curved circular arc. The numerical assembly incorporates the ghost-node technique to enforce derivative boundary conditions accurately on unstructured nodal sets.

---

## 2. Mathematical Formulation

### 2.1 Governing Differential Equation
GFDFlow solves linear second-order Partial Differential Equations (PDEs) in the general operator form:

$$\mathcal{L}u = A u + B u_x + C u_y + D u_{xx} + E u_{xy} + F u_{yy} = f(x, y)$$

where $\mathbf{L} = [A, B, C, D, E, F]^T$ is the coefficient vector. For this benchmark problem, the parameters are defined as:

$$\mathbf{L} = [0, 0, 0, 1, 0, 1]^T$$

This reduces the general differential operator to the 2D Poisson equation:

$$\nabla^2 u = u_{xx} + u_{yy} = f(x, y)$$

with a uniform constant source term $f(x, y) = -2$ across the domain, and an isotropic material permeability $k(p) = 1.0$.

### 2.2 Boundary Conditions
The domain boundary $\partial \Omega$ is decomposed into four distinct segments:

1. **Left Boundary ($x = 0$)**: Dirichlet boundary condition  
   $$u(0, y) = 0$$
2. **Bottom Boundary ($y = 0$)**: Linear Dirichlet boundary condition  
   $$u(x, 0) = 0.5 x$$
3. **Top Boundary ($y = 1$)**: Linear Dirichlet boundary condition  
   $$u(x, 1) = x$$
4. **Right Boundary (Curved Arc)**: Homogeneous Neumann boundary condition  
   $$\frac{\partial u}{\partial n} = \mathbf{n} \cdot \nabla u = 0$$
   where $\mathbf{n} = (n_x, n_y)$ is the outward unit normal vector.

---

## 3. Geometry and Mesh Discretization

### 3.1 Domain Definition
The physical domain $\Omega$ is defined by five key control points:
- $P_0 = (0, 0)$
- $P_1 = (1, 0)$
- $P_2 = (2, 0)$
- $P_3 = (1, 1)$
- $P_4 = (0, 1)$

The right boundary is described by a circular arc centered at $(1,0)$ with radius $R = 1$, connecting point $P_2 = (2,0)$ to point $P_3 = (1,1)$. The remaining boundaries are straight spline segments.

![Domain Geometry](../figures/ex0/geometry.png)

### 3.2 Nodal Cloud and Triangular Mesh
Using `calfem` and `Gmsh`, an unstructured triangular mesh is generated over the geometry with an element size factor of $\delta = 0.08$. The node cloud consists of interior nodes and boundary nodes segregated according to boundary condition type.

![Unstructured Triangular Mesh](../figures/ex0/mesh.png)

### 3.3 Boundary Classification and Normal Vectors
The boundary nodes are categorized into discrete sets:
- **Left Nodes**: Dirichlet ($u = 0$)
- **Bottom Nodes**: Dirichlet ($u = 0.5x$)
- **Top Nodes**: Dirichlet ($u = x$)
- **Right Nodes**: Neumann ($\partial u / \partial n = 0$)
- **Interior Nodes**: Governing Poisson equation ($\nabla^2 u = -2$)

![Boundary Node Classification](../figures/ex0/boundaries.png)

For the curved Neumann boundary (Right Nodes), unit outward normal vectors $\mathbf{n}$ are computed automatically using geometric normals to enforce directional flux constraints.

![Normal Vectors on Neumann Boundary](../figures/ex0/normal_vectors.png)

---

## 4. GFDM Numerical Scheme and Assembly

### 4.1 Local Taylor Expansion & Moment Matrix
At each central node $i = (x_i, y_i)$, a local star stencil $S_i$ is formed by its spatial neighbors. The function $u(x, y)$ around node $i$ is expanded via a second-order Taylor series:

$$u_j \approx u_i + h_j \left.\frac{\partial u}{\partial x}\right|_i + k_j \left.\frac{\partial u}{\partial y}\right|_i + \frac{h_j^2}{2} \left.\frac{\partial u}{\partial x^2}\right|_i + \frac{k_j^2}{2} \left.\frac{\partial u}{\partial y^2}\right|_i + h_j k_j \left.\frac{\partial u}{\partial x \partial y}\right|_i$$

where $h_j = x_j - x_i$ and $k_j = y_j - y_i$ for each $j \in S_i$. The weighted least-squares approximation yields the moment matrix $M_i$:

$$\mathbf{M}_i \boldsymbol{\mathbf{D}} u_i \approx \boldsymbol{\Delta} u_i$$

where $\boldsymbol{\mathbf{D}} u_i = [u_i, u_{x,i}, u_{y,i}, u_{xx,i}, u_{yy,i}, u_{xy,i}]^T$. The pseudo-inverse $\mathbf{M}_i^\dagger$ is pre-calculated to build sparse linear system coefficients efficiently.

### 4.2 Ghost-Node Formulation for Neumann Boundaries
To enforce the Neumann flux condition $\mathbf{n}_i \cdot \nabla u = n_x u_{x,i} + n_y u_{y,i} = 0$ on the right boundary without requiring structured exterior grids, an augmented ghost-node formulation is employed. An extended moment matrix $\mathbf{M}_{\text{augmented}}$ incorporates ghost displacement components projected along normal vector $\mathbf{n}_i$, eliminating the ghost potential analytically during system assembly.

---

## 5. Numerical Results and Discussion

The assembled sparse linear system $\mathbf{K} \mathbf{U} = \mathbf{F}$ is solved using `scipy.sparse.linalg.spsolve`.

### 5.1 Potential Field Contour Distribution
The calculated scalar potential field $U(x, y)$ smoothly transitions between the specified Dirichlet boundaries while satisfying the internal curvature demands of the source term $f(x, y) = -2$.

![Solution Contour Map](../figures/ex0/contourf.png)

Key observations from the contour map:
- **Left boundary ($x=0$)**: $U = 0$ is strictly satisfied.
- **Bottom boundary ($y=0$)**: Linearly increases from $0$ at $x=0$ to $1.0$ at $x=2$.
- **Top boundary ($y=1$)**: Linearly increases from $0$ at $x=0$ to $1.0$ at $x=1$.
- **Right boundary**: Contour lines meet the curved boundary orthogonally, confirming that $\frac{\partial u}{\partial n} = 0$ is accurately enforced.

### 5.2 3D Surface Plot
The 3D surface plot depicts the continuous potential landscape across the irregular domain.

![3D Surface Plot](../figures/ex0/3dplot.png)

---

## 6. Summary

This benchmark demonstrates the capability of **GFDFlow** to model Poisson transport problems on non-rectangular geometries with mixed Dirichlet and Neumann boundary conditions using the Generalized Finite Difference Method. The integration of Gmsh unstructured meshes with GFDM Taylor-expansion stencils and ghost-node derivative constraints provides accurate, stable, and continuous potential field solutions.
