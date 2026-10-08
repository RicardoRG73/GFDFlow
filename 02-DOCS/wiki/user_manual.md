# GFDFlow — User Manual

Comprehensive Guide to Modeling and Solving 2D Partial Differential Equations with the Generalized Finite Difference Method (GFDM) in Python.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation and Environment Setup](#2-installation-and-environment-setup)
3. [Core Mathematical and Data Concepts](#3-core-mathematical-and-data-concepts)
   - [Domain Representation: Nodes and Triangles](#domain-representation-nodes-and-triangles)
   - [Outward Normal Vectors](#outward-normal-vectors)
   - [The Differential Operator Vector L](#the-differential-operator-vector-l)
   - [Support Stencils and M Matrix Pseudo-Inverses](#support-stencils-and-m-matrix-pseudo-inverses)
4. [Standard Workflow Tutorial](#4-standard-workflow-tutorial)
   - [Step 1: Geometry and Mesh Generation](#step-1-geometry-and-mesh-generation)
   - [Step 2: Normal Vectors Assembly](#step-2-normal-vectors-assembly)
   - [Step 3: Initializing `GFDMI_2D_problem`](#step-3-initializing-gfdmi_2d_problem)
   - [Step 4: Defining Materials](#step-4-defining-materials)
   - [Step 5: Applying Boundary Conditions](#step-5-applying-boundary-conditions)
   - [Step 6: Handling Material Interfaces and Intersections](#step-6-handling-material-interfaces-and-intersections)
   - [Step 7: Matrix Assembly and Linear Solving](#step-7-matrix-assembly-and-linear-solving)
   - [Step 8: Visualizing Results](#step-8-visualizing-results)
5. [Complete Working Examples](#5-complete-working-examples)
   - [Example A: 2D Laplace Problem with Mixed Boundaries](#example-a-2d-laplace-problem-with-mixed-boundaries)
   - [Example B: Two-Layer Material with Interface Flux Continuity](#example-b-two-layer-material-with-interface-flux-continuity)
   - [Example C: Precomputed JSON Mesh Loading](#example-c-precomputed-json-mesh-loading)
6. [Best Practices and Numerical Considerations](#6-best-practices-and-numerical-considerations)
7. [Troubleshooting and Common Pitfalls](#7-troubleshooting-and-common-pitfalls)

---

## 1. Overview

`GFDFlow` is a scientific Python library designed to model and solve second-order partial differential equations (PDEs) in two-dimensional domains using the **Generalized Finite Difference Method (GFDM)**. 

Unlike conventional grid-based finite difference schemes that require rectangular structured grids, GFDM operates on arbitrary, scattered node clouds and irregular domains. Compared to standard Finite Element Methods (FEM), GFDM is a point-set collocation technique that eliminates the overhead of global element integration and stiffness assembly loops, resolving PDEs directly through local star Taylor expansions.

`GFDFlow` is specifically engineered for transport phenomena across heterogeneous and multilayer media with:
- Spatially varying permeability / conductivity fields $k(x, y)$.
- Dirichlet and Neumann boundary conditions (via ghost-node formulation).
- Continuous and discontinuous material interfaces with flux balance enforcement.
- High-performance sparse system assembly utilizing `scipy.sparse`.

For a rigorous mathematical derivation of Taylor expansion weights and least-squares formulations, refer to [`theory.md`](theory.md).

---

## 2. Installation and Environment Setup

### System Requirements
- Python 3.9+ (Python 3.11 recommended).
- Recommended: Conda or Miniconda package manager.

### Setting Up a Dedicated Conda Environment

```bash
# Create and activate an isolated environment
conda create -n gfdflow python=3.11 -y
conda activate gfdflow
```

### Installing Dependencies

From the repository root, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core dependencies include:
- `numpy >= 1.20`
- `scipy >= 1.7`
- `matplotlib >= 3.4`

### Editable Package Installation

Install `GFDFlow` in editable development mode:

```bash
pip install -e .
```

Verify that the package is importable:

```bash
python -c "import GFDFlow; print(GFDFlow.__all__)"
```

---

## 3. Core Mathematical and Data Concepts

### Domain Representation: Nodes and Triangles

The computational domain $D \subset \mathbb{R}^2$ is discretized into $N$ points. While GFDM does not evaluate integrals over elements, local topological connectivity is defined via Delaunay triangulation to efficiently query neighboring nodes:

- **`coords`**: NumPy array of shape `(N, 2)` of type `float64`, containing Cartesian coordinates $[x_i, y_i]$ for each node $i \in \{0, \dots, N-1\}$.
- **`triangles`**: NumPy array of shape `(M, 3)` of type `int_`, containing node indices that form Delaunay triangles.

### Outward Normal Vectors

Neumann boundary conditions and material interfaces require normal vectors defined at boundary nodes. In `GFDFlow`, normal vectors are provided as a dense array:

- **`normal_vectors`**: NumPy array of shape `(N, 2)` of type `float64`.
- For boundary nodes, rows contain the unit outward normal vector $\boldsymbol{n} = [n_x, n_y]$ ($||\boldsymbol{n}||_2 = 1$).
- For interior nodes, rows contain $[0.0, 0.0]$.

The helper function `compute_normal_vectors` in `GFDFlow.utils` can automatically compute unit outward normals for straight and curved boundaries.

### The Differential Operator Vector L

`GFDFlow` solves general second-order linear PDEs of the form:

$$L u = A u + B \frac{\partial u}{\partial x} + C \frac{\partial u}{\partial y} + D \frac{\partial^2 u}{\partial x^2} + E \frac{\partial^2 u}{\partial x \partial y} + F \frac{\partial^2 u}{\partial y^2} = f(x, y)$$

The differential operator is specified as a 6-element 1D array:

$$\boldsymbol{L} = [A, B, C, D, E, F]$$

#### Common Operator Examples:
- **Standard Laplacian** ($\Delta u = u_{xx} + u_{yy} = f$):
  ```python
  L = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
  ```
- **Convection-Diffusion Equation** ($\kappa \Delta u - \boldsymbol{v} \cdot \nabla u = f$ with $\boldsymbol{v} = [v_x, v_y]$):
  ```python
  L = np.array([0.0, -v_x, -v_y, kappa, 0.0, kappa])
  ```
- **Helmholtz Equation** ($\Delta u + k^2 u = f$):
  ```python
  L = np.array([k**2, 0.0, 0.0, 1.0, 0.0, 1.0])
  ```

### Support Stencils and M Matrix Pseudo-Inverses

For each node $i$, GFDM constructs a local "star" of $q$ neighboring nodes ($I_i \subset \{0, \dots, N-1\}$). The local spatial offset matrix $M$ is defined as:

$$M_i = \begin{pmatrix} 1 & 1 & \dots & 1 \\ \Delta x_1 & \Delta x_2 & \dots & \Delta x_q \\ \Delta y_1 & \Delta y_2 & \dots & \Delta y_q \\ \Delta x_1^2 & \Delta x_2^2 & \dots & \Delta x_q^2 \\ \Delta x_1 \Delta y_1 & \Delta x_2 \Delta y_2 & \dots & \Delta x_q \Delta y_q \\ \Delta y_1^2 & \Delta y_2^2 & \dots & \Delta y_q^2 \end{pmatrix}$$

By default, `GFDMI_2D_problem` computes the support stars and Moore-Penrose pseudo-inverses $M^+$ during initialization. Alternatively, for large production meshes, these can be precomputed and supplied via the dictionaries `support_stencils` and `M_pinv`.

---

## 4. Standard Workflow Tutorial

### Step 1: Geometry and Mesh Generation

Define or generate the node coordinates and their Delaunay triangulation:

```python
import numpy as np
from scipy.spatial import Delaunay

# Generate regular or irregular node distribution
x = np.linspace(0.0, 1.0, 21)
y = np.linspace(0.0, 1.0, 21)
X, Y = np.meshgrid(x, y)
coords = np.vstack([X.ravel(), Y.ravel()]).T

# Triangulate to establish topological adjacency
tri = Delaunay(coords)
triangles = tri.simplices.astype(int)
n_nodes = coords.shape[0]
```

### Step 2: Normal Vectors Assembly

Identify boundary nodes and assign outward normal vectors:

```python
from GFDFlow.utils import compute_normal_vectors

normal_vectors = np.zeros((n_nodes, 2), dtype=np.float64)

# Locate boundary nodes
bottom_nodes = np.where(np.isclose(coords[:, 1], 0.0))[0]
top_nodes = np.where(np.isclose(coords[:, 1], 1.0))[0]
left_nodes = np.where(np.isclose(coords[:, 0], 0.0))[0]
right_nodes = np.where(np.isclose(coords[:, 0], 1.0))[0]

# Compute and assign unit normals
normal_vectors[bottom_nodes] = compute_normal_vectors(bottom_nodes, coords)
normal_vectors[top_nodes] = compute_normal_vectors(top_nodes, coords)
normal_vectors[left_nodes] = compute_normal_vectors(left_nodes, coords)
normal_vectors[right_nodes] = compute_normal_vectors(right_nodes, coords)
```

### Step 3: Initializing `GFDMI_2D_problem`

Instantiate the core problem class with the operator $L$ and source function $f$:

```python
from GFDFlow.GFDM import GFDMI_2D_problem

# Poisson equation: u_xx + u_yy = source(x, y)
L = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
source = lambda p: -2.0  # Constant volumetric source

problem = GFDMI_2D_problem(
    coords=coords,
    triangles=triangles,
    normal_vectors=normal_vectors,
    L=L,
    source=source
)
```

### Step 4: Defining Materials

Assign permeability or diffusion coefficient functions $k(x, y)$ to material domains:

```python
# Spatially uniform or variable permeability
k_permeability = lambda p: 1.0

# Interior nodes encompass all domain nodes (excluding boundary nodes governed by other rules)
interior_nodes = np.arange(n_nodes)
problem.material("matrix_domain", k_permeability, interior_nodes)
```

### Step 5: Applying Boundary Conditions

#### Dirichlet Conditions ($u = g(x, y)$)
Directly prescribe function values on boundary nodes:

```python
# Left boundary: u(0, y) = 0.0
problem.dirichlet_boundary("left_wall", left_nodes, lambda p: 0.0)

# Right boundary: u(1, y) = 100.0 * p[1]
problem.dirichlet_boundary("right_wall", right_nodes, lambda p: 100.0 * p[1])
```

#### Neumann Conditions ($k \frac{\partial u}{\partial \boldsymbol{n}} = h(x, y)$)
Prescribe the flux normal derivative using the ghost-node formulation:

```python
# Top wall: zero flux (insulated / impermeable)
problem.neumann_boundary(
    label="top_insulation",
    permeability=lambda p: 1.0,
    boundary_nodes=top_nodes,
    condition=lambda p: 0.0
)

# Bottom wall: prescribed inflow flux
problem.neumann_boundary(
    label="bottom_flux",
    permeability=lambda p: 1.0,
    boundary_nodes=bottom_nodes,
    condition=lambda p: 5.0
)
```

### Step 6: Handling Material Interfaces and Intersections

When two distinct media meet at an interface $\Gamma_{\mathrm{int}}$, `GFDFlow` enforces flux equilibrium:

```python
# Define material interface
problem.interface(
    label="interface_1_2",
    k_left=lambda p: 1.0,             # Permeability on material 1 side
    k_right=lambda p: 10.0,           # Permeability on material 2 side
    nodes_left=interface_nodes_sideA, # Nodes on left boundary of interface
    nodes_right=interface_nodes_sideB,# Counterpart nodes on right boundary
    beta=lambda p: 0.0,               # Source flux jump (0 for continuity)
    alpha=lambda p: 0.0,              # Potential jump (0 for continuous head)
    interior_left=mat1_interior,      # Exclude other material nodes from star
    interior_right=mat2_interior
)
```

### Step 7: Matrix Assembly and Linear Solving

Assemble the global sparse system $K \boldsymbol{U} = \boldsymbol{F}$ and solve using SciPy's sparse direct solver:

```python
import scipy.sparse.linalg as spla

# Assemble global stiffness matrix and RHS vector
K, F = problem.discretization_K_F(continuous=True)

# Solve the linear system
U = spla.spsolve(K, F)
```

### Step 8: Visualizing Results

`GFDFlow.visualization` offers high-level plotting utilities tailored for GFDM solutions:

```python
from GFDFlow.visualization import plot_solution_2d, plot_solution_3d

# 2D contour plot
fig_2d, ax_2d = plot_solution_2d(
    coords=coords,
    u=U,
    triangles=triangles,
    cmap="viridis",
    title="Hydraulic Head Distribution"
)

# 3D surface plot
fig_3d, ax_3d = plot_solution_3d(
    coords=coords,
    u=U,
    triangles=triangles,
    cmap="plasma",
    title="3D Elevation Surface"
)
```

---

## 5. Complete Working Examples

### Example A: 2D Laplace Problem with Mixed Boundaries

This standalone script solves $\Delta u = 0$ on the unit square $[0, 1] \times [0, 1]$ with Dirichlet conditions at $x=0$ and $x=1$ and zero-flux Neumann conditions at $y=0$ and $y=1$.

```python
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from GFDFlow.GFDM import GFDMI_2D_problem
from GFDFlow.utils import compute_normal_vectors
from GFDFlow.visualization import plot_solution_2d

# 1. Mesh generation
N_side = 25
x = np.linspace(0.0, 1.0, N_side)
y = np.linspace(0.0, 1.0, N_side)
X, Y = np.meshgrid(x, y)
coords = np.vstack([X.ravel(), Y.ravel()]).T
tri = Delaunay(coords)
triangles = tri.simplices

# 2. Boundary detection & normals
normal_vectors = np.zeros_like(coords)
left = np.where(np.isclose(coords[:, 0], 0.0))[0]
right = np.where(np.isclose(coords[:, 0], 1.0))[0]
bottom = np.where(np.isclose(coords[:, 1], 0.0))[0]
top = np.where(np.isclose(coords[:, 1], 1.0))[0]

normal_vectors[bottom] = compute_normal_vectors(bottom, coords)
normal_vectors[top] = compute_normal_vectors(top, coords)
normal_vectors[left] = compute_normal_vectors(left, coords)
normal_vectors[right] = compute_normal_vectors(right, coords)

# 3. Setup problem
L = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]) # Laplacian: u_xx + u_yy = 0
source = lambda p: 0.0
problem = GFDMI_2D_problem(coords, triangles, normal_vectors, L, source)

# 4. Materials and Boundaries
all_nodes = np.arange(coords.shape[0])
problem.material("domain", lambda p: 1.0, all_nodes)
problem.dirichlet_boundary("left", left, lambda p: 0.0)
problem.dirichlet_boundary("right", right, lambda p: 100.0)
problem.neumann_boundary("bottom", lambda p: 1.0, bottom, lambda p: 0.0)
problem.neumann_boundary("top", lambda p: 1.0, top, lambda p: 0.0)

# 5. Assemble and Solve
K, F = problem.discretization_K_F(continuous=True)
U = spla.spsolve(K, F)

# 6. Plot
fig, ax = plot_solution_2d(coords, U, triangles=triangles, title="Laplacian with Mixed BCs")
plt.show()
```

---

### Example B: Two-Layer Material with Interface Flux Continuity

Solve a 1D-like flow through two layered media with permeabilities $k_1 = 1.0$ (left: $x \in [0, 0.5]$) and $k_2 = 5.0$ (right: $x \in [0.5, 1.0]$).

```python
import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay
from GFDFlow.GFDM import GFDMI_2D_problem
from GFDFlow.utils import compute_normal_vectors

# Generate grid
x = np.linspace(0.0, 1.0, 31)
y = np.linspace(0.0, 1.0, 15)
X, Y = np.meshgrid(x, y)
coords = np.vstack([X.ravel(), Y.ravel()]).T
tri = Delaunay(coords)
triangles = tri.simplices

# Identify regions
idx_mat1 = np.where(coords[:, 0] < 0.5)[0]
idx_mat2 = np.where(coords[:, 0] > 0.5)[0]
idx_interface = np.where(np.isclose(coords[:, 0], 0.5))[0]

normal_vectors = np.zeros_like(coords)
left_nodes = np.where(np.isclose(coords[:, 0], 0.0))[0]
right_nodes = np.where(np.isclose(coords[:, 0], 1.0))[0]
normal_vectors[left_nodes] = compute_normal_vectors(left_nodes, coords)
normal_vectors[right_nodes] = compute_normal_vectors(right_nodes, coords)
normal_vectors[idx_interface] = np.array([1.0, 0.0]) # Outward from mat1 to mat2

L = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
problem = GFDMI_2D_problem(coords, triangles, normal_vectors, L, lambda p: 0.0)

problem.material("mat1", lambda p: 1.0, idx_mat1)
problem.material("mat2", lambda p: 5.0, idx_mat2)

# Interface continuity
problem.interface(
    label="inter",
    k_left=lambda p: 1.0,
    k_right=lambda p: 5.0,
    nodes_left=idx_interface,
    nodes_right=idx_interface,
    beta=lambda p: 0.0,
    alpha=lambda p: 0.0,
    interior_left=idx_mat1,
    interior_right=idx_mat2
)

problem.dirichlet_boundary("inlet", left_nodes, lambda p: 100.0)
problem.dirichlet_boundary("outlet", right_nodes, lambda p: 0.0)

K, F = problem.discretization_K_F(continuous=True)
U = spla.spsolve(K, F)
print(f"Solved successfully. Potential at interface: {np.mean(U[idx_interface]):.2f}")
```

---

### Example C: Precomputed JSON Mesh Loading

In large meshes or legacy examples (`examples/legacy/ex0.py` to `ex11.py`), support stars and Taylor inverse matrices are precomputed and loaded directly from JSON to avoid startup overhead:

```python
import json
import numpy as np
import scipy.sparse.linalg as spla
from GFDFlow.GFDM import GFDMI_2D_problem

with open("examples/legacy/meshes/mesh0.json", "r") as f:
    mesh_data = json.load(f)

coords = np.array(mesh_data["coords"])
triangles = np.array(mesh_data["triangles"])
normal_vectors = np.array(mesh_data["normal_vectors"])
support_stencils = {int(k): np.array(v) for k, v in mesh_data["support_stencils"].items()}
M_pinv = {int(k): np.array(v) for k, v in mesh_data["M_pinv"].items()}

# Direct instant instantiation without matrix inversion overhead
L = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
problem = GFDMI_2D_problem(
    coords, triangles, normal_vectors, L, lambda p: 0.0,
    support_stencils=support_stencils,
    M_pinv=M_pinv
)
```

---

## 6. Best Practices and Numerical Considerations

1. **Node Density and Support Stars:**
   - A minimum of 5 support nodes is required to resolve all 5 second-order spatial derivatives.
   - Using 6 to 9 nodes per star generally provides improved numerical conditioning and avoids ill-conditioned $M$ matrices.
2. **Sparse Matrices:**
   - Always solve the linear system using sparse solvers (`scipy.sparse.linalg.spsolve` or iterative solvers like `bicgstab` or `gmres`).
   - Do not convert the global matrix $K$ to a dense array (`.toarray()`) on meshes with more than 1,000 nodes to prevent memory exhaustion.
3. **Condition Number of $M$:**
   - Ensure nodes in a local star are not collinear. When points are nearly collinear, $\det(M^T M) \approx 0$, which degrades the accuracy of $\Gamma$. Delaunay triangulation provides isotropic neighbor selection that naturally prevents collinear stars.
4. **Normal Vector Alignment:**
   - Outward normal vectors must point away from the domain interior. If normal vectors are inverted, the Neumann ghost node will be placed inside the domain, compromising boundary physics.

---

## 7. Troubleshooting and Common Pitfalls

| Symptom | Probable Cause | Recommended Fix |
|---|---|---|
| `ValueError: normal_vectors must have shape (n, 2)...` | Row dimension of `normal_vectors` does not match `coords.shape[0]`. | Allocate `normal_vectors = np.zeros_like(coords)` and populate only boundary rows. |
| `Singular matrix` in `spsolve` | Underspecified boundary conditions (e.g., pure Neumann without any reference Dirichlet node). | Add at least one Dirichlet constraint to fix the reference potential/gauge. |
| Solution blows up or oscillates near boundary | Normal vectors pointing inwards or ghost node distance too large. | Verify unit normal directions using `plot_normal_vectors` in `GFDFlow.visualization`. |
| Interface discontinuity unexpected | Called `discretization_K_F(continuous=False)` instead of `continuous=True`. | Pass `continuous=True` unless explicitly modeling physical potential jump discontinuities. |
