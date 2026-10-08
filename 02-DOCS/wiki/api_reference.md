# GFDFlow — API Reference

Technical specification and reference manual for all public classes, functions, and modules in `GFDFlow`.

All docstrings follow the **NumPy** documentation standard.

---

## Table of Contents

1. [Package Overview and Imports](#1-package-overview-and-imports)
2. [Module: `GFDFlow.GFDM`](#2-module-gfdflowgfdm)
   - [`GFDMI_2D_problem`](#gfdmi_2d_problem)
     - [`__init__`](#__init__)
     - [`material`](#material)
     - [`dirichlet_boundary`](#dirichlet_boundary)
     - [`neumann_boundary`](#neumann_boundary)
     - [`interface`](#interface)
     - [`intersection`](#intersection)
     - [`discretization_K_F`](#discretization_k_f)
     - [`continuous_discretization`](#continuous_discretization)
     - [`discontinuous_discretization`](#discontinuous_discretization)
     - [`support_nodes`](#support_nodes)
3. [Module: `GFDFlow.utils`](#3-module-gfdflowutils)
   - [`get_support_nodes_2D`](#get_support_nodes_2d)
   - [`compute_normal_vectors`](#compute_normal_vectors)
   - [`compute_M_matrix`](#compute_m_matrix)
   - [`compute_M_matrix_neumann`](#compute_m_matrix_neumann)
4. [Module: `GFDFlow.visualization`](#4-module-gfdflowvisualization)
   - [`plot_solution_2d`](#plot_solution_2d)
   - [`plot_solution_3d`](#plot_solution_3d)
   - [`plot_solution_comparison_3d`](#plot_solution_comparison_3d)
   - [`plot_geometry`](#plot_geometry)
   - [`plot_mesh`](#plot_mesh)
   - [`plot_nodes`](#plot_nodes)
   - [`plot_normal_vectors`](#plot_normal_vectors)
   - [`plot_phreatic_surface`](#plot_phreatic_surface)

---

## 1. Package Overview and Imports

The public entry points for `GFDFlow` are exposed at the top level or via their respective submodules:

```python
# Primary solver class
from GFDFlow.GFDM import GFDMI_2D_problem
# or from top-level package:
from GFDFlow import GFDMI_2D_problem

# Geometry and numerical utility functions
from GFDFlow.utils import (
    get_support_nodes_2D,
    compute_normal_vectors,
    compute_M_matrix,
    compute_M_matrix_neumann,
)

# Plotting and visualization utilities
from GFDFlow.visualization import (
    plot_solution_2d,
    plot_solution_3d,
    plot_solution_comparison_3d,
    plot_geometry,
    plot_mesh,
    plot_nodes,
    plot_normal_vectors,
    plot_phreatic_surface,
)
```

---

## 2. Module: `GFDFlow.GFDM`

### `GFDMI_2D_problem`

```python
class GFDMI_2D_problem:
```

Class to model and solve 2D partial differential equations using the Generalized Finite Difference Method (GFDM) with support for material interfaces and diverse boundary conditions.

The solver discretizes PDEs of the general form:
$$L u = A u + B u_x + C u_y + D u_{xx} + E u_{xy} + F u_{yy} = f(x, y)$$

#### Attributes

* **`coords`** : `numpy.ndarray` of shape `(n_nodes, 2)`, `dtype=float64`  
  Coordinates $[x, y]$ of all nodes in the computational domain.
* **`triangles`** : `numpy.ndarray` of shape `(n_triangles, 3)`, `dtype=int_`  
  Connectivity of nodes forming triangular elements for topological queries.
* **`normal_vectors`** : `numpy.ndarray` of shape `(n_nodes, 2)`, `dtype=float64`  
  Outward unit normal vectors assigned at boundary nodes, and zeros for interior nodes.
* **`L`** : `numpy.ndarray` of shape `(6,)`, `dtype=float64`  
  Differential operator coefficient vector $[A, B, C, D, E, F]$.
* **`source`** : `Callable[[numpy.ndarray], float]`  
  Spatial source term function $f(p)$ receiving coordinate vector $p = [x, y]$.
* **`materials`** : `dict`  
  Registered material configurations mapping labels to `[permeability_func, node_indices]`.
* **`neumann_boundaries`** : `dict`  
  Registered Neumann boundary conditions mapping labels to `[permeability_func, boundary_nodes, condition_func]`.
* **`dirichlet_boundaries`** : `dict`  
  Registered Dirichlet boundary conditions mapping labels to `[boundary_nodes, condition_func]`.
* **`interfaces`** : `dict`  
  Registered material interfaces mapping labels to jump condition definitions and node partitions.
* **`intersections`** : `dict`  
  Registered intersection nodes between multiple interface boundaries.
* **`support_stencils`** : `dict[int, numpy.ndarray]`  
  Map from node index to array of neighboring support star indices.
* **`M_pinv`** : `dict[int, numpy.ndarray]`  
  Map from node index to Moore-Penrose pseudo-inverse of its local spatial Taylor matrix $M$.

---

#### `__init__`

```python
def __init__(
    coords: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    normal_vectors: npt.NDArray[np.float64],
    L: npt.NDArray[np.float64],
    source: Callable[[npt.NDArray[np.float64]], float],
    support_stencils: Optional[Dict[int, npt.NDArray[np.int_]]] = None,
    M_pinv: Optional[Dict[int, npt.NDArray[np.float64]]] = None
) -> None
```

Initializes the 2D GFDM problem instance.

##### Parameters
* **`coords`** : `numpy.ndarray` of shape `(n, 2)`  
  Cartesian coordinates of all $n$ domain nodes.
* **`triangles`** : `numpy.ndarray` of shape `(m, 3)`  
  Triangular simplex node indices.
* **`normal_vectors`** : `numpy.ndarray` of shape `(n, 2)`  
  Unit outward normal vectors for boundary nodes, and zeros for interior nodes. Must have the same number of rows as `coords`.
* **`L`** : `numpy.ndarray` of shape `(6,)`  
  Differential operator coefficients $[A, B, C, D, E, F]$.
* **`source`** : `Callable[[numpy.ndarray], float]`  
  Function mapping a 2D coordinate array $[x, y]$ to a scalar source value $f$.
* **`support_stencils`** : `dict[int, numpy.ndarray]`, optional  
  Precomputed support star indices for each node. If `None`, computed automatically using `get_support_nodes_2D`.
* **`M_pinv`** : `dict[int, numpy.ndarray]`, optional  
  Precomputed pseudo-inverses of matrix $M$ for each node. If `None`, calculated automatically via `np.linalg.pinv`.

##### Raises
* **`ValueError`**  
  If array shapes are inconsistent (e.g., `coords.shape[1] != 2`, `triangles.shape[1] != 3`, `normal_vectors.shape[0] != coords.shape[0]`, or `L.size != 6`).

---

#### `material`

```python
def material(
    label: str,
    permeability: Callable[[npt.NDArray[np.float64]], float],
    interior_nodes: npt.NDArray[np.int_]
) -> None
```

Defines material permeability/diffusivity properties over a specified subset of nodes.

##### Parameters
* **`label`** : `str`  
  Unique identifier for the material.
* **`permeability`** : `Callable[[numpy.ndarray], float]`  
  Spatial function $k(p)$ returning scalar permeability/diffusivity at coordinate $p = [x, y]$.
* **`interior_nodes`** : `numpy.ndarray` of shape `(k,)`  
  Indices of nodes belonging to this material domain.

---

#### `dirichlet_boundary`

```python
def dirichlet_boundary(
    label: str,
    boundary_nodes: npt.NDArray[np.int_],
    condition: Callable[[npt.NDArray[np.float64]], float]
) -> None
```

Applies a Dirichlet boundary condition $u(p) = g(p)$ to a set of boundary nodes.

##### Parameters
* **`label`** : `str`  
  Unique identifier for the boundary condition.
* **`boundary_nodes`** : `numpy.ndarray` of shape `(k,)`  
  Indices of boundary nodes where the Dirichlet condition is enforced.
* **`condition`** : `Callable[[numpy.ndarray], float]`  
  Function returning the prescribed value $g(p)$ at coordinate $p = [x, y]$.

---

#### `neumann_boundary`

```python
def neumann_boundary(
    label: str,
    permeability: Callable[[npt.NDArray[np.float64]], float],
    boundary_nodes: npt.NDArray[np.int_],
    condition: Callable[[npt.NDArray[np.float64]], float]
) -> None
```

Applies a Neumann boundary condition $k \frac{\partial u}{\partial \boldsymbol{n}} = h(p)$ using the GFDM ghost-node formulation.

##### Parameters
* **`label`** : `str`  
  Unique identifier for the boundary condition.
* **`permeability`** : `Callable[[numpy.ndarray], float]`  
  Permeability function $k(p)$ on the boundary.
* **`boundary_nodes`** : `numpy.ndarray` of shape `(k,)`  
  Indices of boundary nodes where the flux is prescribed.
* **`condition`** : `Callable[[numpy.ndarray], float]`  
  Function returning the prescribed normal derivative/flux $h(p)$ at $p = [x, y]$.

---

#### `interface`

```python
def interface(
    label: str,
    k_left: Callable[[npt.NDArray[np.float64]], float],
    k_right: Callable[[npt.NDArray[np.float64]], float],
    nodes_left: npt.NDArray[np.int_],
    nodes_right: npt.NDArray[np.int_],
    beta: Callable[[npt.NDArray[np.float64]], float],
    alpha: Callable[[npt.NDArray[np.float64]], float],
    interior_left: npt.NDArray[np.int_],
    interior_right: npt.NDArray[np.int_]
) -> None
```

Defines an interface between two adjacent materials enforcing flux continuity and potential jump conditions.

##### Parameters
* **`label`** : `str`  
  Unique identifier for the interface.
* **`k_left`** : `Callable[[numpy.ndarray], float]`  
  Permeability function of the material on side A (left).
* **`k_right`** : `Callable[[numpy.ndarray], float]`  
  Permeability function of the material on side B (right).
* **`nodes_left`** : `numpy.ndarray` of shape `(k,)`  
  Node indices along the interface belonging to side A.
* **`nodes_right`** : `numpy.ndarray` of shape `(k,)`  
  Node indices along the interface belonging to side B.
* **`beta`** : `Callable[[numpy.ndarray], float]`  
  Flux jump function across the interface: $\Delta (k \nabla u \cdot \boldsymbol{n}) = \beta(p)$.
* **`alpha`** : `Callable[[numpy.ndarray], float]`  
  Potential jump function across the interface: $u_B - u_A = \alpha(p)$.
* **`interior_left`** : `numpy.ndarray` of shape `(n_left,)`  
  Interior nodes of material A excluded from material B stars.
* **`interior_right`** : `numpy.ndarray` of shape `(n_right,)`  
  Interior nodes of material B excluded from material A stars.

---

#### `intersection`

```python
def intersection(
    label: str,
    intersection_node: int,
    interface_left: str,
    interface_right: str,
    material_between: str,
    f_intersection: Callable[[npt.NDArray[np.float64]], float]
) -> None
```

Defines a singular junction point where two distinct material interfaces intersect.

##### Parameters
* **`label`** : `str`  
  Unique identifier for the intersection.
* **`intersection_node`** : `int`  
  Global node index located at the junction.
* **`interface_left`** : `str`  
  Label of the first intersecting interface.
* **`interface_right`** : `str`  
  Label of the second intersecting interface.
* **`material_between`** : `str`  
  Label of the material occupying the sector between interfaces.
* **`f_intersection`** : `Callable[[numpy.ndarray], float]`  
  Prescribed flux jump or source function at the intersection node.

---

#### `discretization_K_F`

```python
def discretization_K_F(continuous: bool = False) -> Tuple[sp.csr_matrix, npt.NDArray[np.float64]]
```

Assembles the global sparse coefficient matrix $K$ and force/right-hand side vector $F$.

##### Parameters
* **`continuous`** : `bool`, default=`False`  
  If `True`, treats material interfaces with continuous potential fields ($u_A = u_B$).  
  If `False`, enforces jump conditions using dual interface node pairs.

##### Returns
* **`K`** : `scipy.sparse.csr_matrix` of shape `(N, N)`  
  Global stiffness matrix in Compressed Sparse Row format.
* **`F`** : `numpy.ndarray` of shape `(N,)`  
  Global right-hand side vector.

---

#### `continuous_discretization`

```python
def continuous_discretization() -> Tuple[sp.csr_matrix, npt.NDArray[np.float64]]
```

Alias for `discretization_K_F(continuous=True)`.

---

#### `discontinuous_discretization`

```python
def discontinuous_discretization() -> Tuple[sp.csr_matrix, npt.NDArray[np.float64]]
```

Alias for `discretization_K_F(continuous=False)`.

---

#### `support_nodes`

```python
def support_nodes(node_idx: int) -> npt.NDArray[np.int_]
```

Retrieves the indices of support star nodes associated with a given node.

##### Parameters
* **`node_idx`** : `int`  
  Index of the central node.

##### Returns
* **`nodes`** : `numpy.ndarray` of shape `(q,)`, `dtype=int_`  
  Indices of the support nodes forming the star.

---

## 3. Module: `GFDFlow.utils`

Geometric and matrix algebra helper utilities.

### `get_support_nodes_2D`

```python
def get_support_nodes_2D(
    node_idx: int,
    triangles: npt.NDArray[np.int_],
    min_support_nodes: int = 5,
    max_iter: int = 2
) -> npt.NDArray[np.int_]
```

Identifies the star of support nodes for a given central node using triangular adjacency.

##### Parameters
* **`node_idx`** : `int`  
  Index of the central node $p_0$.
* **`triangles`** : `numpy.ndarray` of shape `(m, 3)`  
  Array containing indices of triangular simplices.
* **`min_support_nodes`** : `int`, default=`5`  
  Minimum number of support nodes required to achieve full rank for second-order Taylor systems.
* **`max_iter`** : `int`, default=`2`  
  Maximum breadth-first search iterations across neighboring triangles.

##### Returns
* **`support_nodes`** : `numpy.ndarray` of shape `(q,)`, `dtype=int_`  
  Indices of the support nodes forming the local star.

---

### `compute_normal_vectors`

```python
def compute_normal_vectors(
    boundary_nodes: npt.NDArray[np.int_],
    coords: npt.NDArray[np.float64],
    line_tolerance: float = 0.999
) -> npt.NDArray[np.float64]
```

Computes outward unit normal vectors at boundary nodes with automatic detection of straight and curved boundaries.

##### Parameters
* **`boundary_nodes`** : `numpy.ndarray` of shape `(k,)`  
  Indices of nodes located along the boundary segment.
* **`coords`** : `numpy.ndarray` of shape `(n, 2)`  
  Coordinates of all domain nodes.
* **`line_tolerance`** : `float`, default=`0.999`  
  Collinearity dot-product threshold. When the boundary segment aligns within this tolerance, an exact uniform normal vector is applied. For curved boundaries, normals are computed using 3-nearest-neighbor tangents oriented away from the global centroid.

##### Returns
* **`normal_vecs`** : `numpy.ndarray` of shape `(k, 2)`, `dtype=float64`  
  Unit outward normal vectors for each boundary node in `boundary_nodes`.

---

### `compute_M_matrix`

```python
def compute_M_matrix(
    node_idx: int,
    support_nodes: npt.NDArray[np.int_],
    coords: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]
```

Constructs the unweighted Taylor spatial offset matrix $M$ for an interior node.

$$M = \begin{pmatrix} 1 & \dots & 1 \\ \Delta x_1 & \dots & \Delta x_q \\ \Delta y_1 & \dots & \Delta y_q \\ \Delta x_1^2 & \dots & \Delta x_q^2 \\ \Delta x_1 \Delta y_1 & \dots & \Delta x_q \Delta y_q \\ \Delta y_1^2 & \dots & \Delta y_q^2 \end{pmatrix}$$

##### Parameters
* **`node_idx`** : `int`  
  Index of central node $p_0$.
* **`support_nodes`** : `numpy.ndarray` of shape `(q,)`  
  Indices of support nodes in the star.
* **`coords`** : `numpy.ndarray` of shape `(n, 2)`  
  Node coordinate array.

##### Returns
* **`M`** : `numpy.ndarray` of shape `(6, q)`, `dtype=float64`  
  Taylor expansion offset matrix.

---

### `compute_M_matrix_neumann`

```python
def compute_M_matrix_neumann(
    node_idx: int,
    support_nodes: npt.NDArray[np.int_],
    coords: npt.NDArray[np.float64],
    normal_vec: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]
```

Constructs the augmented Taylor spatial offset matrix $M$ incorporating a fictitious ghost node positioned outside the boundary along the normal vector.

##### Parameters
* **`node_idx`** : `int`  
  Index of central boundary node $p_0$.
* **`support_nodes`** : `numpy.ndarray` of shape `(q,)`  
  Indices of interior support nodes.
* **`coords`** : `numpy.ndarray` of shape `(n, 2)`  
  Node coordinate array.
* **`normal_vec`** : `numpy.ndarray` of shape `(2,)`  
  Unit outward normal vector at node $p_0$.

##### Returns
* **`M`** : `numpy.ndarray` of shape `(6, q + 1)`, `dtype=float64`  
  Augmented Taylor matrix containing the ghost node offset in column 0.

---

## 4. Module: `GFDFlow.visualization`

Plotting routines built on top of Matplotlib for visualizing node meshes, vectors, phreatic surfaces, and 2D/3D solution fields.

### `plot_solution_2d`

```python
def plot_solution_2d(
    coords: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    levels: Union[int, npt.ArrayLike] = 20,
    cmap: str = "inferno",
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    contour_lines: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "2D Solution",
    savepath: Optional[str] = None,
    dpi: int = 300,
    **kwargs: Any
) -> Tuple[Figure, Axes]
```

Plots a 2D scalar field $u(x, y)$ using filled triangular contours (`tricontourf`).

##### Parameters
* **`coords`** : `numpy.ndarray` of shape `(n, 2)`  
  Node coordinates.
* **`u`** : `numpy.ndarray` of shape `(n,)`  
  Scalar solution vector.
* **`triangles`** : `numpy.ndarray` of shape `(m, 3)`, optional  
  Triangular connectivity. If `None`, Delaunay triangulation is calculated automatically.
* **`levels`** : `int` or array-like, default=`20`  
  Contour levels.
* **`cmap`** : `str`, default=`"inferno"`  
  Matplotlib colormap identifier.
* **`colorbar`** : `bool`, default=`True`  
  Whether to render a colorbar.
* **`colorbar_label`** : `str`, optional  
  Text label for the colorbar.
* **`contour_lines`** : `bool`, default=`True`  
  Whether to overlay discrete contour line boundaries.
* **`ax`** : `matplotlib.axes.Axes`, optional  
  Target Matplotlib axis. Created if `None`.
* **`figsize`** : `tuple[float, float]`, default=`(8, 6)`  
  Figure width and height in inches.
* **`title`** : `str`, optional  
  Plot title.
* **`savepath`** : `str`, optional  
  File path to export the plot image.
* **`dpi`** : `int`, default=`300`  
  Resolution for exported raster formats.

##### Returns
* **`fig`** : `matplotlib.figure.Figure`  
  Matplotlib figure object.
* **`ax`** : `matplotlib.axes.Axes`  
  Matplotlib axes object.

---

### `plot_solution_3d`

```python
def plot_solution_3d(
    coords: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 7),
    title: Optional[str] = "3D Solution",
    savepath: Optional[str] = None,
    dpi: int = 300,
    **kwargs: Any
) -> Tuple[Figure, Axes]
```

Renders a 3D elevation surface plot where node elevations represent the solution magnitude $z = u(x, y)$.

---

### `plot_solution_comparison_3d`

```python
def plot_solution_comparison_3d(
    coords: npt.NDArray[np.float64],
    u1: npt.NDArray[np.float64],
    u2: npt.NDArray[np.float64],
    triangles: Optional[npt.NDArray[np.int_]] = None,
    label1: str = "Numerical Solution",
    label2: str = "Analytical / Reference",
    figsize: Tuple[float, float] = (14, 6),
    savepath: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Figure, Tuple[Axes, Axes]]
```

Renders side-by-side 3D elevation plots comparing two solution fields (e.g., numerical vs analytical reference).

---

### `plot_geometry`

```python
def plot_geometry(
    segments: Iterable[npt.NDArray[np.float64]],
    labels: Optional[Iterable[str]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Domain Geometry",
    savepath: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Figure, Axes]
```

Plots domain boundary segments and geometric outlines.

---

### `plot_mesh`

```python
def plot_mesh(
    coords: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    show_nodes: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Computational Mesh",
    savepath: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Figure, Axes]
```

Renders the triangular mesh wireframe and optional node positions.

---

### `plot_nodes`

```python
def plot_nodes(
    coords: npt.NDArray[np.float64],
    node_groups: Optional[Dict[str, npt.NDArray[np.int_]]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Node Cloud Distribution",
    savepath: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Figure, Axes]
```

Visualizes the point cloud, highlighting distinct subsets of nodes (e.g., interior vs specific boundary groups).

---

### `plot_normal_vectors`

```python
def plot_normal_vectors(
    coords: npt.NDArray[np.float64],
    boundary_nodes: npt.NDArray[np.int_],
    normal_vectors: npt.NDArray[np.float64],
    scale: float = 0.05,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Boundary Normal Vectors",
    savepath: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Figure, Axes]
```

Plots boundary nodes with outward normal vectors rendered as directional arrows (quivers) to verify spatial orientation.

---

### `plot_phreatic_surface`

```python
def plot_phreatic_surface(
    coords: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    datum: float = 0.0,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = "Phreatic Surface (Water Table)",
    savepath: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Figure, Axes]
```

Visualizes free phreatic surfaces (e.g., unconfined groundwater tables) derived where potential matches elevation head.
