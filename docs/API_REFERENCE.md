# API Reference - GFDFlow

The `GFDFlow` package provides classes and utilities for solving 2D interface problems using the Generalized Finite Difference Method (GFDM).

## Core Classes

### `GFDMI_2D_problem`

This is the main class used to define and solve 2D PDEs with material interfaces.

#### `__init__(coords, triangles, L, source)`
Initializes the problem.
- **`coords`**: `npt.NDArray[np.float64]` (n, 2) node coordinates.
- **`triangles`**: `npt.NDArray[np.int_]` (m, 3) connectivity.
- **`L`**: `npt.NDArray[np.float64]` (6,) coefficients `[A, B, C, D, E, F]`.
- **`source`**: `Callable[[npt.NDArray[np.float64]], float]` source function `f(x, y)`.

#### `material(label, permeability, interior_nodes)`
Defines material properties for a set of nodes.
- **`label`**: `str` identifier.
- **`permeability`**: `Callable` returning the material's permeability at a point.
- **`interior_nodes`**: `npt.NDArray[np.int_]` indices of nodes with this material.

#### `neumann_boundary(label, permeability, boundary_nodes, condition)`
Defines Neumann boundary conditions (`du/dn = g`).
- **`label`**: `str` identifier.
- **`permeability`**: `Callable` for the material at the boundary.
- **`boundary_nodes`**: `npt.NDArray[np.int_]` indices of boundary nodes.
- **`condition`**: `Callable` returning the flux value at a point.

#### `dirichlet_boundary(label, boundary_nodes, condition)`
Defines Dirichlet boundary conditions (`u = h`).
- **`label`**: `str` identifier.
- **`boundary_nodes`**: `npt.NDArray[np.int_]` indices of boundary nodes.
- **`condition`**: `Callable` returning the fixed value at a point.

#### `interface(label, k_left, k_right, nodes_left, nodes_right, beta, alpha, interior_left, interior_right)`
Defines interface conditions between two materials.
- **`label`**: `str` identifier.
- **`k_left`, `k_right`**: `Callable` permeabilities for both sides.
- **`nodes_left`, `nodes_right`**: `npt.NDArray[np.int_]` nodes along the interface from both sides.
- **`beta`**: `Callable` flux jump condition.
- **`alpha`**: `Callable` potential jump condition.
- **`interior_left`, `interior_right`**: `npt.NDArray[np.int_]` nodes in the respective materials.

#### `discretization_K_F(continuous=False)`
Assembles the global stiffness matrix `K` and force vector `F`.
- **`continuous`**: `bool` (default `False`). If `True`, assumes a continuous interface.
- **Returns**: `Tuple[sp.csr_matrix, npt.NDArray[np.float64]]` (K, F).

---

## Utility Functions

### `get_support_nodes(node_idx, triangles, min_support_nodes=5, max_iter=2)`
Finds the support nodes for a given central node using mesh connectivity.

### `compute_normal_vectors(boundary_nodes, coords)`
Computes outward-pointing normal vectors for a set of boundary nodes.

### `compute_M_matrix(node_idx, support_nodes, coords)`
Builds the `(6, n_support)` Taylor-expansion moment matrix `M`.

### `compute_M_matrix(node_idx, support_nodes, coords, normal_vec)`


---

## Visualization Functions (`GFDFlow.visualization`)

All functions are also exported from the top-level `GFDFlow` namespace.

### `plot_solution_2d(coords, u, triangles=None, levels=20, cmap="plasma", colorbar=True, colorbar_label=None, contour_lines=True, line_colors="k", linewidths=0.5, line_alpha=0.6, clabel=False, overlay_nodes=None, ax=None, figsize=(8, 6), title=None, xlabel="x", ylabel="y", equal_aspect=True, savepath=None, **kwargs)`
Plots a filled 2D contour map (`tricontourf`) of a scalar field over unstructured node coordinates.

- **`coords`**: `(N, 2)` array of node coordinates.
- **`u`**: `(N,)` array of solution values.
- **`triangles`**: Optional `(M, 3)` connectivity matrix.
- **`levels`**: Number or sequence of contour levels.
- **`cmap`**: Colormap name.
- **`colorbar`**, **`colorbar_label`**: Toggle and label the colorbar.
- **`contour_lines`**: Draw isolines over the fill.
- **`line_colors`**, **`linewidths`**, **`line_alpha`**: Isoline style.
- **`clabel`**: Label the isolines numerically.
- **`overlay_nodes`**: `dict {label: node_indices}`, `[(label, node_indices)]`, or plain array — highlights node groups over the contour.
- **`ax`**: Existing `Axes` to draw on.
- **`figsize`**, **`title`**, **`xlabel`**, **`ylabel`**, **`equal_aspect`**: Layout options.
- **`savepath`**: File path to auto-save the figure (creates directories if needed).
- **Returns**: `(fig, ax)`.

---

### `plot_solution_3d(coords, u, triangles=None, cmap="plasma", colorbar=True, colorbar_label=None, view_init=(30, -120), edge_color=None, alpha=1.0, ax=None, figsize=(8, 6), title=None, xlabel="x", ylabel="y", zlabel="U", savepath=None, **kwargs)`
Plots a 3D surface (`plot_trisurf`) of a scalar field.

- **`view_init`**: `(elev, azim)` camera angles.
- **`edge_color`**: Color for triangle edges on the surface.
- **`alpha`**: Surface opacity.
- All other parameters as in `plot_solution_2d`.
- **Returns**: `(fig, ax)`.

---

### `plot_solution_comparison_3d(coords, u_num, u_exact, triangles=None, num_label="Numerical", exact_label="Exact", num_color="r", exact_color="b", alpha=0.5, view_init=(20, -50), ax=None, figsize=(8, 6), title=None, xlabel="x", ylabel="y", zlabel="U", savepath=None, **kwargs)`
Overlays two 3D surfaces (numerical and exact) for visual comparison.

- **`u_num`**: Numerical solution field `(N,)`.
- **`u_exact`**: Exact/analytical solution field `(N,)`.
- **`num_color`**, **`exact_color`**: Surface colors for each solution.
- **Returns**: `(fig, ax)`.

---

### `plot_nodes(coords, node_groups, point_size=20.0, alpha=0.7, legend=True, legend_bbox=None, ax=None, figsize=(8, 6), title="Nodes Distribution", xlabel="x", ylabel="y", equal_aspect=True, savepath=None, **kwargs)`
Scatter-plots classified node groups (materials, boundaries, interfaces) on 2D coordinates.

- **`node_groups`**: `dict {label: node_indices}` or `[(label, node_indices)]`.
- **`legend_bbox`**: Bounding box anchor for legend, e.g. `(1.05, 1)`.
- **Returns**: `(fig, ax)`.

---

### `plot_normal_vectors(coords, normal_vecs, boundary_nodes=None, scatter=True, point_size=15.0, quiver_color="k", quiver_alpha=0.5, ax=None, figsize=(8, 6), title="Normal Vectors", xlabel="x", ylabel="y", equal_aspect=True, savepath=None, **kwargs)`
Plots outward unit normal vectors as quiver arrows at boundary or interface nodes.

- **`normal_vecs`**: Full-size `(N, 2)` normal vector array (zeros at interior nodes).
- **`boundary_nodes`**: Indices to draw; if `None`, all non-zero normals are plotted. Accepts a single array or a list of arrays.
- **`scatter`**: Whether to mark the vector origins with scatter dots.
- **Returns**: `(fig, ax)`.

---

### `plot_phreatic_surface(ax, coords, u, triangles=None, density_g=9.81, level=0.0, color="b", linewidths=2.0, label="Phreatic surface")`
Superimposes the phreatic (free-water) surface on an existing 2D plot by drawing the zero-pressure contour `(u - y) * g = level`.

- **`ax`**: Existing `Axes` to draw on (returned by `plot_solution_2d`).
- **`density_g`**: Unit weight of water `γ = ρ·g` (default 9.81).
- **`level`**: Pressure head contour level (default 0.0, i.e. the phreatic surface).
- Returns `None`; modifies `ax` in place.
