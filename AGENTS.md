# AGENTS.md — GFDFlow

> Guidance for AI coding agents working on this repository.

---

## 1. Project Overview

**GFDFlow** is a Python library that implements the **Generalized Finite Difference Method (GFDM)^[1]** for solving 2D Partial Differential Equations (PDEs) in domains with **multilayer materials and sharp material interfaces**.

The solver targets transport-phenomena problems (heat transfer, mass transport, seepage flow) governed by PDEs of the form:

```
A·u + B·u_x + C·u_y + D·u_xx + E·u_xy + F·u_yy = f(x, y)
```

Key capabilities:
- Arbitrary 2D point clouds / triangle meshes.
- Multiple materials with different permeabilities / conductivities.
- **Dirichlet** and **Neumann** boundary conditions.
- **Continuous and discontinuous interface conditions** (flux and potential jumps).
- **Interface intersections** (triple-point nodes where three material regions meet).
- Ghost-node technique for Neumann and interface stencils.

[1]: See article [**Numerical solution of density-driven groundwater flows using a generalized finite difference method defined by an unweighted least-squares problem**](https://doi.org/10.3389/fams.2022.976958) and [**GFDFlow: An object-oriented Python package for modeling two-dimensional transport phenomena**](https://doi.org/10.1016/j.simpa.2026.100849).

---

## 2. Repository Structure

```text
GFDFlow/
├── src/
│   └── GFDFlow/                # Installable Python package
│       ├── __init__.py         # Exports GFDMI_2D_problem
│       ├── GFDM.py             # Core class: GFDMI_2D_problem
│       └── utils.py            # Helpers: get_support_nodes, compute_normal_vectors, compute_M_matrix
├── tests/                      # pytest unit tests
│   └── test_gfdmi.py
├── examples/                   # different problem solutions for validation and testing
│   ├── basic_example.py        # Minimal working example (2D Laplacian)
│   └── legacy/                 # Adapted research examples (ex0 – ex11)
│       ├── meshes/             # JSON mesh files consumed by legacy examples
│       ├── figures/            # Output plots (not tracked by git)
│       ├── results/            # Numerical output files (not tracked by git)
│       └── ex*.py / ex5.ipynb # different problem solutions for validation and testing
├── docs/
│   ├── API_REFERENCE.md        # Public API documentation
│   ├── USER_MANUAL.md          # Step-by-step usage guide
│   └── workflow_diagram.png    # Visual overview of the solution workflow
├── pyproject.toml              # Build system (setuptools >= 80.10), Python >= 3.11
├── requirements.txt            # Runtime dependencies
├── verify_refactor.py          # Quick sanity-check script (no pytest dependency)
└── AGENTS.md                   # This file
```

---

## 3. Core Architecture

### 3.1 `GFDMI_2D_problem` (`src/GFDFlow/GFDM.py`)

The **single public class** of the library. All problem setup and assembly happen through its methods.

#### Constructor

```python
GFDMI_2D_problem(
    coords,           # (N, 2)  float64  — node coordinates
    triangles,        # (M, 3)  int      — triangle connectivity (Delaunay or structured)
    normal_vectors,   # (N, 2)  float64  — pre-computed unit normals for ALL nodes
                      #                    (interior nodes can hold zeros)
    L,                # (6,)    float64  — [A, B, C, D, E, F] PDE coefficients
    source,           # Callable(p) -> float  — source term f(x, y)
    support_stencils=None,  # Optional pre-computed {node_idx: neighbour indices array}
    M_pinv=None             # Optional pre-computed {node_idx: pseudo-inverse of M matrix}
)
```

> **Important:** `normal_vectors` must be a *full-size* `(N, 2)` array even though only boundary/interface nodes use it. Build the full array first (e.g. `np.zeros((N, 2))`) and then fill the relevant rows using `compute_normal_vectors`.

> **Important:** Inside `discretization_K_F`, `L[3]` and `L[5]` (coefficients D and F) are multiplied by 2 **in-place** on `self.L`. Always pass `L.copy()` to the constructor, or avoid calling `discretization_K_F` more than once on the same instance.

#### Problem-definition methods (call before assembly)

| Method | Purpose |
|---|---|
| `material(label, permeability, interior_nodes)` | Register a material region |
| `dirichlet_boundary(label, boundary_nodes, condition)` | Fixed-value BC |
| `neumann_boundary(label, permeability, boundary_nodes, condition)` | Flux BC using ghost-node technique |
| `interface(label, k_left, k_right, nodes_left, nodes_right, beta, alpha, interior_left, interior_right)` | Jump conditions between two materials |
| `intersection(label, intersection_node, interface_left, interface_right, material_between, f_intersection)` | Special treatment for triple-point nodes |

#### Assembly method

```python
K, F = problem.discretization_K_F(continuous=False)
```

- `continuous=False` — **discontinuous** interface: side-B nodes get the potential-jump row `u_B - u_A = alpha`.
- `continuous=True`  — **continuous** interface: flux contributions from both sides are summed into the same row.

Aliases for backwards compatibility:
- `problem.discontinuous_discretization()` → `discretization_K_F(continuous=False)`
- `problem.continuous_discretization()`    → `discretization_K_F(continuous=True)`

### 3.2 Assembly Order Inside `discretization_K_F`

1. **Interior nodes** — standard GFDM stencil, one row per node.
2. **Neumann boundaries** — ghost-node stencil; eliminates the ghost DOF analytically.
3. **Interface nodes** — ghost-node stencil on each side; coupling depends on `continuous` flag.
4. **Interface intersections** — special multi-interface stencil (continuous mode only).
5. **Dirichlet boundaries** — identity rows overwrite anything assembled earlier.

### 3.3 Utility Functions (`src/GFDFlow/utils.py`)

| Function | Signature | Purpose |
|---|---|---|
| `get_support_nodes` | `(node_idx, triangles, min_support_nodes=5, max_iter=2)` | BFS over triangle connectivity to collect a node's stencil |
| `compute_normal_vectors` | `(boundary_nodes, coords, line_tolerance=0.999)` | Outward unit normals; auto-detects straight vs curved boundaries |
| `compute_M_matrix` | `(node_idx, support_nodes, coords)` | Builds the `(6, n_support)` Taylor-expansion moment matrix `M` |

---

## 4. Data Flow / Typical Workflow

```
1. Build coords (N,2) + triangles (M,3)
         |
2. Identify boundary / interface node indices
         |
3. compute_normal_vectors -> fill full normal_vectors (N,2)
         |
4. Instantiate GFDMI_2D_problem
   (support_stencils and M_pinv computed automatically if not supplied)
         |
5. Register materials, boundaries, interfaces, intersections
         |
6. K, F = problem.discretization_K_F(continuous=...)
         |
7. U = scipy.sparse.linalg.spsolve(K, F)
         |
8. Post-process / visualise with matplotlib
```

Mesh files for legacy examples are stored as **JSON** in `examples/legacy/meshes/` with keys:
`coords`, `triangles`, `normal_vecs`, and node-index arrays for each boundary/material region.

---

## 5. Dependencies

| Package | Min version | Role |
|---|---|---|
| `numpy` | 2.4 | Array math, linear algebra |
| `scipy` | 1.17 | Sparse matrices (`lil_matrix`, `csr_matrix`, `spsolve`), Delaunay triangulation |
| `matplotlib` | 3.10 | Plotting in examples |
| `calfem-python` | latest | Geometry and mesh generation in legacy examples only |

Install in editable mode from the repo root:

```bash
pip install -e .
```

---

## 6. Testing

```bash
# Run the full test suite
pytest tests/

# Quick sanity check without pytest
python verify_refactor.py
```

Tests in `tests/test_gfdmi.py` cover:

- `test_gfdmi_initialization` — constructor validates array shapes.
- `test_support_nodes` — stencil has at least 5 nodes and includes the central node.
- `test_simple_laplacian` — assembles K, F for a 5x5 Laplacian problem; checks shapes.

> **Note:** The test file calls `problem.support_nodes(4)` which is outdated. The correct access is `problem.support_stencils[4]`. Fix this before running a full CI pipeline.

---

## 7. Code Conventions

### Language & Style
- **Python >= 3.11** throughout; use type hints everywhere.
- Type annotations use `numpy.typing` (`npt.NDArray[np.float64]`, `npt.NDArray[np.int_]`).
- `Callable[[npt.NDArray[np.float64]], float]` is the standard signature for permeability and BC functions.
- Private/internal methods are prefixed with a single underscore (e.g., `_assemble_point_discretization`).

### Naming
- Node-index arrays: `*_nodes` suffix (e.g., `left_nodes`, `interface_a_nodes`).
- Permeability callables: `k_*` prefix or `k_fn` (e.g., `k_left`, `kdam`).
- PDE coefficient vector: always `L` of shape `(6,)` with elements `[A, B, C, D, E, F]`.
- Stiffness matrix / force vector: always `K`, `F`.

### Docstrings
- NumPy-style docstrings with `Parameters`, `Returns`, and `Raises` sections.
- All public methods and standalone utility functions must have docstrings.

### Sparse Matrix Pattern
- Assemble into `scipy.sparse.lil_matrix` (efficient for element-wise insertion).
- Convert to `csr_matrix` only at the end via `.tocsr()` before returning.

---

## 8. Known Issues / Gotchas

1. **In-place mutation of `L`:** `discretization_K_F` does `L[3] *= 2; L[5] *= 2` on `self.L`. Calling `discretization_K_F` twice on the same instance will corrupt the coefficients. Workaround: pass `L.copy()` to the constructor.

2. **`normal_vectors` full-size requirement:** Must have `shape == (N, 2)` matching `coords`. A common mistake is passing only the boundary-node subset.

3. **Interface `nodes_right` / `alpha` may be `None` in legacy examples:** The discontinuous assembler accesses these by positional index from the stored list, so `None` is valid for unused arguments but will crash if iterated over inadvertently.

4. **Outdated test API:** `problem.support_nodes(i)` should be `problem.support_stencils[i]`.

5. **`calfem-python` is only used in legacy examples:** The core library (`GFDM.py`, `utils.py`) does not import it. Do not add it as a hard core dependency.

6. **Ghost-node logic is duplicated:** The same ghost-point stencil construction appears in the Neumann block, the discontinuous-interface block (side A and side B), and the continuous-interface block. This is the main technical debt.

---

## 9. Agent Guidelines

### Environment
- Check the conda environment. Current is `gfdfm`.
    - `conda activate gfdfm`
- Check `requirements.txt` for dependencies.
- Check if `gfdflow` is instaled in the current conda environment.
    - If not, install it using `pip install -e .`.

### Before Making Changes
- Read `src/GFDFlow/GFDM.py` and `src/GFDFlow/utils.py` in full.
- Check `tests/test_gfdmi.py` to understand expected behaviour.
- Consult `docs/API_REFERENCE.md` for the public API contract.

### When Adding Features
- New numerical methods or operators belong in `GFDM.py` (class methods) or `utils.py` (standalone helpers).
- Preserve existing method signatures for backwards compatibility; use optional keyword arguments for new parameters.
- Add a corresponding test in `tests/test_gfdmi.py`.
- Update `docs/API_REFERENCE.md` with the new method/function signature and description.

### When Refactoring
- The **ghost-node technique** is replicated in multiple assembly blocks. The highest-impact refactor is extracting it into a private helper, e.g.:
  ```python
  def _ghost_stencil(self, i, normal_vec, support_nodes, k_val, L): ...
  ```
- The **in-place mutation of `L`** is a bug. Fix with `L = self.L.copy()` at the start of `discretization_K_F`.

### When Adding Examples
- Adapted research scripts go into `examples/legacy/`.
- Mesh data should be JSON files in `examples/legacy/meshes/` following the existing key schema.
- Results should be stored in `examples/legacy/results/`. DO NOT add these files to the git repository.
- Figures should be saved as png files using `plt.savefig('example_name.png')` and should be stored in `examples/legacy/figures/`. DO NOT add these files to the git repository.
- Use `from GFDFlow.GFDM import GFDMI_2D_problem` (not `sys.path` hacks) once the package is installed.

### Do NOT
- Add heavy dependencies (e.g., FEniCS, PETSc) to the core package.
- Change the `L = [A, B, C, D, E, F]` convention — it is used throughout examples and docs.
- Remove the `discontinuous_discretization` / `continuous_discretization` alias methods (backwards compatibility).
- Commit generated output files to `examples/legacy/figures/` or `examples/legacy/results/`.
- Call `discretization_K_F` more than once on the same problem instance without re-instantiating (due to the `L` mutation bug).
