---
author: Ricardo
branch: rsc
status: completed
---
# Neumann Support Stencils and M_pinv Computation in GFDM

## Intent
Correct the initialization of `support_stencils` and `M_pinv` in `GFDM.py` (`GFDMI_2D_problem.__init__`) so that nodes lying on Neumann boundaries (identified by non-zero normal vectors `np.linalg.norm(normal_vectors[i]) > 1e-10`) compute `M_pinv` using `compute_M_matrix_neumann`, consistent with mesh precomputations in `mesh2.py` and `mesh5.py`.

## Scope
- In scope:
  - Import `compute_M_matrix_neumann` from `.utils` in `src/GFDFlow/GFDM.py`.
  - Update `M_pinv` precomputation logic in `GFDMI_2D_problem.__init__` to branch on `np.linalg.norm(self.normal_vectors[i]) > 1e-10`.
  - Update docstring in `GFDM.py` to reflect `normal_vectors`, `support_stencils`, and `M_pinv` parameters.
  - Add test coverage in `tests/test_gfdmi.py` verifying that Neumann nodes receive ghost-node augmented `M_pinv` stencils while other nodes use standard `compute_M_matrix`.
  - Ensure compatibility with existing legacy examples (`ex0.py` - `ex11.py`).
- Out of scope:
  - Modifying `utils.py` implementation of `compute_M_matrix_neumann`.
  - Changing JSON mesh files or precomputed datasets.

## Checklist
- [x] Create FTD feature document
- [x] Update imports in `src/GFDFlow/GFDM.py` to include `compute_M_matrix_neumann`
- [x] Update `self.M_pinv` initialization in `src/GFDFlow/GFDM.py` to handle Neumann boundaries with `compute_M_matrix_neumann`
- [x] Update `GFDMI_2D_problem` docstring
- [x] Add unit test verifying Neumann `M_pinv` dimension and values
- [x] Run test suite and legacy example checks
- [x] Mark FTD document completed

## Evidence
- `compute_M_matrix_neumann` imported from `.utils` in `src/GFDFlow/GFDM.py`.
- `GFDMI_2D_problem.__init__` iterates through `range(self.coords.shape[0])`:
  - When `np.linalg.norm(self.normal_vectors[i]) > 1e-10`, `compute_M_matrix_neumann(i, self.support_stencils[i], self.coords, self.normal_vectors[i])` is used to assemble `M`, yielding shape `(len(stencil) + 1, 6)` for `self.M_pinv[i]`.
  - Otherwise, `compute_M_matrix(i, self.support_stencils[i], self.coords)` is used, yielding shape `(len(stencil), 6)`.
- `test_neumann_stencils_and_m_pinv` added to `tests/test_gfdmi.py` verifying exact shape and numerical values via `np.testing.assert_allclose`.
- `pytest tests/test_gfdmi.py` ran with 4 passed tests (exit code 0).
- `pytest tests/test_visualization.py` ran with 9 passed tests (exit code 0).

## Next
All tasks complete.
