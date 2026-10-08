# Decisions Log


### Spec: legacy-stencils-refactor
- **Date:** 2026-10-08
- **Decision:** Legacy examples 2-11 will have their GFDM stencils and matrices precomputed in mesh generation scripts.
- **Options considered:** Dynamic computation vs precomputation.
- **Why:** Recomputing stencils is slow and breaks architectural consistency with ex0 and ex1.
