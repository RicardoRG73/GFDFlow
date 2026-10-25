# Decisions Log


### Spec: legacy-stencils-refactor
- **Date:** 2026-10-08
- **Decision:** Legacy examples 2-11 will have their GFDM stencils and matrices precomputed in mesh generation scripts.
- **Options considered:** Dynamic computation vs precomputation.
- **Why:** Recomputing stencils is slow and breaks architectural consistency with ex0 and ex1.

### Spec: paper-dam-5c
- **Date:** 2026-10-23
- **Decision:** Calibrate mesh geometry to match the exact coordinates of Chavez et al. (2023) Section 4.2.3 and precompute stencils and pseudo-inverses in `stat_mesh1.py`.
- **Options considered:** Reusing uncalibrated legacy mesh4 vs calibrated paper dimensions.
- **Why:** Legacy `mesh4.py` used placeholder dimensions (e.g. y=-5, x=0..55) whereas the paper specifies foundation depth y=-4, left toe x=3, right boundary x=60, and ~1503 nodes. Calibrating ensures exact reproduction of Figures 16 and 17.
