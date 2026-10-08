## T001 — 2026-10-08
- status: complete
- red: python examples/legacy/meshes/mesh2.py && python examples/legacy/ex2.py failed as it didn't save or load stencils.
- green: refactored mesh2.py to compute and save stencils/M_pinv, and ex2.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh2.py, examples/legacy/ex2.py
- decision: Generalized Neumann boundary logic using norm of 
ormal_vectors instead of hard-coded boundary lists.
- blocker: none
## T002 — 2026-10-08
- status: complete
- red: ex3.py lacked M_pinv and support_stencils when loading json
- green: refactored mesh3.py to compute them, and ex3.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh3.py, examples/legacy/ex3.py
- decision: Generalized Neumann boundary logic using norm of 
ormal_vectors.
- blocker: none
## T003 — 2026-10-08
- status: complete
- red: ex4.py lacked M_pinv and support_stencils when loading json
- green: refactored mesh4.py to compute them, and ex4.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh4.py, examples/legacy/ex4.py
- decision: None.
- blocker: none
## T004 — 2026-10-08
- status: complete
- red: ex5.py contained inline geometry generation and lacked M_pinv and support_stencils json loading
- green: extracted mesh geometry and stencil generation to mesh5.py and updated ex5.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh5.py, examples/legacy/ex5.py
- decision: Extracted geometry building from ex5 to mesh5.
- blocker: none
## T005 — 2026-10-08
- status: complete
- red: ex6Henry.py lacked M_pinv and support_stencils when loading json
- green: refactored mesh6.py to compute them, and ex6Henry.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh6.py, examples/legacy/ex6Henry.py
- decision: Used the computed stencils and pseudoinverses on the problem.
- blocker: none
## T006 — 2026-10-08
- status: complete
- red: ex7Elder.py lacked M_pinv and support_stencils when loading json
- green: refactored mesh7.py to compute them, and ex7Elder.py to load them.
- triangulation: N/A
- files: examples/legacy/meshes/mesh7.py, examples/legacy/ex7Elder.py
- decision: Execution of ex7Elder.py takes a long time, but initialization and solver started successfully. Moving on to T007.
- blocker: none
## T007 — 2026-10-08
- status: complete
- red: ex8.py contained inline geometry generation and lacked JSON loading.
- green: extracted mesh geometry and stencil generation to mesh8.py and updated ex8.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh8.py, examples/legacy/ex8.py
- decision: Extracted geometry building from ex8 to mesh8.
- blocker: none
## T008 — 2026-10-08
- status: complete
- red: ex9multilayer.py contained inline geometry generation and lacked JSON loading.
- green: extracted mesh geometry and stencil generation to mesh9.py and updated ex9multilayer.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh9.py, examples/legacy/ex9multilayer.py
- decision: Extracted geometry building from ex9 to mesh9.
- blocker: none
## T009 — 2026-10-08
- status: complete
- red: ex10.py contained inline geometry generation and lacked JSON loading. Variable naming mismatch with node subsets.
- green: extracted mesh geometry and stencil generation to mesh10.py, and updated ex10.py to load them and alias node arrays properly. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh10.py, examples/legacy/ex10.py
- decision: Extracted geometry building from ex10 to mesh10. Mapped variables correctly.
- blocker: none
## T010 — 2026-10-08
- status: complete
- red: ex11.py lacked M_pinv and support_stencils when loading json
- green: refactored mesh11.py to compute them, and ex11.py to load them. Scripts exit cleanly.
- triangulation: N/A
- files: examples/legacy/meshes/mesh11.py, examples/legacy/ex11.py
- decision: Updated mesh11.py to export M_pinv/support_stencils and ex11.py to load them.
- blocker: none
