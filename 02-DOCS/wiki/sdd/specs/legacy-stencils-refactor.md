# Spec — Legacy Examples Stencils Refactor

## Problem & why
In `examples/legacy/` from `ex2.py` through `ex11.py`, the GFDM support stencils (`support_stencils`) and the pseudo-inverse matrices (`M_pinv`) are computed inside the solver scripts during the instantiation of the `GFDMI_2D_problem`. This computation is expensive and violates the separation of concerns established in `ex0.py` and `ex1.py`, where geometry pre-processing happens in `meshX.py` and the outputs are saved to a JSON file to be loaded quickly by the solver.

## Cost of not building it
Running solver scripts is artificially slow. The codebase remains inconsistent, forcing new users to parse different architectural patterns across examples and making future upgrades to the GFDM core harder.

## The cheapest alternative
Leave the examples as they are. This solves nothing and leaves technical debt in the repository.

## Goals
- Standardize all legacy examples (ex2 to ex11) to load `support_stencils` and `M_pinv` from their precomputed JSON files.
- Move the computation of these geometric properties to the respective `meshX.py` scripts.

## Non-goals / out of scope
- Refactoring the internal workings of the `GFDMI_2D_problem` solver itself.
- Deleting or changing the mathematical setup of the legacy examples.

## Users & context
Researchers and developers running the legacy examples to test or learn about GFDFlow's capabilities.

## Behaviour
- **Main**: A user runs `python examples/legacy/meshes/mesh2.py`. The script parses the mesh, computes normal vectors, computes `support_stencils` and `M_pinv` for all boundaries, and saves the full data payload to `mesh2.json`. The user then runs `python examples/legacy/ex2.py`. The script instantly loads the JSON file and initializes the `gfdm` problem without recomputing the matrices.
- **Edge**: For examples like 8, 9, and 10 that currently lack a separate `meshX.py` file, a new script is created to handle mesh generation and JSON serialization.

## Acceptance criteria
- Given any legacy solver script `ex2.py` through `ex11.py`, When executed, Then it loads `support_stencils` and `M_pinv` from its JSON mesh file instead of computing them.
- Given any legacy mesh script (or newly split mesh script for ex8, ex9, ex10), When executed, Then it correctly computes and writes `support_stencils` and `M_pinv` to the JSON output file.
- Given the execution of the refactored solver scripts, When run, Then they complete successfully without errors.

## Points to clarify
- **assumption made** — The implementation of `M_pinv` for internal nodes vs Neumann boundary nodes follows the exact logic shown in `mesh0.py`. *Base:* `mesh0.py` sets the pattern for computing these matrices.
- **assumption made** — For `ex8.py`, `ex9 multilayer.py`, and `ex10.py` which define meshes inline using `calfem`, we will split out the mesh generation logic into `mesh8.py`, `mesh9.py`, and `mesh10.py` to match the pattern. *Base:* Consistency across all 12 examples.
