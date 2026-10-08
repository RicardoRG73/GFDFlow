# Spec — Comprehensive Documentation for the GFDFlow Project

## Problem & why

The repository's `README.md` explicitly references `docs/USER_MANUAL.md` and `docs/API_REFERENCE.md`, but the `docs/` directory does not exist in the repository. Furthermore, the project implements advanced GFDM (Generalized Finite Difference Method) mathematical algorithms featuring material interfaces and partial differential equation (PDE) solving on scattered node clouds, yet lacks a structured user guide and unified reference documentation for developers and researchers.

Theres a need of GFDM theory into `docs/` directory.

## Cost of not building it

New users and researchers cannot understand or adopt the library without manually reading and interpreting the source code of legacy examples. The absence of the referenced files in the `README` results in broken links and projects a lack of project maturity.

## The cheapest alternative

Create only minimal skeletons or empty stubs in `docs/`. This fixes the broken links, but provides no value to users and fails to document the library's behavior.

## Goals

* Create `docs/USER_MANUAL.md` featuring an accessible mathematical overview of GFDM, a step-by-step installation guide, problem setup, boundary condition definition (Dirichlet and Neumann), material interfaces, assembly, sparse system solving, and visualization.
* Create `docs/API_REFERENCE.md` with exhaustive technical documentation for the public API (`GFDMI_2D_problem`, support utilities and normal vector calculations in `utils.py`, and plotting functions in `visualization.py`), following NumPy docstring standards.
* Create a guide for practical examples and mesh/node cloud workflows (`examples/`).
* Update and verify `README.md` and the knowledge map in `02-DOCS/wiki/index.md` to properly link all documentation.

## Non-goals / out of scope

* Modifying the mathematical logic or solver source code in `src/GFDFlow/`.
* Generating static HTML websites with Sphinx or MkDocs during this phase (retaining pure standard Markdown in the repository).

## Users & context

Researchers in computational physics, transport phenomena modeling, fluid mechanics, and scientific developers implementing GFDM in Python.

## Behaviour

* **Main**: The user or researcher opens `README.md` and navigates to `docs/USER_MANUAL.md` to follow a quickstart tutorial and understand the mathematical foundations of GFDM. They consult `docs/API_REFERENCE.md` to review types, parameters, return values, and exceptions for each class and method.
* **Edge**: Users working with complex clouds/meshes (featuring multiple materials or jump conditions) find documented examples and clear guidelines on structuring their input data.

## Acceptance criteria

* Given the `docs/` directory, both `docs/USER_MANUAL.md` and `docs/API_REFERENCE.md` are created and populated with detailed content.
* Given the API in `src/GFDFlow/`, all core classes (`GFDMI_2D_problem`), support and normal vector functions (`utils.py`), and visualization utilities (`visualization.py`) have parameter descriptions, types, and usage examples in `docs/API_REFERENCE.md`.
* Given the `README.md` file, all internal documentation links are valid and resolve to existing files.
* Given the project map in `02-DOCS/wiki/index.md`, the new specification and documentation structure are fully reflected.

## Points to clarify
- **Scope confirmation**: Written in English with standard technical register conforming to scientific publications and NumPy docstring conventions.
- **Theory inclusion**: Incorporated dedicated `docs/theory.md` covering GFDM mathematical foundations and boundary formulations.