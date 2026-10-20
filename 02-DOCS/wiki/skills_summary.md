# Available Skills & Harness Capabilities in GFDFlow

This document provides a comprehensive summary of all skills, specialized agents, and automated commands configured in the GFDFlow repository via the RSC (Risk, Strategy, Craft) harness.

---

## 1. Executive Summary

The GFDFlow repository incorporates **30 active skills**, **6 specialized evaluation agents**, and **17 harness commands**. These tools govern development workflows, architectural design, mathematical modeling standards, code verification, and documentation integrity.

Workflows operate under two primary execution paths:
1. **Spec-Driven Development (SDD)**: A rigorous 10-phase pipeline designed for multi-component features, architectural shifts, and complex mathematical modeling in the Generalized Finite Difference Method (GFDM).
2. **Fast-Track Development (FTD)**: A streamlined single-document workflow (`02-DOCS/wiki/ftd/<slug>.md`) designed for bounded, ordinary tasks and documentation updates.

---

## 2. Master Skills Catalog

| Skill Name | Primary Category | Purpose & Description | Input / Output Artifacts |
| :--- | :--- | :--- | :--- |
| [`analyze`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/analyze/SKILL.md) | SDD Gate | Cross-checks constitution, spec, plan, and tasks against each other to identify gaps, contradictions, and scope creep before coding begins. | Reads SDD artifacts; emits gap analysis report |
| [`automation-strategy`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/automation-strategy/SKILL.md) | Strategy & Ops | Evaluates automation viability, calculates ROI, guides build-vs-buy choices, and specifies platform resilience patterns. | Decision brief & architecture guidelines |
| [`clarify`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/clarify/SKILL.md) | SDD Phase | Hunts ambiguities, unstated assumptions, and edge cases in draft specifications; updates the specification with concrete answers. | `02-DOCS/wiki/sdd/specs/*.md` |
| [`constitution`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/constitution/SKILL.md) | SDD Phase | Establishes and amends repository non-negotiables, code quality bars, architectural conventions, and safety standards. | `02-DOCS/wiki/sdd/constitution.md` |
| [`debug`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/debug/SKILL.md) | Quality & Triage | Systematic root-cause isolation. Reproduces bugs using minimal failing tests before attempting code modification. | Failing tests, root-cause proof, minimal diff |
| [`decision-challenge`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/decision-challenge/SKILL.md) | Governance | Conducts bounded adversarial reviews of high-impact architectural choices or migrations to evaluate risk before commitment. | Decision assessment: proceed / hold / stop |
| [`deprecation`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/deprecation/SKILL.md) | Maintenance | Manages safe sunsetting of public APIs or legacy interfaces with usage auditing, migration windows, and compatibility tooling. | Deprecation schedule & migration guides |
| [`fable-operator`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/fable-operator/SKILL.md) | Epistemics & Rigor | Enforces calibrated reasoning and formal verification when addressing irreversible or mission-critical problems. | Grounded analytical briefs & verified audits |
| [`ftd`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/ftd/SKILL.md) | Workflow Lane | Fast-Track Development for ordinary tasks. Tracks intent, scope, verified checklists, and proof in a single document. | `02-DOCS/wiki/ftd/*.md` |
| [`harness`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/harness/SKILL.md) | Repository Ops | Governs the repository control plane, including `01-TOOLS/`, `02-DOCS/` wiki knowledge models, and assistant configurations. | `02-DOCS/wiki/*`, `.gemini/*` |
| [`idea-refinement`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/idea-refinement/SKILL.md) | Ideation | Explores divergent options and challenges assumptions before formulating a formal specification. | Decision brief with cheap testable experiments |
| [`implement`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/implement/SKILL.md) | SDD Phase | Translates approved task lists into production code using strict Test-Driven Development (TDD). | Source code, unit tests, progress ledgers |
| [`init`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/init/SKILL.md) | Onboarding | Repository bootstrap front door. Calibrates user communication profiles and provisions essential harness skills. | `02-DOCS/wiki/harness/user-profile.md` |
| [`orient`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/orient/SKILL.md) | Always-On Protocol | Governs tone, communication clarity (Simplified Technical English), explanation depth, and turn-ending compass blocks. | Conversational output & compass blocks |
| [`parallel`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/parallel/SKILL.md) | Concurrency | Analyzes task graphs to identify disjoint subtasks and dispatches them across concurrent subagents safely. | Subagent dispatch plans & reconciliation suites |
| [`plan`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/plan/SKILL.md) | SDD Phase | Converts clarified specifications into technical blueprints, interface contracts, data models, and verification plans. | `02-DOCS/wiki/sdd/plans/*.md` |
| [`python`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/python/SKILL.md) | Language Canon | Enforces Python standards: strict `mypy` typing, PEP 695 generics, `dataclass`/`Protocol` design, `uv` packaging, and `pytest`. | Python source code, type annotations, tests |
| [`review`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/review/SKILL.md) | SDD Gate | Conducts adversarial code reviews against the specification, plan, and constitution before code lands on main. | Review findings & resolution verdicts |
| [`sdd`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/sdd/SKILL.md) | Workflow Engine | Master coordinator for Spec-Driven Development. Routes feature work through all ten SDD phases and on-demand gates. | Living knowledge base & SDD records |
| [`sdd-init`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/sdd-init/SKILL.md) | Stack Detection | Analyzes build tools, test runners, and package managers to initialize or update the repository SDD configuration. | `02-DOCS/wiki/sdd/config.yaml` |
| [`ship`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/ship/SKILL.md) | Release & Land | Final SDD phase. Performs pre-landing safety checks, merges branches, generates pull requests, and maintains author attribution. | Git commits, PRs, merge records |
| [`simplify-code`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/simplify-code/SKILL.md) | Refactoring | Reduces accidental complexity and improves readability while strictly preserving observable behavior and test coverage. | Refactored code with regression test proof |
| [`source-grounded-development`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/source-grounded-development/SKILL.md) | Verification | Validates technical implementations against exact installed library versions and official primary documentation. | Verified API contracts & citations |
| [`specify`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/specify/SKILL.md) | SDD Phase | Brainstorms and defines WHAT to build and WHY, producing clear user stories, scope boundaries, and acceptance criteria. | `02-DOCS/wiki/sdd/specs/*.md` |
| [`suggest`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/suggest/SKILL.md) | Always-On Protocol | Monitors task requirements, identifies uninstalled capabilities, and offers one-click installation from the RSC catalog. | `.rsc.json` package additions |
| [`tasks`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/tasks/SKILL.md) | SDD Phase | Breaks technical plans into ordered, granular implementation tasks with verifiable acceptance checks and parallel tags. | `02-DOCS/wiki/sdd/plans/*.md` (Task ledger) |
| [`ui-engineering`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/ui-engineering/SKILL.md) | Interface Design | Guides UI architecture, state allocation (URL vs. local component), optimistic updates, and component boundaries. | Frontend architecture & component designs |
| [`unslop`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/unslop/SKILL.md) | Editing & Tone | Polishes user-facing text, documentation, and communications by removing AI clichés, filler words, and unnatural phrasing. | Clear, publication-grade documentation |
| [`verify`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/verify/SKILL.md) | SDD Gate | Runs automated quality gates (`verify.sh`, linters, test suites) and records objective proof against acceptance criteria. | Test execution logs & verification records |
| [`worktrees`](file:///c:/Users/rick_/Documents/repos/GFDFlow/.rsc/skills/worktrees/SKILL.md) | Git Isolation | Manages git worktrees and isolated branches to ensure implementation work never runs directly on dirty or trunk branches. | Dedicated working trees & feature branches |

---

## 3. Detailed Skills Breakdown by Domain

### 3.1 Spec-Driven Development (SDD) Pipeline

The SDD chain ensures that complex scientific algorithms—such as GFDM star calculations, normal vector approximations, and sparse system assembly—are fully analyzed, specified, and tested before being integrated.

```mermaid
flowchart LR
    A[constitution] --> B[specify]
    B --> C[clarify]
    C --> D[plan]
    D --> E[tasks]
    E --> F[analyze]
    F --> G[implement]
    G --> H[verify]
    H --> I[review]
    I --> J[ship]
```

1. **`constitution`**: Defines foundational architectural principles, such as pure NumPy/SciPy scientific computing standards, test coverage thresholds, and numerical tolerance requirements.
2. **`specify`**: Drafts high-level feature specifications in `02-DOCS/wiki/sdd/specs/`. Focuses strictly on functional requirements and numerical acceptance criteria without premature coding.
3. **`clarify`**: Probes the specification for edge cases (e.g., singular matrices, irregular boundary stencils, non-convex domains) and embeds resolutions into the specification.
4. **`plan`**: Formulates architectural blueprints in `02-DOCS/wiki/sdd/plans/`, detailing data structures, matrix assembly strategies, and public API signatures.
5. **`tasks`**: Decomposes the plan into sequential, independently verifiable units of work with explicit criteria for completion.
6. **`analyze`**: Reviews the constitution, spec, plan, and task list together to catch contradictions, gaps, or drift prior to implementation.
7. **`implement`**: Executes the task list using Test-Driven Development (TDD). Every algorithm must have a failing test before solver implementation.
8. **`verify`**: Executes full test suites (`pytest tests/`) and checks numerical accuracy against benchmarks.
9. **`review`**: Performs adversarial scrutiny of code diffs against the initial specification and design constraints.
10. **`ship`**: Merges verified code into the working branch, updates documentation indexes, and maintains clean git history.

### 3.2 Fast-Track Development (FTD)

For routine updates, bug fixes, or documentation tasks that do not alter core solver architecture, GFDFlow uses **FTD**.
- Feature documents are stored in `02-DOCS/wiki/ftd/<slug>.md`.
- Documents outline intent, scope, checklist, and evidence.
- Every task checkbox must be substantiated by recorded command output or test runs.

### 3.3 Scientific & Python Engineering Canon

GFDFlow implements specialized scientific computing practices through the **`python`** skill:
- **Typing**: Strict `mypy` conformance, explicit types on public functions, and generic coordinates (`npt.NDArray[np.float64]`).
- **Data Modeling**: Use of `@dataclass(frozen=True)` or `Protocol` interfaces for node clouds and boundary descriptors.
- **Environment & Build**: Modern Python packaging managed via `pyproject.toml` and `uv`.
- **Testing**: Deterministic mathematical tests with controlled random seeds, verifying matrix conditioning, convergence rates, and solution residual norms.

### 3.4 Always-On Session Infrastructure

Two skills remain active across all agent turns:
- **`orient`**: Regulates agent output to be direct, structured, and compliant with ASD-STE100 principles. Every turn concludes with a compass block:
  - 📍 Where you are
  - ✅ What was done
  - 🧭 Why a decision was made
  - ➡️ Concrete next steps
- **`suggest`**: Scans user prompts for missing capabilities and recommends installing relevant tools from the RSC catalog.

---

## 4. Specialized Agents Matrix

The repository harness configures 6 automated review agents:

| Agent Name | Specialty | Role & Core Verification Focus |
| :--- | :--- | :--- |
| **`developer`** | Implementation | Executes engineering tasks, develops solver logic, and writes unit tests following TDD principles. |
| **`python-reviewer`** | Python Best Practices | Reviews code for idiomatic Python, proper typing, memory efficiency, and NumPy vectorization. |
| **`finding-verifier`** | Issue Verification | Confirms whether identified bugs or lint findings can be reproduced before code edits are applied. |
| **`refuter-correctness`** | Mathematical Logic | Adversarial agent that searches for mathematical edge cases, boundary condition flaws, and numerical instabilities. |
| **`refuter-security`** | Code Integrity | Inspects input handling, file operations, and external dependencies for security vulnerabilities. |
| **`refuter-tests`** | Test Quality | Evaluates test suites for mock overuse, missing edge cases, assertion validity, and test flakiness. |

---

## 5. Harness Command Reference

Harness commands provide direct terminal and chat shortcuts within `.gemini/commands/`:

| Command | Corresponding Skill | Operational Description |
| :--- | :--- | :--- |
| `/specify` | `specify` | Starts specification brainstorming for a new feature or solver capability. |
| `/clarify` | `clarify` | Discovers and clarifies unstated assumptions in a draft specification. |
| `/plan` | `plan` | Designs the technical architecture and component contracts for an approved spec. |
| `/tasks` | `tasks` | Slices an approved plan into ordered, verifiable engineering tasks. |
| `/analyze` | `analyze` | Validates consistency across constitution, spec, plan, and tasks. |
| `/implement` | `implement` | Executes implementation tasks one by one using TDD. |
| `/verify` | `verify` | Executes the complete verification suite and records test evidence. |
| `/review` | `review` | Conducts an adversarial diff review against specifications. |
| `/ship` | `ship` | Prepares, verifies, and lands the current feature branch. |
| `/debug` | `debug` | Reproduces, diagnoses, and verifies a fix for a defect. |
| `/python-review` | `python` | Runs a dedicated Python code quality and typing review. |
| `/refactor-clean` | `simplify-code` | Simplifies working code while proving behavior preservation. |
| `/checkpoint` | `worktrees` | Takes an atomic git checkpoint of current progress. |
| `/harness-audit` | `harness` | Audits the repository wiki, tools, and control plane. |
| `/learn` | `init` / `harness` | Records project conventions and persistent preferences. |
| `/save-session` | `harness` | Persists current session memory and progress state. |
| `/resume-session`| `harness` | Restores context from a previously saved session. |

