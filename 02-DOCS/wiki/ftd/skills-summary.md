---
author: Ricardo
branch: rsc
status: completed
---
# skills-summary

## Intent
Document all available skills, agents, and harness capabilities in the GFDFlow repository within `02-DOCS/wiki/skills_summary.md` to give contributors and users a complete, structured reference of repository capabilities.

## Scope
- Create comprehensive documentation in `02-DOCS/wiki/skills_summary.md`.
- Detail the 30 installed RSC harness skills, specialized agents, and harness commands.
- Index `skills_summary.md` in `02-DOCS/wiki/index.md`.
- Out of scope: Modifying code in `src/` or changing `.rsc.json`.

## Checklist
- [x] Create `02-DOCS/wiki/ftd/skills-summary.md` feature document
- [x] Generate comprehensive skill summary in `02-DOCS/wiki/skills_summary.md`
- [x] Update `02-DOCS/wiki/index.md` with link to `skills_summary.md`
- [x] Verify markdown syntax and file integrity

## Evidence
- `02-DOCS/wiki/ftd/skills-summary.md` created with intent, scope, checklist, and evidence sections.
- `02-DOCS/wiki/skills_summary.md` generated containing:
  - Master catalog of all 30 installed RSC skills with artifact inputs/outputs.
  - Domain-specific breakdown covering SDD, FTD, Python scientific standards, and always-on protocols.
  - Mermaid diagram of the 10-phase SDD pipeline.
  - Agent matrix covering 6 review/evaluation roles (`developer`, `python-reviewer`, `finding-verifier`, `refuter-correctness`, `refuter-security`, `refuter-tests`).
  - Command table detailing 17 harness commands in `.gemini/commands/`.
- `02-DOCS/wiki/index.md` updated to include `[Skills Summary](skills_summary.md)` under Documentation.

## Next
Present summary to the user.
