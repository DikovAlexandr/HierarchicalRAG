# Team Board

Last updated: 2026-07-11

## Team contributions

Each member maintains one row. Use repository paths, commit IDs, or experiment IDs as evidence.

| Team member | Role / focus | Pre-project preparation completed | Evidence | Current task | Next contribution | Availability / risks | Last updated |
|---|---|---|---|---|---|---|---|
| `abdurrahman` | _Area_ | _Describe concrete work completed during project preparation_ | _Path/link_ | _Task ID_ | _Planned outcome_ | _Constraint_ | _YYYY-MM-DD_ |
| `alexander_dikov` | Repository preparation and research planning | Reviewed the project proposal and curator presentation; prepared the repository structure, project brief, phased roadmap, experiment/result templates, agent rules, literature map, team board, and pre-defense source text | `README.md`; `docs/project.md`; `planning/roadmap.md`; `references/`; `reports/pre-defense-content.md` | REPORT-01 | Collect verified teammate contributions, finalize the preparation report, and support the team review of scope and first experiment ownership | Requires input from the other team members before the contribution section can be finalized | 2026-07-11 |
| `skudarmaria` | _Area_ | _Describe concrete work completed during project preparation_ | _Path/link_ | _Task ID_ | _Planned outcome_ | _Constraint_ | _YYYY-MM-DD_ |
| `advanced_12iq` | _Area_ | _Describe concrete work completed during project preparation_ | _Path/link_ | _Task ID_ | _Planned outcome_ | _Constraint_ | _YYYY-MM-DD_ |

## Tasks

Allowed statuses: `Backlog`, `Ready`, `In progress`, `Review`, `Done`, `Blocked`.

| ID | Task | Status | Owner | Due | Dependencies | Output / acceptance check | Evidence / notes |
|---|---|---|---|---|---|---|---|
| PREP-02 | Prepare the research repository and initial project materials | Done | `alexander_dikov` | 2026-07-11 | — | Research structure, working rules, literature, roadmap, experiment templates, board, and report source are ready and validated | `README.md`; `AGENTS.md`; `docs/`; `planning/`; `references/`; `reports/`; `experiments/`; `results/` |
| PREP-01 | Review goal, hypotheses, and scope with the team | Ready | _Unassigned_ | _Date_ | — | Agreed scope recorded in project brief | _Link/notes_ |
| ENV-01 | Pin HRM source revision and reproducible environment | Ready | _Unassigned_ | _Date_ | PREP-01 | Clean setup and reference-task command documented | _Link/notes_ |
| DATA-01 | Add deterministic HotpotQA small-slice loader | Backlog | _Unassigned_ | _Date_ | ENV-01 | Smoke test validates schema and split | _Link/notes_ |
| EVAL-01 | Implement normalized EM and token F1 | Backlog | _Unassigned_ | _Date_ | DATA-01 | Unit tests cover canonical examples | _Link/notes_ |
| RET-01 | Add gold-context and BM25 retrieval adapters | Backlog | _Unassigned_ | _Date_ | DATA-01 | Recall@k and serialized contexts are saved | _Link/notes_ |
| BASE-01 | Select and run a compute-feasible CoT baseline | Backlog | _Unassigned_ | _Date_ | RET-01, EVAL-01 | Configured small-slice result row | _Link/notes_ |
| HRM-01 | Define frozen HRM evidence/answer interface | Backlog | _Unassigned_ | _Date_ | ENV-01, RET-01 | Same examples and contexts run end to end | _Link/notes_ |
| ROB-01 | Implement deterministic distractor injection | Backlog | _Unassigned_ | _Date_ | BASE-01, HRM-01 | Noise-level comparison is reproducible | _Link/notes_ |
| REPORT-01 | Update pre-defense content with verified progress | In progress | `alexander_dikov` | _Date_ | PREP-01 | Every claim links to evidence; member rows complete | Initial content is prepared; waiting for the other members' verified contributions |

## Decisions and blockers

| Date | Type | Owner | Decision or blocker | Rationale / resolution |
|---|---|---|---|---|
| _YYYY-MM-DD_ | _Decision/Blocked_ | _Name_ | _Statement_ | _Reason or required action_ |
