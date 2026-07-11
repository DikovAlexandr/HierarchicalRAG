# Research Roadmap

Dates are assigned on the team board when owners confirm availability.

| Phase | Work | Exit criterion |
|---|---|---|
| P0 — Preparation | Repository structure, project brief, literature map, board, and report outline | Team agrees on scope and claims tasks |
| P1 — Feasibility | Pin HRM revision; reproduce one reference task; load a small HotpotQA slice; implement answer metrics | Reproduction and dataset smoke checks are documented |
| P2 — Retrieval baseline | Build BM25/gold-context adapters and a size-matched CoT baseline | One command produces retrieval and QA metrics |
| P3 — Frozen HRM | Define evidence serialization and HRM answer interface | E2 runs on the same examples and contexts as the baseline |
| P4 — Robustness | Add deterministic distractor injection and efficiency instrumentation | E3 produces a clean/noisy comparison with uncertainty |
| P5 — Adaptation | Train the low-level module with frozen high-level module; rerun reasoning check | E4 tests grounding gain and reasoning retention |
| P6 — Scale and report | Run full splits/seeds, analyze errors, populate article tables | Results are reproducible and linked to configurations |
| P7 — Optional code track | Run a small SWE-bench Lite feasibility study | Proceed only if text milestones and compute budget are secure |

## Immediate next actions

1. Assign owners and dates in `planning/board.md`.
2. Record the exact HRM repository revision and environment requirements.
3. Decide the parameter-matched baseline from models that the available hardware can run.
4. Implement E0 on a tiny, deterministic dataset slice.
5. Review feasibility and lock the first experiment configuration before larger runs.

## Risk controls

- **HRM text interface is immature:** validate context ingestion before dataset-scale work.
- **Compute is insufficient:** use small slices and comparable small baselines; estimate cost before training.
- **Retriever confounds model quality:** use shared retrieved candidates and report recall separately.
- **Scope expands:** keep the code track behind the P7 gate.
- **Results are irreproducible:** pin revisions and log config, seed, environment, hardware, and command.
