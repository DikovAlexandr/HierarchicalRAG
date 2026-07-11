# Agent Instructions

These rules apply to the whole repository.

1. Read `docs/project.md`, `planning/roadmap.md`, and `planning/board.md` before changing research code or plans.
2. Claim one task on the board, keep its status current, and attach evidence when it is done.
3. Preserve the shared evaluation contract: identical retrieved context and comparable decoding budgets across model families.
4. Keep reusable code in `src/`, configurations in `experiments/configs/`, and exploratory work in `notebooks/`.
5. Never invent metrics or citations. Record seeds, dataset versions, model revisions, commands, and hardware for every reported result.
6. Do not commit secrets, datasets, checkpoints, indexes, or raw run directories.
7. Make focused changes, add proportional tests, and avoid unrelated rewrites.

See `agents/README.md` for the working protocol and completion checklist.
