# Agent Instructions

These rules apply to the whole repository.

1. Read `docs/project.md`, `docs/decisions.md`, and `.agents/README.md` before changing research code, protocols, or claims.
2. Do not deviate from the mentor-approved goal, hypotheses, benchmark scope, or shared evaluation contract without a motivated entry in `docs/decisions.md`.
3. Never invent, hide, cherry-pick, or manually alter results. Preserve failures and negative findings.
4. Every experiment must be reproducible from a versioned configuration, exact command, code revision, data/model revisions, seeds, environment, and hardware record.
5. Use fair controls, explicit ablations, uncertainty estimates, and appropriate statistical tests before making comparative claims.
6. Keep reusable code in `src/`, tests in `tests/`, and experiment definitions in `experiments/configs/`.
7. Do not commit secrets, datasets, model weights, indexes, or raw run outputs.

The binding experimental protocol and completion checklist are in `.agents/README.md`.
