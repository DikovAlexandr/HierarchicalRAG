# Experiments

Each experiment must have a unique ID and a versioned configuration in `configs/`. The configuration is written before the main run and pins the hypothesis/decision ID, dataset and sample selection, retriever/index, context construction, model/tokenizer revisions, training or decoding parameters, seeds, metrics, statistical plan, environment, hardware, and output path.

## Required design

- State the hypothesis, primary metric, baseline, controls, exclusions, stopping rule, and expected comparison.
- Keep retrieved evidence and inference budgets identical for the main HRM/baseline comparison.
- Include ablations that isolate every claimed component; change one factor at a time unless testing an interaction.
- Use at least three independent training seeds or record a resource-based exception before running.
- Preselect confidence intervals, paired significance tests, effect sizes, and any multiple-comparison correction.
- Run a deterministic smoke test before scaling and estimate compute/storage cost.

## Required record

Store the exact command, Git commit, fully resolved config, data/model/index revisions, environment, hardware, per-example predictions, retrieved IDs/scores, logs, failures, timing, memory, and statistical-analysis inputs under `results/runs/<experiment-id>/`. Raw run outputs are ignored by Git but must remain immutable and recoverable.

Each completed run directory must contain these logical records, using the listed names unless the runner documents an equivalent format:

- `manifest.json`: experiment ID, timestamps, owner, Git state, checksums, and file inventory;
- `resolved-config.yaml`: every explicit value and inherited default;
- `command.txt` and `environment.txt`: invocation, dependency lock/checksum, OS/drivers, and hardware;
- `predictions.jsonl` and `retrieval.jsonl`: per-example outputs, retrieved IDs, ranks, and scores;
- `metrics.json` and `statistics.json`: aggregate/per-example metrics, intervals, tests, effect sizes, and analysis settings;
- `run.log`: warnings, failures, exclusions, runtime, and peak resources.

It is forbidden to cherry-pick seeds/checkpoints/examples, change the primary analysis after seeing outcomes, discard failed runs, or report an aggregate without traceable raw evidence. Add a row to `results/summary.csv` only after validation.

## E0 metric validation

E0 checks the local HotpotQA answer, supporting-fact, and joint metrics against outputs pinned from the official evaluator. It uses a small manually authored fixture and makes no model-quality or retrieval claim.

From a clean committed worktree:

```text
python -m pip install -e .
python -m pytest -q
python -m hierarchical_rag.run_e0 --config experiments/configs/e0-hotpotqa-metrics-v1.yaml
```

The last command refuses a dirty worktree, verifies input and dependency-lock checksums, and refuses to overwrite an existing run directory. Preserve `results/runs/e0-hotpotqa-metrics-v1/` after the run; it is intentionally ignored by Git.
