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

## E1 fullwiki retrieval diagnostic

E1 measures retrieval independently of any reader. It uses the pinned HotpotQA
fullwiki validation export and the immutable schema-v2 FTS5 index. The smoke run
must succeed before the full validation run; neither run permits retriever tuning.

From a clean committed worktree in the pinned CPU container:

```text
python -m hierarchical_rag.run_e1 --config experiments/configs/e1-hotpotqa-fullwiki-bm25-smoke-v1.yaml
python -m hierarchical_rag.run_e1 --config experiments/configs/e1-hotpotqa-fullwiki-bm25-smoke-v2.yaml
python -m hierarchical_rag.run_e1 --config experiments/configs/e1-hotpotqa-fullwiki-bm25-v2.yaml
```

All commands verify dataset, index, index-manifest, and dependency-lock
checksums. E1 reports Recall@k, uncertainty, latency, throughput, and peak memory;
answer EM/F1 remains explicitly not applicable until a reader is introduced.
The sequential v1 smoke is the immutable ranking reference. The v2 smoke adds
only eight-way CPU execution and must exactly match every v1 rank, document ID,
and score before the v2 full run is allowed. The unexecuted full v1 config is
retained as the preregistered sequential alternative.

## D028 dense-retrieval resource calibration

D028 freezes the single modern dense ablation before any dense validation metric
is visible. Its first run is deliberately corpus-only: 256 warmup documents and
8,192 measured documents sampled systematically across the complete Wikipedia
index. It records throughput, truncation, peak memory, the approximately 5 GiB
FP16 vector-store size, and a conservative full-corpus DataSphere unit
projection. It performs no search and observes no benchmark questions or labels.

The calibration must stop after writing its artifact. A full 5,233,235-document
embedding pass requires a separate cost approval based on the live unit balance;
the calibration result cannot be used to change the frozen model, serialization,
pooling, dimension, normalization, or input length.
