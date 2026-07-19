# Research and Agent Protocol

This file contains binding rules for human and LLM-assisted work. A result that violates these rules must not appear in the paper, presentation, or aggregate result table.

## Project alignment

- Read `docs/project.md` before implementation or experiment design.
- Preserve the mentor-approved goal: evaluate HRMs as reasoning components for RAG, using shared retrieval evidence and comparable generation baselines.
- Do not silently change hypotheses, benchmark splits, primary metrics, model interfaces, or comparison budgets after observing results.
- Record every material choice or deviation in `docs/decisions.md`, including alternatives, evidence, consequences, and the person responsible.
- Prefer experimental evidence. When experiments are not yet possible, cite a verified primary paper or official benchmark/model documentation and mark the decision provisional.

## Honest experimentation

- Define the hypothesis, primary metric, comparison, controls, exclusions, seeds, and stopping rule before the main run.
- Never cherry-pick seeds, examples, checkpoints, metrics, or subsets. Report failed and negative runs.
- Never reuse test data for training, tuning, prompt selection, threshold selection, or debugging.
- Do not tune one model more extensively than another without reporting the asymmetry.
- Keep retrieved evidence, ordering, context limit, dataset split, metric implementation, and decoding budget identical for the main HRM/baseline comparison.
- Separate retrieval quality from reader quality; report retrieval recall and answer quality independently.

## Ablations and controls

- Each claimed contribution needs a control or ablation that isolates it.
- At minimum, compare the CoT baseline, frozen HRM, and adapted HRM under shared evidence.
- For adaptation, compare frozen vs adapted modules and evaluate reasoning retention on an unchanged reasoning task.
- For robustness, use deterministic distractor generation and predefined noise levels.
- Change one factor at a time unless the interaction is the stated research question.
- Document omitted ablations and explain why they were infeasible.

## Statistical evidence

- Use at least three independent seeds for stochastic training comparisons; use more when variance is high. Any exception requires a recorded compute-based justification.
- Report sample size, point estimate, dispersion or confidence interval, and effect size where applicable.
- Use paired tests when systems are evaluated on the same examples. Select the test before inspecting the final comparison and verify its assumptions.
- Prefer bootstrap confidence intervals for EM/F1 differences and paired permutation/bootstrap tests when distributional assumptions are unclear.
- Correct for multiple comparisons when testing several models, datasets, metrics, or noise levels.
- Do not equate statistical significance with practical importance; report both uncertainty and effect magnitude.

## Reproducibility record

Every run must preserve:

- experiment ID, hypothesis/decision ID, timestamp, owner, exact command, and Git commit;
- complete resolved configuration, including all defaults;
- dataset name, split, revision, checksum/fingerprint, preprocessing, filtering, and sample IDs;
- retriever, corpus/index revision, top-k, query construction, ranking parameters, and context serialization;
- model/checkpoint identifier and revision, tokenizer, precision, decoding parameters, context/output limits, and prompt/template revision;
- training parameters, optimizer state, checkpoint-selection rule, seeds, and deterministic settings when training is used;
- package/driver versions, operating system/container, hardware, peak memory, latency, and throughput;
- predictions, retrieved item IDs/scores, logs, failures, metric outputs, and statistical-analysis inputs.

Raw run data belongs under `results/runs/<experiment-id>/` and is not committed. It must remain immutable after completion. Track only reviewed aggregates and lightweight metadata in Git.

Confirmatory runs must use a clean committed worktree. If an exploratory run uses uncommitted code, save the exact diff with the run and mark the result exploratory; such a run cannot support a final claim until repeated from committed code. Record a checksum for the resolved config, environment lock, and run manifest.

## Code and reporting quality

- Put reusable logic in `src/`; notebooks and ad hoc snippets cannot be the only implementation of a reported method.
- Add tests for metrics, data boundaries, serialization, randomness, configuration validation, and critical model/retrieval interfaces.
- Keep changes focused. Do not add unused abstractions, decorative artifacts, duplicated documentation, or speculative dependencies.
- Every number in reports must resolve to an experiment ID and configuration. Label exploratory results and do not mix them with confirmatory results.
- Cite only sources that were checked against a primary publication or official repository. Do not fabricate citation metadata.

## Definition of done

An experimental task is complete only when the relevant tests pass, the run is reproducible from recorded metadata, raw outputs are preserved, aggregate metrics are independently checkable, statistical analysis is complete, failures are documented, and any resulting decision or claim is linked to evidence.
