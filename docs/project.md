# Project Brief

## Goal

Determine whether a small Hierarchical Reasoning Model can use retrieved evidence as an effective and efficient reasoning component for RAG, compared with parameter-matched chain-of-thought language models under a shared retrieval pipeline.

## Research questions

- **H1 — Reasoning:** with identical evidence, an HRM performs better on multi-step reasoning tasks than comparable CoT baselines.
- **H2 — Robustness:** an HRM loses less accuracy when irrelevant passages are added to the retrieved context.
- **H3 — Adaptation:** lightweight low-level-module adaptation improves evidence grounding without materially reducing reasoning ability.

## Initial scope

The primary track is text RAG. Start with a small HotpotQA validation subset, then scale to the official validation split. Add 2WikiMultihopQA as a multi-hop replication and Natural Questions as a single-hop control only after the first end-to-end comparison is reproducible.

Repository-level code repair is a stretch track. Start it only after the text pipeline passes the gate defined in the roadmap; SWE-bench setup and patch validation would otherwise dominate the available time.

## Shared evaluation contract

- Models receive the same question, retrieved passages, ordering, and context limit.
- Primary text metrics are Exact Match and token-level F1 using benchmark-normalized answers.
- Report retrieval recall separately so reader quality is not confused with retrieval quality.
- Measure robustness at predefined distractor counts and report the change from clean context.
- Report latency, throughput, peak memory, parameter count, and hardware alongside quality.
- Use fixed dataset splits and at least three seeds for stochastic training comparisons.
- Compare frozen and adapted HRM variants; keep an unchanged reasoning-task check for H3.

## First experiment matrix

| ID | Question | Comparison | Required output |
|---|---|---|---|
| E0 | Does the pipeline compute trusted metrics? | Published/example predictions vs local scorer | Metric unit tests and smoke report |
| E1 | How much does retrieval limit performance? | Gold evidence vs BM25 top-k | Recall@k and EM/F1 upper-bound gap |
| E2 | Does HRM reasoning help with shared evidence? | Frozen HRM vs size-matched CoT baseline | EM/F1 and efficiency table |
| E3 | Is HRM robust to retrieval noise? | 0, 1, 2, and 4 seeded distractors | Accuracy-degradation curve |
| E4 | Does lightweight adaptation improve grounding? | Frozen vs low-level adapted HRM | EM/F1 plus retained reasoning score |

## Success and stop conditions

The preparation phase succeeds when the team can run a documented small-slice baseline and produce a validated result row. Before scaling, confirm that the HRM can ingest the required context and emit benchmark-compatible answers within available compute. If it cannot, document the failure and test a narrower input interface before committing to full training.

## Expected deliverables

1. Reusable retrieval, reader, metric, and experiment-runner code.
2. Reproducible frozen and adapted HRM evaluations against fair baselines.
3. Robustness and efficiency analyses with machine-readable results.
4. A proceedings article and supporting pre-defense/final presentation materials.
