# Research Progress

Last updated: 2026-07-26.

## Current position

The repository preparation, reproducibility infrastructure, metric validation, HRM-Text interface work, full BM25 retrieval run, and train-only reader feasibility studies are complete or substantially complete. The project has not yet produced a confirmatory HRM-versus-baseline comparison on shared validation evidence.

The end-to-end project is approximately **40% complete** by research-stage exit criteria. This estimate is not based on code volume: feasibility and infrastructure are advanced, while the claim-bearing H1–H3 experiments remain mostly ahead.

| Stage | Status | Completed | Remaining exit work |
|---|---|---|---|
| P1 — Feasibility | Complete | Pinned HRM-Text reference, deterministic train slice, metric harness, GPU environment, reader interfaces, and preserved failures | No remaining P1 work; baseline selection gates P3/E2 |
| P2 — Retrieval | Mostly complete | Full 7,405-example BM25 run and bootstrap CI | Independent artifact audit, error analysis, shared-reader cache binding, and dense-ablation decision |
| P3 — Frozen HRM | In progress | HRM adapter and gold-evidence train smoke | Identical BM25-evidence smoke and confirmatory validation comparison |
| P4 — Robustness | Prepared only | Deterministic distractor machinery | Shared-reader runs at 0, 1, 2, and 4 distractors with paired analysis |
| P5 — Adaptation | Not started | Hypothesis and safeguards defined | Training data, low-level adaptation, seeds, ablations, and reasoning-retention evaluation |
| P6 — Scale and report | Early | Audited result index and reproducibility records | Confirmatory tables, statistical analysis, replications, error analysis, and paper conclusions |

## Evidence obtained

### Retrieval

The complete `e1-hotpotqa-fullwiki-bm25-v2` run contains 7,405 validation rankings. Its current recorded diagnostics are:

- paragraph Recall@10: 0.5519; 95% bootstrap CI [0.5443, 0.5596];
- supporting-fact Recall@10: 0.5575;
- all supporting paragraphs present at top 10: 0.2783.

These values indicate a likely retrieval bottleneck and motivate a dense-retrieval ablation. They do not measure reader quality, and they require an independent artifact audit and error analysis before a new retrieval decision is accepted.

### Native-thinking baseline feasibility

All three D013 candidates completed their single final train-only D017 attempt under the shared 2,048-input + 2,048-output ceiling. None met the preregistered requirement of valid reasoning and final answers for all 16 targets with zero exhaustion and zero input truncation.

| Model | Valid answers | Reasoning | Exhausted | Truncated | Gate |
|---|---:|---:|---:|---:|---|
| LFM2.5-1.2B-Thinking | 16/16 | 16/16 | 1/16 | 0/16 | Failed |
| Qwen3.5-0.8B | 9/16 | 16/16 | 12/16 | 0/16 | Failed |
| Qwen3.5-2B | 15/16 | 16/16 | 6/16 | 0/16 | Failed |

This supports a limited negative conclusion: these three native-thinking checkpoints are not feasible primary E2 controls under the frozen 4,096-token contract and interface criteria. It does not show that they are generally weak models, that HRM is better, or that another resource contract would fail.

The exploratory train-only answer scores are retained for diagnosis but cannot be used for H1 or model selection. Decisions D020–D022 prevent output-dependent reruns or relaxation of the gate.

## Relation to the research hypotheses

- **H1 — HRM reasoning advantage:** not tested. HRM and an accepted baseline have not yet been compared on identical cached validation evidence.
- **H2 — robustness to irrelevant passages:** not tested. The distractor implementation exists, but paired model runs have not started.
- **H3 — low-level adaptation:** not tested. No adaptation training or reasoning-retention evaluation has started.

The completed work materially reduces implementation, environment, baseline-feasibility, and retrieval uncertainty. It brings the project to the point where confirmatory experiments can be designed responsibly, but it does not yet justify a global HRM effectiveness claim.

## Preserved Notebook results

Raw archives are stored locally under `results/runs/_archives/`; extracted immutable records remain under `results/runs/<experiment-id>/`. Raw outputs are intentionally excluded from Git. Reviewed checksums and independently reproduced aggregates are tracked in:

- `results/reviews/p1-lfm2.5-thinking-gold-train-smoke-v3.audit.json`;
- `results/reviews/p1-qwen3.5-0.8b-thinking-gold-train-smoke-v5.audit.json`;
- `results/reviews/p1-qwen3.5-2b-thinking-gold-train-smoke-v5.audit.json`;
- `results/summary.csv`.

## Next gates

1. Independently audit the complete E1 artifact and perform retrieval error analysis.
2. Ask the mentor to approve the final E2 baseline family. The recommended proposal is to retain the native-thinking failures as negative feasibility evidence and use the already validated Qwen2.5-1.5B-Instruct model as the CoT control.
3. Record a new decision that freezes the E2 models, shared evidence and token contract, sample size, seeds, paired tests, confidence intervals, effect size, and multiple-comparison correction.
4. Run a train-only HRM/Qwen2.5 smoke on byte-identical cached BM25 evidence.
5. Only after that gate passes, run the frozen validation comparison for H1, followed by H2 robustness and H3 adaptation.

No additional D017 GPU run is authorized. A GPU is needed again only after the E2 decision and shared-evidence smoke configuration are ready.
