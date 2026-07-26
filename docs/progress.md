# Research Progress

Last updated: 2026-07-26.

## Current position

The repository preparation, reproducibility infrastructure, metric validation, HRM-Text interface work, full BM25 retrieval run, and train-only reader feasibility studies are complete or substantially complete. The project has not yet produced a confirmatory HRM-versus-baseline comparison on shared validation evidence.

The end-to-end project is approximately **45% complete** by research-stage exit criteria. This estimate is not based on code volume: feasibility, infrastructure, sparse retrieval, and reader-budget diagnosis are complete, while the claim-bearing H1–H3 experiments remain mostly ahead.

| Stage | Status | Completed | Remaining exit work |
|---|---|---|---|
| P1 — Feasibility | Complete | Pinned HRM-Text reference, deterministic train slice, metric harness, GPU environment, reader interfaces, and preserved failures | No remaining P1 work; baseline selection gates P3/E2 |
| P2 — Retrieval | Sparse baseline complete | Full 7,405-example BM25 run, independent audit, bootstrap CI, and error analysis | Frozen dense-retrieval ablation required by D027; then bind both immutable caches to readers |
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

The independent audit at revision `8bf5108` reproduced every ranking-derived metric and the 10,000-resample interval from the immutable raw records. Both supporting paragraphs are present in top 10 for only 2,061/7,405 examples (27.83%); 4,052 (54.72%) contain one supporting paragraph and 1,292 (17.45%) contain none. Across 14,810 gold paragraphs, 6,636 (44.81%) are absent from top 10. D027 therefore keeps BM25 as the frozen sparse baseline and requires one pretrained dense-retrieval ablation before confirmatory reader conclusions. These values still do not measure reader quality.

### Native-thinking baseline feasibility

All three D013 candidates completed their single final train-only D017 attempt under the shared 2,048-input + 2,048-output ceiling. None met the preregistered requirement of valid reasoning and final answers for all 16 targets with zero exhaustion and zero input truncation.

| Model | Valid answers | Reasoning | Exhausted | Truncated | Gate |
|---|---:|---:|---:|---:|---|
| LFM2.5-1.2B-Thinking | 16/16 | 16/16 | 1/16 | 0/16 | Failed |
| Qwen3.5-0.8B | 9/16 | 16/16 | 12/16 | 0/16 | Failed |
| Qwen3.5-2B | 15/16 | 16/16 | 6/16 | 0/16 | Failed |

This supports a limited negative conclusion: these three native-thinking checkpoints are not feasible primary E2 controls under the frozen 4,096-token contract and interface criteria. It does not show that they are generally weak models, that HRM is better, or that another resource contract would fail.

The exploratory train-only answer scores are retained for diagnosis but cannot be used for H1 or model selection. Decisions D020–D022 prevent output-dependent reruns or relaxation of the gate.

### Expanded-output sensitivity

D023 tested the two preregistered modern readers at 4,096 and 8,192 output-token ceilings on the unchanged 16-example train slice with cap-stable per-example seeds.

| Model | Output ceiling | Valid answers | Exhausted | Generated tokens | Mean latency | Gate |
|---|---:|---:|---:|---:|---:|---|
| LFM2.5-1.2B-Thinking | 4,096 | 14/16 | 2/16 | 20,644 | 17.0 s | Failed |
| LFM2.5-1.2B-Thinking | 8,192 | 14/16 | 2/16 | 28,836 | 24.4 s | Failed |
| Qwen3.5-2B | 4,096 | 15/16 | 1/16 | 37,071 | 78.0 s | Failed |
| Qwen3.5-2B | 8,192 | 16/16 | 0/16 | 37,737 | 80.5 s | Passed |

For Qwen, 15/16 raw outputs were identical across ceilings; the remaining example continued from 4,096 to 4,762 tokens and then terminated normally. Thus 8,192 is a safety ceiling rather than a cost paid on every example. For LFM, the same two examples consumed the full 4,096 and 8,192 ceilings, so the added budget increased cost without resolving the looping behavior. Qwen3.5-2B is therefore feasible only as an expanded-budget secondary control under D026; LFM is closed after this negative result. The train-only EM/F1 values are diagnostic and cannot support H1 or model selection.

Peak allocated memory was 4.61 GB for Qwen and 2.48 GB for LFM on an 80 GB A100. This confirms that the current sequential runner is not memory-limited and must not be scaled to validation before a batching and cost-per-example hardware benchmark.

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
- `results/reviews/d023-expanded-output-series.audit.json`;
- `results/reviews/e1-hotpotqa-fullwiki-bm25-v2.audit.json`;
- `results/summary.csv`.

## Next gates

1. Select and freeze one pretrained dense retriever under D027, then build its corpus index and immutable HotpotQA rankings.
2. Benchmark a batched reader implementation on fixed train-only prompts across available GPUs and select hardware by cost per completed example, not hourly price or VRAM capacity.
3. Ask the mentor to approve the final E2 baseline family. The recommended proposal is Qwen2.5-1.5B-Instruct under the shared primary budget, with Qwen3.5-2B at the expanded ceiling only as a labeled secondary control.
4. Record a new decision that freezes the E2 models, shared evidence and token contract, sample size, seeds, paired tests, confidence intervals, effect size, and multiple-comparison correction.
5. Run a train-only HRM/primary-baseline smoke on byte-identical cached evidence.
6. Only after that gate passes, run the frozen validation comparison for H1, followed by H2 robustness and H3 adaptation.

No additional D017 or D023 GPU run is authorized. The next GPU work is a train-only throughput benchmark, followed by the E2 shared-evidence smoke only after its protocol is frozen.
