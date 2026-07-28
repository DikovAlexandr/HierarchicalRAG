# Research Progress

Last updated: 2026-07-28.

## Current position

The repository preparation, reproducibility infrastructure, metric validation, HRM-Text interface work, full BM25 retrieval run, and train-only reader feasibility studies are complete or substantially complete. The project has not yet produced a confirmatory HRM-versus-baseline comparison on shared validation evidence.

The current reported DataSphere balance is 3,409,593 units. D031's first command stopped before model loading because the 10 GB project volume had only 3.7 GB available, and Jobs cannot retain the multi-gigabyte index without writable object storage. The D033 local RTX 4060 BF16 calibration has now completed and passed independent audit: 59.5847 documents/second, a conservative 30.50-hour full-build projection, and 4.40 GiB peak reserved VRAM. D034 selects the resumable local build, preserving all DataSphere units for the later claim-bearing work; validation remains closed.

The end-to-end project is approximately **47% complete** by research-stage exit criteria. This estimate is not based on code volume: feasibility, infrastructure, sparse retrieval, dense resource feasibility, and reader-budget diagnosis are complete, while the claim-bearing H1–H3 experiments remain mostly ahead.

| Stage | Status | Completed | Remaining exit work |
|---|---|---|---|
| P1 — Feasibility | Complete | Pinned HRM-Text reference, deterministic train slice, metric harness, GPU environment, reader interfaces, and preserved failures | No remaining P1 work; baseline selection gates P3/E2 |
| P2 — Retrieval | Full build ready | Full 7,405-example BM25 run, independent audit, bootstrap CI, error analysis, audited A100 and local dense calibrations, tested resumable builder, disk guards, and a frozen local execution config | Execute the D034 local full build; then independently audit the index and run exact dense validation search |
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

The D028 label-free calibration successfully loaded the pinned `Qwen/Qwen3-Embedding-0.6B` revision on one A100 and encoded 8,192 measured systematic corpus documents in 27.312 seconds: 299.939 documents/second and 20,783 tokens/second. The independently recomputed 1.25x-reserved full-corpus projection is 21,809.6 seconds, or 2,529,915 DataSphere units. The D033 local calibration encoded the same measured sample at 59.5847 documents/second and 4.40 GiB peak reserved VRAM; its audited 1.25x projection is 109,785.62 seconds (30.50 hours). Both local gates passed, so D034 preserves the same quality-relevant protocol while moving this throughput-bound corpus-only stage to the RTX 4060 at zero DataSphere cost. Dense retrieval quality remains unknown.

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
- `results/reviews/p2-qwen3-embedding-fullwiki-calibration-v1.audit.json`;
- `results/reviews/p2-qwen3-embedding-local-calibration-v1.audit.json`;
- `results/summary.csv`.

## Next gates

1. Run `p2-qwen3-embedding-fullwiki-build-local-v1` from its clean committed revision. Preserve and resume its immutable 32,768-document checkpoints if interrupted.
2. Independently audit the completed vector/metadata manifest and add their exact checksums to a new dense E1 config before opening validation questions.
3. Run exact dense search on all 7,405 examples, audit retrieval metrics, and freeze immutable BM25/dense/gold evidence caches.
4. Benchmark a batched reader implementation on fixed synthetic/train-only prompts and select execution by units per completed example.
5. Freeze the mentor-facing E2 baseline family, shared evidence/token contract, sample size, seeds, paired tests, confidence intervals, effect size, and multiple-comparison correction.
6. Run the train-only shared-evidence gate, then H1 validation, H2 robustness, and H3 adaptation in that order.

No additional D017 or D023 run is authorized. The next GPU work is the D034 corpus-only local full build; it must not open validation labels. Query search and E2 remain blocked until the completed index is independently audited and pinned.

### Expanded compute envelopes (D031)

| Work | Units | A100-equivalent time | Rule |
|---|---:|---:|---|
| Full dense corpus build | 0 | not applicable | D034 local RTX 4060 run; 30.50 h projected with reserve, 48 h attempt cap |
| Dense query encoding and exact search | 300,000 | 43.1 min | Blocked until completed index checksums are audited and pinned |
| H1 primary comparison | 1,000,000 | 2.39 h | Highest claim-bearing reader priority |
| H2 robustness | 400,000 | 57.5 min | Freeze sample/noise matrix before execution |
| H3 adaptation and retention | 350,000 | 50.3 min | Minimum defensible seeds/ablations first |
| Failure and reproduction reserve | at least 1,200,000 | at least 2.87 h | Cannot be consumed by ordinary successful runs |
