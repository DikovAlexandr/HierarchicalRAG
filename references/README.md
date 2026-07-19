# References

## Why this directory exists

`references/` is the evidence base for the project, not a storage folder for PDFs. It keeps the article bibliography and the primary works used to justify benchmarks, metrics, baselines, ablations, and architecture decisions. Without it, later decisions can become opinion-based and citations in the paper are harder to audit.

Keep the directory if the project will produce a scientific article or if methodological decisions may rely on prior work. It can be removed only if bibliography management moves to another explicit, version-controlled system and `docs/decisions.md` still links every literature-based decision to a verified source.

The initial `bibliography.bib` is derived from the mentor proposal. Verify each entry against the publisher, official proceedings, or official repository before citing it. Do not cite unread secondary summaries and never invent missing metadata.

## Read first

1. **Wang et al. — Hierarchical Reasoning Model.** Understand the two-timescale architecture, input/output constraints, training recipe, ACT mechanism, reference tasks, and reported compute. This determines feasibility.
2. **Lewis et al. — Retrieval-Augmented Generation.** Establish the retrieve-then-read framing and the separation between parametric and non-parametric knowledge.
3. **Yang et al. — HotpotQA.** Confirm splits, supporting-fact structure, official answer normalization, and multi-hop categories.
4. **Trivedi et al. — MuSiQue.** Use as evidence for controlling shortcut-based multi-hop reasoning; defer implementation until HotpotQA works.
5. **Karpukhin et al. — DPR.** Dense-retrieval baseline; start with BM25 to reduce setup cost, then add dense retrieval if it answers a research question.

## Supporting papers

- **Ho et al. — 2WikiMultihopQA:** replication benchmark for compositional multi-hop QA.
- **Kwiatkowski et al. — Natural Questions:** single-hop/open-domain control.
- **Joshi et al. — TriviaQA:** additional knowledge-intensive QA benchmark.
- **Wei et al. — Chain-of-Thought Prompting:** conceptual baseline for token-space reasoning.
- **Jimenez et al. — SWE-bench:** optional repository-repair evaluation.
- **Zhang et al. — RepoCoder:** optional repository-level retrieval approach.

## Reading-note rule

When a paper motivates a decision, record its citation key, research question, relevant method, dataset/metric, assumptions, threat to validity, and concrete implication in `docs/decisions.md` or a focused note created under `references/notes/`. A title in the bibliography alone is not evidence that the work was reviewed.

## Open literature questions

- Can the reference HRM accept variable-length natural-language evidence without a major architectural change?
- What is a defensible parameter- and compute-matched CoT baseline?
- Which benchmark evaluation scripts can be reused without altering answer normalization?
- How should reasoning retention be measured after low-level-module adaptation?
