# Literature Map

The initial bibliography is derived from the project proposal and stored in `bibliography.bib`. Verify metadata against the publisher or official repository before submission.

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

## Reading-note template

For each paper, record: citation key, research question, relevant method, dataset/metric, assumptions, reusable implementation, threat to validity, and one concrete implication for this project. Store notes as `references/notes/<citation-key>.md` when reading begins.

## Open literature questions

- Can the reference HRM accept variable-length natural-language evidence without a major architectural change?
- What is a defensible parameter- and compute-matched CoT baseline?
- Which benchmark evaluation scripts can be reused without altering answer normalization?
- How should reasoning retention be measured after low-level-module adaptation?
