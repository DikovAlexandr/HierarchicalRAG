# Decision Log

Record material research decisions before they affect confirmatory experiments. Decisions must be motivated by pilot evidence, a verified primary source, an official benchmark/model requirement, or a documented resource constraint.

Required fields: ID, date, owner, status, decision, alternatives, evidence, expected impact, and revisit condition. Never rewrite an accepted entry after results are known; append a superseding decision.

## Accepted decisions

### D001 — Text RAG is the primary track

- **Status:** accepted during preparation.
- **Decision:** establish the shared text-RAG evaluation pipeline before considering repository-level code repair.
- **Alternatives:** start Text and Code RAG simultaneously; start Code RAG first.
- **Evidence:** the mentor proposal permits either or both tracks; text QA directly tests multi-hop reasoning with lower infrastructure cost.
- **Impact:** HotpotQA is the feasibility benchmark; the code track remains gated.
- **Revisit when:** the frozen-HRM text comparison is reproducible and compute/storage budgets are known.

### D002 — Main model comparisons share retrieved evidence

- **Status:** accepted; part of the mentor-aligned evaluation contract.
- **Decision:** HRM and CoT readers receive identical examples, passages, ordering, context limits, and comparable decoding budgets.
- **Alternatives:** allow model-specific retrieval or context selection.
- **Evidence:** shared evidence isolates reader reasoning from retrieval quality and directly tests H1/H2.
- **Impact:** retrieval recall is reported separately; model-specific retrieval may appear only as a labeled secondary experiment.
- **Revisit when:** a new hypothesis explicitly concerns joint retriever-reader optimization.

### D003 — Start retrieval diagnostics with gold context and BM25

- **Status:** provisional until E1.
- **Decision:** use gold evidence as an upper-bound diagnostic and BM25 as the first reproducible retriever.
- **Alternatives:** begin with DPR or another dense retriever.
- **Evidence:** BM25 reduces setup and tuning confounds; dense retrieval remains justified if E1 shows retrieval is the limiting factor or the literature requires it.
- **Impact:** no claim that BM25 is the best retriever; E1 determines whether a dense-retrieval ablation is needed.
- **Revisit when:** retrieval recall materially limits reader evaluation.

## New decision template

### DXXX — Short title

- **Date / owner:** YYYY-MM-DD / name.
- **Status:** proposed, accepted, rejected, or superseded by DXXX.
- **Decision:** exact choice.
- **Alternatives:** options considered.
- **Evidence:** experiment IDs, verified citations, official documentation, or measured resource constraints.
- **Impact:** affected configs, datasets, metrics, claims, and comparability.
- **Revisit when:** explicit condition.
