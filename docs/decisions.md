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

### D004 — Separate controlled HotpotQA smoke tests from primary open-domain retrieval

- **Date / owner:** 2026-07-23 / alexander_dikov.
- **Status:** accepted before E1 and before any model comparison.
- **Decision:** use the official HotpotQA distractor validation context only for metric validation, interface smoke tests, and controlled robustness checks. Use HotpotQA fullwiki validation with one fixed BM25 index over a pinned official Wikipedia-introduction corpus as the primary retrieval setting for E1–E4. Gold supporting paragraphs remain a labeled diagnostic upper bound. Cache one retrieval result per example and give the identical ranked passages, ordering, and context budget to every reader.
- **Alternatives:** use distractor mode as the primary result; make distractor and fullwiki co-primary; begin directly with a dense retriever.
- **Evidence:** the mentor proposal defines the text pipeline as top-k retrieval from an open-domain index and requires a shared retriever for HRM/CoT comparisons. The official HotpotQA documentation distinguishes the ten-paragraph distractor setting from fullwiki retrieval over Wikipedia: https://hotpotqa.github.io/ and https://github.com/hotpotqa/hotpot. D003 already selects BM25 as the low-confound first retriever.
- **Impact:** distractor candidate-reranking numbers are exploratory diagnostics and cannot fill the primary HotpotQA cells in Table 1. Preparing a pinned corpus and index is required before E1; dense retrieval is an ablation only if BM25 recall is limiting. This decision does not change H1–H3, the benchmark, or the shared evaluation contract.
- **Revisit when:** the official corpus cannot be obtained or indexed within the documented storage/compute budget; any replacement requires mentor confirmation and a superseding decision before confirmatory runs.

### D005 — Use text-only SQLite FTS5 BM25 for the first fullwiki index

- **Date / owner:** 2026-07-23 / alexander_dikov.
- **Status:** accepted before the full corpus build and E1.
- **Decision:** stream the official HotpotQA `enwiki-20171001` introduction archive and index only its plain `text` field. Store the NFKC-normalized, whitespace-normalized, case-folded title as the stable document identifier, but do not index or boost the title separately. Use SQLite FTS5 with `unicode61 remove_diacritics 2`, its fixed BM25 parameters (`k1=1.2`, `b=0.75`), and an OR query over unique tokenized question terms. Build the index atomically, record corpus and index checksums, and cache one immutable ranking per question for every reader.
- **Alternatives:** a separate title field or title boost; the in-memory Python BM25 smoke implementation; Pyserini/Lucene; a dense retriever as the first fullwiki baseline.
- **Evidence:** the official HotpotQA corpus specification identifies the 2017-10-01 Wikipedia snapshot, publishes the archive size and MD5, and instructs users to index `text` rather than `text_with_links`: https://hotpotqa.github.io/wiki-readme.html. SQLite documents the FTS5 BM25 constants and score ordering: https://sqlite.org/fts5.html. A four-document fixture completed in the pinned Linux/amd64 CPU container at commit `3ba9e13d88a5ccdd697faf6641a19733806d35ff`; all 41 tests passed, the environment check matched image and repository revisions, and the mismatch control failed as required. The fixture validates mechanics and portability only, not retrieval quality or full-corpus throughput. The local machine had 53.35 GiB free disk and 15,611 MiB RAM during the resource check, making a streaming disk-backed implementation preferable to an in-memory index.
- **Impact:** E1 will measure paragraph and supporting-fact Recall@k on the full validation retrieval setting before any reader comparison. No Table 1 model result follows from this decision or the fixture smoke test. Dense retrieval remains a planned ablation only if the frozen BM25 baseline is retrieval-limited. Removing title weighting avoids an unmotivated tuning parameter.
- **Revisit when:** the verified full corpus and index do not fit the recorded storage budget, construction is not practically recoverable on the cluster, or E1 recall is insufficient. Any tokenizer, field weighting, backend, or dense-retrieval replacement requires a superseding decision before reindexing or confirmatory runs.

## New decision template

### DXXX — Short title

- **Date / owner:** YYYY-MM-DD / name.
- **Status:** proposed, accepted, rejected, or superseded by DXXX.
- **Decision:** exact choice.
- **Alternatives:** options considered.
- **Evidence:** experiment IDs, verified citations, official documentation, or measured resource constraints.
- **Impact:** affected configs, datasets, metrics, claims, and comparability.
- **Revisit when:** explicit condition.
