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

### D006 — Preserve title case and explicitly exclude empty fullwiki records

- **Date / owner:** 2026-07-23 / alexander_dikov.
- **Status:** accepted; supersedes only the identifier and empty-record handling in D005.
- **Decision:** use the NFKC- and whitespace-normalized title with original case as the benchmark document identifier, and retain the official Wikipedia page ID as a separate immutable source key. Exclude a record only when the official plain `text` field is empty after the documented sentence join. Do not substitute `text_with_links`. Require the full build to observe exactly 5,233,329 source records and 94 empty-text exclusions; otherwise fail before publishing the index. Record every excluded page ID, title, source shard, line, and reason in the index manifest.
- **Alternatives:** continue using a case-folded title; use only the Wikipedia page ID; keep an empty FTS document; derive text by stripping links from `text_with_links`; silently skip invalid rows.
- **Evidence:** a complete streaming audit of the checksum-verified official archive found 5,233,329 records, 94 empty `text` fields, 2,637 case-folded-title collision occurrences, zero exact normalized-title collisions, zero duplicate Wikipedia IDs, and no invalid titles or IDs. The first full build correctly stopped on `2006 in organized crime` rather than hiding the anomaly. Its `text` is empty while `text_with_links` contains only an external empty anchor, so substituting that field would invent a preprocessing rule without useful text.
- **Impact:** index schema v2 contains 5,233,235 searchable documents if and only if the audited counts are reproduced. Retrieved document identifiers remain directly comparable with case-sensitive HotpotQA supporting titles. E1 must report whether any gold supporting title belongs to the 94 exclusions. Existing schema-v1 indexes cannot be used for E1.
- **Revisit when:** a checksum-identical rebuild produces different counts, an official HotpotQA evaluator requires another title mapping, or a gold supporting title is absent because of an excluded empty record. Any replacement rule requires a superseding decision before E1.

### D007 — Parallelize E1 queries without changing retrieval

- **Date / owner:** 2026-07-23 / alexander_dikov.
- **Status:** accepted after the preregistered E1 smoke and before the full validation run.
- **Decision:** execute independent fullwiki queries concurrently through read-only SQLite connections while preserving input-example order. Keep the schema-v2 index, question text, tokenizer, OR query, BM25 ranking, tie-breaking, and top-k unchanged. Before the full run, repeat the same 16-example smoke slice and require exact equality of document IDs, ranks, and scores against `e1-hotpotqa-fullwiki-bm25-smoke-v1`; timing fields are expected to differ.
- **Alternatives:** run the full split sequentially; request a CPU cluster allocation; change the query by removing stopwords or tuning retrieval parameters.
- **Evidence:** the clean-commit smoke at `04f5ced7d45b5c403395331e168e070e28fd3ba6` completed all provenance checks and measured 5.2929 seconds mean retrieval latency and 0.1889 examples/second over 16 examples. A sequential 7,405-example run is therefore estimated at 10.9 hours. Query changes are rejected because metrics have already been observed and D005 freezes the retrieval method.
- **Impact:** only execution scheduling and throughput may change. A new v2 smoke/full config records the worker count; v1 raw outputs remain immutable. If ranking equivalence fails, the parallel result is invalid and the sequential implementation remains authoritative.
- **Revisit when:** concurrent reads do not improve wall time, exceed available RAM, or change any ranking field.

### D008 — Use the official HRM-Text-1B checkpoint for the primary text track

- **Date / owner:** 2026-07-24 / alexander_dikov.
- **Status:** accepted after the P1 reference smoke and before implementing the text-reader adapter.
- **Decision:** use `sapientinc/HRM-Text-1B` at Hugging Face revision `9f082d68b8cd0ebc56e33f1c88c45609174c272c` as the frozen HRM backbone for the primary text-RAG feasibility study. Run the native `hrm_text` implementation from Transformers 5.9.0 in BF16, preserve the checkpoint's PrefixLM interface by marking all prompt positions with `token_type_ids = 1`, and use deterministic greedy decoding for the first interface tests. The original puzzle-only HRM remains architectural background and is not treated as a natural-language reader.
- **Alternatives:** build a new text tokenizer and decoder around the original 27M puzzle HRM; use an unofficial instruction-tuned HRM derivative; train HRM-Text from scratch; abandon the HRM text track.
- **Evidence:** the official checkpoint documents a natural-language PrefixLM interface and a 4,096-token context. DataSphere Job `bt1h5hqe0qmi3fe9trpo` reproduced the official reference prompt on one A100-SXM4-80GB using checkpoint revision `9f082d68b8cd0ebc56e33f1c88c45609174c272c`, Transformers 5.9.0, PyTorch 2.5.1+cu121, and BF16. The resolved model type was `hrm_text`, the parameter count was 1,182,795,264, peak allocated generation memory was 2,451,062,272 bytes, and 84 tokens were generated in 8.80 seconds. This is an interface and resource result, not evidence of HotpotQA quality or H1.
- **Impact:** P1 has a feasible, pinned text backbone and GPU environment. Before any E2 claim, the project must still freeze a shared evidence prompt, validate benchmark-compatible answer extraction on development data, select a defensible size-matched baseline, and run both readers on identical cached evidence and budgets. The reference output cannot populate an article result table.
- **Revisit when:** the frozen checkpoint cannot reliably emit benchmark-compatible answers from the fixed evidence interface, its 4,096-token limit prevents a fair shared-context comparison, or a mentor-approved backbone change is supported by stronger feasibility evidence.

### D009 — Develop the HRM evidence interface only on a deterministic train slice

- **Date / owner:** 2026-07-24 / alexander_dikov.
- **Status:** accepted before selecting examples or observing evidence-grounded HRM outputs.
- **Decision:** build P1 interface version 1 from the pinned HotpotQA `distractor/train` Parquet export at revision `1908d6afbbead072334abe2965f91bd2709910ab`. Select 18 examples by the existing SHA256 procedure with seed 42 and no manual filtering: deterministic ranks 1–2 are fixed in-context demonstrations and ranks 3–18 are the 16 evaluated development targets. Use only the gold supporting paragraphs in original benchmark context order. Follow the official raw-checkpoint guidance with the `direct` condition and exactly two few-shot examples. Use a 4,032-token maximum input, reserve 64 tokens for deterministic greedy generation, mark every prompt token as PrefixLM input, and truncate only target evidence at sentence boundaries in original order when necessary. The smoke runs once; all empty, malformed, truncated, or otherwise failed outputs count and remain preserved. Report answer EM/F1, extraction status, latency, throughput, and GPU memory as exploratory diagnostics only.
- **Alternatives:** tune prompts on HotpotQA validation; manually choose easy demonstrations; start with zero-shot or `synth,cot`; evaluate BM25 and gold evidence simultaneously; run a large split before validating answer extraction.
- **Evidence:** the HotpotQA train split is disjoint from the validation split used for primary evaluation. The official HRM-Text model card states that short-form QA should use `direct` with two to eight few-shot examples and that pure zero-shot is weaker. Two demonstrations are the minimum supported count and preserve the largest evidence budget under the checkpoint's 4,096-token limit. Gold evidence isolates the reader interface before retrieval failures are introduced.
- **Impact:** prompt or parser changes informed by this smoke are allowed only in a new train-only development version and must not retroactively replace v1. These 16 outputs cannot support H1, an article table, baseline selection by observed superiority, or any validation claim. After the interface is frozen, HRM and the selected baseline must receive identical target evidence text and context limits in E2.
- **Revisit when:** v1 cannot produce parseable short answers, fixed demonstrations alone exceed the input budget, or sentence-boundary truncation removes all target evidence. A revised interface must keep development on train and record the reason for superseding v1.

## New decision template

### DXXX — Short title

- **Date / owner:** YYYY-MM-DD / name.
- **Status:** proposed, accepted, rejected, or superseded by DXXX.
- **Decision:** exact choice.
- **Alternatives:** options considered.
- **Evidence:** experiment IDs, verified citations, official documentation, or measured resource constraints.
- **Impact:** affected configs, datasets, metrics, claims, and comparability.
- **Revisit when:** explicit condition.
