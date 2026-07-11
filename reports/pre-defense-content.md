# Pre-defense Presentation Content

This is source text, not a slide deck. Replace bracketed fields with verified team information before presenting.

## 1. Project goal

We investigate whether Hierarchical Reasoning Models can serve as compact reasoning components for Retrieval-Augmented Generation. Retrieval provides external evidence, while the HRM performs multi-step latent reasoning over it. We compare HRMs with size-matched chain-of-thought models under identical retrieved context and evaluation conditions.

## 2. What we have learned

- HRMs separate slow abstract planning from fast detailed computation and report strong reasoning at a small parameter count, but their interface to variable-length natural-language evidence is the main feasibility risk.
- The project proposal supports both text QA and repository-level code repair. A simultaneous start would duplicate infrastructure and weaken early evidence.
- Multi-hop QA is the most direct first test of the reasoning/knowledge decomposition. HotpotQA provides answer and supporting-evidence annotations; 2WikiMultihopQA can later test replication.
- Retrieval quality and reader reasoning must be measured separately. All readers therefore receive the same context, while retrieval recall and answer EM/F1 are reported independently.
- The three useful comparisons are shared-context model quality, controlled retrieval-noise robustness, and frozen-versus-adapted HRM performance.

## 3. Preparation completed

- Defined the project goal, three falsifiable hypotheses, scope, metrics, controls, and stop/go conditions.
- Chosen Text RAG as the primary track and gated Code RAG as an extension.
- Designed a phased roadmap from environment reproduction to retrieval, frozen HRM evaluation, robustness, and lightweight adaptation.
- Prepared a reproducible repository layout for code, tests, data stages, experiment configs, artifacts, aggregate results, literature, and reports.
- Created agent collaboration rules, a team task board, an initial annotated bibliography, and result-recording templates.

## 4. Individual contributions

Use concrete outcomes and evidence; do not list attendance or vague participation.

- **abdurrahman — [role/focus]:** during project preparation, completed [artifact, review, decision, or experiment]. Evidence: [path, commit, or experiment ID]. Next: [owned task and due date].
- **alexander_dikov — repository preparation and research planning:** reviewed the project proposal and curator presentation; designed the research-ready repository structure; formalized the goal, hypotheses, Text-RAG-first scope, evaluation contract, and phased roadmap; and prepared the experiment/result templates, LLM-agent rules, literature map, team board, article workspace, and pre-defense source text. Evidence: `README.md`, `docs/project.md`, `planning/`, `references/`, `experiments/`, `results/`, and `reports/`. Next: collect verified contributions from the other team members, finalize the preparation report, and help assign owners for the environment and first smoke experiment.
- **skudarmaria — [role/focus]:** during project preparation, completed [artifact, review, decision, or experiment]. Evidence: [path, commit, or experiment ID]. Next: [owned task and due date].
- **advanced_12iq — [role/focus]:** during project preparation, completed [artifact, review, decision, or experiment]. Evidence: [path, commit, or experiment ID]. Next: [owned task and due date].

## 5. Planned experiments

1. Validate official EM/F1 scoring and dataset loading on a deterministic HotpotQA slice.
2. Measure the gap between gold evidence and BM25 retrieval to separate retrieval errors from reader errors.
3. Compare a frozen HRM and a compute-feasible CoT baseline using exactly the same contexts.
4. Inject 0, 1, 2, and 4 seeded irrelevant passages and compare degradation.
5. Adapt only the HRM low-level module, then measure both QA improvement and retained performance on an unchanged reasoning task.
6. Scale to full validation data and multiple seeds only after the small-slice pipeline is reproducible.

## 6. Next steps

In the next work cycle, the team will assign owners, pin the HRM implementation and environment, reproduce one reference task, implement dataset and metric smoke tests, and lock the first baseline configuration. The immediate proof of progress will be one reproducible command that produces a validated result row on a small dataset slice.

## 7. Readiness statement

The goal and hypotheses are explicit, the first experiments isolate meaningful causal comparisons, and the plan includes feasibility gates for the two largest risks: HRM context ingestion and compute. Work is divided into independently reviewable tasks with owners, evidence fields, and acceptance checks.
