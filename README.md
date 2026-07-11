# HierarchicalRAG

Research on whether Hierarchical Reasoning Models (HRMs) can act as compact reasoning engines in Retrieval-Augmented Generation (RAG). Retrieval supplies external knowledge; the HRM is evaluated on reasoning over the same evidence provided to comparable chain-of-thought baselines.

The first milestone targets text RAG on multi-hop question answering. Repository-level code repair remains a gated extension after the text evaluation pipeline is working.

## Start here

- [Project brief](docs/project.md): goal, hypotheses, scope, and evaluation contract.
- [Roadmap](planning/roadmap.md): phased, feasible research plan.
- [Team board](planning/board.md): tasks, owners, and contribution fields.
- [Literature map](references/README.md): papers and reading priorities.
- [Pre-defense content](reports/pre-defense-content.md): concise source text for the future presentation.
- [Agent rules](AGENTS.md): instructions for LLM-assisted work.

## Repository layout

```text
agents/       Detailed collaboration rules for LLM agents
artifacts/    Local checkpoints and retrieval indexes (ignored by Git)
data/         Raw, intermediate, and processed datasets (ignored by Git)
docs/         Stable project documentation
experiments/  Run definitions and configuration files
notebooks/    Exploratory analysis only
planning/     Roadmap and team board
references/   Annotated literature and bibliography
reports/      Pre-defense, paper, tables, and figures
results/      Ignored run outputs plus tracked aggregate summaries
src/          Reusable implementation
tests/        Automated tests and smoke checks
```

Before starting work, claim a task in `planning/board.md`. Do not commit datasets, model weights, indexes, secrets, or unreviewed generated results.
