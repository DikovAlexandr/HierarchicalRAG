# HierarchicalRAG

Research on whether Hierarchical Reasoning Models (HRMs) can act as compact reasoning engines in Retrieval-Augmented Generation (RAG). Retrieval supplies external knowledge; the HRM is evaluated on reasoning over the same evidence provided to comparable chain-of-thought baselines.

The first milestone targets text RAG on multi-hop question answering. Repository-level code repair remains a gated extension after the text evaluation pipeline is working.

## Start here

- [Project brief](docs/project.md): goal, hypotheses, scope, and evaluation contract.
- [Research protocol](.agents/README.md): mandatory rules for honest and reproducible experiments.
- [Decision log](docs/decisions.md): motivated protocol and scope decisions.
- [Research progress](docs/progress.md): verified evidence, stage readiness, and next gates.
- [Literature map](references/README.md): papers and reading priorities.
- [Agent rules](AGENTS.md): instructions for LLM-assisted work.
- [Cluster execution](cluster/README.md): pinned Linux container and Slurm/Apptainer contract.

## Repository layout

```text
.agents/      Binding research and agent protocol
artifacts/    Local checkpoints and retrieval indexes (ignored by Git)
data/         Local datasets and preprocessing records (data ignored by Git)
docs/         Project contract and motivated decisions
experiments/  Run definitions and configuration files
references/   Annotated literature and bibliography
reports/      Paper, tables, figures, and final reporting
results/      Ignored run outputs plus tracked aggregate summaries
src/          Reusable implementation
tests/        Automated tests and smoke checks
cluster/      Slurm/Apptainer launchers and cluster execution contract
containers/   Pinned container definitions; CPU evaluation now, GPU after HRM selection
environments/ Exact package locks used by containers
```

Before starting work, read the project contract and research protocol. Do not report a result unless its configuration, source revisions, seeds, raw outputs, environment, uncertainty, and statistical comparison are recoverable. Do not commit datasets, model weights, indexes, secrets, or raw run outputs.
