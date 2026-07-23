# Cluster execution

The cluster contract keeps code, data, model, environment, and output revisions independent and explicit. A run is valid only when the container revision matches the clean repository checkout and all data/model checksums match the versioned configuration.

## Environment split

- `eval-cpu` is the current Linux environment for data preparation, SQLite FTS5 retrieval, metrics, statistics, and CPU smoke tests.
- A GPU/HRM image is intentionally not defined yet. It must pin the selected HRM repository/checkpoint, CUDA, PyTorch, FlashAttention, compiler, and GPU architecture after the HRM feasibility decision. Do not use the CPU image as evidence that HRM itself is reproducible.

The CPU image uses Linux/amd64 Python 3.11.11 and an architecture-specific immutable base-image digest. Python packages are pinned in `environments/eval-cpu.lock`. The build embeds the exact project commit in `HIERARCHICAL_RAG_IMAGE_REVISION`.

## Build and verify

Run from a clean checkout:

```bash
REVISION=$(git rev-parse HEAD)
docker build --platform linux/amd64 \
  --build-arg SOURCE_REVISION="${REVISION}" \
  --file containers/eval-cpu.Dockerfile \
  --tag "hierarchical-rag-eval:${REVISION}" .

docker run --rm \
  --volume "${PWD}:/workspace:ro" \
  "hierarchical-rag-eval:${REVISION}" \
  python -m hierarchical_rag.check_environment \
  --profile eval-cpu --repository-root /workspace
```

The environment check fails if Linux, Python, FTS5, package versions, embedded image revision, or mounted repository revision differ. Push the image to an approved registry by digest or convert it to SIF; record the OCI digest or SIF SHA256 in every cluster run.

## Slurm and Apptainer

`cluster/slurm_eval.sh` mounts the repository read-only and mounts only `data/` and `results/` as writable locations. It runs the environment check before the requested command.

```bash
export PROJECT_ROOT=/shared/projects/HierarchicalRAG
export DATA_ROOT=/shared/datasets/hierarchical-rag
export RESULTS_ROOT=/shared/results/hierarchical-rag
export CONTAINER_IMAGE=/shared/containers/hierarchical-rag-eval-<commit>.sif

sbatch --output="${RESULTS_ROOT}/slurm-%j.out" \
  cluster/slurm_eval.sh \
  python -m hierarchical_rag.run_e0 \
  --config experiments/configs/e0-hotpotqa-metrics-v1.yaml
```

Cluster-specific partition, account, QoS, wall-time, CPU, RAM, and GPU requests must be supplied by the operator or scheduler profile and recorded with the run. Never place secrets, datasets, indexes, or model weights inside Git or the container build context.

## Fullwiki storage

The pinned HotpotQA introduction archive is 1,553,565,403 bytes with official MD5 `01edf64cd120ecc03a2745352779514c`. Keep it in the writable data mount and build the SQLite index in the ignored `artifacts/indexes/` or a cluster scratch path that is copied to durable storage with its manifest and checksum.
