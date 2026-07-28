# Cluster execution

The cluster contract keeps code, data, model, environment, and output revisions independent and explicit. A run is valid only when the container revision matches the clean repository checkout and all data/model checksums match the versioned configuration.

## Environment split

- `eval-cpu` is the current Linux environment for data preparation, SQLite FTS5 retrieval, metrics, statistics, and CPU smoke tests.
- `hrm-text-gpu` uses the immutable PyTorch 2.5.1 CUDA 12.1 image digest recorded in the experiment config, the hash-pinned `environments/hrm-text-gpu.lock`, and the exact `sapientinc/HRM-Text-1B` checkpoint revision. The job must receive the clean Git SHA through `HIERARCHICAL_RAG_SOURCE_REVISION`; the runner verifies all local input hashes and records the resolved model revision and GPU before accepting a run.

The CPU image uses Linux/amd64 Python 3.11.11 and an architecture-specific immutable base-image digest. Python wheels and their SHA256 hashes are pinned in `environments/eval-cpu.lock`; source distributions are rejected. The build embeds the exact project commit in `HIERARCHICAL_RAG_IMAGE_REVISION`.

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

The environment check fails if Linux, Python, FTS5, package versions, embedded image revision, or mounted repository revision differ. Push the image to an approved registry by digest or convert it to SIF; record the OCI digest or SIF SHA256 in every cluster run. A mutable image tag alone is not a valid environment identifier.

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

## Yandex DataSphere

DataSphere job definitions are versioned under `cluster/datasphere/`. Run them only from a clean committed worktree, exporting the full source SHA without storing credentials in the repository:

```bash
export HIERARCHICAL_RAG_SOURCE_REVISION="$(git rev-parse HEAD)"
datasphere --profile <profile> project job execute \
  -p <project-id> \
  -c cluster/datasphere/p1-hrm-text-gold-train-smoke-v1.yaml
```

The job definition uploads only declared code, configuration, lock, and ignored processed inputs, then downloads all nine required run records. Preserve failed jobs and their logs; never rerun or alter the prompt because an observed answer is inconvenient.

The D018 Notebook fallback was restricted to D017 pre-container Job failures. D020–D022 preserve the completed failed LFM2.5, Qwen3.5-0.8B, and Qwen3.5-2B gates and forbid repeating them. D017 is closed; no further Notebook runner command is authorized until a mentor-approved baseline or resource-contract decision is recorded.

### Local dense-resource gate

D033 uses the same pinned Linux/PyTorch environment and frozen D028 corpus sample to decide whether the full dense corpus build can run on the local RTX 4060 without consuming the claim-bearing DataSphere budget. It is corpus-only and must not open HotpotQA questions or labels. From a clean committed checkout, run:

```powershell
pwsh -File cluster/run_dense_local_calibration.ps1
```

The runner builds `containers/dense-gpu.Dockerfile` at the exact Git revision, mounts the repository for checksummed inputs and immutable run outputs, and keeps the Hugging Face cache in a named Docker volume. The full build remains forbidden until the calibration passes both preregistered D033 gates: a 1.25x projected wall time of at most 48 hours and at most 7 GiB peak reserved GPU memory.

The historical script now rejects all three consumed attempts. Its source, configs, terminal logs, raw artifacts, and reviewed audits remain preserved for reproducibility. Notebook efficiency measurements are not directly comparable with pinned-container Job measurements.

### D023 expanded-output Notebook series

D023 is a separate train-only budget-sensitivity study and does not reopen D017 or change primary E2. Its fixed matrix contains LFM2.5-1.2B-Thinking and Qwen3.5-2B at 4,096 and 8,192 generated-token ceilings, with the same 2,048-token input ceiling and all other reader fields frozen. The series runner installs or reuses the pinned Notebook environment once, executes all four cells sequentially, emits example-level heartbeat progress, preserves failed cells, and creates both per-run and combined archives.

Create the upload archive only from a clean committed worktree. The builder checks the embedded source revision, LF line endings, dependency lock, and the two ignored prepared-data files against the checksums recorded by all four configs:

```powershell
python cluster/datasphere/prepare_d023_notebook_bundle.py
```

Upload the resulting `d023-notebook-bundle-<revision>.tar.gz` to `/home/jupyter/project`, select the A100 VM, and run one Python Notebook cell. Replace the bundle name with the exact uploaded filename:

```python
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

project_dir = Path("/home/jupyter/project")
bundle = project_dir / "d023-notebook-bundle-<revision>.tar.gz"

with tarfile.open(bundle, "r:gz") as archive:
    members = archive.getmembers()
    if not members or any(
        (path := PurePosixPath(member.name)).is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "HierarchicalRAG"
        for member in members
    ):
        raise RuntimeError("Unsafe or unexpected bundle layout")
    archive.extractall(project_dir)

completed = subprocess.run(
    ["bash", "cluster/datasphere/run-d023-notebook.sh", "all"],
    cwd=project_dir / "HierarchicalRAG",
    check=False,
)
print(f"runner_exit_status={completed.returncode}")
```

The command prints `series_archive=...` when it finishes. Download that combined archive even if `series_exit_status` is nonzero; it contains the terminal logs and every available immutable run artifact. A single preregistered cell can be executed only for technical recovery before any model output exists:

```bash
bash cluster/datasphere/run-d023-notebook.sh \
  experiments/configs/p1-lfm2.5-thinking-gold-train-budget-4096-v1.yaml
```

The runner refuses to overwrite a non-empty run directory. Do not delete or repeat a completed or model-output-producing D023 run merely because its result is unfavorable.

### D028 dense encoder calibration

D028 is independent of the closed D017/D023 reader gates. It measures only
corpus-encoding cost and never opens HotpotQA questions or labels. Build the
bundle from a clean committed worktree; the ignored 8,448-document sample and its
manifest are checksum-verified and embedded automatically:

```powershell
python cluster/datasphere/prepare_d028_notebook_bundle.py
```

Upload `d028-notebook-bundle-<revision>.tar.gz` to `/home/jupyter/project`, select
the A100 VM, and run this Python notebook cell after substituting the exact name:

```python
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

project_dir = Path("/home/jupyter/project")
bundle = project_dir / "d028-notebook-bundle-<revision>.tar.gz"

with tarfile.open(bundle, "r:gz") as archive:
    members = archive.getmembers()
    if not members or any(
        (path := PurePosixPath(member.name)).is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "HierarchicalRAG"
        for member in members
    ):
        raise RuntimeError("Unsafe or unexpected bundle layout")
    archive.extractall(project_dir)

completed = subprocess.run(
    ["bash", "cluster/datasphere/run-d028-notebook.sh"],
    cwd=project_dir / "HierarchicalRAG",
    check=False,
)
print(f"runner_exit_status={completed.returncode}")
```

Download the printed `artifact_archive` even on failure. Release the Notebook VM
immediately afterward: the calibration is not authorization for the full corpus
build, and an idle Notebook continues consuming units.

## Fullwiki storage

The pinned HotpotQA introduction archive is 1,553,565,403 bytes with official MD5 `01edf64cd120ecc03a2745352779514c`. Keep it in the writable data mount and build the SQLite index in the ignored `artifacts/indexes/` or a cluster scratch path that is copied to durable storage with its manifest and checksum.

### D031 resumable fullwiki dense build

D031 reopens the unchanged D028 corpus build after the compute budget was
expanded. This command remains corpus-only: it does not bundle validation
questions and cannot produce retrieval quality. The single upload bundle embeds
the checksum-verified 1.55 GB official corpus, so its construction and upload
take materially longer than earlier bundles:

Before uploading, resize persistent project storage to at least 25 GB; 30 GB is
recommended. The free 10 GB volume cannot contain the 5.36 GB vector matrix,
SQLite metadata, corpus, environment, and filesystem headroom together. The
runner rejects a fresh allocation unless at least 12 GiB is free and rejects a
resume unless at least 3 GiB remains. It automatically removes only a matching
zero-shard partial allocation left by the preserved D032 storage failure.

```powershell
python cluster/datasphere/prepare_d031_notebook_bundle.py
```

Upload `d031-notebook-bundle-<revision>.tar.gz` to `/home/jupyter/project`,
select `g2.1` (A100), and run the following Python cell after substituting the
exact filename:

```python
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

project_dir = Path("/home/jupyter/project")
bundle = project_dir / "d031-notebook-bundle-<revision>.tar.gz"

with tarfile.open(bundle, "r:gz") as archive:
    members = archive.getmembers()
    if not members or any(
        (path := PurePosixPath(member.name)).is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "HierarchicalRAG"
        for member in members
    ):
        raise RuntimeError("Unsafe or unexpected bundle layout")
    archive.extractall(project_dir)

# The verified corpus is now inside HierarchicalRAG; remove only the exact
# uploaded bundle to reclaim its duplicate 1.55 GB before allocating vectors.
bundle.unlink()

completed = subprocess.run(
    ["bash", "cluster/datasphere/run-d031-notebook.sh"],
    cwd=project_dir / "HierarchicalRAG",
    check=False,
)
print(f"runner_exit_status={completed.returncode}")
```

The progress bar covers all 5,233,235 documents and each committed 32,768-item
shard prints its checksum. If the attempt stops, download the printed artifact
archive, leave `artifacts/indexes/qwen3-embedding-0.6b-fullwiki-v1.building`
untouched, and rerun the same committed bundle: it resumes at the next shard.
On success, download the small artifact archive and release the VM. Keep the
large completed index on persistent project storage; validation search is a
separate command that will be frozen only after its manifest is audited.
