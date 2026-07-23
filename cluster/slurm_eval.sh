#!/usr/bin/env bash
#SBATCH --job-name=hierarchical-rag-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the clean repository checkout}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to a pinned SIF or OCI image}"

if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch cluster/slurm_eval.sh <command> [args...]" >&2
  exit 2
fi

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}"

binds=(
  "${PROJECT_ROOT}:/workspace:ro"
  "${DATA_ROOT}:/workspace/data"
  "${RESULTS_ROOT}:/workspace/results"
)
bind_argument=$(IFS=,; echo "${binds[*]}")

apptainer exec --cleanenv \
  --bind "${bind_argument}" \
  --pwd /workspace \
  "${CONTAINER_IMAGE}" \
  python -m hierarchical_rag.check_environment \
  --profile eval-cpu \
  --repository-root /workspace

apptainer exec --cleanenv \
  --bind "${bind_argument}" \
  --pwd /workspace \
  "${CONTAINER_IMAGE}" \
  "$@"
