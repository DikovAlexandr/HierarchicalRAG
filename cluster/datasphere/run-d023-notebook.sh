#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

ALL_CONFIGS=(
  experiments/configs/p1-lfm2.5-thinking-gold-train-budget-4096-v1.yaml
  experiments/configs/p1-qwen3.5-2b-thinking-gold-train-budget-4096-v1.yaml
  experiments/configs/p1-lfm2.5-thinking-gold-train-budget-8192-v1.yaml
  experiments/configs/p1-qwen3.5-2b-thinking-gold-train-budget-8192-v1.yaml
)

usage() {
  echo "Usage: bash cluster/datasphere/run-d023-notebook.sh [all|<supported-config>]" >&2
}

experiment_id_for_config() {
  case "$1" in
    experiments/configs/p1-lfm2.5-thinking-gold-train-budget-4096-v1.yaml)
      echo "p1-lfm2.5-thinking-gold-train-budget-4096-v1"
      ;;
    experiments/configs/p1-qwen3.5-2b-thinking-gold-train-budget-4096-v1.yaml)
      echo "p1-qwen3.5-2b-thinking-gold-train-budget-4096-v1"
      ;;
    experiments/configs/p1-lfm2.5-thinking-gold-train-budget-8192-v1.yaml)
      echo "p1-lfm2.5-thinking-gold-train-budget-8192-v1"
      ;;
    experiments/configs/p1-qwen3.5-2b-thinking-gold-train-budget-8192-v1.yaml)
      echo "p1-qwen3.5-2b-thinking-gold-train-budget-8192-v1"
      ;;
    *)
      return 2
      ;;
  esac
}

if [[ "$#" -gt 1 ]]; then
  usage
  exit 2
fi

if [[ "$#" -eq 0 || "${1:-}" == "all" ]]; then
  CONFIGS=("${ALL_CONFIGS[@]}")
else
  if ! experiment_id_for_config "$1" >/dev/null; then
    echo "Unsupported D023 config: $1" >&2
    usage
    exit 2
  fi
  CONFIGS=("$1")
fi

if [[ ! -f SOURCE_REVISION.txt ]]; then
  echo "SOURCE_REVISION.txt is missing from the prepared bundle" >&2
  exit 2
fi

SOURCE_REVISION="$(tr -d '\r\n' < SOURCE_REVISION.txt)"
if [[ ! "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION.txt does not contain a full Git SHA" >&2
  exit 2
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing versioned config: ${config}" >&2
    exit 2
  fi
  experiment_id="$(experiment_id_for_config "${config}")"
  run_dir="results/runs/${experiment_id}"
  if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty run directory: ${run_dir}" >&2
    exit 2
  fi
done

LOCK_DIR="${ROOT}/.d023-run-lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another D023 runner appears to be active: ${LOCK_DIR}" >&2
  exit 2
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

LOG_DIR="notebook-logs"
SERIES_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
BOOTSTRAP_LOG="${LOG_DIR}/d023-bootstrap.${SERIES_UTC}.terminal.log"
mkdir -p "${LOG_DIR}"

BOOTSTRAP_PYTHON=""
REJECTED_PYTHONS=()
for candidate in \
  "${D023_BOOTSTRAP_PYTHON:-}" \
  /usr/local/bin/python3 \
  /usr/bin/python3 \
  /kernel/bin/python \
  /kernel/bin/python3 \
  python3 \
  python
do
  if [[ -n "${candidate}" ]] && resolved="$(command -v "${candidate}" 2>/dev/null)"; then
    if "${resolved}" -c \
      'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' \
      >/dev/null 2>&1
    then
      BOOTSTRAP_PYTHON="${resolved}"
      break
    fi
    REJECTED_PYTHONS+=("${resolved}")
  fi
done
if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
  echo "No runnable CPython 3.10 interpreter found; set D023_BOOTSTRAP_PYTHON explicitly" >&2
  if [[ "${#REJECTED_PYTHONS[@]}" -gt 0 ]]; then
    printf 'Rejected Python candidates: %s\n' "${REJECTED_PYTHONS[*]}" >&2
  fi
  exit 127
fi

package_dir_is_valid() {
  local package_dir="$1"
  [[ -f "${package_dir}/.install-complete" ]] || return 1
  PYTHONPATH="${package_dir}" "${BOOTSTRAP_PYTHON}" -c \
    'import torch, transformers; assert torch.__version__ == "2.5.1+cu121"; assert transformers.__version__ == "5.9.0"' \
    >/dev/null 2>&1
}

bootstrap_environment() {
  echo "stage=bootstrap_python_ready executable=${BOOTSTRAP_PYTHON}"
  if [[ "${#REJECTED_PYTHONS[@]}" -gt 0 ]]; then
    echo "stage=bootstrap_python_candidates_rejected candidates=${REJECTED_PYTHONS[*]}"
  fi

  if package_dir_is_valid "${PACKAGE_DIR}"; then
    echo "stage=package_environment_ready source=existing_complete_install path=${PACKAGE_DIR}"
  else
    if [[ -d "${PACKAGE_DIR}" ]]; then
      echo "Incomplete or invalid package directory exists: ${PACKAGE_DIR}" >&2
      exit 2
    fi
    PACKAGE_TMP="${PACKAGE_DIR}.tmp-${SERIES_UTC}"
    mkdir -p "${PACKAGE_TMP}"

    echo "stage=torch_install_start version=2.5.1+cu121"
    "${BOOTSTRAP_PYTHON}" -m pip install --disable-pip-version-check --no-cache-dir \
      --target "${PACKAGE_TMP}" --no-warn-conflicts \
      --index-url https://download.pytorch.org/whl/cu121 \
      "torch==2.5.1"
    echo "stage=torch_install_complete"

    echo "stage=locked_dependencies_install_start"
    "${BOOTSTRAP_PYTHON}" -m pip install --disable-pip-version-check --no-cache-dir \
      --target "${PACKAGE_TMP}" --upgrade --no-warn-conflicts --no-deps \
      --progress-bar on --require-hashes \
      -r environments/hrm-text-gpu-py310.lock
    echo "stage=locked_dependencies_install_complete"

    PYTHONPATH="${PACKAGE_TMP}" "${BOOTSTRAP_PYTHON}" - <<'PY'
import torch
import transformers

if torch.__version__ != "2.5.1+cu121":
    raise RuntimeError(f"Expected torch 2.5.1+cu121, found {torch.__version__}")
if transformers.__version__ != "5.9.0":
    raise RuntimeError(f"Expected transformers 5.9.0, found {transformers.__version__}")
PY

    mv "${PACKAGE_TMP}" "${PACKAGE_DIR}"
    touch "${PACKAGE_DIR}/.install-complete"
    echo "stage=package_environment_published path=${PACKAGE_DIR}"
  fi

  PYTHONPATH="${PACKAGE_DIR}" "${BOOTSTRAP_PYTHON}" - <<'PY'
import torch
import transformers

if torch.__version__ != "2.5.1+cu121":
    raise RuntimeError(f"Expected torch 2.5.1+cu121, found {torch.__version__}")
if transformers.__version__ != "5.9.0":
    raise RuntimeError(f"Expected transformers 5.9.0, found {transformers.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
if "A100" not in name:
    raise RuntimeError(f"Expected an A100 GPU, found {name}")
print(
    f"environment_ok torch={torch.__version__} "
    f"transformers={transformers.__version__} gpu={name}"
)
PY
}

PACKAGE_DIR=""
for candidate in "${ROOT}/.d023-python-packages" "${ROOT}/.d017-python-packages"; do
  if package_dir_is_valid "${candidate}"; then
    PACKAGE_DIR="${candidate}"
    break
  fi
done
if [[ -z "${PACKAGE_DIR}" ]]; then
  PACKAGE_DIR="${ROOT}/.d023-python-packages"
fi

set +e
(set -e; bootstrap_environment) 2>&1 | tee "${BOOTSTRAP_LOG}"
BOOTSTRAP_STATUS=${PIPESTATUS[0]}
set -e
if [[ "${BOOTSTRAP_STATUS}" -ne 0 ]]; then
  BOOTSTRAP_ARCHIVE="d023-bootstrap-failure-${SERIES_UTC}.tar.gz"
  tar -czf "${BOOTSTRAP_ARCHIVE}" "${BOOTSTRAP_LOG}"
  echo "bootstrap_exit_status=${BOOTSTRAP_STATUS}"
  echo "artifact_archive=${BOOTSTRAP_ARCHIVE}"
  exit "${BOOTSTRAP_STATUS}"
fi

export HIERARCHICAL_RAG_SOURCE_REVISION="${SOURCE_REVISION}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

SERIES_STARTED="$(date +%s)"
OVERALL_STATUS=0
COMPLETED=0
SERIES_ITEMS=("${BOOTSTRAP_LOG}")
RESULT_ROWS=()
TOTAL="${#CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
  experiment_id="$(experiment_id_for_config "${config}")"
  run_dir="results/runs/${experiment_id}"
  attempt_utc="$(date -u +%Y%m%dT%H%M%SZ)"
  log_file="${LOG_DIR}/${experiment_id}.${attempt_utc}.terminal.log"
  archive="${experiment_id}-artifacts-${attempt_utc}.tar.gz"
  run_number=$((COMPLETED + 1))
  filled=$((24 * COMPLETED / TOTAL))
  empty=$((24 - filled))
  bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  echo "series_progress=[${bar}] runs=${COMPLETED}/${TOTAL} status=run_start run=${run_number} experiment_id=${experiment_id}"

  set +e
  (
    echo "stage=model_runner_start experiment_id=${experiment_id}"
    PYTHONPATH="${PACKAGE_DIR}:src" "${BOOTSTRAP_PYTHON}" \
      -m hierarchical_rag.run_native_thinking_smoke \
      --config "${config}"
    status=$?
    if [[ "${status}" -eq 0 ]]; then
      echo "stage=model_runner_complete experiment_id=${experiment_id}"
    else
      echo "stage=model_runner_failed experiment_id=${experiment_id} exit_status=${status}"
    fi
    exit "${status}"
  ) 2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e

  archive_items=("${BOOTSTRAP_LOG}" "${log_file}")
  if [[ -d "${run_dir}" ]]; then
    archive_items+=("${run_dir}")
  fi
  tar -czf "${archive}" "${archive_items[@]}"
  SERIES_ITEMS+=("${log_file}" "${archive}")
  if [[ -d "${run_dir}" ]]; then
    SERIES_ITEMS+=("${run_dir}")
  fi
  RESULT_ROWS+=("${experiment_id}\t${status}\t${archive}")
  if [[ "${status}" -ne 0 ]]; then
    OVERALL_STATUS=1
  fi

  COMPLETED=$((COMPLETED + 1))
  elapsed=$(( $(date +%s) - SERIES_STARTED ))
  remaining=$((TOTAL - COMPLETED))
  eta=0
  if [[ "${COMPLETED}" -gt 0 ]]; then
    eta=$((elapsed * remaining / COMPLETED))
  fi
  filled=$((24 * COMPLETED / TOTAL))
  empty=$((24 - filled))
  bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  echo "series_progress=[${bar}] runs=${COMPLETED}/${TOTAL} status=run_complete experiment_id=${experiment_id} exit_status=${status} elapsed_seconds=${elapsed} estimated_remaining_seconds=${eta}"
  echo "artifact_archive=${archive}"
done

SUMMARY_FILE="notebook-logs/d023-series.${SERIES_UTC}.summary.tsv"
{
  printf 'experiment_id\texit_status\tartifact_archive\n'
  for row in "${RESULT_ROWS[@]}"; do
    printf '%b\n' "${row}"
  done
} > "${SUMMARY_FILE}"
SERIES_ITEMS+=("${SUMMARY_FILE}")

SERIES_ARCHIVE="d023-expanded-output-series-artifacts-${SERIES_UTC}.tar.gz"
tar -czf "${SERIES_ARCHIVE}" "${SERIES_ITEMS[@]}"

echo "series_exit_status=${OVERALL_STATUS}"
echo "series_archive=${SERIES_ARCHIVE}"
echo "summary_file=${SUMMARY_FILE}"
exit "${OVERALL_STATUS}"
