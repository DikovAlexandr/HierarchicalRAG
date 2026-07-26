#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CONFIG="experiments/configs/p2-qwen3-embedding-fullwiki-build-v1.yaml"
EXPERIMENT_ID="p2-qwen3-embedding-fullwiki-build-v1"
RUN_DIR="results/runs/${EXPERIMENT_ID}"
INDEX_DIR="artifacts/indexes/qwen3-embedding-0.6b-fullwiki-v1"
BUILDING_DIR="${INDEX_DIR}.building"
SERIES_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="notebook-logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_ID}.${SERIES_UTC}.terminal.log"
ARCHIVE="${EXPERIMENT_ID}-artifacts-${SERIES_UTC}.tar.gz"
mkdir -p "${LOG_DIR}"

if [[ ! -f SOURCE_REVISION.txt ]]; then
  echo "SOURCE_REVISION.txt is missing from the prepared bundle" >&2
  exit 2
fi
SOURCE_REVISION="$(tr -d '\r\n' < SOURCE_REVISION.txt)"
if [[ ! "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION.txt does not contain a full Git SHA" >&2
  exit 2
fi
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to repeat completed/non-empty run directory: ${RUN_DIR}" >&2
  exit 2
fi

BOOTSTRAP_PYTHON=""
for candidate in /usr/local/bin/python3 /usr/bin/python3 /kernel/bin/python python3 python; do
  if resolved="$(command -v "${candidate}" 2>/dev/null)" && \
    "${resolved}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' >/dev/null 2>&1
  then
    BOOTSTRAP_PYTHON="${resolved}"
    break
  fi
done
if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
  echo "No runnable CPython 3.10 interpreter found" >&2
  exit 127
fi
echo "stage=bootstrap_python_ready executable=${BOOTSTRAP_PYTHON}"

package_dir_is_valid() {
  local package_dir="$1"
  [[ -f "${package_dir}/.install-complete" ]] || return 1
  PYTHONPATH="${package_dir}" "${BOOTSTRAP_PYTHON}" -c \
    'import torch, transformers; assert torch.__version__ == "2.5.1+cu121"; assert transformers.__version__ == "5.9.0"' \
    >/dev/null 2>&1
}

PACKAGE_DIR=""
for candidate in "${ROOT}/.d023-python-packages" "${ROOT}/.d017-python-packages" "${ROOT}/.d028-python-packages" "${ROOT}/.d031-python-packages"; do
  if package_dir_is_valid "${candidate}"; then
    PACKAGE_DIR="${candidate}"
    break
  fi
done
if [[ -z "${PACKAGE_DIR}" ]]; then
  PACKAGE_DIR="${ROOT}/.d031-python-packages"
  if [[ -e "${PACKAGE_DIR}" ]]; then
    echo "Incomplete D031 package directory already exists: ${PACKAGE_DIR}" >&2
    exit 2
  fi
  PACKAGE_TMP="${PACKAGE_DIR}.tmp-${SERIES_UTC}"
  mkdir -p "${PACKAGE_TMP}"
  echo "stage=torch_install_start version=2.5.1+cu121"
  "${BOOTSTRAP_PYTHON}" -m pip install --disable-pip-version-check --no-cache-dir \
    --target "${PACKAGE_TMP}" --no-warn-conflicts \
    --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
  echo "stage=locked_dependencies_install_start"
  "${BOOTSTRAP_PYTHON}" -m pip install --disable-pip-version-check --no-cache-dir \
    --target "${PACKAGE_TMP}" --upgrade --no-warn-conflicts --no-deps \
    --progress-bar on --require-hashes -r environments/hrm-text-gpu-py310.lock
  PYTHONPATH="${PACKAGE_TMP}" "${BOOTSTRAP_PYTHON}" -c \
    'import torch, transformers; assert torch.__version__ == "2.5.1+cu121"; assert transformers.__version__ == "5.9.0"'
  mv "${PACKAGE_TMP}" "${PACKAGE_DIR}"
  touch "${PACKAGE_DIR}/.install-complete"
fi
echo "stage=package_environment_ready path=${PACKAGE_DIR}"
df -h "${ROOT}"

export HIERARCHICAL_RAG_SOURCE_REVISION="${SOURCE_REVISION}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

set +e
(
  echo "stage=dense_fullwiki_build_start experiment_id=${EXPERIMENT_ID}"
  PYTHONPATH="${PACKAGE_DIR}:src" "${BOOTSTRAP_PYTHON}" \
    -m hierarchical_rag.run_dense_fullwiki_build --config "${CONFIG}"
) 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

ARCHIVE_ITEMS=("${LOG_FILE}")
if [[ -d "${RUN_DIR}" ]]; then
  ARCHIVE_ITEMS+=("${RUN_DIR}")
fi
for item in "${INDEX_DIR}/manifest.json" "${INDEX_DIR}/build-info.json" "${INDEX_DIR}/progress.json" "${BUILDING_DIR}/build-info.json" "${BUILDING_DIR}/progress.json"; do
  if [[ -f "${item}" ]]; then
    ARCHIVE_ITEMS+=("${item}")
  fi
done
tar -czf "${ARCHIVE}" "${ARCHIVE_ITEMS[@]}"
echo "runner_exit_status=${STATUS}"
echo "artifact_archive=${ARCHIVE}"
if [[ -f "${BUILDING_DIR}/progress.json" ]]; then
  echo "resume_state=${BUILDING_DIR}/progress.json"
fi
if [[ -f "${INDEX_DIR}/manifest.json" ]]; then
  echo "completed_index_manifest=${INDEX_DIR}/manifest.json"
fi
exit "${STATUS}"
