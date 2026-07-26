#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash cluster/datasphere/run-d017-notebook.sh <supported-config>" >&2
  exit 2
fi

CONFIG="$1"
case "${CONFIG}" in
  experiments/configs/p1-lfm2.5-thinking-gold-train-smoke-v3.yaml)
    echo "Refusing to repeat the completed LFM2.5 D017 attempt (D020)" >&2
    exit 2
    ;;
  experiments/configs/p1-qwen3.5-0.8b-thinking-gold-train-smoke-v5.yaml)
    echo "Refusing to repeat the completed Qwen3.5-0.8B D017 attempt (D021)" >&2
    exit 2
    ;;
  experiments/configs/p1-qwen3.5-2b-thinking-gold-train-smoke-v5.yaml)
    EXPERIMENT_ID="p1-qwen3.5-2b-thinking-gold-train-smoke-v5"
    ;;
  *)
    echo "Unsupported D017 config: ${CONFIG}" >&2
    exit 2
    ;;
esac

if [[ ! -f SOURCE_REVISION.txt ]]; then
  echo "SOURCE_REVISION.txt is missing from the prepared bundle" >&2
  exit 2
fi

SOURCE_REVISION="$(tr -d '\r\n' < SOURCE_REVISION.txt)"
if [[ ! "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION.txt does not contain a full Git SHA" >&2
  exit 2
fi

LOG_DIR="notebook-logs"
RUN_DIR="results/runs/${EXPERIMENT_ID}"
ATTEMPT_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_ID}.${ATTEMPT_UTC}.terminal.log"
ARCHIVE="${EXPERIMENT_ID}-artifacts-${ATTEMPT_UTC}.tar.gz"
mkdir -p "${LOG_DIR}"

if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2
  exit 2
fi

LOCK_DIR="${ROOT}/.d017-run-lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another D017 runner appears to be active: ${LOCK_DIR}" >&2
  exit 2
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

set +e
(
  set -e

  BOOTSTRAP_PYTHON=""
  for candidate in \
    "${D017_BOOTSTRAP_PYTHON:-}" \
    /kernel/bin/python \
    /kernel/bin/python3 \
    python3 \
    python
  do
    if [[ -n "${candidate}" ]] && command -v "${candidate}" >/dev/null 2>&1; then
      BOOTSTRAP_PYTHON="$(command -v "${candidate}")"
      break
    fi
  done
  if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
    echo "No Python interpreter found; set D017_BOOTSTRAP_PYTHON explicitly" >&2
    exit 127
  fi
  echo "stage=bootstrap_python_ready executable=${BOOTSTRAP_PYTHON}"

  PACKAGE_DIR="${ROOT}/.d017-python-packages"
  PACKAGE_MARKER="${PACKAGE_DIR}/.install-complete"
  if [[ -d "${PACKAGE_DIR}" && ! -f "${PACKAGE_MARKER}" ]]; then
    if PYTHONPATH="${PACKAGE_DIR}" "${BOOTSTRAP_PYTHON}" -c \
      'import torch, transformers; assert torch.__version__ == "2.5.1+cu121"; assert transformers.__version__ == "5.9.0"' \
      >/dev/null 2>&1
    then
      touch "${PACKAGE_MARKER}"
    else
      echo "Incomplete package directory exists: ${PACKAGE_DIR}" >&2
      exit 2
    fi
  fi
  if [[ ! -f "${PACKAGE_MARKER}" ]]; then
    PACKAGE_TMP=""
    for candidate in "${PACKAGE_DIR}".tmp-*; do
      if [[ -d "${candidate}" ]] && { [[ -z "${PACKAGE_TMP}" ]] || [[ "${candidate}" -nt "${PACKAGE_TMP}" ]]; }; then
        PACKAGE_TMP="${candidate}"
      fi
    done
    if [[ -z "${PACKAGE_TMP}" ]]; then
      PACKAGE_TMP="${PACKAGE_DIR}.tmp-${ATTEMPT_UTC}"
      mkdir -p "${PACKAGE_TMP}"
    fi

    if PYTHONPATH="${PACKAGE_TMP}" "${BOOTSTRAP_PYTHON}" -c \
      'import torch; assert torch.__version__ == "2.5.1+cu121"' \
      >/dev/null 2>&1
    then
      echo "stage=torch_ready source=reused_partial_install"
    else
      echo "stage=torch_install_start version=2.5.1+cu121"
      "${BOOTSTRAP_PYTHON}" -m pip install --disable-pip-version-check --no-cache-dir \
        --target "${PACKAGE_TMP}" --no-warn-conflicts \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.5.1"
      echo "stage=torch_install_complete"
    fi

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
    touch "${PACKAGE_MARKER}"
    echo "stage=package_environment_published path=${PACKAGE_DIR}"
  else
    echo "stage=package_environment_ready source=existing_complete_install"
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
print(f"environment_ok torch={torch.__version__} transformers={transformers.__version__} gpu={name}")
PY

  export HIERARCHICAL_RAG_SOURCE_REVISION="${SOURCE_REVISION}"
  export PYTHONHASHSEED=0
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_DISABLE_TELEMETRY=1
  echo "stage=model_runner_start experiment_id=${EXPERIMENT_ID}"
  PYTHONPATH="${PACKAGE_DIR}:src" "${BOOTSTRAP_PYTHON}" \
    -m hierarchical_rag.run_native_thinking_smoke \
    --config "${CONFIG}"
  echo "stage=model_runner_complete experiment_id=${EXPERIMENT_ID}"
) 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

ARCHIVE_ITEMS=("${LOG_FILE}")
if [[ -d "${RUN_DIR}" ]]; then
  ARCHIVE_ITEMS+=("${RUN_DIR}")
fi
tar -czf "${ARCHIVE}" "${ARCHIVE_ITEMS[@]}"

echo "exit_status=${STATUS}"
echo "artifact_archive=${ARCHIVE}"
exit "${STATUS}"
