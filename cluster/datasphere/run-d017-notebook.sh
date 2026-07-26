#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

CONFIG="${1:-experiments/configs/p1-lfm2.5-thinking-gold-train-smoke-v2.yaml}"
case "${CONFIG}" in
  experiments/configs/p1-lfm2.5-thinking-gold-train-smoke-v2.yaml)
    EXPERIMENT_ID="p1-lfm2.5-thinking-gold-train-smoke-v2"
    ;;
  experiments/configs/p1-qwen3.5-0.8b-thinking-gold-train-smoke-v4.yaml)
    EXPERIMENT_ID="p1-qwen3.5-0.8b-thinking-gold-train-smoke-v4"
    ;;
  experiments/configs/p1-qwen3.5-2b-thinking-gold-train-smoke-v4.yaml)
    EXPERIMENT_ID="p1-qwen3.5-2b-thinking-gold-train-smoke-v4"
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
LOG_FILE="${LOG_DIR}/${EXPERIMENT_ID}.terminal.log"
ARCHIVE="${EXPERIMENT_ID}-artifacts.tar.gz"
mkdir -p "${LOG_DIR}"

if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2
  exit 2
fi

set +e
(
  set -e
  python -m pip install --disable-pip-version-check --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1"
  python -m pip install --disable-pip-version-check --no-cache-dir --no-deps \
    --progress-bar off --require-hashes -r environments/hrm-text-gpu.lock

  python - <<'PY'
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
  PYTHONPATH=src python -m hierarchical_rag.run_native_thinking_smoke \
    --config "${CONFIG}"
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
