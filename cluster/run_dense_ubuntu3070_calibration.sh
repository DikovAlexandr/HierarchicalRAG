#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

revision="$(git rev-parse HEAD)"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error=invalid_git_revision" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "error=tracked_worktree_must_be_clean" >&2
  exit 2
fi

sample="data/interim/hotpotqa/qwen3-embedding-calibration-s8448.jsonl"
sample_manifest="data/interim/hotpotqa/qwen3-embedding-calibration-s8448.manifest.json"
printf '%s  %s\n' \
  "70ad85afdb16e6ec8e2f9aaa2649f9156c61b6e6f6901c89d9ec46c9b38200a1" \
  "$sample" | sha256sum --check --strict
printf '%s  %s\n' \
  "fa0f52fcd872bdd900abe8c32f90f410a2ab05669834ab131336280bf5c53294" \
  "$sample_manifest" | sha256sum --check --strict

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)"
if [[ "$gpu_name" != *"RTX 3070"* ]]; then
  echo "error=unexpected_gpu observed=$gpu_name" >&2
  exit 2
fi

host_memory_bytes="$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))"
minimum_host_memory_bytes=$((7 * 1024 * 1024 * 1024))
if (( host_memory_bytes < minimum_host_memory_bytes )); then
  echo "error=insufficient_host_memory observed_bytes=$host_memory_bytes" >&2
  exit 2
fi

available_bytes="$(df --output=avail -B1 "$repo" | tail -n 1 | tr -d ' ')"
minimum_free_bytes=$((20 * 1024 * 1024 * 1024))
if (( available_bytes < minimum_free_bytes )); then
  echo "error=insufficient_disk available_bytes=$available_bytes" >&2
  exit 2
fi

if docker info >/dev/null 2>&1; then
  docker_cmd=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  docker_cmd=(sudo -n docker)
else
  echo "error=docker_unavailable_or_not_running" >&2
  exit 2
fi

image="hierarchical-rag-dense-gpu:${revision:0:7}"
echo "stage=image_build_start revision=$revision"
"${docker_cmd[@]}" build \
  --build-arg "SOURCE_REVISION=$revision" \
  --file containers/dense-gpu.Dockerfile \
  --tag "$image" \
  .
echo "stage=image_build_complete image=$image"

cache_dir="${HOME}/.cache/hierarchical-rag/huggingface"
mkdir -p "$cache_dir"
run_id="p2-qwen3-embedding-ubuntu3070-calibration-v1"
run_dir="results/runs/${run_id}"
terminal_dir="results/runs/_terminal"
archive_dir="results/runs/_archives"
mkdir -p "$terminal_dir" "$archive_dir"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
terminal_log="${terminal_dir}/${run_id}-${timestamp}.log"

echo "stage=calibration_start gpu=$gpu_name host_memory_bytes=$host_memory_bytes available_bytes=$available_bytes"
set +e
"${docker_cmd[@]}" run --rm --init --gpus all \
  --user "$(id -u):$(id -g)" \
  --env "HOME=/tmp" \
  --env "HF_HOME=/hf-cache" \
  --env "HIERARCHICAL_RAG_SOURCE_REVISION=$revision" \
  --env "HF_HUB_DISABLE_TELEMETRY=1" \
  --env "PYTHONHASHSEED=0" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --volume "${repo}:/workspace" \
  --volume "${cache_dir}:/hf-cache" \
  --workdir /workspace \
  "$image" \
  hierarchical_rag.run_dense_calibration \
  --config experiments/configs/p2-qwen3-embedding-ubuntu3070-calibration-v1.yaml \
  2>&1 | tee "$terminal_log"
runner_status="${PIPESTATUS[0]}"
set -e

archive="${archive_dir}/${run_id}-artifacts-${timestamp}.tar.gz"
archive_inputs=("$terminal_log")
if [[ -d "$run_dir" ]]; then
  archive_inputs+=("$run_dir")
fi
tar -czf "$archive" "${archive_inputs[@]}"
archive_sha256="$(sha256sum "$archive" | awk '{print $1}')"

echo "runner_exit_status=$runner_status"
echo "artifact_archive=$archive"
echo "artifact_sha256=$archive_sha256"
if (( runner_status != 0 )); then
  echo "error=calibration_failed_preserve_artifact" >&2
  exit "$runner_status"
fi
echo "stage=calibration_complete"
