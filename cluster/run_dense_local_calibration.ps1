[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitRepo = $repo.Replace("\", "/")
$revision = (& git -c "safe.directory=$gitRepo" -C $repo rev-parse HEAD).Trim()
$dirty = & git -c "safe.directory=$gitRepo" -C $repo status --porcelain --untracked-files=no
if ($dirty) {
    throw "Tracked worktree changes must be committed before calibration."
}

$image = "hierarchical-rag-dense-gpu:$($revision.Substring(0, 7))"
docker build `
    --build-arg "SOURCE_REVISION=$revision" `
    --file (Join-Path $repo "containers/dense-gpu.Dockerfile") `
    --tag $image `
    $repo
if ($LASTEXITCODE -ne 0) {
    throw "Dense GPU image build failed."
}

docker run --rm --gpus all `
    --env "HIERARCHICAL_RAG_SOURCE_REVISION=$revision" `
    --env "HF_HUB_DISABLE_TELEMETRY=1" `
    --env "PYTHONHASHSEED=0" `
    --env "TOKENIZERS_PARALLELISM=false" `
    --volume "${repo}:/workspace" `
    --volume "hierarchical-rag-hf-cache:/root/.cache/huggingface" `
    --workdir /workspace `
    $image `
    --config experiments/configs/p2-qwen3-embedding-local-calibration-v1.yaml
if ($LASTEXITCODE -ne 0) {
    throw "Local dense calibration failed. Preserve results/runs before retrying."
}
