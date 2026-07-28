[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitRepo = $repo.Replace("\", "/")
$revision = (& git -c "safe.directory=$gitRepo" -C $repo rev-parse HEAD).Trim()
$dirty = & git -c "safe.directory=$gitRepo" -C $repo status --porcelain --untracked-files=no
if ($dirty) {
    throw "Tracked worktree changes must be committed before the full build."
}

$auditPath = Join-Path $repo "results/reviews/p2-qwen3-embedding-local-calibration-v1.audit.json"
$audit = Get-Content -Raw $auditPath | ConvertFrom-Json
if (
    $audit.audit.integrity_status -ne "passed_for_resource_projection" -or
    $audit.resource_policy.decision -ne "authorize_separate_local_full_dense_corpus_build" -or
    -not $audit.resource_policy.wall_time_gate_passed -or
    -not $audit.resource_policy.memory_gate_passed
) {
    throw "The pinned local calibration audit does not authorize the full build."
}

$requiredFreeBytes = 12884901888
$drive = (Get-Item $repo).PSDrive
if ($drive.Free -lt $requiredFreeBytes) {
    throw "At least 12 GiB of free local disk is required before a fresh build."
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

Write-Host "stage=fullwiki_local_start projected_hours_with_reserve=$($audit.resource_policy.projected_hours_with_reserve)"
docker run --rm --gpus all `
    --env "HIERARCHICAL_RAG_SOURCE_REVISION=$revision" `
    --env "HF_HUB_DISABLE_TELEMETRY=1" `
    --env "PYTHONHASHSEED=0" `
    --env "TOKENIZERS_PARALLELISM=false" `
    --volume "${repo}:/workspace" `
    --volume "hierarchical-rag-hf-cache:/root/.cache/huggingface" `
    --workdir /workspace `
    $image `
    hierarchical_rag.run_dense_fullwiki_build `
    --config experiments/configs/p2-qwen3-embedding-fullwiki-build-local-v1.yaml
if ($LASTEXITCODE -ne 0) {
    throw "Local fullwiki build stopped. Preserve the .building directory and rerun the same committed revision to resume."
}
