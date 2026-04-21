<#
Summary: Build and start the isolated Isaac Gym Python 3.8 Docker stack.
Created: 2026-04-17
Last Updated: 2026-04-17
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$composeFile = Join-Path $repoRoot "tools\docker\isaacgym-py38\docker-compose.yml"

if (-not (Test-Path $composeFile)) {
    throw "Compose file not found: $composeFile"
}

Write-Host "[isaacgym] Building image..."
docker compose -f $composeFile build
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with exit code $LASTEXITCODE"
}

Write-Host "[isaacgym] Starting container..."
docker compose -f $composeFile up -d
if ($LASTEXITCODE -ne 0) {
    throw "Docker compose up failed with exit code $LASTEXITCODE"
}

Write-Host "[isaacgym] Running smoke checks..."
docker compose -f $composeFile exec -T isaacgym-py38 /bin/bash -lc `
    "python /workspace/tools/docker/isaacgym-py38/smoke_check.py"
if ($LASTEXITCODE -ne 0) {
    throw "Smoke check failed with exit code $LASTEXITCODE"
}
docker compose -f $composeFile exec -T isaacgym-py38 /bin/bash -lc `
    "python /workspace/tools/docker/isaacgym-py38/task_registry_check.py"
if ($LASTEXITCODE -ne 0) {
    throw "Task registry smoke failed with exit code $LASTEXITCODE"
}

Write-Host "[isaacgym] Ready. Container: rewardlab-isaacgym-py38"
