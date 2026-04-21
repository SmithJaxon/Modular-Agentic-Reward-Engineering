<#
Summary: Run Isaac Gym import and task-registry smoke checks in dockerized py38 env.
Created: 2026-04-17
Last Updated: 2026-04-17
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$composeFile = Join-Path $repoRoot "tools\docker\isaacgym-py38\docker-compose.yml"

docker compose -f $composeFile exec -T isaacgym-py38 /bin/bash -lc `
    "python /workspace/tools/docker/isaacgym-py38/task_registry_check.py"
if ($LASTEXITCODE -ne 0) {
    throw "Isaac Gym task registry smoke failed with exit code $LASTEXITCODE"
}
