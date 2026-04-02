<#
Summary: Run the full local validation suite for the RewardLab worktree.
Created: 2026-04-02
Last Updated: 2026-04-02
#>

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Expected local interpreter at $python. Create the worktree-local .venv first."
}

Push-Location $repoRoot
try {
    & $python tools\quality\validate_contracts.py
    & $python tools\quality\check_headers.py src tools tests
    & $python -m ruff check src\rewardlab tests tools\quality
    & $python -m mypy src
    & $python -m pytest tests\unit tests\contract tests\integration tests\e2e -q
}
finally {
    Pop-Location
}
