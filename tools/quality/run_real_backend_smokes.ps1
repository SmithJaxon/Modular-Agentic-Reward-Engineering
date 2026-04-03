<#
Summary: Run opt-in real backend smoke tests from the local RewardLab worktree.
Created: 2026-04-02
Last Updated: 2026-04-02
#>

param(
    [switch]$Gymnasium,
    [switch]$IsaacGym,
    [string]$IsaacEnvId,
    [string]$IsaacFactory,
    [string]$IsaacValidator
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$tmpRoot = Join-Path $repoRoot ".tmp"

if (-not (Test-Path $python)) {
    throw "Expected local interpreter at $python. Create the worktree-local .venv first."
}

if (-not $Gymnasium -and -not $IsaacGym) {
    $Gymnasium = $true
}

New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null
$resolvedTmpRoot = (Resolve-Path $tmpRoot).Path
$env:TMP = $resolvedTmpRoot
$env:TEMP = $resolvedTmpRoot
$env:TMPDIR = $resolvedTmpRoot

function Invoke-CheckedPython {
    param(
        [string]$Label,
        [string[]]$ArgumentList
    )

    Write-Host "==> $Label"
    & $python @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    if ($Gymnasium) {
        $gymBaseTemp = Join-Path $tmpRoot "pytest-real-gymnasium"
        Remove-Item -Recurse -Force $gymBaseTemp -ErrorAction SilentlyContinue
        Invoke-CheckedPython "Run real Gymnasium smoke" @(
            "-m",
            "pytest",
            "tests\e2e\test_gymnasium_actual_experiment.py",
            "-q",
            "--run-real-gymnasium",
            "--basetemp",
            $gymBaseTemp,
            "-p",
            "no:cacheprovider"
        )
    }

    if ($IsaacGym) {
        if (-not $IsaacFactory) {
            $IsaacFactory = $env:REWARDLAB_ISAAC_ENV_FACTORY
        }
        if (-not $IsaacValidator) {
            $IsaacValidator = $env:REWARDLAB_ISAAC_ENV_VALIDATOR
        }
        if (-not $IsaacEnvId) {
            $IsaacEnvId = $env:REWARDLAB_TEST_ISAAC_ENV_ID
        }
        if (-not $IsaacFactory) {
            throw "Set -IsaacFactory or REWARDLAB_ISAAC_ENV_FACTORY before Isaac smoke."
        }
        if (-not $IsaacEnvId) {
            throw "Set -IsaacEnvId or REWARDLAB_TEST_ISAAC_ENV_ID before Isaac smoke."
        }

        $env:REWARDLAB_ISAAC_ENV_FACTORY = $IsaacFactory
        $env:REWARDLAB_TEST_ISAAC_ENV_ID = $IsaacEnvId
        if ($IsaacValidator) {
            $env:REWARDLAB_ISAAC_ENV_VALIDATOR = $IsaacValidator
        }

        $isaacBaseTemp = Join-Path $tmpRoot "pytest-real-isaacgym"
        Remove-Item -Recurse -Force $isaacBaseTemp -ErrorAction SilentlyContinue
        Invoke-CheckedPython "Run real Isaac smoke" @(
            "-m",
            "pytest",
            "tests\e2e\test_isaac_actual_experiment.py",
            "-q",
            "--run-real-isaacgym",
            "--basetemp",
            $isaacBaseTemp,
            "-p",
            "no:cacheprovider"
        )
    }
}
finally {
    Pop-Location
}
