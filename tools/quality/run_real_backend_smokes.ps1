<#
Summary: Run opt-in real backend smoke tests from the local RewardLab worktree.
Created: 2026-04-02
Last Updated: 2026-04-02
#>

param(
    [switch]$Gymnasium
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$tmpRoot = Join-Path $repoRoot ".tmp"

if (-not (Test-Path $python)) {
    throw "Expected local interpreter at $python. Create the worktree-local .venv first."
}

if (-not $Gymnasium) {
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

}
finally {
    Pop-Location
}
