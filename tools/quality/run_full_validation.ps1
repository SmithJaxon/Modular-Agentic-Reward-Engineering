<#
Summary: Run the full local validation suite for the RewardLab worktree.
Created: 2026-04-02
Last Updated: 2026-04-02
#>

param(
    [switch]$IncludeRealGymnasium
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$smokeScript = Join-Path $repoRoot "tools\quality\run_real_backend_smokes.ps1"
$tmpRoot = Join-Path $repoRoot ".tmp"
$pytestBaseTemp = Join-Path $tmpRoot "pytest-full-validation"

if (-not (Test-Path $python)) {
    throw "Expected local interpreter at $python. Create the worktree-local .venv first."
}
if (-not (Test-Path $smokeScript)) {
    throw "Expected backend smoke wrapper at $smokeScript."
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
    Remove-Item -Recurse -Force $pytestBaseTemp -ErrorAction SilentlyContinue

    Invoke-CheckedPython "Validate contracts" @("tools\quality\validate_contracts.py")
    Invoke-CheckedPython "Audit file headers" @("tools\quality\check_headers.py", "src", "tools", "tests")
    Invoke-CheckedPython "Run ruff" @("-m", "ruff", "check", "src\rewardlab", "tests", "tools\quality")
    Invoke-CheckedPython "Run mypy" @("-m", "mypy", "src")
    Invoke-CheckedPython "Run offline pytest suite" @(
        "-m",
        "pytest",
        "tests\unit",
        "tests\contract",
        "tests\integration",
        "tests\e2e",
        "-q",
        "--basetemp",
        $pytestBaseTemp,
        "-p",
        "no:cacheprovider"
    )

    if ($IncludeRealGymnasium) {
        $smokeArguments = @()
        $smokeArguments += "-Gymnasium"

        Write-Host "==> Run requested real backend smokes"
        & $smokeScript @smokeArguments
    }
}
finally {
    Pop-Location
}
