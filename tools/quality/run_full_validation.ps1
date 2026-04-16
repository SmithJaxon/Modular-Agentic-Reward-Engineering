param(
    [switch]$DeterministicOnly,
    [switch]$EnableOpenAISmoke,
    [switch]$RequireRuntimeSuites,
    [switch]$RequireOpenAISmoke,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"

function Invoke-ValidationStep {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host "==> $Label"
    & $script:PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Validation step failed: $Label"
    }
}

function Test-OptionalModule {
    param(
        [string]$ModuleName
    )

    & $script:PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)"
    return $LASTEXITCODE -eq 0
}

function Import-DotEnvFile {
    param(
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $false
    }

    foreach ($rawLine in Get-Content $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).TrimStart()
        }
        if (-not $line.Contains("=")) {
            continue
        }

        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        if (-not $key) {
            continue
        }

        $value = $parts[1].Trim()
        if (
            $value.Length -ge 2 -and (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            )
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if (-not (Test-Path "Env:$key")) {
            Set-Item "Env:$key" $value
        }
    }

    return $true
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $PythonExe) {
    $PythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"
}
$script:PythonExe = $PythonExe

if (-not (Test-Path $script:PythonExe)) {
    throw "Python executable not found: $script:PythonExe"
}

$skippedRuntimeSuites = [System.Collections.Generic.List[string]]::new()

Push-Location $repoRoot
try {
    Invoke-ValidationStep "ruff" @(
        "-m", "ruff", "check", "src", "tests", "tools"
    )
    Invoke-ValidationStep "mypy" @(
        "-m", "mypy", "src"
    )
    Invoke-ValidationStep "pytest (deterministic suites)" @(
        "-m", "pytest",
        "tests\unit",
        "tests\contract",
        "tests\integration",
        "tests\e2e",
        "-q",
        "-p", "no:cacheprovider",
        "-p", "no:tmpdir",
        "--ignore", "tests\integration\test_gymnasium_runtime.py",
        "--ignore", "tests\integration\test_isaacgym_runtime.py",
        "--ignore", "tests\integration\test_openai_runtime.py"
    )
    Invoke-ValidationStep "contract validation" @(
        "tools\quality\validate_contracts.py"
    )
    Invoke-ValidationStep "header audit" @(
        "tools\quality\check_headers.py",
        "src\rewardlab",
        "tests",
        "tools"
    )

    if ($DeterministicOnly) {
        Write-Host "Deterministic validation completed successfully."
        return
    }

    $runtimeSuites = @(
        @{
            Label = "Gymnasium runtime smoke"
            Module = "gymnasium"
            TestPath = "tests\integration\test_gymnasium_runtime.py"
        },
        @{
            Label = "Isaac Gym runtime smoke"
            Module = "isaacgym"
            TestPath = "tests\integration\test_isaacgym_runtime.py"
        }
    )

    foreach ($suite in $runtimeSuites) {
        if (Test-OptionalModule $suite.Module) {
            Invoke-ValidationStep $suite.Label @(
                "-m", "pytest",
                $suite.TestPath,
                "-q",
                "-p", "no:cacheprovider",
                "-p", "no:tmpdir"
            )
            continue
        }

        $skippedRuntimeSuites.Add($suite.Label)
        Write-Host "Skipping $($suite.Label): optional module '$($suite.Module)' is unavailable."
    }

    if ($RequireRuntimeSuites -and $skippedRuntimeSuites.Count -gt 0) {
        throw (
            "Required runtime suites were unavailable: " +
            ($skippedRuntimeSuites -join ", ")
        )
    }

    $runtimeStatusMessage = if ($skippedRuntimeSuites.Count -gt 0) {
        (
            "Validation completed successfully with runtime suites skipped: " +
            ($skippedRuntimeSuites -join ", ")
        )
    } else {
        "Validation completed successfully, including runtime smoke suites."
    }

    $openAISmokeStatusMessage = $null
    if ($EnableOpenAISmoke -or $RequireOpenAISmoke) {
        $null = Import-DotEnvFile (Join-Path $repoRoot ".env")
        $env:REWARDLAB_ENABLE_OPENAI_LIVE_TESTS = "1"

        if (-not (Test-OptionalModule "openai")) {
            if ($RequireOpenAISmoke) {
                throw "Required OpenAI smoke suite was unavailable: optional module 'openai' is missing."
            }
            $openAISmokeStatusMessage = "Skipping OpenAI live smoke: optional module 'openai' is unavailable."
        } elseif (-not $env:OPENAI_API_KEY) {
            if ($RequireOpenAISmoke) {
                throw "Required OpenAI smoke suite was unavailable: OPENAI_API_KEY was not set and no .env value could be loaded."
            }
            $openAISmokeStatusMessage = "Skipping OpenAI live smoke: OPENAI_API_KEY was not set."
        } else {
            Invoke-ValidationStep "OpenAI live smoke" @(
                "-m", "pytest",
                "tests\integration\test_openai_runtime.py",
                "-q",
                "-p", "no:cacheprovider",
                "-p", "no:tmpdir"
            )
            $openAISmokeStatusMessage = "OpenAI live smoke completed successfully."
        }
    }

    Write-Host $runtimeStatusMessage
    if ($openAISmokeStatusMessage) {
        Write-Host $openAISmokeStatusMessage
    }
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Pop-Location
}
