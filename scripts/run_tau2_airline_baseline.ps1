[CmdletBinding()]
param(
    [string]$AgentLlm = "openai/gpt-4.1-mini",
    [string]$UserLlm = "openai/gpt-4.1-mini",
    [string]$AgentLlmArgs = "",
    [string]$UserLlmArgs = "",
    [int]$NumTasks = 5,
    [int]$NumTrials = 1,
    [int]$MaxConcurrency = 1,
    [int]$MaxSteps = 200,
    [int]$MaxRetries = 0,
    [double]$RetryDelay = 1.0,
    [int]$Seed = 300,
    [string]$TaskIds = "",
    [string]$SaveTo = "airline_smoke",
    [switch]$VerboseLogs,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Tau2Root = Join-Path $Root "third_party\tau2-bench"
$Uv = Join-Path $env:USERPROFILE "AppData\Roaming\Python\Python312\Scripts\uv.exe"
$SummaryScript = Join-Path $Root "scripts\summarize_tau2_results.py"
$ReportsDir = Join-Path $Root "reports"

if (-not (Test-Path $Tau2Root)) {
    throw "tau2-bench checkout not found: $Tau2Root"
}

if (-not (Test-Path $Uv)) {
    throw "uv.exe not found: $Uv. Run: py -m pip install --user uv"
}

$EnvFile = Join-Path $Tau2Root ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Warning "No .env found in tau2-bench. Copy .env.example to .env and set your API key before running real LLM evals."
}

$env:PYTHONUTF8 = "1"

$TauArgs = @(
    "run",
    "tau2",
    "run",
    "--domain", "airline",
    "--agent", "llm_agent",
    "--user", "user_simulator",
    "--agent-llm", $AgentLlm,
    "--user-llm", $UserLlm,
    "--num-trials", "$NumTrials",
    "--max-concurrency", "$MaxConcurrency",
    "--max-steps", "$MaxSteps",
    "--max-retries", "$MaxRetries",
    "--retry-delay", "$RetryDelay",
    "--seed", "$Seed",
    "--save-to", $SaveTo,
    "--auto-resume"
)

if ($AgentLlmArgs.Trim().Length -gt 0) {
    $TauArgs += "--agent-llm-args"
    $TauArgs += $AgentLlmArgs
}

if ($UserLlmArgs.Trim().Length -gt 0) {
    $TauArgs += "--user-llm-args"
    $TauArgs += $UserLlmArgs
}

if ($TaskIds.Trim().Length -gt 0) {
    $TauArgs += "--task-ids"
    $TauArgs += $TaskIds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 }
} else {
    $TauArgs += "--num-tasks"
    $TauArgs += "$NumTasks"
}

if ($VerboseLogs) {
    $TauArgs += "--verbose-logs"
    $TauArgs += "--llm-log-mode"
    $TauArgs += "latest"
}

Write-Host "Running tau2 airline baseline..."
Write-Host "  agent: $AgentLlm"
Write-Host "  user:  $UserLlm"
Write-Host "  save:  data/simulations/$SaveTo"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run command:"
    Write-Host "$Uv $($TauArgs -join ' ')"
    return
}

Push-Location $Tau2Root
try {
    & $Uv @TauArgs
} finally {
    Pop-Location
}

$ResultsPath = Join-Path $Tau2Root "data\simulations\$SaveTo"
if (Test-Path $ResultsPath) {
    New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
    $SummaryOut = Join-Path $ReportsDir "$SaveTo.summary.json"
    py $SummaryScript $ResultsPath --out $SummaryOut
} else {
    Write-Warning "Expected results directory was not found: $ResultsPath"
}
