param(
    [string]$Domain = "airline",
    [int]$NumTasks = 3,
    [int]$NumTrials = 1,
    [string]$TaskSplitName = "base",
    [string]$AgentLlm = $env:AGENT_LLM,
    [string]$UserLlm = $env:USER_LLM,
    [string]$AgentLlmArgs = '{"temperature": 0.0}',
    [string]$UserLlmArgs = '{"temperature": 0.7}',
    [int]$MaxConcurrency = 1,
    [int]$Seed = 300,
    [int]$MaxSteps = 200,
    [string]$SaveTo = "",
    [string]$Tau2Dir = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($Tau2Dir)) {
    $Tau2Dir = Join-Path $repoRoot "third_party\tau2-bench"
}

if (-not (Test-Path (Join-Path $Tau2Dir "pyproject.toml"))) {
    throw "Cannot find tau2-bench at '$Tau2Dir'. Expected pyproject.toml there."
}

if ([string]::IsNullOrWhiteSpace($AgentLlm)) {
    throw "Set -AgentLlm or environment variable AGENT_LLM, for example: openai/gpt-4.1-mini"
}
if ([string]::IsNullOrWhiteSpace($UserLlm)) {
    throw "Set -UserLlm or environment variable USER_LLM, for example: openai/gpt-4.1-mini"
}

if ([string]::IsNullOrWhiteSpace($SaveTo)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeAgent = ($AgentLlm -replace "[^A-Za-z0-9_.-]", "_")
    $SaveTo = "baseline_${Domain}_${safeAgent}_${stamp}"
}

$tau2Args = @(
    "run",
    "--domain", $Domain,
    "--agent-llm", $AgentLlm,
    "--user-llm", $UserLlm,
    "--agent-llm-args", $AgentLlmArgs,
    "--user-llm-args", $UserLlmArgs,
    "--num-trials", "$NumTrials",
    "--task-split-name", $TaskSplitName,
    "--max-concurrency", "$MaxConcurrency",
    "--seed", "$Seed",
    "--max-steps", "$MaxSteps",
    "--save-to", $SaveTo,
    "--log-level", "ERROR"
)

if ($NumTasks -gt 0) {
    $tau2Args += @("--num-tasks", "$NumTasks")
}

Write-Host "Running tau2 baseline..."
Write-Host "  domain: $Domain"
Write-Host "  agent:  $AgentLlm"
Write-Host "  user:   $UserLlm"
Write-Host "  save:   $SaveTo"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run only. Command that would be executed:"
    Write-Host "  python -m uv run tau2 $($tau2Args -join ' ')"
    exit 0
}

Push-Location $Tau2Dir
try {
    & python -m uv run tau2 @tau2Args
}
finally {
    Pop-Location
}

$runPath = Join-Path $Tau2Dir "data\simulations\$SaveTo"
Write-Host ""
Write-Host "Summarizing $runPath"
& python (Join-Path $repoRoot "scripts\summarize_tau2_results.py") $runPath
