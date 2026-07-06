[CmdletBinding()]
param(
    [string]$Models = "deepseek/deepseek-chat,anthropic/glm-5.1",
    [string]$UserModels = "",
    [int]$NumTasks = 50,
    [string]$TaskIds = "",
    [int]$NumTrials = 1,
    [int]$MaxConcurrency = 1,
    [int]$MaxSteps = 120,
    [int]$MaxRetries = 0,
    [double]$RetryDelay = 1.0,
    [int]$Seed = 300,
    [string]$NamePrefix = "airline_baseline",
    [switch]$VerboseLogs,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Runner = Join-Path $PSScriptRoot "run_tau2_airline_baseline.ps1"
if (-not (Test-Path $Runner)) {
    throw "Runner not found: $Runner"
}

function Split-ModelList([string]$Value) {
    return $Value.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 }
}

function Get-SafeName([string]$Model) {
    $safe = $Model -replace '[^A-Za-z0-9]+', '_'
    return $safe.Trim("_").ToLowerInvariant()
}

$AgentModels = @(Split-ModelList $Models)
if ($AgentModels.Count -eq 0) {
    throw "No models provided."
}

$UserModelList = @(Split-ModelList $UserModels)
if ($UserModelList.Count -eq 0) {
    $UserModelList = $AgentModels
}

if ($UserModelList.Count -ne $AgentModels.Count) {
    throw "UserModels must be empty or have the same count as Models."
}

for ($i = 0; $i -lt $AgentModels.Count; $i++) {
    $AgentModel = $AgentModels[$i]
    $UserModel = $UserModelList[$i]
    $SafeAgent = Get-SafeName $AgentModel
    $SizeLabel = if ($TaskIds.Trim().Length -gt 0) { "taskids" } else { "${NumTasks}tasks" }
    $SaveTo = "${NamePrefix}_${SafeAgent}_${SizeLabel}_${NumTrials}trials"

    Write-Host ""
    Write-Host "=== Running baseline sweep item ==="
    Write-Host "agent: $AgentModel"
    Write-Host "user:  $UserModel"
    Write-Host "save:  $SaveTo"

    $RunnerParams = @{
        AgentLlm       = $AgentModel
        UserLlm        = $UserModel
        NumTasks       = $NumTasks
        NumTrials      = $NumTrials
        MaxConcurrency = $MaxConcurrency
        MaxSteps       = $MaxSteps
        MaxRetries     = $MaxRetries
        RetryDelay     = $RetryDelay
        Seed           = $Seed
        SaveTo         = $SaveTo
    }

    if ($TaskIds.Trim().Length -gt 0) {
        $RunnerParams["TaskIds"] = $TaskIds
    }
    if ($VerboseLogs) {
        $RunnerParams["VerboseLogs"] = $true
    }
    if ($DryRun) {
        $RunnerParams["DryRun"] = $true
    }

    & $Runner @RunnerParams
}
