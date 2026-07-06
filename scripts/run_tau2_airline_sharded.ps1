[CmdletBinding()]
param(
    [string]$AgentLlm = "anthropic/glm-5.1",
    [string]$UserLlm = "anthropic/glm-5.1",
    [string]$AgentLlmArgs = "",
    [string]$UserLlmArgs = "",
    [string]$TaskIds = "",
    [int]$NumTrials = 4,
    [int]$MaxSteps = 80,
    [int]$TimeoutSeconds = 300,
    [int]$MaxRetries = 0,
    [double]$RetryDelay = 1.0,
    [int]$Seed = 300,
    [string]$ShardPrefix = "airline_sharded",
    [string]$MergedSaveTo = "",
    [switch]$NoMerge,
    [switch]$Force,
    [switch]$VerboseLogs,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Tau2Root = Join-Path $Root "third_party\tau2-bench"
$Uv = Join-Path $env:USERPROFILE "AppData\Roaming\Python\Python312\Scripts\uv.exe"
$SummaryScript = Join-Path $Root "scripts\summarize_tau2_results.py"
$MergeScript = Join-Path $Root "scripts\merge_tau2_shards.py"
$ReportsDir = Join-Path $Root "reports"
$SimulationsRoot = Join-Path $Tau2Root "data\simulations"
$LogsDir = Join-Path $ReportsDir "sharded_logs\$ShardPrefix"

if (-not (Test-Path $Tau2Root)) {
    throw "tau2-bench checkout not found: $Tau2Root"
}

if (-not (Test-Path $Uv)) {
    throw "uv.exe not found: $Uv. Run: py -m pip install --user uv"
}

if (-not (Test-Path $MergeScript)) {
    throw "merge script not found: $MergeScript"
}

if ($TaskIds.Trim().Length -eq 0) {
    throw "TaskIds is required for sharded runs, e.g. -TaskIds '1,2,7'"
}

$TaskIdList = $TaskIds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 }
if ($TaskIdList.Count -eq 0) {
    throw "No valid task ids parsed from: $TaskIds"
}

if ($TimeoutSeconds -le 0) {
    throw "TimeoutSeconds must be positive."
}

$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null

$ManifestPath = Join-Path $LogsDir "manifest.csv"
if (-not (Test-Path $ManifestPath) -or $Force) {
    '"timestamp","task_id","trial","seed","status","exit_code","duration_seconds","save_to","stdout","stderr"' |
        Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

function Stop-ProcessTree {
    param([int]$RootProcessId)

    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $RootProcessId }
    foreach ($child in $children) {
        Stop-ProcessTree -RootProcessId ([int]$child.ProcessId)
    }

    $process = Get-Process -Id $RootProcessId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Test-CompletedShard {
    param([string]$SaveTo)

    $resultsJson = Join-Path $SimulationsRoot "$SaveTo\results.json"
    if (-not (Test-Path $resultsJson)) {
        return $false
    }

    try {
        $json = Get-Content -LiteralPath $resultsJson -Raw | ConvertFrom-Json
        return ($json.simulations -and $json.simulations.Count -gt 0)
    } catch {
        return $false
    }
}

function Add-CsvLine {
    param(
        [string]$TaskId,
        [int]$Trial,
        [int]$ShardSeed,
        [string]$Status,
        [string]$ExitCode,
        [double]$DurationSeconds,
        [string]$SaveTo,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $line = '"{0}","{1}","{2}","{3}","{4}","{5}","{6:N1}","{7}","{8}","{9}"' -f `
        (Get-Date).ToString("s"),
        $TaskId,
        $Trial,
        $ShardSeed,
        $Status,
        $ExitCode,
        $DurationSeconds,
        $SaveTo,
        $StdoutPath,
        $StderrPath
    Add-Content -LiteralPath $ManifestPath -Encoding UTF8 -Value $line
}

Write-Host "Running timeout-safe tau2 airline shards..."
Write-Host "  agent:   $AgentLlm"
Write-Host "  user:    $UserLlm"
Write-Host "  tasks:   $($TaskIdList -join ',')"
Write-Host "  trials:  $NumTrials"
Write-Host "  timeout: ${TimeoutSeconds}s per shard"
Write-Host "  prefix:  $ShardPrefix"
Write-Host "  logs:    $LogsDir"

$total = $TaskIdList.Count * $NumTrials
$index = 0

for ($trial = 0; $trial -lt $NumTrials; $trial++) {
    foreach ($taskId in $TaskIdList) {
        $index += 1
        $ShardSeed = $Seed + $trial
        $SaveTo = "${ShardPrefix}_task_${taskId}_trial_${trial}"
        $StdoutPath = Join-Path $LogsDir "$SaveTo.stdout.log"
        $StderrPath = Join-Path $LogsDir "$SaveTo.stderr.log"

        if ((-not $Force) -and (Test-CompletedShard -SaveTo $SaveTo)) {
            Write-Host "[$index/$total] skip completed task=$taskId trial=$trial save=$SaveTo"
            Add-CsvLine -TaskId $taskId -Trial $trial -ShardSeed $ShardSeed -Status "skipped" `
                -ExitCode "" -DurationSeconds 0 -SaveTo $SaveTo -StdoutPath $StdoutPath -StderrPath $StderrPath
            continue
        }

        $TauArgs = @(
            "run",
            "tau2",
            "run",
            "--domain", "airline",
            "--agent", "llm_agent",
            "--user", "user_simulator",
            "--agent-llm", $AgentLlm,
            "--user-llm", $UserLlm,
            "--num-trials", "1",
            "--max-concurrency", "1",
            "--max-steps", "$MaxSteps",
            "--max-retries", "$MaxRetries",
            "--retry-delay", "$RetryDelay",
            "--seed", "$ShardSeed",
            "--save-to", $SaveTo,
            "--auto-resume",
            "--task-ids", "$taskId"
        )

        if ($AgentLlmArgs.Trim().Length -gt 0) {
            $TauArgs += "--agent-llm-args"
            $TauArgs += $AgentLlmArgs
        }

        if ($UserLlmArgs.Trim().Length -gt 0) {
            $TauArgs += "--user-llm-args"
            $TauArgs += $UserLlmArgs
        }

        if ($VerboseLogs) {
            $TauArgs += "--verbose-logs"
            $TauArgs += "--llm-log-mode"
            $TauArgs += "latest"
        }

        Write-Host "[$index/$total] run task=$taskId trial=$trial seed=$ShardSeed save=$SaveTo"
        if ($DryRun) {
            Write-Host "$Uv $($TauArgs -join ' ')"
            continue
        }

        $start = Get-Date
        $process = Start-Process -FilePath $Uv `
            -ArgumentList $TauArgs `
            -WorkingDirectory $Tau2Root `
            -RedirectStandardOutput $StdoutPath `
            -RedirectStandardError $StderrPath `
            -WindowStyle Hidden `
            -PassThru

        $timedOut = $false
        while (-not $process.HasExited) {
            Start-Sleep -Seconds 2
            $elapsed = ((Get-Date) - $start).TotalSeconds
            if ($elapsed -ge $TimeoutSeconds) {
                Write-Warning "Timeout task=$taskId trial=$trial after $([math]::Round($elapsed, 1))s. Stopping process tree."
                Stop-ProcessTree -RootProcessId $process.Id
                $timedOut = $true
                break
            }
        }

        if ($timedOut) {
            $duration = ((Get-Date) - $start).TotalSeconds
            Add-CsvLine -TaskId $taskId -Trial $trial -ShardSeed $ShardSeed -Status "timeout" `
                -ExitCode "" -DurationSeconds $duration -SaveTo $SaveTo -StdoutPath $StdoutPath -StderrPath $StderrPath
            continue
        }

        $process.Refresh()
        try {
            $process.WaitForExit()
        } catch {
            # The process has already exited; keep going and judge by the result file.
        }

        $duration = ((Get-Date) - $start).TotalSeconds
        $exitCode = $process.ExitCode
        $exitCodeText = if ($null -eq $exitCode) { "" } else { "$exitCode" }
        $hasCompletedResult = Test-CompletedShard -SaveTo $SaveTo
        $status = if ($hasCompletedResult) { "ok" } else { "failed" }
        Add-CsvLine -TaskId $taskId -Trial $trial -ShardSeed $ShardSeed -Status $status `
            -ExitCode $exitCodeText -DurationSeconds $duration -SaveTo $SaveTo -StdoutPath $StdoutPath -StderrPath $StderrPath
        Write-Host "[$index/$total] $status task=$taskId trial=$trial duration=$([math]::Round($duration, 1))s"
    }
}

if ($DryRun -or $NoMerge) {
    return
}

if ($MergedSaveTo.Trim().Length -eq 0) {
    $MergedSaveTo = "${ShardPrefix}_merged"
}

$MergedOut = Join-Path $SimulationsRoot $MergedSaveTo
py $MergeScript `
    --shards-root $SimulationsRoot `
    --prefix $ShardPrefix `
    --out $MergedOut `
    --num-trials $NumTrials `
    --task-ids ($TaskIdList -join ",") `
    --rewrite-trial-from-name

$SummaryOut = Join-Path $ReportsDir "$MergedSaveTo.summary.json"
py $SummaryScript $MergedOut --out $SummaryOut
