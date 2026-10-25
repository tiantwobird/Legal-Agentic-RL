<#
.SYNOPSIS
    End-to-end LegalAgent pipeline: export -> SQLite -> tasks -> tests -> train -> eval -> analysis.

.DESCRIPTION
    Each stage is a module invocation, so every step is also runnable on its own
    (the exact command is echoed before it runs).

    Stages 1-3 are data, 4 is tests, 5-6 are training + scoring, 7 is analysis.
    Re-exporting the corpus is opt-in (-Export): it needs the local chinalaw DB
    and is the only stage that touches a network-independent external CLI.

.EXAMPLE
    .\scripts\run_all.ps1                      # data + tests + 30-step runs + eval
    .\scripts\run_all.ps1 -Export -MaxSteps 1  # full plumbing check from scratch
    .\scripts\run_all.ps1 -SkipTrain            # data + tests + analysis only
#>
param(
    [switch]$Export,
    [switch]$SkipTrain,
    [switch]$SkipTests,
    [switch]$Download,
    [int]$MaxSteps = 30,
    [string]$SingleRun = "grpo_single",
    [string]$AgenticRun = "grpo_agentic"
)

$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv-ml\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "ML venv not found: $python"
}

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
# -Download 让每个子进程都允许联网拉模型（默认只用本地缓存）
if ($Download) { $env:LEGALAGENT_ALLOW_DOWNLOAD = "1" }

function Invoke-Step {
    param([string]$Title, [string[]]$PyArgs)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host "  python $($PyArgs -join ' ')" -ForegroundColor DarkGray
    & $python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "$Title failed (exit $LASTEXITCODE)" }
}

Push-Location $root
try {
    if ($Export) {
        Invoke-Step "1/6 export corpus" @("legalagent.data.export_subset")
    } else {
        Write-Host "[skip] export (use -Export to refresh data/labor_subset.jsonl)"
    }

    Invoke-Step "2/6 build SQLite snapshot" @("legalagent.data.build_db")
    Invoke-Step "3/6 validate task splits" @("legalagent.data.validate_tasks")

    if ($SkipTests) {
        Write-Host "[skip] unit tests"
    } else {
        Invoke-Step "4/6 unit tests" @("-m", "unittest", "discover", "-s", "tests", "-t", ".")
    }

    if ($SkipTrain) {
        Write-Host "[skip] training + eval"
    } else {
        Invoke-Step "5a/6 single-turn baseline" @(
            "-m", "legalagent.eval.score_rollout", "--split", "dev", "--grounded",
            "--out", "results/eval_grounded_before.json")
        Invoke-Step "5b/6 single-turn GRPO" @(
            "-m", "legalagent.train.train_grpo", "--max-steps", "$MaxSteps",
            "--out", "runs/$SingleRun")
        Invoke-Step "5c/6 single-turn eval" @(
            "-m", "legalagent.eval.score_rollout", "--split", "dev", "--grounded",
            "--adapter", "runs/$SingleRun", "--out", "results/eval_grounded_after.json")

        Invoke-Step "5d/6 agentic baseline" @(
            "-m", "legalagent.eval.eval_agentic", "--split", "dev",
            "--out", "results/eval_agentic_before.json")
        Invoke-Step "5e/6 agentic GRPO" @(
            "-m", "legalagent.train.train_grpo_agentic", "--max-steps", "$MaxSteps",
            "--out", "runs/$AgenticRun")
        Invoke-Step "5f/6 agentic eval" @(
            "-m", "legalagent.eval.eval_agentic", "--split", "dev",
            "--adapter", "runs/$AgenticRun", "--out", "results/eval_agentic_after.json")
    }

    Invoke-Step "6/6 bad-case analysis" @("legalagent.analysis.bad_case_analyzer")

    Write-Host ""
    Write-Host "[done] reports under results/" -ForegroundColor Green
}
finally {
    Pop-Location
}
