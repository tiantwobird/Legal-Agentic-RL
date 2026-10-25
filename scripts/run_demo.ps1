# Run one tool-calling rollout with the local Qwen3 model.
#
#   .\scripts\run_demo.ps1 --query "员工严重违反规章制度被辞退，公司要赔钱吗？" --date 2024-01-01
#   .\scripts\run_demo.ps1 --few-shot
#
# KMP_DUPLICATE_LIB_OK: the venv reuses Anaconda's CUDA torch via anaconda.pth;
# Anaconda and torch each ship libiomp5md.dll, and without this flag the import
# aborts with "OMP: Error #15". It only silences OpenMP duplicate detection.

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
# 默认离线，只用本地 HF 缓存。首次运行要下载模型时传 --download：
#   .\scripts\run_demo.ps1 --download --query "..." --date 2024-01-01

$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv-ml\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "ML venv not found: $python"
}

# `-m` keeps the package imports working regardless of the caller's cwd.
Push-Location $root
try {
    & $python -m legalagent.tools.rollout_demo @args
}
finally {
    Pop-Location
}
exit $LASTEXITCODE
