# Reproduce everything: task validation, unit tests, the full experiment and the analysis.
#
# Usage (PowerShell, from any folder):
#     powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1
#
# The experiment step is resumable: run the script again after an interruption and it
# continues results\main where it stopped.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Tag = "main"

function Stop-With([string]$Message) {
    Write-Host ""
    Write-Host "STOP: $Message" -ForegroundColor Red
    exit 1
}

function Invoke-Step([string]$Name, [string[]]$Arguments) {
    Write-Host ""
    Write-Host "== $Name" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-With "$Name failed (exit code $LASTEXITCODE). Fix the problem above and run the script again."
    }
}

if (-not (Test-Path $Python)) {
    Stop-With "the virtual environment is missing. Create it with: py -3.11 -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

# Ollama must be running and every model in config.yaml must exist; agents.llm.check_ollama
# names the missing model and the ollama create command for it.
Write-Host "== Checking Ollama and the models in config.yaml" -ForegroundColor Cyan
$check = "import sys; from agents.llm import check_ollama, load_config, native_base, ollama_get; " +
    "c = load_config(); p = check_ollama(c['models'], c); " +
    "print(p) if p else print('ollama', ollama_get(native_base(c['ollama_base_url']) + '/api/version', 10)['version'], 'with', ', '.join(c['models'])); " +
    "sys.exit(1 if p else 0)"
& $Python -c $check
if ($LASTEXITCODE -ne 0) {
    $exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    $hint = if (Get-Command ollama -ErrorAction SilentlyContinue) { "ollama" } elseif (Test-Path $exe) { $exe } else { $null }
    if ($null -eq $hint) {
        Stop-With "Ollama is not installed or not running. Install it from ollama.com, start it, create the models (see README.md) and run the script again."
    }
    Stop-With "Ollama or a model is missing (see the message above). Start Ollama ($hint serve) and create the models with: $hint create <tag> -f models\<tag>.Modelfile"
}

Invoke-Step "Task validation" @("eval\validate_tasks.py")
Invoke-Step "Unit tests" @("-m", "pytest", "-q")
Invoke-Step "Full experiment (about 15 GPU hours)" @("run.py", "--tag", $Tag, "--runs", "3", "--resume")
Invoke-Step "Analysis" @("-m", "analysis.run_all", "--tag", $Tag)

Write-Host ""
Write-Host "Done. Tables, figures and summary.md are in results\$Tag\analysis\" -ForegroundColor Green
