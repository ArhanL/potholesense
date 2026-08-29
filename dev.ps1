<#
    Windows entry point. Creates the virtual environment if it is missing,
    installs anything absent, then starts the server.

    Usage:  .\dev.ps1            start with the classical-CV stand-in
            .\dev.ps1 -Real      use the trained model in models/
            .\dev.ps1 -Test      run the test suite instead
            .\dev.ps1 -Sim       drive the simulator against a running server

    The environment lives in .venv-win, deliberately not .venv: a virtual
    environment holds binaries compiled for one operating system, so a folder
    shared between machines - through OneDrive, a network drive, anything -
    must not have them collide. macOS uses .venv-mac via dev.sh.
#>
param([switch]$Real, [switch]$Test, [switch]$Sim, [int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$venv = ".venv-win"

if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Write-Host "Creating $venv ..." -ForegroundColor Cyan
    py -3 -m venv $venv
}
$py = Resolve-Path "$venv\Scripts\python.exe"

$version = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python $version at $py" -ForegroundColor DarkGray

# Stub mode, the simulator and the tests need none of PyTorch, so only
# -Real pays for the model stack.
$reqs = if ($Real) { "requirements-model.txt" } else { "requirements-dev.txt" }
$probe = if ($Real) { "import fastapi, cv2, ultralytics" } else { "import fastapi, cv2, reportlab, requests, cryptography, pytest" }
& $py -c $probe 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies from $reqs (once) ..." -ForegroundColor Cyan
    & $py -m pip install --quiet --upgrade pip
    & $py -m pip install --quiet -r $reqs
}

if ($Test) { & $py -m pytest tests/ -q; exit $LASTEXITCODE }
if ($Sim)  { & $py scripts/simulate_drive.py --frames 220 --potholes 10 --oracle `
                    --server "https://127.0.0.1:$Port"; exit $LASTEXITCODE }

$stub = if ($Real) { @() } else { @("--stub") }
& $py run.py --https --port $Port @stub
