param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $Python --version
& $Python -m venv (Join-Path $ProjectRoot ".venv")
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "Environment ready: $VenvPython"
Write-Host "Select this interpreter in PyCharm, then run: python run_all.py"
