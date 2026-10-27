<#
Launch the resumable pipeline (set $env:N1_PYTHON or pass -Python to use a specific interpreter, e.g. a conda env's python.exe).
  .\scripts\windows\run_all.ps1                                  # full config, foreground
  .\scripts\windows\run_all.ps1 -Background                      # full config, background (survives closing the window)
  .\scripts\windows\run_all.ps1 -Config configs/quick.yaml       # quick GPU check
  .\scripts\windows\run_all.ps1 -Extra "--estimate"              # just print the runtime estimate
Re-running the same command after a crash/reboot resumes where it stopped.
#>
param(
  [string]$Config = "configs/full.yaml",
  [string]$Python = $(if ($env:N1_PYTHON) { $env:N1_PYTHON } else { "python" }),
  [switch]$Background,
  [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -Path $RepoRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python not found: $Python. Pass -Python <path to python.exe> or set $env:N1_PYTHON." }
$argsList = @("run_all.py", "--config", $Config) + ($Extra -split " " | Where-Object { $_ -ne "" })
if ($Background) {
  New-Item -ItemType Directory -Force -Path logs | Out-Null
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $p = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput "logs\stdout_$stamp.txt" -RedirectStandardError "logs\stderr_$stamp.txt" `
        -WindowStyle Hidden -PassThru
  Write-Host "Started PID $($p.Id). Check progress with:  .\scripts\windows\check_progress.ps1"
  $p.Id | Out-File -Encoding ascii "logs\run_all.pid"
} else {
  & $Python @argsList
}
