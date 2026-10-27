<# Show progress of the background run:  .\scripts\windows\check_progress.ps1  [-LogDir logs] [-Config configs/full.yaml] #>
param([string]$LogDir = "logs", [string]$OutDir = "outputs", [int]$Tail = 15)
Set-Location -Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$csv = Join-Path $LogDir "progress.csv"
if (-not (Test-Path $csv)) { Write-Host "No $csv yet."; exit }
$rows = Import-Csv $csv
$done = ($rows | Where-Object { $_.status -eq "done" -and $_.stage -eq "train" })
$failed = ($rows | Where-Object { $_.status -eq "FAILED" })
$last = $rows | Select-Object -Last 1
$nDone = (Get-ChildItem -Path (Join-Path $OutDir "runs") -Filter done.json -Recurse -ErrorAction SilentlyContinue).Count
Write-Host "Finished runs on disk: $nDone   | failed events: $($failed.Count)"
$lastDone = $done | Select-Object -Last 1
if ($lastDone) { Write-Host "Last finished: $($lastDone.run_id)  test_acc=$($lastDone.test_acc)  ETA(h)=$($lastDone.eta_h)  [$($lastDone.time)]" }
Write-Host "Latest event : $($last.time) $($last.stage) $($last.run_id) $($last.status) epoch $($last.epoch)/$($last.epochs) $($last.note)"
$pidFile = Join-Path $LogDir "run_all.pid"
if (Test-Path $pidFile) {
  $procId = Get-Content $pidFile
  if (Get-Process -Id $procId -ErrorAction SilentlyContinue) { Write-Host "Process $procId is RUNNING" } else { Write-Host "Process $procId is NOT running (finished or stopped; re-launch to resume)" }
}
Write-Host "`n--- last $Tail lines of run_all.log ---"
Get-Content (Join-Path $LogDir "run_all.log") -Tail $Tail
try { nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv } catch {}
