<# Verify / complete the Python environment (set $env:N1_PYTHON or pass -Python to use a specific interpreter, e.g. a conda env's python.exe). Run once:  .\scripts\windows\setup_check.ps1 #>
param([string]$Python = $(if ($env:N1_PYTHON) { $env:N1_PYTHON } else { "python" }))
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -Path $RepoRoot
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python not found: $Python. Pass -Python <path> or set $env:N1_PYTHON." }
& $Python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
if ($LASTEXITCODE -ne 0) {
  Write-Host "torch missing -> installing CUDA 12.4 wheels"
  & $Python -m pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
}
& $Python -m pip install -r requirements.txt
& $Python -m pytest -q tests
& $Python -c "import timm, imagehash, sklearn, scipy, pandas, matplotlib, yaml; print('timm', timm.__version__, 'OK')"
