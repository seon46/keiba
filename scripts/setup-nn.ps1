# Installs PyTorch (GPU build) into .venv for the neural experiments and checks that CUDA works.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-nn.ps1            (CUDA 12.6 build)
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-nn.ps1 -Cuda cu118 (older NVIDIA drivers)
param([string]$Cuda = "cu126")
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectDirectory = Split-Path -Parent $PSScriptRoot
$keibaPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $keibaPython -PathType Leaf)) { throw "Run scripts/setup.ps1 and scripts/setup-ml.ps1 first." }
& $keibaPython -X utf8 -m pip install "torch==2.7.1" --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $keibaPython -X utf8 -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else '-'); raise SystemExit(0 if ok else 3)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch is installed but CUDA is not available. Update the NVIDIA driver or retry with -Cuda cu118." }
