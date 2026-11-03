# Installs PyTorch (GPU build) into .venv for the neural experiments and checks that CUDA works.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-nn.ps1            (CUDA 12.8 build: RTX 50xx and older GPUs)
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-nn.ps1 -Cuda cu126 (drivers older than 570, GPUs before RTX 50xx)
#   -Version selects another PyTorch release available for this Python (default 2.9.1)
param([string]$Cuda = "cu128", [string]$Version = "2.9.1")
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectDirectory = Split-Path -Parent $PSScriptRoot
$keibaPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $keibaPython -PathType Leaf)) { throw "Run scripts/setup.ps1 and scripts/setup-ml.ps1 first." }
& $keibaPython -X utf8 -m pip install "torch==$Version" --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $keibaPython -X utf8 -m pip install --force-reinstall --no-deps "torch==$Version" --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $keibaPython -X utf8 -c "import torch; ok = torch.cuda.is_available(); ok = ok and float((torch.ones(4, device='cuda') * 2).sum()) == 8.0; print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); raise SystemExit(0 if ok else 3)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch is installed but cannot run on this GPU. Update the NVIDIA driver (570 or later for CUDA 12.8) and run this script again." }
