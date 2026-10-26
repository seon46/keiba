$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    $keibaPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $keibaPython -PathType Leaf)) {
        throw "Run scripts/setup.ps1 first to create the Python environment."
    }
    & $keibaPython -X utf8 -m pip install -r requirements-ml.txt
    if ($LASTEXITCODE -ne 0) { throw "ML dependency installation failed." }
    & $keibaPython -X utf8 -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }
    & $keibaPython -X utf8 -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency validation failed." }
    & $keibaPython -X utf8 -m keiba_ai --help
    if ($LASTEXITCODE -ne 0) { throw "ML command is unavailable." }
} finally {
    Pop-Location
}
