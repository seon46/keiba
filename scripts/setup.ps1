param(
    [string]$Python = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedNative {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    $pythonArguments = @()
    if ($Python) {
        $pythonCommand = Get-Command -Name $Python -CommandType Application -ErrorAction Stop
    } else {
        $pythonCommand = Get-Command -Name py -CommandType Application -ErrorAction SilentlyContinue
        if ($null -ne $pythonCommand) {
            $pythonArguments = @("-3")
        } else {
            $pythonCommand = Get-Command -Name python -CommandType Application -ErrorAction SilentlyContinue
            if ($null -eq $pythonCommand) {
                $pythonCommand = Get-Command -Name python3 -CommandType Application -ErrorAction SilentlyContinue
            }
        }
    }
    if ($null -eq $pythonCommand) {
        throw "Python 3.11 or newer was not found. Install Python and reopen PowerShell."
    }

    $versionCheck = "import sys; sys.exit('Python 3.11 or newer is required.' if sys.version_info < (3, 11) else 0)"
    Invoke-CheckedNative $pythonCommand.Source ($pythonArguments + @("-c", $versionCheck))
    Invoke-CheckedNative $pythonCommand.Source ($pythonArguments + @("-m", "venv", ".venv"))

    $venvPython = Join-Path $projectDirectory ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "This setup script requires Windows. On Linux/macOS use bash scripts/setup.sh."
    }

    Invoke-CheckedNative $venvPython @("-X", "utf8", "-m", "pip", "--disable-pip-version-check", "install", "-r", "requirements.txt")
    Invoke-CheckedNative $venvPython @("-X", "utf8", "-m", "pip", "--disable-pip-version-check", "install", "--no-deps", "--no-build-isolation", "-e", ".")
    Invoke-CheckedNative $venvPython @("-X", "utf8", "-m", "pip", "check")
    Invoke-CheckedNative $venvPython @("-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v")
    Invoke-CheckedNative $venvPython @("-X", "utf8", "-m", "keiba_scraper", "--help")
} finally {
    Pop-Location
}
