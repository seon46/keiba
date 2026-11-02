# Registers two Windows scheduled tasks for keiba_app (current user, no administrator rights needed):
#   KeibaApp Nightly : every day at the given time, wakes the PC, adds results / updates / retrains / predicts today
#   KeibaApp Server  : starts the browser app at logon (http://127.0.0.1:8765)
# Remove them with: .\scripts\install-app-tasks.ps1 -Remove
param(
    [string]$NightlyTime = "02:30",
    [int]$Port = 8765,
    [switch]$Remove
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $projectDirectory ".venv\Scripts\pythonw.exe"
$nightlyName = "KeibaApp Nightly"
$serverName = "KeibaApp Server"

if ($Remove) {
    foreach ($name in @($nightlyName, $serverName)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed: $name"
        }
    }
    return
}
if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw "Run scripts/setup.ps1 and scripts/setup-ml.ps1 first ($pythonw not found)."
}

$nightlyAction = New-ScheduledTaskAction -Execute $pythonw -Argument "-X utf8 -m keiba_app nightly" -WorkingDirectory $projectDirectory
$nightlyTrigger = New-ScheduledTaskTrigger -Daily -At $NightlyTime
$nightlySettings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 8) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $nightlyName -Action $nightlyAction -Trigger $nightlyTrigger -Settings $nightlySettings `
    -Description "keiba_app: results, database update, scoring, weekly retraining, today's predictions" -Force | Out-Null
Write-Host "Registered: $nightlyName (daily $NightlyTime)"

$serverAction = New-ScheduledTaskAction -Execute $pythonw -Argument "-X utf8 -m keiba_app serve --port $Port" -WorkingDirectory $projectDirectory
$serverTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$serverSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $serverName -Action $serverAction -Trigger $serverTrigger -Settings $serverSettings `
    -Description "keiba_app browser UI on http://127.0.0.1:$Port" -Force | Out-Null
Start-ScheduledTask -TaskName $serverName
Write-Host "Registered and started: $serverName (http://127.0.0.1:$Port)"
