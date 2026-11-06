# Restarts the keiba_app browser server: stops every running "keiba_app serve" (the scheduled task's and any
# started by hand), waits until port 8765 is free, then starts the "KeibaApp Server" task again.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\restart-app.ps1
param([int]$Port = 8765)
$ErrorActionPreference = "Stop"
$taskName = "KeibaApp Server"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($task) { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue }
$servers = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
    Where-Object { $_.CommandLine -like '*keiba_app*serve*' })
foreach ($process in $servers) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host "停止したアプリ: $($servers.Count)個"

for ($i = 0; $i -lt 20; $i++) {
    if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 500
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    throw "ポート$Port をまだ別のプログラムが使っています: $($owner.Name) (PID $($owner.ProcessId)) $($owner.CommandLine)"
}

if ($task) {
    Start-ScheduledTask -TaskName $taskName
    Write-Host "起動しました。ブラウザを再読み込みしてください: http://127.0.0.1:$Port"
} else {
    Write-Host "タスク「$taskName」がないので、別のPowerShellで次を実行してください:"
    Write-Host "  .\.venv\Scripts\python.exe -X utf8 -m keiba_app serve"
}
