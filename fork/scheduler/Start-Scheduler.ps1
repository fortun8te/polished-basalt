<#
 Start the request scheduler in front of a running Basalt server. The scheduler listens on its own port, so clients
 are pointed at it (default http://127.0.0.1:8081) and it forwards to the server (default http://127.0.0.1:8080).
 It never starts, stops or changes the Basalt server, and it refuses to start while its own port is taken.
 It needs only Python 3.9 or newer; no packages.
 Usage:  .\Start-Scheduler.ps1 [-Config scheduler.json] [-Listen 127.0.0.1:8081] [-Upstream http://127.0.0.1:8080]
         .\Start-Scheduler.ps1 -Stop      (stops only the scheduler this script recorded, checked by pid and start time)
 Status: Invoke-RestMethod http://127.0.0.1:8081/_sched/status
#>
param(
  [string]$Config = (Join-Path $PSScriptRoot 'scheduler.example.json'),
  [string]$Listen,
  [string]$Upstream,
  [string]$Python = 'python',
  [string]$LogDir = (Join-Path $PSScriptRoot 'logs'),
  [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$record = Join-Path $LogDir 'scheduler.pid.json'

if ($Stop) {
  if (-not (Test-Path $record)) { Write-Output 'No scheduler was recorded.'; exit 0 }
  $rec = Get-Content $record | ConvertFrom-Json
  $p = Get-Process -Id $rec.pid -ErrorAction SilentlyContinue
  if (-not $p) { Write-Output 'Scheduler is not running.'; exit 0 }
  if ($p.StartTime.ToUniversalTime().ToString('o') -ne $rec.startUtc) { throw 'Process id was reused by something else. Nothing was stopped.' }
  Stop-Process -Id $p.Id
  Write-Output 'Scheduler stopped.'
  exit 0
}

$settings = Get-Content $Config -Raw | ConvertFrom-Json
if (-not $Listen) { $Listen = if ($settings.listen) { $settings.listen } else { '127.0.0.1:8081' } }
if (-not $Upstream) { $Upstream = if ($settings.upstream) { $settings.upstream } else { 'http://127.0.0.1:8080' } }
$port = [int]($Listen -split ':')[-1]
if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
  throw "Port $port is already in use. Nothing was started."
}
try { Invoke-RestMethod "$Upstream/health" -TimeoutSec 2 | Out-Null }
catch { Write-Warning "Nothing answers at $Upstream yet; requests get a 502 until the server is up." }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$argsList = @('-u', (Join-Path $PSScriptRoot 'scheduler_proxy.py'), '--config', $Config, '--listen', $Listen, '--upstream', $Upstream)
$p = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru `
  -RedirectStandardOutput (Join-Path $LogDir "scheduler-$stamp.out.log") -RedirectStandardError (Join-Path $LogDir "scheduler-$stamp.log")
[ordered]@{ pid = $p.Id; listen = $Listen; upstream = $Upstream; startUtc = $p.StartTime.ToUniversalTime().ToString('o') } |
  ConvertTo-Json | Set-Content $record
Start-Sleep -Seconds 2
if ($p.HasExited) { throw "Scheduler exited with code $($p.ExitCode); see $LogDir\scheduler-$stamp.log" }
Write-Output "Scheduler on http://$Listen -> $Upstream (pid $($p.Id)). Log: $LogDir\scheduler-$stamp.log"
