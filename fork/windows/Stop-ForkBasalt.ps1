<# Stop only the fork server that Start-ForkBasalt.ps1 recorded (identity-checked by pid and start time). #>
$ErrorActionPreference = 'Stop'
$rec = Get-Content 'C:/Users/pc-agent/AgentWorkspace/basalt-5080/current-fork-server.json' | ConvertFrom-Json
$p = Get-Process -Id $rec.pid -ErrorAction SilentlyContinue
if (-not $p) { Write-Output 'Fork server is not running.'; exit 0 }
if ($p.StartTime.ToUniversalTime().ToString('o') -ne $rec.startUtc) { throw 'Process id was reused by something else. Nothing was stopped.' }
Stop-Process -Id $p.Id
Write-Output 'Fork server stopped.'
