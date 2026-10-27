<#
 Start the rtx5080-win fork on its OWN port (18892) with a chosen profile arm, for A/B runs.
 It never stops or replaces the production server. It refuses to start while anything answers on 18891
 or 18892, or while the GPU has less than 14 GiB free (one 16 GB model at a time).
 Kill switches: pass -Disable a,b,c to set BASALT_FORK_<NAME>=0 for each name (see fork/FORK.md for names).
 Usage: .\Start-ForkBasalt.ps1 -Profile a1-ple-ram [-Disable pdep,staging]
#>
param(
  [Parameter(Mandatory)][ValidateSet('a0-daily-control','a1-ple-ram','a2-adapt-slow','a3-reserve-lean')][string]$Profile,
  [string[]]$Disable = @()
)
$ErrorActionPreference = 'Stop'
$task = 'C:/Users/pc-agent/AgentWorkspace/local-ai-20261026'
$fork = 'C:/Users/pc-agent/AgentWorkspace/basalt-5080'
foreach ($port in 18891, 18892) {
  try { Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2 | Out-Null; throw "Port $port answers; stop that server first. Nothing was stopped." }
  catch [System.Net.WebException] { if ($_.Exception.Response) { throw "Port $port is occupied. Nothing was stopped." } }
}
$free = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
if ($free -lt 14336) { throw "Only $free MiB of GPU memory is free; another model is loaded. Nothing was started." }
$profilePath = "$fork/src/fork/profiles/$Profile.json"
$python = "$task/../ai200-20261026/.venv/Scripts/python.exe"
$cuda = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1'
$env:PATH = "$cuda/bin;$cuda/bin/x64;" + $env:PATH
$env:HF_HUB_OFFLINE = '1'; $env:PYTHONNOUSERSITE = '1'
foreach ($name in $Disable) { Set-Item -Path "Env:BASALT_FORK_$($name.ToUpper())" -Value '0' }
$log = "$fork/logs"; New-Item -ItemType Directory -Force -Path $log | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$argsList = @('-u', '-m', 'serve.server', '--engine-bin', "$fork/build/bin/basalt.exe", '--config', $profilePath)
$p = Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory "$fork/src" -NoNewWindow -PassThru `
  -RedirectStandardOutput "$log/$Profile-$stamp.out.log" -RedirectStandardError "$log/$Profile-$stamp.err.log"
[ordered]@{ pid = $p.Id; profile = $Profile; disabled = $Disable; startUtc = $p.StartTime.ToUniversalTime().ToString('o') } |
  ConvertTo-Json | Set-Content "$fork/current-fork-server.json"
Write-Output "Fork Basalt starting on port 18892 with profile $Profile. Logs: $log"
