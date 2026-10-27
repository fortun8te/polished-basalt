<# Stop only the fork server that Start-ForkBasalt.ps1 recorded: fork_launch.py checks the pid against its start time. #>
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:/Users/pc-agent/AgentWorkspace/ai200-20261026/.venv/Scripts/python.exe'
& $python "$root/fork/tools/fork_launch.py" stop --root $root
exit $LASTEXITCODE
