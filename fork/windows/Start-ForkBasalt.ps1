<#
 Start the fork on its arm's own port, only when the files on disk are the candidate that Build-Fork.cmd recorded.
 Every check and the launch itself are in fork/tools/fork_launch.py; this wrapper only locates Python and CUDA.
 It refuses, and starts nothing, unless: the engine (bin\basalt.exe in the source tree), the source commit and tree,
 the arm's profile and the model files match fork-identity.json; nothing listens on 18891 or on this arm's port; and
 14 GiB of GPU memory are free. Logs, the receipt and every file the engine writes go to fork-runs\<arm>\<time>\.
 Switches (BASALT_FORK_<NAME>; the caller's own BASALT_FORK_* are never inherited):
   -Disable STAGING,PDEP_SIGNS   set to 0 (on by default; names in docs/fork.md and docs/polished-basalt/PATCHES.md)
   -Enable GDN_REC2              set to 1 (off by default)
   -Number DRAFT_CUM_MULT=1.2    a number switch
 Model files are checked by sha256. -WaitSeconds N waits for a loaded Basalt /health response.
 Usage: .\Start-ForkBasalt.ps1 -Profile a1-ple-ram [-Disable staging] [-WaitSeconds 600]
#>
param(
  [Parameter(Mandatory)][string]$Profile,
  [string[]]$Disable = @(), [string[]]$Enable = @(), [string[]]$Number = @(),
  [int]$WaitSeconds = 0
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'C:/Users/pc-agent/AgentWorkspace/ai200-20261026/.venv/Scripts/python.exe'
$cuda = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1'
$argv = @("$root/fork/tools/fork_launch.py", 'start', '--root', $root, '--arm', $Profile, '--cuda-path', $cuda, '--wait', $WaitSeconds)
$argv += @('--models', 'full')
foreach ($n in $Disable) { $argv += @('--disable', $n) }
foreach ($n in $Enable) { $argv += @('--enable', $n) }
foreach ($n in $Number) { $argv += @('--set', $n) }
& $python @argv
exit $LASTEXITCODE
