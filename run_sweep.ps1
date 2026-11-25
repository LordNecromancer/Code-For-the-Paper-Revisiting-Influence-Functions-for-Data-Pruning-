# Detached sweep launcher for this Windows PC. Keeps the PC awake only while the sweep runs
# (no power-plan change), keeps caches on G: (env.ps1), and logs to logs\<log>.log.
#   powershell -File run_sweep.ps1 <log-name> <sweep.yaml> [extra dpif.sweep args...]
param([Parameter(Mandatory)][string]$LogName, [Parameter(Mandatory)][string]$Sweep,
      [Parameter(ValueFromRemainingArguments)][string[]]$Rest)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
. .\env.ps1
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][W.P]::SetThreadExecutionState([uint32]'0x80000001')   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$log = "logs\$LogName.log"
$utf8 = New-Object System.Text.UTF8Encoding $false
# live report under runs/<name>/REPORT_<log-name>.md, refreshed after every finished job
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
    "`"$PSScriptRoot\report_watch.ps1`"", $Sweep, $LogName, "REPORT_$LogName.md", $PID)
[IO.File]::AppendAllText("$PSScriptRoot\$log", "[launcher] $(Get-Date -Format s) start: dpif.sweep $Sweep $Rest (pid $PID)`r`n", $utf8)
try {
    # cmd does the redirection so the log is plain bytes (PowerShell 5.1 '>>' would write UTF-16)
    cmd /c ".venv\Scripts\python.exe -u -m dpif.sweep $Sweep $Rest >> $log 2>&1"
    [IO.File]::AppendAllText("$PSScriptRoot\$log", "[launcher] $(Get-Date -Format s) exit code $LASTEXITCODE`r`n", $utf8)
} finally {
    [void][W.P]::SetThreadExecutionState([uint32]'0x80000000')  # back to normal sleep policy
}
