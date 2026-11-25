# Re-runs sweep_report.py whenever a new result / synthetic-merge file appears, until the sweep
# launcher (process id $WatchPid) exits; then writes one final report.
#   powershell -File report_watch.ps1 <sweep.yaml> <log-name> <report.md> <launcher-pid>
param([string]$Sweep, [string]$LogName, [string]$Report, [int]$WatchPid)
Set-Location $PSScriptRoot
. .\env.ps1
$sw = Get-Content $Sweep -Raw
$name = if ($sw -match '(?m)^name:\s*(\S+)') { $Matches[1] } else { '' }
function Sig { $f = @(Get-ChildItem "runs\$name" -Recurse -Include *.json, *_synth.npz -File -ErrorAction SilentlyContinue); "$($f.Count)" }
function Report { & .\.venv\Scripts\python.exe sweep_report.py $Sweep --log "logs\$LogName.log" --report $Report *>> "logs\${LogName}_report.log" }
$last = ''
while (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue) {
    $s = Sig
    if ($s -ne $last) { Report; $last = $s }
    Start-Sleep 60
}
Report
