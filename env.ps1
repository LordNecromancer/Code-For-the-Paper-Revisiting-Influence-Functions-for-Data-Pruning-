# Dot-source before any command: keeps temp files, caches and downloaded weights on G: (C: is nearly full).
#   . .\env.ps1
$root = 'G:\projects\influence-pruning\.cache'
foreach ($d in 'tmp','pip','torch','hf','mpl','xdg') { New-Item -ItemType Directory -Force (Join-Path $root $d) | Out-Null }
$env:TMP = "$root\tmp"; $env:TEMP = "$root\tmp"
$env:PIP_CACHE_DIR = "$root\pip"
$env:TORCH_HOME = "$root\torch"          # torchvision pretrained weights + torch.hub (DINOv2)
$env:HF_HOME = "$root\hf"
$env:MPLCONFIGDIR = "$root\mpl"
$env:XDG_CACHE_HOME = "$root\xdg"
