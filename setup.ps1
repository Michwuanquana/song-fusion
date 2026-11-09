# Song Fusion - instalace. Spust jednou: powershell -ExecutionPolicy Bypass -File setup.ps1
# Vse jde na Z: (venv + checkpointy ~12 GB). LM model (3.5 GB) se nestahuje, nepouziva se.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$ace  = Join-Path $root 'vendor\ACE-Step-1.5'
$env:UV_LINK_MODE = 'copy'

if (-not (Test-Path $ace)) {
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5 $ace
}
Push-Location $ace
uv sync --python 3.11
uv pip install --python .venv\Scripts\python.exe demucs librosa requests
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; d('ACE-Step/Ace-Step1.5', local_dir='checkpoints', allow_patterns=['acestep-v15-turbo/*','vae/*','Qwen3-Embedding-0.6B/*','config.json'])"
# ACE-Step kontroluje pritomnost LM vah; prazdny placeholder ho uspokoji (LM se nenacita).
New-Item -ItemType Directory -Force checkpoints\acestep-5Hz-lm-1.7B | Out-Null
New-Item -ItemType File -Force checkpoints\acestep-5Hz-lm-1.7B\model.safetensors | Out-Null
Pop-Location
Write-Host "Hotovo. Spust Fusion.bat"
