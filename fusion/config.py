"""Cesty a výchozí nastavení pro Song Fusion."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "ACE-Step-1.5"
CHECKPOINTS = VENDOR / "checkpoints"
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"
LORAS = ROOT / "loras"
MUSIC_DIR = ROOT.parent / "music"

SR = 48000  # ACE-Step pracuje nativně ve 48 kHz stereo

DIT_MODEL = "acestep-v15-turbo"
QUANTIZATION = "int8_weight_only"

OLLAMA_URL = "http://localhost:11434"
# Preferované pořadí; první nainstalovaný vyhrává.
OLLAMA_MODELS = [
    "huihui_ai/qwen3-abliterated:30b-a3b-instruct-2507-q4_K_M",
    "huihui_ai/qwen3-abliterated:8b",
    "qwen3:8b",
    "dolphin3:latest",
    "gemma3:4b",
]

for _d in (CACHE, OUTPUTS, LORAS):
    _d.mkdir(parents=True, exist_ok=True)
