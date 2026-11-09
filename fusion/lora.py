"""Vibe pack: dataset z jedné skladby (řezy po taktech) + trénink LoRA přes Side-Step (train.py).

Experimentální na 6 GB VRAM (oficiální minimum je 8 GB). Používá Adafactor (bez bitsandbytes),
gradient checkpointing a offload enkodéru. Pokud trénink spadne na OOM, zkus nižší rank
nebo kratší řezy.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from . import audio_ops as ao
from . import monitor
from .analysis import analyze
from .config import LORAS, SR, VENDOR

DATASETS = LORAS / "_datasets"
PYTHON = VENDOR / ".venv" / "Scripts" / "python.exe"


def make_dataset(song: str, name: str, caption: str, slice_bars: int = 8,
                 max_slices: int = 24) -> Path:
    """Nařeže skladbu na řezy po `slice_bars` taktech (nejenergičtější max_slices) + dataset.json."""
    an = analyze(song)
    audio = ao.load(song)
    d = DATASETS / name
    (d / "audio").mkdir(parents=True, exist_ok=True)
    for old in (d / "audio").glob("*.wav"):
        old.unlink()

    n_slices = an.n_bars // slice_bars
    energy = [float(np.mean(an.bar_energy[i * slice_bars:(i + 1) * slice_bars]))
              for i in range(n_slices)]
    floor = np.percentile(energy, 25) if energy else 0.0  # vyhodit ticho / intra
    order = [i for i in np.argsort(energy)[::-1] if energy[i] > floor][:max_slices]

    samples = []
    for k, i in enumerate(sorted(order)):
        seg = ao.cut(audio, an.bar_time(i * slice_bars), slice_bars * an.bar_len)
        fn = f"{name}_{k:02d}.wav"
        ao.save(d / "audio" / fn, seg)
        samples.append({"filename": fn, "audio_path": str(d / "audio" / fn), "caption": caption,
                        "lyrics": "[Instrumental]", "bpm": int(round(an.bpm)), "keyscale": an.key,
                        "timesignature": "4"})
    (d / "dataset.json").write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    return d


def _stream(cmd: list[str], detail: str = "train.py") -> Iterator[str]:
    """Spustí proces ve VENDOR, sleduje ho v monitoru a yielduje řádky výstupu."""
    jid = monitor.start_job("lora", detail)
    try:
        p = subprocess.Popen(cmd, cwd=str(VENDOR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             env={**__import__("os").environ, "PYTHONUTF8": "1",
                                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        monitor.update_job(jid, pid=p.pid)
        assert p.stdout
        for line in p.stdout:
            yield line.rstrip()
        p.wait()
        monitor.end_job(jid, ok=(p.returncode == 0), extra=f"exit {p.returncode}")
        yield f"[exit {p.returncode}]"
    except Exception as e:
        monitor.end_job(jid, ok=False, extra=str(e)[:120])
        raise


def train(song: str, name: str, caption: str, epochs: int = 30, rank: int = 16,
          slice_bars: int = 8) -> Iterator[str]:
    """Generátor logu: dataset -> preprocessing -> trénink. Výstup do loras/<name>/."""
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "vibe"
    yield f"Dělám dataset z {Path(song).name} …"
    d = make_dataset(song, name, caption, slice_bars)
    n = len(json.loads((d / "dataset.json").read_text(encoding="utf-8"))["samples"])
    yield f"Dataset: {n} řezů po {slice_bars} taktech → {d}"

    tensors = d / "tensors"
    out = LORAS / name
    common = ["--checkpoint-dir", "checkpoints", "--model-variant", "turbo"]
    yield "— Preprocessing (dvouprůchodový, šetří VRAM) —"
    last = ""
    for line in _stream([str(PYTHON), "train.py", "--preprocess", "--dataset-json",
                         str(d / "dataset.json"), "--tensor-output", str(tensors), *common]):
        last = line
        yield line
    if "[exit 0]" not in last:
        yield "✘ Preprocessing selhal."
        return
    yield f"— Trénink: rank {rank}, {epochs} epoch —"
    for line in _stream([str(PYTHON), "train.py", "fixed", *common, "--dataset-dir", str(tensors),
                         "--output-dir", str(out), "--rank", str(rank), "--alpha", str(rank * 2),
                         "--epochs", str(epochs), "--optimizer-type", "adafactor",
                         "--offload-encoder", "--gradient-checkpointing", "--save-every",
                         str(max(epochs // 3, 1)), "--batch-size", "1",
                         "--gradient-accumulation", "2", "--warmup-steps", "10"]):
        last = line
        yield line
    yield ("✔ Hotovo. LoRA je v dropdownu v Pokročilé." if "[exit 0]" in last
           else "✘ Trénink selhal (viz log výše, nejspíš OOM).")


def resolve_adapter(folder: str | Path) -> str | None:
    """Najde adaptér v adresáři tréninku (preferuje 'final', pak nejnovější checkpoint)."""
    root = Path(folder)
    cands = [p.parent for p in root.rglob("adapter_model.safetensors")]
    cands += [p.parent for p in root.rglob("*lokr*.safetensors")]
    if not cands:
        return None
    final = [c for c in cands if "final" in c.name.lower()]
    best = final[0] if final else max(cands, key=lambda c: c.stat().st_mtime)
    return str(best)
