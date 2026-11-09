"""Separace stemů přes demucs (htdemucs) s cache. Běží na GPU po segmentech kvůli 6 GB VRAM."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import monitor
from .audio_ops import file_hash, load
from .config import CACHE

STEM_NAMES = ["drums", "bass", "other", "vocals"]
DEMUCS_SR = 44100


def _stem_dir(path: str):
    return CACHE / "stems" / file_hash(path)


def have_stems(path: str) -> bool:
    d = _stem_dir(path)
    return all((d / f"{n}.flac").exists() for n in STEM_NAMES)


def separate(path: str, progress=None) -> dict[str, np.ndarray]:
    """Vrátí {stem: (2, n) float32 @ 48 kHz}. Výsledek se cachuje na disk jako FLAC."""
    d = _stem_dir(path)
    if have_stems(path):
        return {n: sf.read(str(d / f"{n}.flac"), dtype="float32")[0].T.copy() for n in STEM_NAMES}

    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if progress:
        progress(0.0, "demucs: načítám model")
    with monitor.track("demucs", f"htdemucs separace {Path(path).name[:36]}") as jid:
        model = get_model("htdemucs")
        model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)

        wav = torch.from_numpy(load(path, DEMUCS_SR))  # (2, n)
        ref_mean, ref_std = wav.mean(), wav.std() + 1e-8
        wav = (wav - ref_mean) / ref_std
        monitor.update_job(jid, progress=f"separace na {dev}")
        # demucs umí segmentovat sám (split=True), segment 7.8 s drží VRAM pod ~2 GB
        with torch.no_grad():
            out = apply_model(model, wav[None].to(dev), split=True, overlap=0.25,
                              shifts=0, progress=False, device=dev)[0]
        out = (out * ref_std + ref_mean).cpu().numpy()  # (4, 2, n)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        d.mkdir(parents=True, exist_ok=True)
        res: dict[str, np.ndarray] = {}
        for i, name in enumerate(model_sources()):
            tmp44 = d / f"{name}_44.wav"
            sf.write(str(tmp44), out[i].T, DEMUCS_SR)
            tgt = d / f"{name}.flac"
            monitor.run_proc(["ffmpeg", "-v", "error", "-y", "-i", str(tmp44), "-ar", "48000",
                              str(tgt)], "ffmpeg", f"resample stem {name} 48k")
            tmp44.unlink()
            res[name] = sf.read(str(tgt), dtype="float32")[0].T.copy()
    if progress:
        progress(1.0, "demucs hotovo")
    return res


def model_sources() -> list[str]:
    """Pořadí stemů v htdemucs."""
    return STEM_NAMES
