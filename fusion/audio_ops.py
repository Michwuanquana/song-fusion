"""Audio I/O a DSP operace: ffmpeg dekódování, stretch/pitch, crossfade, mix."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SR
from . import monitor


def file_hash(path: str | Path) -> str:
    """Levný otisk souboru (velikost + začátek + konec), stačí na cache klíč."""
    p = Path(path)
    h = hashlib.sha1(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, p.stat().st_size - (1 << 20)))
        h.update(f.read())
    return h.hexdigest()[:16]


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    """Dekóduje libovolný formát přes ffmpeg na float32 (2, n) v daném SR."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", "2", "-ar", str(sr), "-"]
    raw = monitor.run_proc(cmd, "ffmpeg", f"load {Path(path).name[:36]} ({sr} Hz)")
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def save(path: str | Path, audio: np.ndarray, sr: int = SR, subtype: str | None = None) -> Path:
    """Uloží (2, n) audio; formát podle přípony (flac/wav), subtype např. 'PCM_24'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio.T, -1.0, 1.0), sr, subtype=subtype)
    return path


def tmp_wav(audio: np.ndarray, sr: int = SR, prefix: str = "fz_") -> str:
    """Zapíše audio do dočasného WAV a vrátí cestu (pro modely, které chtějí soubor)."""
    fd, name = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    save(name, audio, sr)
    return name


def cut(audio: np.ndarray, start_s: float, dur_s: float, sr: int = SR) -> np.ndarray:
    """Výřez [start, start+dur); mimo rozsah doplní tichem."""
    a, n = int(round(start_s * sr)), int(round(dur_s * sr))
    out = np.zeros((audio.shape[0], n), dtype=np.float32)
    src = audio[:, max(a, 0): max(a, 0) + n]
    off = max(-a, 0)
    out[:, off: off + src.shape[1]] = src[:, : n - off]
    return out


def fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Ořízne nebo doplní tichem na přesně n vzorků."""
    if audio.shape[1] >= n:
        return audio[:, :n]
    return np.pad(audio, ((0, 0), (0, n - audio.shape[1])))


def stretch_pitch(audio: np.ndarray, tempo_ratio: float = 1.0, semitones: float = 0.0,
                  sr: int = SR) -> np.ndarray:
    """Time-stretch a pitch shift přes ffmpeg rubberband (kvalitní, formant-safe pro bicí).

    tempo_ratio > 1 zrychlí (cílové BPM / zdrojové BPM).
    """
    if abs(tempo_ratio - 1.0) < 1e-4 and abs(semitones) < 1e-3:
        return audio
    pitch = 2.0 ** (semitones / 12.0)
    src = tmp_wav(audio, sr)
    try:
        flt = f"rubberband=tempo={tempo_ratio:.6f}:pitch={pitch:.6f}:transients=crisp:window=short"
        cmd = ["ffmpeg", "-v", "error", "-i", src, "-af", flt, "-f", "f32le",
               "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
        dur_s = audio.shape[1] / sr
        raw = monitor.run_proc(
            cmd, "ffmpeg",
            f"rubberband {dur_s:.0f}s · tempo ×{tempo_ratio:.2f} · pitch {semitones:+.0f}st",
        )
    finally:
        os.remove(src)
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def rms(audio: np.ndarray) -> float:
    """RMS přes oba kanály."""
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def match_loudness(audio: np.ndarray, ref: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Srovná RMS na referenci a pohlídá peak pod stropem (měkký limiter přes tanh)."""
    out = audio * (rms(ref) / rms(audio))
    ceiling = 10 ** (ceiling_db / 20)
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > ceiling:
        out = np.tanh(out / ceiling) * ceiling
    return out.astype(np.float32)


def mix(*tracks: np.ndarray, gains: list[float] | None = None) -> np.ndarray:
    """Sečte stopy (ořízne na nejkratší)."""
    n = min(t.shape[1] for t in tracks)
    gains = gains or [1.0] * len(tracks)
    return sum(g * t[:, :n] for g, t in zip(gains, tracks)).astype(np.float32)


def equal_power_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Vrátí (fade_out, fade_in) křivky délky n s konstantním výkonem."""
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    return np.cos(t), np.sin(t)


def crossfade_join(a: np.ndarray, b: np.ndarray, xfade: int) -> np.ndarray:
    """Spojí a + b, posledních xfade vzorků a přelije do prvních xfade vzorků b."""
    xfade = min(xfade, a.shape[1], b.shape[1])
    if xfade <= 0:
        return np.concatenate([a, b], axis=1)
    fo, fi = equal_power_fade(xfade)
    mid = a[:, -xfade:] * fo + b[:, :xfade] * fi
    return np.concatenate([a[:, :-xfade], mid, b[:, xfade:]], axis=1)
