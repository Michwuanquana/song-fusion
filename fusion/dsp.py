"""DSP stavebnice aranžéru: filtry, echo, gate, sidechain obálka a syntéza FX."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from . import audio_ops as ao
from .config import SR


def rbj(kind: str, fc: float, q: float):
    """Koeficienty biquadu (RBJ cookbook), kind 'lp' nebo 'hp'."""
    fc = float(np.clip(fc, 20.0, SR * 0.45))
    w0 = 2 * np.pi * fc / SR
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if kind == "lp":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    else:
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def sweep_filter(x: np.ndarray, f0: float, f1: float, kind: str = "lp", q: float = 0.8,
                 block: int = 256) -> np.ndarray:
    """Filtr, jehož frekvence se plynule (geometricky) mění z f0 na f1 přes celý úsek."""
    n = x.shape[1]
    nb = max(1, -(-n // block))
    fcs = np.geomspace(f0, f1, nb)
    out = np.empty_like(x)
    zi = np.zeros((x.shape[0], 2))
    for k in range(nb):
        s, e = k * block, min((k + 1) * block, n)
        b, a = rbj(kind, fcs[k], q)
        for c in range(x.shape[0]):
            out[c, s:e], zi[c] = lfilter(b, a, x[c, s:e], zi=zi[c])
    return out


def static_filter(x: np.ndarray, kind: str, fc: float, order: int = 2) -> np.ndarray:
    sos = butter(order, fc, btype="low" if kind == "lp" else "high", fs=SR, output="sos")
    return sosfilt(sos, x, axis=1).astype(np.float32)


def echo(x: np.ndarray, delay_l: int, delay_r: int, fb: float = 0.42, taps: int = 6) -> np.ndarray:
    """Ping-pong echo s různým zpožděním L/R a utlumenými výškami v ozvěnách."""
    n = x.shape[1]
    tail = np.zeros_like(x)
    for ch, d in enumerate((delay_l, delay_r)):
        for k in range(1, taps + 1):
            sh = d * k
            if sh >= n:
                break
            tail[ch, sh:] += (fb ** k) * x[1 - ch if k % 2 else ch, : n - sh]
    return x + static_filter(tail, "lp", 3200, 1)


def lvl(x: np.ndarray, target: float) -> np.ndarray:
    """Normalizuje RMS na `target`; ticho nechá tichem."""
    r = ao.rms(x)
    return x * (target / r) if r > 1e-5 else np.zeros_like(x)


def ramp(n: int, a: float, b: float) -> np.ndarray:
    return np.linspace(a, b, n, dtype=np.float32)


def fade_edges(x: np.ndarray, ms: float = 4.0) -> np.ndarray:
    k = min(int(SR * ms / 1000), x.shape[1] // 2)
    if k > 0:
        x = x.copy()
        x[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)
        x[:, -k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return x


def kick_env(drums: np.ndarray) -> np.ndarray:
    """Obálka kicku 0..1 z pásma 35-140 Hz (pro sidechain pumpu)."""
    sos = butter(2, [35, 140], btype="band", fs=SR, output="sos")
    k = np.abs(sosfilt(sos, drums.mean(axis=0)))
    e = sosfilt(butter(1, 18, fs=SR, output="sos"), k)
    return np.clip(e / (np.percentile(e, 97) + 1e-9), 0.0, 1.0).astype(np.float32)


def gate(x: np.ndarray, pattern: list[int], bar_samples: float) -> np.ndarray:
    """Rytmický gate po 16 dobách taktu; x musí začínat na začátku taktu."""
    step = bar_samples / 16
    idx = (np.arange(x.shape[1]) / step).astype(int) % 16
    g = np.asarray(pattern, dtype=np.float32)[idx]
    k = int(0.006 * SR)
    g = np.convolve(g, np.ones(k, dtype=np.float32) / k, mode="same")
    return x * g


def retrigger(x: np.ndarray, slice_len: int) -> np.ndarray:
    """Stutter: opakuje začátek úseku po celé jeho délce."""
    s = fade_edges(x[:, :max(slice_len, 64)], 3.0)
    return ao.fit_length(np.tile(s, (1, -(-x.shape[1] // s.shape[1]))), x.shape[1])


def noise_riser(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((2, n)).astype(np.float32)
    x = sweep_filter(x, 400, 9000, "hp", 0.9)
    return lvl(x * ((np.arange(n) / n) ** 2.2).astype(np.float32), 0.10)


def noise_downlifter(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((2, n)).astype(np.float32)
    x = sweep_filter(x, 9000, 300, "lp", 0.9)
    return lvl(x * np.exp(-3.0 * np.arange(n) / n).astype(np.float32), 0.09)


def impact(rng: np.random.Generator, dur: float = 1.3) -> np.ndarray:
    """Sub boom s klesající výškou + krátký šumový úder."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(32 + 38 * np.exp(-t * 5.0)) / SR) * np.exp(-t * 3.0)
    noise = rng.standard_normal((1, n)).astype(np.float32)
    hit = static_filter(noise, "lp", 3500)[0] * np.exp(-t * 11)
    x = (0.9 * boom + 0.35 * hit).astype(np.float32)
    return np.stack([x, x])


def snare_hit(rng: np.random.Generator, dur: float = 0.14) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = static_filter(rng.standard_normal((1, n)).astype(np.float32), "hp", 1200)[0]
    x = (0.8 * noise * np.exp(-t * 28) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40))
    return np.stack([x, x]).astype(np.float32)


def snare_roll(bars: float, bar_samples: float, rng: np.random.Generator,
               end_gap_beats: int = 1) -> np.ndarray:
    """Zrychlující snare roll na gridu: osminy -> šestnáctiny -> poslední doba 32iny.

    Končí `end_gap_beats` dob před koncem (tam je ticho před dropem).
    """
    n = int(round(bars * bar_samples))
    out = np.zeros((2, n), dtype=np.float32)
    hit = snare_hit(rng)
    total32 = int(bars * 32) - end_gap_beats * 8
    i = 0
    while i < total32:
        s = int(round(i * bar_samples / 32))
        e = min(s + hit.shape[1], n)
        out[:, s:e] += hit[:, : e - s] * (0.3 + 0.7 * (i / max(total32, 1)) ** 1.5)
        rem = total32 - i
        i += 1 if rem <= 8 else (2 if i >= total32 / 2 else 4)
    return lvl(sweep_filter(out, 250, 2500, "hp", 0.7), 0.07)


def kick_click(bars: int, bar_samples: float) -> np.ndarray:
    """Jednoduchý syntetický 4/4 kick (jen pro náhledy riffů)."""
    n = int(round(bars * bar_samples))
    out = np.zeros(n, dtype=np.float32)
    m = int(0.25 * SR)
    t = np.arange(m) / SR
    k = (np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-t * 32)) / SR) * np.exp(-t * 9)).astype(np.float32)
    for beat in range(bars * 4):
        s = int(round(beat * bar_samples / 4))
        e = min(s + m, n)
        out[s:e] += k[: e - s]
    return np.stack([out, out]) * 0.5
