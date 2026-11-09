"""Acid riff: generátor vzoru ve stylu 303, syntéza a export do MIDI."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from .config import SR
from .dsp import rbj


def _pattern(seed: int, tonic: int, mode: str, density: float, octave: int,
             steps: int = 32) -> list[dict]:
    rng = np.random.default_rng(seed)
    scale = [0, 3, 5, 7, 10] if mode == "minor" else [0, 2, 4, 7, 9]
    base = 36 + tonic + 12 * octave
    pat = []
    for i in range(steps):
        on = (i % 4 == 0 and rng.random() < min(1.0, density + 0.35)) or rng.random() < density
        deg = int(rng.choice(5, p=[.38, .17, .17, .18, .10]))
        pat.append({"on": bool(on), "note": base + scale[deg] + (12 if rng.random() < 0.18 else 0),
                    "acc": bool(rng.random() < 0.28), "slide": bool(rng.random() < 0.18)})
    return pat


def acid_steps(bars: int, seed: int, tonic: int, mode: str, density: float,
               octave: int = 0) -> list[dict]:
    """Šestnáctiny pro `bars` taktů: dvoutaktový vzor, každá druhá čtyřtaktová fráze lehce mutovaná.

    Stejný seed = stejná identita riffu napříč sekcemi.
    """
    base = _pattern(seed, tonic, mode, density, octave)
    steps: list[dict] = []
    for bar in range(bars):
        phrase = bar // 4
        pat = [dict(s) for s in base]
        if phrase % 2:
            rng = np.random.default_rng(seed * 1000 + phrase)
            for i in rng.choice(32, 4, replace=False):
                pat[i]["on"] = not pat[i]["on"]
        steps += pat[(bar % 2) * 16:(bar % 2) * 16 + 16]
    return steps


def acid_synth(steps: list[dict], n: int, bar_samples: float, sweep: tuple[float, float] = (0.6, 1.4),
               cutoff: float = 380.0, reso: float = 7.0, decay: float = 0.14) -> np.ndarray:
    """Saw + rezonanční low-pass s obálkou na akcentech, glide na slide. Vrací stereo (2, n)."""
    step_len = bar_samples / 16
    f_inst = np.zeros(n)
    amp = np.zeros(n)
    since = np.zeros(n)
    acc_scale = np.ones(n)
    prev_f, prev_slide = None, False
    for i, st in enumerate(steps):
        s, e = int(round(i * step_len)), min(int(round((i + 1) * step_len)), n)
        if s >= n:
            break
        t = np.arange(e - s) / SR
        f_t = 440.0 * 2 ** ((st["note"] - 69) / 12)
        if st["on"]:
            f0 = prev_f if prev_f is not None else f_t
            f_inst[s:e] = f_t + (f0 - f_t) * np.exp(-t / (0.055 if prev_slide else 0.004))
            prev_f = f_t
            glen = (e - s) if st["slide"] else int((e - s) * 0.62)
            g = np.zeros(e - s)
            g[:glen] = 1.0
            a_n, r_n = min(int(0.003 * SR), len(g)), int(0.008 * SR)
            g[:a_n] *= np.linspace(0, 1, a_n)
            r0 = max(glen - r_n, 0)
            g[r0:glen] *= np.linspace(1, 0, glen - r0)
            amp[s:e] = g * (1.0 if st["acc"] else 0.68)
            acc_scale[s:e] = 1.7 if st["acc"] else 1.0
            prev_slide = st["slide"]
        else:
            f_inst[s:e] = prev_f if prev_f is not None else f_t
            prev_slide = False
        since[s:e] = t
    f_inst[f_inst == 0] = 110.0
    saw = (2 * (np.cumsum(f_inst / SR) % 1.0) - 1) * amp

    block = 128
    nb = -(-n // block)
    out = np.empty(n)
    zi = np.zeros(2)
    curve = np.geomspace(sweep[0], sweep[1], nb)
    for k in range(nb):
        s, e = k * block, min((k + 1) * block, n)
        env = np.exp(-since[s] / max(decay, 0.02))
        fc = cutoff * curve[k] * (1.0 + 3.2 * env * acc_scale[s])
        b, a = rbj("lp", min(fc, 9000.0), reso)
        out[s:e], zi = lfilter(b, a, saw[s:e], zi=zi)
    y = np.tanh(1.1 * out / (np.std(out) + 1e-9)).astype(np.float32)
    return np.stack([y, y])


def midi_events(steps: list[dict], step_offset: int) -> list[tuple[int, int, int, int]]:
    """(start, délka, nota, velocity) v šestnáctinách; slide = legato do další noty."""
    ev = []
    for i, st in enumerate(steps):
        if st["on"]:
            length = 2 if st["slide"] else 1
            ev.append((step_offset + i, length, int(st["note"]), 120 if st["acc"] else 90))
    return ev


def _vlq(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.insert(0, (v & 0x7F) | 0x80)
        v >>= 7
    return bytes(out)


def write_midi(path: str | Path, events: list[tuple[int, int, int, int]], bpm: float,
               ppq: int = 96) -> Path:
    """Minimální SMF (typ 0): tempo + noty. Délky v šestnáctinách, gate 60 % (slide = přesah)."""
    tick16 = ppq // 4
    msgs = []
    for start, length, note, vel in events:
        on = start * tick16
        off = on + (length * tick16 if length > 1 else int(tick16 * 0.6))
        msgs.append((on, 1, bytes([0x90, note, vel])))
        msgs.append((off, 0, bytes([0x80, note, 0])))
    msgs.sort(key=lambda m: (m[0], m[1]))
    track = bytearray()
    track += _vlq(0) + b"\xff\x03" + _vlq(9) + b"Acid 303 "
    track += _vlq(0) + b"\xff\x51\x03" + int(round(60e6 / bpm)).to_bytes(3, "big")
    last = 0
    for tick, _, data in msgs:
        track += _vlq(tick - last) + data
        last = tick
    track += _vlq(0) + b"\xff\x2f\x00"
    path = Path(path)
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
                     + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
    return path
