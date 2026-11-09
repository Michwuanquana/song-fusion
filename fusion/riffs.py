"""Riffy ze songu B: hledání opakujících se smyček, natažení na cílové tempo a tóninu, variace."""
from __future__ import annotations

import librosa
import numpy as np

from . import audio_ops as ao
from .analysis import Analysis
from .config import SR
from .dsp import fade_edges, gate, retrigger

GATES = {  # rytmické gate vzory po 16 dobách taktu
    "a": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    "b": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    "c": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
}

_CACHE: dict = {}


def _bar_features(b: Analysis, other: np.ndarray, bass: np.ndarray):
    """Chroma po osminách taktu (nb, 8, 12) a RMS stemu `other` po taktech; cachováno."""
    key = (b.path, round(b.bpm, 3), round(b.downbeat0, 4))
    if key in _CACHE:
        return _CACHE[key]
    mono = librosa.resample((other + bass).mean(axis=0), orig_sr=SR, target_sr=22050)
    hop = 512
    chroma = librosa.feature.chroma_stft(y=mono, sr=22050, hop_length=hop)
    fps = 22050 / hop
    nb = b.n_bars
    C = np.zeros((nb, 8, 12), dtype=np.float32)
    for i in range(nb):
        for j in range(8):
            f0 = int((b.bar_time(i) + j * b.bar_len / 8) * fps)
            f1 = max(int((b.bar_time(i) + (j + 1) * b.bar_len / 8) * fps), f0 + 1)
            seg = chroma[:, f0:f1]
            if seg.shape[1]:
                C[i, j] = seg.mean(axis=1)
    rms = np.array([ao.rms(ao.cut(other, b.bar_time(i), b.bar_len)) for i in range(nb)])
    _CACHE.clear()
    _CACHE[key] = (C, rms)
    return C, rms


def candidates(b: Analysis, other: np.ndarray, bass: np.ndarray, n_bars: int = 4,
               top: int = 8) -> list[dict]:
    """Nepřekrývající se kandidáti na riff, seřazení podle skóre.

    Skóre = podobnost chroma okna s následujícím oknem (riff se opakuje) x hlasitost stemu other.
    """
    C, rms = _bar_features(b, other, bass)
    nb = len(rms)
    cand = []
    for s in range(0, nb - 2 * n_bars + 1, 2):
        u, v = C[s:s + n_bars].ravel(), C[s + n_bars:s + 2 * n_bars].ravel()
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        cand.append((s, cos, float(rms[s:s + n_bars].mean())))
    if not cand:
        return [{"start_bar": 0, "score": 0.0, "time_s": round(b.bar_time(0), 2)}]
    loud = np.array([c[2] for c in cand])
    rank = loud.argsort().argsort() / max(len(loud) - 1, 1)
    scored = sorted(((c[1] * (0.4 + 0.6 * r), c[0]) for c, r in zip(cand, rank)), reverse=True)
    out: list[dict] = []
    for sc, s in scored:
        if all(abs(s - o["start_bar"]) >= n_bars for o in out):
            out.append({"start_bar": int(s), "score": round(float(sc), 3),
                        "time_s": round(b.bar_time(s), 2)})
        if len(out) == top:
            break
    return out


def label(c: dict, i: int, n_bars: int) -> str:
    t = c["time_s"]
    return (f"#{i + 1} · {int(t // 60)}:{int(t % 60):02d} · takty {c['start_bar'] + 1}-"
            f"{c['start_bar'] + n_bars} · skóre {c['score']:.2f}")


def make_loops(b: Analysis, other: np.ndarray, bass: np.ndarray, start_bar: int, n_bars: int,
               ratio: float, semis: int, out_bar_len: float) -> dict[str, np.ndarray]:
    """Smyčky z B natažené (ratio) a přeladěné (semis) na cílový grid. Klíče riff, oct, bass."""
    n = int(round(n_bars * out_bar_len * SR))
    t0, dur = b.bar_time(start_bar), n_bars * b.bar_len
    o, bs = ao.cut(other, t0, dur), ao.cut(bass, t0, dur)
    return {
        "riff": ao.fit_length(ao.stretch_pitch(o, ratio, semis), n),
        "oct": ao.fit_length(ao.stretch_pitch(o, ratio, semis + 12), n),
        "bass": ao.fit_length(ao.stretch_pitch(bs, ratio, semis), n),
    }


def riff_layer(kind: str, loops: dict[str, np.ndarray], n: int, bar_samples: float,
               variant: int) -> np.ndarray:
    """Riff vrstva sekce složená po frázích (délka smyčky) s variacemi.

    variant 0 = první drop, 1+ = další dropy (jiné pořadí variací). V buildu poslední fráze
    stutteruje (1/2, 1/2, 1/4, 1/8 taktu na poslední čtyři takty).
    """
    L = loops["riff"].shape[1]
    bars_per_loop = max(1, int(round(L / bar_samples)))
    phrases = -(-n // L)
    pieces = []
    for p in range(phrases):
        base = loops["riff"]
        if kind == "drop":
            seq = (["base", "gate_a", "base", "gate_b_oct"] if variant == 0
                   else ["gate_b", "oct_a", "base", "gate_c_oct"])
            v = seq[p % 4]
        elif kind == "build" and p >= phrases - 1:
            v = "retrig"
        else:
            v = "base"
        x = base
        if v.startswith("gate"):
            x = gate(base, GATES[v.split("_")[1]], bar_samples)
        if v.endswith("oct"):
            x = x + 0.55 * loops["oct"]
        if v == "oct_a":
            x = base + 0.55 * gate(loops["oct"], GATES["a"], bar_samples)
        if v == "retrig":
            bar = int(round(bar_samples))
            fr = [None] * max(0, bars_per_loop - 4) + [0.5, 0.5, 0.25, 0.125][-min(4, bars_per_loop):]
            segs = []
            for i, f in enumerate(fr):
                seg = base[:, i * bar:(i + 1) * bar]
                segs.append(retrigger(seg, int(bar * f)) if f and seg.shape[1] > 64 else seg)
            x = np.concatenate(segs, axis=1)
        pieces.append(fade_edges(x, 1.5))
    return ao.fit_length(np.concatenate(pieces, axis=1), n)
