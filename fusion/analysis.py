"""Analýza skladby: přesné BPM, pevný beat/bar grid, downbeat, tónina, energie po taktech.

Techno má konstantní tempo, takže místo trackovaných beatů (jitter) fitujeme pevný grid
lineární regresí: t = t0 + i * T. Chyba 0.01 BPM = drift ~30 ms za 7 minut.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import librosa
import numpy as np

from .audio_ops import file_hash, load
from .config import CACHE

AN_SR = 22050
HOP = 256
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler profily
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Analysis:
    """Výsledek analýzy jedné skladby (časy v sekundách)."""
    path: str
    duration: float
    bpm: float
    beat_period: float
    downbeat0: float  # první downbeat >= 0
    key_tonic: int  # 0 = C
    key_mode: str  # "major" / "minor"
    bar_energy: list[float] = field(default_factory=list)

    @property
    def bar_len(self) -> float:
        return 4 * self.beat_period

    @property
    def key(self) -> str:
        """Tónina ve formátu ACE-Step, např. 'F# minor'."""
        return f"{NOTES[self.key_tonic]} {self.key_mode}"

    @property
    def n_bars(self) -> int:
        return int((self.duration - self.downbeat0) // self.bar_len)

    def bar_time(self, i: int) -> float:
        """Čas začátku i-tého taktu."""
        return self.downbeat0 + i * self.bar_len

    def best_window(self, n_bars: int, align: int = 8) -> int:
        """Index taktu, kde začíná nejenergičtější okno n_bars (start zarovnaný na `align`)."""
        e = np.asarray(self.bar_energy)
        if len(e) <= n_bars:
            return 0
        starts = range(0, len(e) - n_bars + 1, align) or [0]
        return max(starts, key=lambda s: float(e[s: s + n_bars].mean()))

    def summary(self) -> str:
        m, s = divmod(int(self.duration), 60)
        return (f"{self.bpm:.2f} BPM · {self.key} · {m}:{s:02d} · "
                f"{self.n_bars} taktů · první downbeat {self.downbeat0:.3f} s")


def _fit_grid(beat_times: np.ndarray, bpm_guess: float) -> tuple[float, float]:
    """Lineární fit pevného gridu na trackované beaty. Vrací (perioda, fáze t0)."""
    period = 60.0 / bpm_guess
    for _ in range(2):  # dvě iterace: hrubé indexy -> fit -> přesnější indexy -> fit
        idx = np.round((beat_times - beat_times[0]) / period)
        resid = beat_times - (beat_times[0] + idx * period)
        keep = np.abs(resid) < 0.25 * period  # zahodit ujeté beaty (breaky)
        period, t0 = np.polyfit(idx[keep], beat_times[keep], 1)
    return float(period), float(t0)


def _fold_bpm(bpm: float, lo: float = 80.0, hi: float = 160.0) -> float:
    """Srovná půl/dvojnásobné tempo do rozsahu [lo, hi); hi/lo musí být 2, jinak díra."""
    while bpm < lo:
        bpm *= 2
    while bpm >= hi:
        bpm /= 2
    return bpm


def _downbeat_phase(y: np.ndarray, period: float, t0: float, duration: float) -> float:
    """Z 4 možných fází vybere tu, kde začínají změny (nové prvky nastupují na 1. dobu)."""
    beats = np.arange(t0, duration, period)
    beats = beats[beats >= 0]
    frames = librosa.time_to_frames(beats, sr=AN_SR, hop_length=HOP)
    S = np.abs(librosa.stft(y, hop_length=HOP, n_fft=2048))
    bands = np.log1p(S[:1024].reshape(8, -1, S.shape[1]).mean(axis=1))  # 8 hrubých pásem
    seg = np.stack([bands[:, f: f + int(period * AN_SR / HOP)].mean(axis=1)
                    for f in frames[:-1]], axis=1)
    novelty = np.r_[0.0, np.abs(np.diff(seg, axis=1)).sum(axis=0)]
    scores = [novelty[k::4].mean() + 0.5 * novelty[k::16].mean() for k in range(4)]
    k = int(np.argmax(scores))
    first = beats[k]
    while first - 4 * period >= 0:
        first -= 4 * period
    return float(first)


def _key(y: np.ndarray) -> tuple[int, str]:
    """Krumhansl-Schmuckler nad průměrným chroma."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=AN_SR, hop_length=4096).mean(axis=1)
    best = (-2.0, 0, "minor")
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(chroma, np.roll(prof, tonic))[0, 1]
            if r > best[0]:
                best = (r, tonic, mode)
    return best[1], best[2]


def _bar_energy(y: np.ndarray, downbeat0: float, bar_len: float) -> list[float]:
    """RMS v dB pro každý celý takt."""
    out = []
    t = downbeat0
    while t + bar_len <= len(y) / AN_SR:
        seg = y[int(t * AN_SR): int((t + bar_len) * AN_SR)]
        out.append(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)))
        t += bar_len
    return out


def analyze(path: str, force: bool = False) -> Analysis:
    """Analyzuje soubor (s cache podle hashe)."""
    cache = CACHE / "analysis" / f"{file_hash(path)}.json"
    if cache.exists() and not force:
        d = json.loads(cache.read_text(encoding="utf-8"))
        d["path"] = str(path)
        return Analysis(**d)

    y = load(path, AN_SR).mean(axis=0)
    duration = len(y) / AN_SR
    onset = librosa.onset.onset_strength(y=y, sr=AN_SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=AN_SR, hop_length=HOP,
                                                 start_bpm=128, tightness=400)
    beat_times = librosa.frames_to_time(beat_frames, sr=AN_SR, hop_length=HOP)
    period, t0 = _fit_grid(beat_times, _fold_bpm(float(np.atleast_1d(tempo)[0])))
    bpm = _fold_bpm(60.0 / period)
    period = 60.0 / bpm
    downbeat0 = _downbeat_phase(y, period, t0 % period, duration)
    tonic, mode = _key(y)

    an = Analysis(path=str(path), duration=duration, bpm=round(bpm, 3), beat_period=period,
                  downbeat0=downbeat0, key_tonic=tonic, key_mode=mode,
                  bar_energy=_bar_energy(y, downbeat0, 4 * period))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(asdict(an)), encoding="utf-8")
    return an


_CAMELOT_MINOR = {8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7, 9: 8, 4: 9, 11: 10, 6: 11, 1: 12}
_CAMELOT_MAJOR = {11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7, 0: 8, 7: 9, 2: 10, 9: 11, 4: 12}


def camelot(tonic: int, mode: str) -> str:
    """Camelot kód tóniny (např. G moll = 6A)."""
    if mode == "minor":
        return f"{_CAMELOT_MINOR[tonic % 12]}A"
    return f"{_CAMELOT_MAJOR[tonic % 12]}B"


def camelot_relation(k1: str, k2: str) -> str:
    """Slovní vztah dvou Camelot kódů pro harmonický mix."""
    n1, l1, n2, l2 = int(k1[:-1]), k1[-1], int(k2[:-1]), k2[-1]
    dist = min((n1 - n2) % 12, (n2 - n1) % 12)
    if k1 == k2:
        return "stejná tónina"
    if n1 == n2:
        return "relativní dur/moll, sedí"
    if l1 == l2 and dist == 1:
        return "sousední, sedí"
    if l1 == l2 and dist == 2:
        return "o dva kroky, napětí"
    return "střet, doporučuji srovnat tóninu"


def key_name(tonic: int, mode: str) -> str:
    return f"{NOTES[tonic % 12]} {mode}"


@lru_cache(maxsize=4)
def _mono(path: str) -> np.ndarray:
    return load(path, AN_SR).mean(axis=0)


def retime(an: Analysis, bpm: float | None = None, nudge_beats: float = 0.0) -> Analysis:
    """Ruční oprava detekce: jiné BPM (např. x2 pro half-time) a posun první doby o N dob.

    Energie po taktech se přepočítá pro nový grid.
    """
    same_bpm = bpm is None or abs(float(bpm) - an.bpm) < 1e-3
    if same_bpm and not nudge_beats:
        return an
    bpm = an.bpm if same_bpm else float(bpm)
    period = 60.0 / bpm
    d0 = (an.downbeat0 + float(nudge_beats) * period) % (4 * period)
    return replace(an, bpm=round(bpm, 3), beat_period=period, downbeat0=d0,
                   bar_energy=_bar_energy(_mono(an.path), d0, 4 * period))


def scaled(an: Analysis, ratio: float) -> Analysis:
    """Analýza pro audio natažené na `ratio` x tempo (časy se zkrátí, počet taktů zůstane)."""
    if abs(ratio - 1.0) < 1e-6:
        return an
    return replace(an, bpm=an.bpm * ratio, beat_period=an.beat_period / ratio,
                   downbeat0=an.downbeat0 / ratio, duration=an.duration / ratio)


def grid_lock(audio, bpm: float) -> tuple[float, float]:
    """Jak moc výstup sedí na tempo `bpm`? Vrací (síla pulzu na periodě bpm, BPM nejsilnější pulzace).

    Autokorelace náběhů nízkého pásma (kick, 30-120 Hz) do 2 s. Skóre blízko nejsilnějšího píku
    znamená, že kick jede na daném tempu; výrazně nižší znamená, že model tempo přepsal.
    """
    y = librosa.resample(audio.mean(axis=0), orig_sr=48000, target_sr=AN_SR)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=AN_SR, n_fft=2048)
    low = S[(f > 30) & (f < 120)].sum(axis=0)
    on = np.maximum(0, np.diff(low, prepend=low[0]))
    on = on / (on.max() + 1e-9)
    ac = librosa.autocorrelate(on - on.mean(), max_size=int(2.0 * AN_SR / HOP))
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(len(ac)) * HOP / AN_SR
    lo, hi = int(0.25 * AN_SR / HOP), len(ac) - 1  # perioda od 0.25 s (240 BPM)
    best = lo + int(np.argmax(ac[lo:hi]))
    target = 60.0 / bpm
    near = [i for i in range(lo, hi) if abs(lags[i] - target) < 0.03 * target]
    at_target = float(max((ac[i] for i in near), default=0.0))
    return at_target, float(60.0 / lags[best])


def semitone_shift(src: Analysis, dst: Analysis) -> int:
    """Nejmenší posun (v půltónech, -6..+5), který přenese tóninu src do tóniny dst.

    Moll se převádí na paralelní dur (relativní dur), aby šlo kombinovat dur s moll.
    """
    def rel_major(a: Analysis) -> int:
        return (a.key_tonic + 3) % 12 if a.key_mode == "minor" else a.key_tonic

    d = (rel_major(dst) - rel_major(src)) % 12
    return d - 12 if d > 5 else d
