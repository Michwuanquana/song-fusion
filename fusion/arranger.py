"""Aranžér: dlouhý mashup A+B podle scénáře, v cílovém tempu a tónině.

Čistě DSP, takže drží grid: bicí A jsou páteř, celé stemy A se jednou natáhnou na cílové tempo
(rubberband, bicí se nepřelaďují) a nad nimi se podle sekcí skládají vrstvy z A a riffy z B.
Dynamiku dělají sekce (vrstvy, filtrové sweepy, ticho před dropem, sidechain na kick, risery,
impakty, snare roll). Export: master FLAC 24 bit, volitelně stemy, MIDI acid riffu a markery.
"""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio_ops as ao
from . import monitor, stems
from .acid import acid_steps, acid_synth, midi_events, write_midi
from .analysis import Analysis, camelot, key_name, scaled, semitone_shift
from .config import OUTPUTS, SR
from .dsp import (echo, impact, kick_click, kick_env, lvl, noise_downlifter, noise_riser, ramp,
                  snare_roll, static_filter, sweep_filter)
from .riffs import candidates, make_loops, riff_layer
from .scenario import PRESETS, parse_scenario, total_bars

Progress = Callable[[float, str], None]

# cílové RMS vrstev před gainem sekce (poměr mezi vrstvami)
LEVEL = {"drums": 0.20, "bass": 0.16, "other": 0.085, "riff": 0.11, "acid": 0.085, "vox": 0.09}
BUS = {"drums": "drums", "bass": "bass", "other": "music", "vox": "music", "riff": "riff",
       "acid": "acid", "fx": "fx"}


@dataclass
class ArrangeSettings:
    """Všechno, co ovlivňuje render. Ukládá se do params.json a jde načíst zpět do UI."""
    scenario: str = PRESETS["Klub ~6 min (128 taktů)"]
    tempo_mode: str = "A"          # A | B | vlastni
    bpm: float = 128.0             # cílové BPM pro tempo_mode=vlastni
    key_mode: str = "A"            # A | B (do čí tóniny se ladí)
    transpose: int = 0             # celkový posun v půltónech
    pitch_drums: bool = False      # ladit i bicí (obvykle ne)
    riff1: int | None = None       # počáteční takt riffu v B (None = automaticky)
    riff2: int | None = None
    riff_bars: int = 4
    riff_mode: str = "oba"         # oba | b | acid
    acid_density: float = 0.55
    acid_cutoff: float = 380.0
    acid_reso: float = 7.0
    acid_decay: float = 0.14
    acid_octave: int = 0
    seed: int = 42
    duck: float = 0.55
    vox: bool = True
    roll: bool = True
    fx_level: float = 1.0
    export_stems: bool = False
    export_midi: bool = True
    export_markers: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ArrangeSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def plan(a: Analysis, b: Analysis, s: ArrangeSettings) -> dict:
    """Cílové tempo a tónina + kolik se co natahuje a přelaďuje (pro UI i render)."""
    target = {"A": a.bpm, "B": b.bpm}.get(s.tempo_mode, float(s.bpm or a.bpm))
    target = float(np.clip(target, 60.0, 200.0))
    base_a, base_b = (semitone_shift(a, b), 0) if s.key_mode == "B" else (0, semitone_shift(b, a))
    t = int(s.transpose)
    semis_a, semis_b = base_a + t, base_b + t
    tonic = (a.key_tonic + semis_a) % 12
    ratio_a, ratio_b = target / a.bpm, target / b.bpm
    warn = []
    for name, r in (("A", ratio_a), ("B", ratio_b)):
        if r > 1.25 or r < 0.8:
            warn.append(f"{name} se natahuje ×{r:.2f}, zvuk se výrazně změní")
    for name, st in (("A", semis_a), ("B", semis_b)):
        if abs(st) > 4:
            warn.append(f"{name} se přelaďuje o {st:+d} půltónů")
    return {"bpm": target, "ratio_a": ratio_a, "ratio_b": ratio_b, "semis_a": semis_a,
            "semis_b": semis_b, "tonic": tonic, "mode": a.key_mode,
            "key": key_name(tonic, a.key_mode), "camelot": camelot(tonic, a.key_mode),
            "bar_len": 240.0 / target, "warnings": warn}


# --------------------------------------------------------------------------- materiál z A

_STRETCH: dict = {}


def stretched_stems(path: str, ratio: float, semis: int, pitch_drums: bool) -> dict[str, np.ndarray]:
    """Stemy A natažené na cílové tempo (paralelně, cache v paměti pro poslední nastavení)."""
    raw = stems.separate(path)
    if abs(ratio - 1.0) < 1e-4 and semis == 0:
        return raw
    key = (path, round(ratio, 5), int(semis), bool(pitch_drums))
    if key not in _STRETCH:
        def job(name: str):
            st = 0 if (name == "drums" and not pitch_drums) else semis
            return name, ao.stretch_pitch(raw[name], ratio, st)
        with ThreadPoolExecutor(4) as ex:
            out = dict(ex.map(job, raw))
        _STRETCH.clear()
        _STRETCH[key] = out
    return _STRETCH[key]


def pick_window(a: Analysis, n: int, mode: str, used: dict) -> int:
    """Počáteční takt okna `n` taktů v A: high = nejhlasitější, low = tišší ale živé, mid = střed."""
    e = np.asarray(a.bar_energy)
    if len(e) <= n:
        return 0
    cands = [(s, float(e[s:s + n].mean())) for s in range(0, len(e) - n + 1, 4)]
    ms = np.array([m for _, m in cands])
    if mode == "high":
        order = sorted(cands, key=lambda c: -c[1])
    else:
        q = np.percentile(ms, 25 if mode == "low" else 55)
        order = sorted(cands, key=lambda c: abs(c[1] - q))
    taken = used.setdefault(mode, [])
    for s, _ in order:
        if all(s + n <= u0 or s >= u1 for u0, u1 in taken):
            taken.append((s, s + n))
            return s
    return order[0][0]


def window(stem: np.ndarray, a: Analysis, start_bar: int, bars: int, n: int) -> np.ndarray:
    """Okno stemu zarovnané na takty, přesně n vzorků (při nedostatku materiálu se opakuje)."""
    avail = max(a.n_bars - start_bar, 1)
    seg = ao.cut(stem, a.bar_time(start_bar), min(bars, avail) * a.bar_len)
    if seg.shape[1] < n:
        seg = np.tile(seg, (1, -(-n // seg.shape[1])))
    return ao.fit_length(seg, n)


def _tile(x: np.ndarray, n: int) -> np.ndarray:
    return ao.fit_length(np.tile(x, (1, -(-n // x.shape[1]))), n)


# --------------------------------------------------------------------------- render

def render(a: Analysis, b: Analysis, a_path: str, b_path: str, s: ArrangeSettings,
           progress: Progress | None = None) -> tuple[np.ndarray, dict, dict]:
    """Vyrenderuje aranž. Vrací (master, meta, extras{buses, midi})."""
    say = progress or (lambda f, m: None)
    secs = parse_scenario(s.scenario)
    p = plan(a, b, s)
    jid = monitor.start_job("arranger", f"render {total_bars(secs)} taktů @ {p['bpm']:.1f} BPM ({p['key']})")
    try:
        return _render_inner(a, b, a_path, b_path, s, secs, p, say, jid)
    except Exception as e:
        monitor.end_job(jid, ok=False, extra=str(e)[:120])
        raise
    else:
        monitor.end_job(jid, ok=True)


def _render_inner(a: Analysis, b: Analysis, a_path: str, b_path: str, s: ArrangeSettings,
                  secs: list, p: dict, say: Progress, jid: int) -> tuple[np.ndarray, dict, dict]:
    rng = np.random.default_rng(int(s.seed))
    t_start = time.time()

    say(0.02, "stemy (poprvé demucs ~1 min na song)")
    sb = stems.separate(b_path)
    stems.separate(a_path)
    if abs(p["ratio_a"] - 1) > 1e-4 or p["semis_a"]:
        say(0.06, f"natahuji A na {p['bpm']:.1f} BPM, {p['semis_a']:+d} pt (poprvé ~30 s)")
    sa = stretched_stems(a_path, p["ratio_a"], p["semis_a"], s.pitch_drums)
    a2 = scaled(a, p["ratio_a"])
    bar_samples = a2.bar_len * SR
    beat = int(bar_samples / 4)

    offs = [0]
    for sec in secs:
        offs.append(int(round(offs[-1] + sec.bars * bar_samples)))
    total = offs[-1]
    master = np.zeros((2, total + 2 * SR), dtype=np.float32)
    buses = ({k: np.zeros_like(master) for k in set(BUS.values())} if s.export_stems else None)

    def add(name: str, x: np.ndarray, o: int):
        o = max(o, 0)
        e = min(o + x.shape[1], master.shape[1])
        master[:, o:e] += x[:, : e - o]
        if buses is not None:
            buses[BUS[name]][:, o:e] += x[:, : e - o]

    say(0.3, "riffy z B")
    starts = [s.riff1, s.riff2]
    if None in starts:
        c = candidates(b, sb["other"], sb["bass"], int(s.riff_bars), top=2)
        auto = [c[0]["start_bar"], c[min(1, len(c) - 1)]["start_bar"]]
        starts = [st if st is not None else auto[i] for i, st in enumerate(starts)]
    loops = {}
    for i, st in enumerate(starts, 1):
        loops[i] = loops.get(1) if (i == 2 and st == starts[0]) else make_loops(
            b, sb["other"], sb["bass"], int(st), int(s.riff_bars), p["ratio_b"], p["semis_b"],
            a2.bar_len)

    used: dict = {}
    drop_idx = 0
    timeline, midi = [], []
    d1, d2 = int(0.75 * beat), int(0.5 * beat)
    for i, sec in enumerate(secs):
        k, f = sec.kind, sec.flags
        msg = f"{i + 1}/{len(secs)} {k} ({sec.bars} taktů)"
        say(0.35 + 0.55 * i / len(secs), msg)
        monitor.update_job(jid, progress=msg)
        n, o = offs[i + 1] - offs[i], offs[i]
        sbar = pick_window(a2, sec.bars, {"drop": "high", "build": "mid", "outro": "mid"}.get(k, "low"), used)
        drums = window(sa["drums"], a2, sbar, sec.bars, n)
        a_bass = window(sa["bass"], a2, sbar, sec.bars, n)
        a_other = window(sa["other"], a2, sbar, sec.bars, n)
        drums_k = static_filter(drums, "hp", 180) if "nokick" in f else drums
        duck_g = (1.0 - float(s.duck) * kick_env(drums_k))[None, :]

        which = 2 if ("riff2" in f or (k == "drop" and drop_idx >= 1 and "riff1" not in f)) else 1
        lp = loops[which]
        riff = (riff_layer(k, lp, n, bar_samples, 0 if drop_idx == 0 else 1)
                if s.riff_mode in ("b", "oba") and "noriff" not in f else None)
        bass_b = "bassb" in f and "bassa" not in f
        bass_x = _tile(lp["bass"], n) if bass_b else a_bass
        bass_duck = duck_g if bass_b else 1.0

        L: dict[str, np.ndarray] = {}
        if k == "intro":
            L["drums"] = lvl(static_filter(drums, "hp", 500), LEVEL["drums"]) * 0.8 * ramp(n, 0.3, 1.0)
            L["other"] = sweep_filter(lvl(a_other, LEVEL["other"]), 300, 4500) * 0.9
            if riff is not None:
                L["riff"] = sweep_filter(lvl(riff, LEVEL["riff"]), 250, 4000) * 0.75 * ramp(n, 0.2, 1.0) * duck_g
        elif k == "build":
            g = ramp(n, 0.55, 1.0)
            L["drums"] = lvl(drums_k, LEVEL["drums"]) * g
            L["bass"] = lvl(bass_x, LEVEL["bass"]) * 0.85 * g * bass_duck
            L["other"] = lvl(a_other, LEVEL["other"]) * 0.5
            if riff is not None:
                L["riff"] = lvl(riff, LEVEL["riff"]) * 0.95 * g * duck_g
        elif k == "drop":
            L["drums"] = lvl(drums_k, LEVEL["drums"])
            L["bass"] = lvl(bass_x, LEVEL["bass"]) * bass_duck
            L["other"] = lvl(a_other, LEVEL["other"]) * 0.45 * duck_g
            if riff is not None:
                L["riff"] = lvl(riff, LEVEL["riff"]) * 0.9 * duck_g
        elif k == "break":
            L["drums"] = lvl(static_filter(drums, "hp", 700), LEVEL["drums"]) * 0.4
            L["other"] = lvl(static_filter(a_other, "lp", 2600), LEVEL["other"]) * 1.1
            if riff is not None:
                L["riff"] = echo(static_filter(lvl(riff, LEVEL["riff"]), "lp", 1800), d1, d2) * 0.9
        else:  # outro
            g = ramp(n, 1.0, 0.0) ** 1.4
            L["drums"] = lvl(drums_k, LEVEL["drums"]) * g
            L["bass"] = lvl(bass_x, LEVEL["bass"]) * g
            L["other"] = lvl(a_other, LEVEL["other"]) * 0.4 * g
            if riff is not None:
                L["riff"] = lvl(riff, LEVEL["riff"]) * 0.6 * g * duck_g

        use_vox = ("vox" in f or (k == "break" and s.vox)) and "novox" not in f
        if use_vox:
            v = echo(lvl(window(sa["vocals"], a2, sbar, sec.bars, n), LEVEL["vox"]), d1, beat)
            L["vox"] = v * (0.8 if k == "break" else 0.55) * (duck_g if k == "drop" else 1.0)

        use_acid = ((s.riff_mode == "acid" and k in ("drop", "build"))
                    or (s.riff_mode == "oba" and k == "drop") or "acid" in f) and "noacid" not in f
        if use_acid:
            dens = float(s.acid_density) * (0.55 if k == "break" else 1.0)
            sweep = (0.5, 2.3) if k == "build" else (0.4, 1.6) if "filter" in f else (0.7, 1.5)
            steps = acid_steps(sec.bars, int(s.seed), p["tonic"], p["mode"], dens, int(s.acid_octave))
            ac = acid_synth(steps, n, bar_samples, sweep, float(s.acid_cutoff), float(s.acid_reso),
                            float(s.acid_decay))
            ac = lvl(ac, LEVEL["acid"]) * (0.55 if k == "break" else 0.85)
            L["acid"] = echo(ac, d1, d2, 0.3, 3) * duck_g
            midi += midi_events(steps, sum(x.bars for x in secs[:i]) * 16)

        gsec = 10 ** (sec.db / 20)
        cut_g = None
        if k == "build":  # ticho na poslední dobu před dropem
            cut_g = np.ones(n, dtype=np.float32)
            cut_g[-beat:] = 0.0
            cut_g[-beat - 240:-beat] = np.linspace(1, 0, 240)
        for name, x in L.items():
            if "filter" in f:
                x = sweep_filter(x, 250, 16000, "lp", 0.9)
            x = x * gsec
            if cut_g is not None:
                x = x * cut_g
            add(name, x.astype(np.float32), o)

        fx = float(s.fx_level)
        if (k == "build" and s.roll and "noroll" not in f) or "roll" in f:
            rb = min(2, sec.bars)
            roll = snare_roll(rb, bar_samples, rng) * fx * gsec
            add("fx", roll, o + n - roll.shape[1])
        if k == "build":
            rn = int(min(sec.bars, 8) * bar_samples)
            add("fx", noise_riser(rn, rng) * 0.9 * fx, o + n - rn)
        if k == "drop":
            add("fx", impact(rng) * 0.8 * fx, o)
            drop_idx += 1
        if k == "break":
            add("fx", noise_downlifter(int(min(2, sec.bars) * bar_samples), rng) * fx, o)
        timeline.append({"kind": k, "bars": sec.bars, "start_s": round(o / SR, 2),
                         "end_s": round((o + n) / SR, 2), "flags": sorted(f), "db": sec.db,
                         "comment": sec.comment, "a_src_bar": int(sbar), "riff": which})

    say(0.93, "mastering")
    end = total + SR // 2
    master = master[:, :end]
    ref = ao.cut(ao.load(a_path), a.bar_time(a.best_window(32)), 32 * a.bar_len)
    blk = int(8 * bar_samples)
    blocks = [ao.rms(master[:, q:q + blk]) for q in range(0, max(total - blk // 2, 1), blk)]
    gain = ao.rms(ref) / (float(np.percentile(blocks, 90)) + 1e-9)
    ceiling = 10 ** (-1.0 / 20)
    master = (np.tanh(master * gain / ceiling) * ceiling).astype(np.float32)
    if buses is not None:
        buses = {k: (v[:, :end] * gain).astype(np.float32) for k, v in buses.items()}

    for t, sec_i in zip(timeline, range(len(secs))):
        x = master[:, offs[sec_i]:offs[sec_i + 1]]
        t["rms_db"] = round(float(20 * np.log10(ao.rms(x) + 1e-9)), 1)
    meta = {"plan": p, "timeline": timeline, "bars": total_bars(secs),
            "riffs": [{"which": i, "b_start_bar": int(st), "b_time_s": round(b.bar_time(int(st)), 2)}
                      for i, st in enumerate(starts, 1)],
            "render_seconds": round(time.time() - t_start, 1)}
    say(1.0, "hotovo")
    return master, meta, {"buses": buses, "midi": midi}


def save_render(master: np.ndarray, meta: dict, extras: dict, s: ArrangeSettings,
                songs: dict) -> tuple[Path, list[Path]]:
    """Uloží výsledek do outputs/<čas>_arrange_<bpm>bpm/. Vrací (master, seznam souborů)."""
    bpm = meta["plan"]["bpm"]
    d = OUTPUTS / f"{time.strftime('%Y%m%d_%H%M%S')}_arrange_{int(round(bpm))}bpm"
    out = ao.save(d / "out.flac", master, subtype="PCM_24")
    files = [out]
    info = {"recipe": "arrange", "label": "Aranž", "a": songs.get("a"), "b": songs.get("b"),
            "songs": songs, "duration_s": round(master.shape[1] / SR, 2),
            "settings": asdict(s), **meta}
    (d / "params.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    files.append(d / "params.json")
    if s.export_markers:
        lab = d / "markery_audacity.txt"
        lab.write_text("".join(f"{t['start_s']:.3f}\t{t['end_s']:.3f}\t{t['kind']} {t['bars']} "
                               f"{' '.join(t['flags'])}\n" for t in meta["timeline"]), encoding="utf-8")
        with open(d / "markery.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["start_s", "end_s", "druh", "takty", "flagy", "rms_db"])
            for t in meta["timeline"]:
                w.writerow([t["start_s"], t["end_s"], t["kind"], t["bars"], " ".join(t["flags"]),
                            t["rms_db"]])
        files += [lab, d / "markery.csv"]
    if s.export_midi and extras.get("midi"):
        files.append(write_midi(d / "acid.mid", extras["midi"], bpm))
    if extras.get("buses"):
        for name, x in extras["buses"].items():
            if ao.rms(x) > 1e-5:
                files.append(ao.save(d / "stems" / f"{name}.flac", x, subtype="PCM_24"))
    return out, files


def preview_riff(a: Analysis, b: Analysis, b_path: str, s: ArrangeSettings, which: int = 1,
                 with_kick: bool = True) -> Path:
    """Rychlý 8taktový náhled vybraného riffu z B v cílovém tempu a tónině (4 takty čistě + 4 s gate)."""
    p = plan(a, b, s)
    sb = stems.separate(b_path)
    st = s.riff1 if which == 1 else s.riff2
    if st is None:
        c = candidates(b, sb["other"], sb["bass"], int(s.riff_bars), top=2)
        st = c[min(which - 1, len(c) - 1)]["start_bar"]
    bar_samples = p["bar_len"] * SR
    n = int(round(8 * bar_samples))
    lp = make_loops(b, sb["other"], sb["bass"], int(st), int(s.riff_bars), p["ratio_b"],
                    p["semis_b"], p["bar_len"])
    rf = lvl(riff_layer("drop", lp, n, bar_samples, 0 if which == 1 else 1), LEVEL["riff"])
    if with_kick:
        k = kick_click(8, bar_samples)
        duck_g = (1.0 - float(s.duck) * kick_env(k))[None, :]
        out = k + rf * duck_g
    else:
        out = rf
    out = (np.tanh(out * 1.4) * 0.88).astype(np.float32)
    return ao.save(OUTPUTS / "_preview_riff.wav", out)


def preview_acid(a: Analysis, b: Analysis, s: ArrangeSettings, with_kick: bool = True) -> Path:
    """Rychlý 8taktový náhled 303 acid linky v cílovém tempu a tónině."""
    p = plan(a, b, s)
    bar_samples = p["bar_len"] * SR
    beat = int(bar_samples / 4)
    n = int(round(8 * bar_samples))
    steps = acid_steps(8, int(s.seed), p["tonic"], p["mode"], float(s.acid_density),
                       int(s.acid_octave))
    ac = acid_synth(steps, n, bar_samples, (0.65, 1.55), float(s.acid_cutoff), float(s.acid_reso),
                    float(s.acid_decay))
    ac = echo(lvl(ac, LEVEL["acid"]), int(0.75 * beat), int(0.5 * beat), 0.3, 3)
    if with_kick:
        k = kick_click(8, bar_samples)
        duck_g = (1.0 - float(s.duck) * kick_env(k))[None, :]
        out = k + ac * duck_g
    else:
        out = ac
    out = (np.tanh(out * 1.4) * 0.88).astype(np.float32)
    return ao.save(OUTPUTS / "_preview_acid.wav", out)
