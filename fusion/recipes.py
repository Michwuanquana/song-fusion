"""Recepty fúze. Každý recept: Ctx -> (audio (2,n)@48k, meta dict)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio_ops as ao
from . import engine, stems
from .analysis import Analysis, analyze, grid_lock, semitone_shift
from .config import OUTPUTS, SR
from .longform import generate_long
from .prompting import build_caption

Progress = Callable[[float, str], None]


@dataclass
class Ctx:
    """Vstupy jednoho běhu."""
    a_path: str
    b_path: str
    command: str = ""
    strength: float = 0.5          # kolik zůstane z A (audio_cover_strength)
    seed: int = 42
    steps: int = 8
    shift: float = 3.0
    preview: bool = True           # True = jen okno `window_bars`, False = celá skladba
    window_bars: int = 32
    chunk_bars: int = 32
    overlap_bars: int = 2
    ref_seconds: float = 30.0      # délka reference z B
    lora: str | None = None
    lora_scale: float = 1.0
    use_llm: bool = True
    caption_override: str = ""
    progress: Progress | None = None
    a: Analysis = field(init=False)
    b: Analysis = field(init=False)

    def __post_init__(self):
        self.a, self.b = analyze(self.a_path), analyze(self.b_path)

    def say(self, f: float, msg: str):
        if self.progress:
            self.progress(f, msg)


# ----------------------------------------------------------------------------- pomocné

def _segment(an: Analysis, audio: np.ndarray, start_bar: int, n_bars: int) -> np.ndarray:
    """Výřez n_bars taktů od start_bar, přesně na gridu."""
    return ao.cut(audio, an.bar_time(start_bar), n_bars * an.bar_len)


def _a_window(ctx: Ctx) -> tuple[int, int]:
    """(start_bar, n_bars) skladby A pro tento běh."""
    if not ctx.preview:
        return 0, ctx.a.n_bars
    n = min(ctx.window_bars, ctx.a.n_bars)
    return ctx.a.best_window(n), n


def _ref_clip(ctx: Ctx, vibe: Analysis, vibe_path: str, target: Analysis) -> np.ndarray:
    """~30 s nejenergičtějšího úseku `vibe` songu jako reference.

    Referenční audio v ACE-Step strhává tempo výstupu (naměřeno: cover A@87.6 s referencí B@117.5
    skončil na 117 BPM). Proto referenci natáhneme na tempo `target` a přeladíme do jeho tóniny,
    model si pak vezme zvuk, ne rytmus.
    """
    n = max(4, int(round(ctx.ref_seconds / vibe.bar_len)) // 4 * 4)
    s = vibe.best_window(n, align=4)
    seg = _segment(vibe, ao.load(vibe_path), s, n)
    return ao.stretch_pitch(seg, target.bpm / vibe.bpm, semitone_shift(vibe, target))


def _b_ref(ctx: Ctx) -> np.ndarray:
    """Reference z B srovnaná na tempo a tóninu A."""
    return _ref_clip(ctx, ctx.b, ctx.b_path, ctx.a)


def _caption(ctx: Ctx) -> tuple[str, str]:
    if ctx.caption_override.strip():
        return ctx.caption_override.strip(), "ručně"
    return build_caption(ctx.command, ctx.a, ctx.b, use_llm=ctx.use_llm)


def _gen_kwargs(ctx: Ctx, caption: str, bpm: float, key: str, strength: float | None = None):
    return dict(caption=caption, task="cover", strength=ctx.strength if strength is None else strength,
                bpm=bpm, key=key, seed=ctx.seed, steps=ctx.steps, shift=ctx.shift)


def _setup_lora(ctx: Ctx):
    engine.set_lora(ctx.lora, ctx.lora_scale)


def _regen(ctx: Ctx, src: np.ndarray, ref: np.ndarray | None, bar_len: float, bpm: float,
           key: str, caption: str, strength: float | None = None) -> np.ndarray:
    return generate_long(src, bar_len, ref, chunk_bars=ctx.chunk_bars,
                         overlap_bars=ctx.overlap_bars, progress=ctx.progress,
                         **_gen_kwargs(ctx, caption, bpm, key, strength))


def _b_stems_aligned_to_a(ctx: Ctx, start_a: int, n_bars: int) -> dict[str, np.ndarray]:
    """Stemy B natažené na tempo A a posunuté do tóniny A, přesně n_bars taktů A."""
    sb = stems.separate(ctx.b_path)
    s_b = ctx.b.best_window(n_bars)
    ratio = ctx.a.bpm / ctx.b.bpm
    semis = semitone_shift(ctx.b, ctx.a)
    n_out = int(round(n_bars * ctx.a.bar_len * SR))
    res = {}
    for name in ("bass", "other"):
        seg = _segment(ctx.b, sb[name], s_b, n_bars)
        res[name] = ao.fit_length(ao.stretch_pitch(seg, ratio, semis), n_out)
    return res


# ----------------------------------------------------------------------------- recepty

def r_convert(ctx: Ctx) -> tuple[np.ndarray, dict]:
    """1) Převlek A -> B: struktura A, zvuk B (cover A s referencí B)."""
    caption, csrc = _caption(ctx)
    _setup_lora(ctx)
    s, n = _a_window(ctx)
    src = _segment(ctx.a, ao.load(ctx.a_path), s, n)
    out = _regen(ctx, src, _b_ref(ctx), ctx.a.bar_len, ctx.a.bpm, ctx.a.key, caption)
    return out, {"caption": caption, "caption_source": csrc, "start_bar": s, "bars": n}


def r_reverse(ctx: Ctx) -> tuple[np.ndarray, dict]:
    """2) Obrácený převlek: struktura B, zvuk A."""
    caption, csrc = _caption(ctx)
    _setup_lora(ctx)
    n = min(ctx.window_bars, ctx.b.n_bars) if ctx.preview else ctx.b.n_bars
    s = ctx.b.best_window(n) if ctx.preview else 0
    src = _segment(ctx.b, ao.load(ctx.b_path), s, n)
    ref = _ref_clip(ctx, ctx.a, ctx.a_path, ctx.b)
    out = _regen(ctx, src, ref, ctx.b.bar_len, ctx.b.bpm, ctx.b.key, caption)
    return out, {"caption": caption, "caption_source": csrc, "start_bar_b": s, "bars": n}


def r_skeleton(ctx: Ctx) -> tuple[np.ndarray, dict]:
    """3) Kostra A: bicí A zůstanou, bas + ostatní se přegenerují s vibe B."""
    caption, csrc = _caption(ctx)
    s, n = _a_window(ctx)
    sa = stems.separate(ctx.a_path)
    engine.free_vram()
    _setup_lora(ctx)
    drums = _segment(ctx.a, sa["drums"], s, n)
    harmonic = _segment(ctx.a, sa["bass"] + sa["other"], s, n)
    regen = _regen(ctx, harmonic, _b_ref(ctx), ctx.a.bar_len, ctx.a.bpm, ctx.a.key, caption)
    out = ao.mix(drums, regen)
    return ao.match_loudness(out, _segment(ctx.a, ao.load(ctx.a_path), s, n)), \
        {"caption": caption, "caption_source": csrc, "start_bar": s, "bars": n}


def r_mashup(ctx: Ctx) -> tuple[np.ndarray, dict]:
    """4) Mashup bez AI: bicí A + bas/harmonie B (tempo a tónina srovnané na A)."""
    s, n = _a_window(ctx)
    sa = stems.separate(ctx.a_path)
    drums = _segment(ctx.a, sa["drums"], s, n)
    bs = _b_stems_aligned_to_a(ctx, s, n)
    out = ao.mix(drums, bs["bass"], bs["other"], gains=[1.0, 0.9, 0.8])
    ref = _segment(ctx.a, ao.load(ctx.a_path), s, n)
    return ao.match_loudness(out, ref), {
        "start_bar": s, "bars": n, "tempo_ratio": ctx.a.bpm / ctx.b.bpm,
        "semitones": semitone_shift(ctx.b, ctx.a)}


def r_mashup_glue(ctx: Ctx) -> tuple[np.ndarray, dict]:
    """5) Mashup + AI lepidlo: výsledek mashupu prohnaný coverem, ať zní jako jedna skladba."""
    mash, meta = r_mashup(ctx)
    caption, csrc = _caption(ctx)
    engine.free_vram()
    _setup_lora(ctx)
    glue = max(ctx.strength, 0.7)
    out = _regen(ctx, mash, None, ctx.a.bar_len, ctx.a.bpm, ctx.a.key, caption, strength=glue)
    meta.update({"caption": caption, "caption_source": csrc, "glue_strength": glue})
    return out, meta


def r_morph(ctx: Ctx) -> tuple[np.ndarray, dict]:
    """6) Morph A -> B: přechod 16 taktů, spojení přes repaint."""
    caption, csrc = _caption(ctx)
    _setup_lora(ctx)
    pre, trans, post = 8, 16, 8
    bar = ctx.a.bar_len
    a_audio = ao.load(ctx.a_path)
    a_start = max(ctx.a.n_bars - pre - trans, 0)
    a_part = _segment(ctx.a, a_audio, a_start, pre + trans)

    ratio, semis = ctx.a.bpm / ctx.b.bpm, semitone_shift(ctx.b, ctx.a)
    b_audio = ao.load(ctx.b_path)
    b_start = ctx.b.best_window(trans + post)
    b_seg = _segment(ctx.b, b_audio, b_start, trans + post)
    b_part = ao.stretch_pitch(b_seg, ratio, semis)
    b_part = ao.fit_length(b_part, int(round((trans + post) * bar * SR)))

    rough = ao.crossfade_join(a_part, b_part, int(round(trans * bar * SR)))
    t0, t1 = pre * bar, (pre + trans) * bar
    out = engine.generate(
        src=rough, ref=_b_ref(ctx), task="repaint", repaint=(t0, t1), caption=caption,
        strength=ctx.strength, bpm=ctx.a.bpm, key=ctx.a.key, seed=ctx.seed, steps=ctx.steps,
        shift=ctx.shift, duration=rough.shape[1] / SR)
    out = ao.fit_length(out, rough.shape[1])
    return out, {"caption": caption, "caption_source": csrc, "a_start_bar": a_start,
                 "b_start_bar": b_start, "repaint_s": [t0, t1]}


RECIPES: dict[str, tuple[str, Callable]] = {
    "skeleton": ("1 · Kostra A: bicí A, bas+harmonie nově s vibe B (drží tempo)", r_skeleton),
    "convert": ("2 · Převlek A→B: cover celé A s referencí B (tempo může ujet)", r_convert),
    "reverse": ("3 · Obrácený převlek: struktura B, zvuk A (tempo může ujet)", r_reverse),
    "mashup": ("4 · Mashup bez AI: bicí A + bas/harmonie B", r_mashup),
    "glue": ("5 · Mashup + AI lepidlo", r_mashup_glue),
    "morph": ("6 · Morph A→B: 16taktový přechod", r_morph),
}


def run(recipe: str, ctx: Ctx, tag: str = "") -> tuple[Path, dict]:
    """Spustí recept, uloží out.flac + params.json do outputs/<čas>_<recept>/."""
    label, fn = RECIPES[recipe]
    t = time.time()
    ctx.say(0.0, f"{label}")
    audio, meta = fn(ctx)
    audio = ao.match_loudness(audio, audio)  # jen limiter, hlasitost zachovat
    out_bpm_ref = ctx.b.bpm if recipe == "reverse" else ctx.a.bpm
    lock, strongest = grid_lock(audio, out_bpm_ref)
    meta["grid_lock"] = round(lock, 2)
    meta["strongest_pulse_bpm"] = round(strongest, 1)
    meta["tempo_warning"] = bool(lock < 0.4)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = OUTPUTS / f"{stamp}_{recipe}{('_' + tag) if tag else ''}"
    out = ao.save(d / "out.flac", audio)
    info = {
        "recipe": recipe, "label": label, "seconds": round(time.time() - t, 1),
        "a": ctx.a_path, "b": ctx.b_path, "a_bpm": ctx.a.bpm, "b_bpm": ctx.b.bpm,
        "a_key": ctx.a.key, "b_key": ctx.b.key, "command": ctx.command,
        "strength": ctx.strength, "seed": ctx.seed, "steps": ctx.steps, "shift": ctx.shift,
        "preview": ctx.preview, "lora": ctx.lora, "lora_scale": ctx.lora_scale,
        "duration_s": round(audio.shape[1] / SR, 2), **meta,
    }
    (d / "params.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, info


def run_strength_sweep(ctx: Ctx, strengths=(0.2, 0.4, 0.6, 0.8)) -> list[tuple[Path, dict]]:
    """7) Sweep síly: recept 1 na více hodnotách strength, vše se stejným seedem."""
    res = []
    for i, st in enumerate(strengths):
        ctx.strength = st
        ctx.say(i / len(strengths), f"sweep strength {st}")
        res.append(run("convert", ctx, tag=f"s{int(st * 100)}"))
    return res
