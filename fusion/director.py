"""AI Režisér (Autopilot): z volného popisu v češtině vyplní celé rozhraní a zdůvodní svá rozhodnutí."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import riffs, stems
from .analysis import Analysis, camelot, camelot_relation, semitone_shift
from .prompting import chat
from .scenario import PRESETS, _clean, energy_profile, from_structure, parse_scenario, structure_sections, total_bars

DIRECTOR_SYSTEM = """You are an expert electronic/techno music producer, DJ, and sound designer ("AI Režisér").
Based on the user's Czech description and the musical analysis of Song A (backbone/drums/structure) and Song B (riffs/vibe), configure the ENTIRE music arranger & AI fusion interface.

Respond ONLY with a valid JSON object matching this exact schema (no markdown fences outside JSON):
{
  "tempo_mode": "A" | "B" | "vlastni",
  "bpm": <number 60..190, e.g. 128.0>,
  "key_mode": "A" | "B",
  "transpose": <integer -6..6, usually 0>,
  "pitch_drums": <boolean, usually false>,
  "riff_mode": "oba" | "b" | "acid",
  "riff_bars": 2 | 4 | 8,
  "riff1_rank": <integer 1..6, usually 1>,
  "riff2_rank": <integer 1..6, usually 2>,
  "acid_density": <number 0.2..0.9, e.g. 0.55>,
  "acid_cutoff": <number 150..1200, e.g. 380>,
  "acid_reso": <number 1.0..12.0, e.g. 7.5>,
  "acid_decay": <number 0.05..0.40, e.g. 0.15>,
  "acid_octave": -1 | 0 | 1,
  "duck": <number 0.0..0.9, sidechain pump on kick, e.g. 0.6>,
  "vox": <boolean, vocals from A in breakdowns>,
  "roll": <boolean, snare roll in builds>,
  "fx_level": <number 0.0..1.5, risers/impacts/downlifters volume, e.g. 1.0>,
  "export_stems": <boolean>,
  "export_midi": <boolean>,
  "export_markers": <boolean>,
  "seed": <integer 1..9999>,
  "scenario": "<multiline string, each line: kind bars [flags] # Czech comment>",
  "fusion_recipes": ["skeleton", ...],
  "fusion_caption": "<8-14 English tags for ACE-Step cover/skeleton>",
  "fusion_strength": <number 0.15..0.95>,
  "fusion_window_bars": 16 | 32 | 64,
  "explanation_cz": "<detailed Czech markdown explaining what you set, how, and why>"
}

RULES FOR "scenario":
- Allowed section kinds: intro, build, drop, break, outro
- Bars must be multiples of 4 (e.g. 8, 16, 24, 32).
- Allowed flags on a line:
  * acid (enables 303 acid line in this section)
  * noacid (disables 303 acid)
  * bassb (uses bass loop from Song B instead of Song A)
  * bassa (uses bass from Song A)
  * riff1 (uses Riff #1 from B)
  * riff2 (uses Riff #2 from B)
  * noriff (mutes B riff in this section)
  * nokick (high-passes drums to remove kick — great for tension intros, dips, or builds)
  * filter (gradually opens a low-pass filter 250Hz -> 16kHz across the section)
  * roll / noroll (snare roll at end of section)
  * vox / novox (echoed vocals from A)
  * db=-2 (gain adjustment in dB)
- Example scenario line:
  drop 32 acid bassb riff2 # Druhý vrchol s 303 acidem a basou z B
- Start with intro, put a build before drops, vary drops (e.g. first drop with riff1, second drop with acid + riff2 + bassb, maybe a short 8-bar drop nokick filter bridge), end with outro.

RULES FOR "explanation_cz":
- Write in natural, engaging Czech for a music producer.
- Structure into 3-4 short bullet sections:
  1. **Tempo a harmonie** (jaké BPM a tóninu jsi zvolil a proč, jak vychází natažení A a B).
  2. **Dramaturgie a scénář** (kolik taktů/minut má aranž, proč jsi takto poskládal sekce a flagy).
  3. **Zvuk, riffy a 303 Acid** (jak jsi nastavil riffy z B, cutoff/rezonanci/oktávu 303 acidu, sidechain pumpu a FX).
  4. **Nastavení AI Fúze (ACE-Step)** (jaké recepty a caption jsi připravil pro porovnání)."""


def _clamp(val, lo, hi, default):
    try:
        v = float(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _snap(val: float, step: float) -> float:
    return round(round(val / step) * step, 4)


def _extract_json(text: str) -> dict:
    """Vyřízne JSON objekt i když model přidá obalový text nebo markdown blok."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Odpověď LLM neobsahuje JSON objekt: {text[:200]}")
    end = text.rfind("}")
    candidate = text[start:end + 1] if end > start else (text[start:] + '"}')
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Oprava případných neescapovaných odřádkování uvnitř stringů nebo uříznutého konce
        if not candidate.endswith("}"):
            candidate = candidate.rstrip(', \n\r\t"') + '"}'
        return json.loads(candidate, strict=False)


def direct_all(wish: str, a: Analysis, b: Analysis, a_path: str, b_path: str,
               model: str | None = None, riff_cands: list[dict] | None = None) -> dict:
    """Spustí LLM na GPU, nechá ho vyplnit celý formulář a hned model odpojí z GPU."""
    t0 = time.time()
    cam_a = camelot(a.key_tonic, a.key_mode)
    cam_b = camelot(b.key_tonic, b.key_mode)
    rel = camelot_relation(cam_a, cam_b)
    shift_b_to_a = semitone_shift(b, a)
    struct_a = ", ".join(f"{k} {n} bars" for k, _, n in structure_sections(a))

    if riff_cands is None and stems.have_stems(b_path):
        sb = stems.separate(b_path)
        riff_cands = riffs.candidates(b, sb["other"], sb["bass"], 4, top=6)

    if riff_cands:
        cand_desc = "; ".join(
            f"Rank #{i + 1}: start_bar={c['start_bar']} (@{c['time_s']}s, score={c['score']})"
            for i, c in enumerate(riff_cands[:6])
        )
    else:
        cand_desc = f"Auto-detected from Song B energy (best window around bar {b.best_window(4)})"

    user_prompt = (
        f"/no_think\n"
        f"USER VISION (Czech): {wish.strip() or 'Udělej z toho šlapavý klubový techno/전자 track s gradací, riffy z B a acidovým vrcholem.'}\n\n"
        f"SONG A (drums/backbone): '{Path(a_path).stem}' | {a.bpm:.2f} BPM | Key: {a.key} ({cam_a}) | "
        f"Duration: {a.duration:.0f}s ({a.n_bars} bars)\n"
        f"Song A structure: {struct_a}\n"
        f"Song A energy per 8 bars (dB rel to peak): {energy_profile(a)}\n\n"
        f"SONG B (riffs/vibe): '{Path(b_path).stem}' | {b.bpm:.2f} BPM | Key: {b.key} ({cam_b}) | "
        f"Duration: {b.duration:.0f}s ({b.n_bars} bars)\n"
        f"Harmonic relation A vs B: {rel} (shift B->A is {shift_b_to_a:+d} semitones)\n"
        f"Song B riff candidates: {cand_desc}\n\n"
        f"Note on stretching: if Song A is around 85-95 BPM and the user wants club techno (124-135 BPM), "
        f"set tempo_mode='vlastni' and bpm around 126-132 (or if B is already club tempo ~118-130, tempo_mode='B' or 'vlastni')."
    )

    raw_text, used_model = chat(
        DIRECTOR_SYSTEM, user_prompt, max_tokens=1100, temperature=0.45,
        model=model, use_gpu=True, unload_after=True, json_mode=False,
    )
    data = _extract_json(raw_text)

    # Validace a sanitizace všech hodnot pro UI
    tempo_mode = data.get("tempo_mode", "vlastni")
    if tempo_mode not in ("A", "B", "vlastni"):
        tempo_mode = "vlastni"
    bpm = _snap(_clamp(data.get("bpm"), 60.0, 200.0, 128.0), 0.5)
    key_mode = "B" if str(data.get("key_mode", "A")).upper() == "B" else "A"
    transpose = int(_clamp(data.get("transpose"), -6, 6, 0))
    pitch_drums = bool(data.get("pitch_drums", False))

    riff_mode = data.get("riff_mode", "oba")
    if riff_mode not in ("oba", "b", "acid"):
        riff_mode = "oba"
    riff_bars = int(data.get("riff_bars", 4))
    if riff_bars not in (2, 4, 8):
        riff_bars = 4

    r1_rank = int(_clamp(data.get("riff1_rank"), 1, 8, 1)) - 1
    r2_rank = int(_clamp(data.get("riff2_rank"), 1, 8, 2)) - 1
    riff1_bar = riff_cands[min(r1_rank, len(riff_cands) - 1)]["start_bar"] if riff_cands else None
    riff2_bar = riff_cands[min(r2_rank, len(riff_cands) - 1)]["start_bar"] if riff_cands else None

    acid_density = _snap(_clamp(data.get("acid_density"), 0.2, 0.9, 0.55), 0.05)
    acid_cutoff = _snap(_clamp(data.get("acid_cutoff"), 150.0, 1200.0, 380.0), 10.0)
    acid_reso = _snap(_clamp(data.get("acid_reso"), 1.0, 12.0, 7.0), 0.5)
    acid_decay = _snap(_clamp(data.get("acid_decay"), 0.05, 0.40, 0.14), 0.01)
    acid_octave = int(_clamp(data.get("acid_octave"), -1, 1, 0))

    duck = _snap(_clamp(data.get("duck"), 0.0, 0.9, 0.55), 0.05)
    vox = bool(data.get("vox", True))
    roll = bool(data.get("roll", True))
    fx_level = _snap(_clamp(data.get("fx_level"), 0.0, 1.5, 1.0), 0.05)
    export_stems = bool(data.get("export_stems", False))
    export_midi = bool(data.get("export_midi", True))
    export_markers = bool(data.get("export_markers", True))
    seed = int(_clamp(data.get("seed"), 1, 99999, 42))

    # Scénář (LLM může vrátit string s \n, středníky nebo JSON pole řádků)
    scn_val = data.get("scenario", "")
    if isinstance(scn_val, list):
        raw_scn = "\n".join(str(x) for x in scn_val)
    else:
        raw_scn = str(scn_val).replace("\\n", "\n").replace(";", "\n").strip()
    cleaned_scn = _clean(raw_scn)
    try:
        secs = parse_scenario(cleaned_scn)
        eff_bpm = {"A": a.bpm, "B": b.bpm}.get(tempo_mode, bpm)
        dur_min = total_bars(secs) * (240.0 / eff_bpm) / 60.0
        scn_header = (f"# AI Režisér ({used_model.split('/')[-1].split(':')[0]}) · "
                      f"{total_bars(secs)} taktů (~{dur_min:.1f} min @ {eff_bpm:.0f} BPM)")
        scenario_txt = scn_header + "\n" + cleaned_scn
    except ValueError:
        scenario_txt = from_structure(a)

    # AI Fúze (ACE-Step)
    valid_recipes = {"skeleton", "convert", "reverse", "mashup", "glue", "morph", "sweep"}
    f_recipes = [r for r in (data.get("fusion_recipes") or ["skeleton"]) if r in valid_recipes]
    if not f_recipes:
        f_recipes = ["skeleton"]
    f_caption = str(data.get("fusion_caption") or
                    "driving dark techno, hypnotic groove, punchy kick, acid 303 line, club mix").strip()
    f_strength = _snap(_clamp(data.get("fusion_strength"), 0.05, 1.0, 0.5), 0.05)
    f_win = int(_clamp(data.get("fusion_window_bars"), 8, 64, 32)) // 4 * 4

    dt = time.time() - t0
    short_m = used_model.split("/")[-1]
    expl = str(data.get("explanation_cz") or "").strip()
    if not expl:
        expl = "Nastavení bylo vygenerováno podle analýzy obou skladeb a tvého zadání."

    explanation_md = (
        f"### 🎬 Co a proč AI Režisér nastavil\n"
        f"*Model `{short_m}` zpracoval návrh za **{dt:.1f} s** a byl **odpojen z GPU/VRAM**.*\n\n"
        f"{expl}"
    )

    return {
        "tempo_mode": tempo_mode,
        "bpm": bpm,
        "key_mode": key_mode,
        "transpose": transpose,
        "pitch_drums": pitch_drums,
        "riff_mode": riff_mode,
        "riff_bars": riff_bars,
        "riff1": riff1_bar,
        "riff2": riff2_bar,
        "acid_density": acid_density,
        "acid_cutoff": acid_cutoff,
        "acid_reso": acid_reso,
        "acid_decay": acid_decay,
        "acid_octave": acid_octave,
        "duck": duck,
        "vox": vox,
        "roll": roll,
        "fx_level": fx_level,
        "export_stems": export_stems,
        "export_midi": export_midi,
        "export_markers": export_markers,
        "seed": seed,
        "scenario": scenario_txt,
        "fusion_recipes": f_recipes,
        "fusion_command": wish.strip(),
        "fusion_caption": f_caption,
        "fusion_strength": f_strength,
        "fusion_window_bars": f_win,
        "explanation_md": explanation_md,
        "model": used_model,
        "elapsed_s": round(dt, 1),
    }

