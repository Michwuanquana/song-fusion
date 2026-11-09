"""Song Fusion & Aranžér — Desktop Studio UI (http://localhost:7861)."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import gradio as gr

from fusion import arranger, director, engine, monitor, plots, riffs, stems, ui_state
from fusion import lora as lora_mod
from fusion import scenario as scn_mod
from fusion.analysis import Analysis, analyze, camelot, camelot_relation, retime
from fusion.config import LORAS, MUSIC_DIR, OUTPUTS
from fusion.prompting import available_models, build_caption
from fusion.recipes import RECIPES, Ctx, run, run_strength_sweep

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
SWEEP = "sweep"
RECIPE_CHOICES = [(label, key) for key, (label, _) in RECIPES.items()] + [
    ("7 · Sweep síly: cover (recept 2) na 0.2 / 0.4 / 0.6 / 0.8", SWEEP)
]

MODEL_NOTES = {
    "gemma3:4b": "⚡ gemma3:4b (3.9B · 100% VRAM · ~65 tok/s · nejrychlejší)",
    "dolphin3:latest": "🚀 dolphin3:latest (8B · GPU+RAM · ~18 tok/s)",
    "huihui_ai/qwen3-abliterated:8b": "🚀 qwen3-abliterated:8b (8B · GPU+RAM · ~12 tok/s)",
    "qwen3:8b": "🚀 qwen3:8b (8B · GPU+RAM · ~12 tok/s)",
    "huihui_ai/qwen3-abliterated:30b-a3b-instruct-2507-q4_K_M":
        "🧠 qwen3-abliterated:30b-a3b (30.5B MoE · GPU+RAM · ~16 tok/s · nejpečlivější)",
}

DESKTOP_CSS = """
.gradio-container {
    max-width: 1760px !important;
    margin: 0 auto !important;
    padding: 8px 16px !important;
}
.studio-card {
    border: 1px solid rgba(128, 128, 128, 0.24) !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    background: rgba(128, 128, 128, 0.03) !important;
}
.mono-box textarea {
    font-family: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace !important;
    font-size: 12px !important;
    line-height: 1.35 !important;
}
/* Zákaz blikání / šednutí při ticku timeru */
.no-flicker, .no-flicker *, .mono-box, .mono-box * {
    transition: none !important;
}
.no-flicker.pending, .no-flicker .pending, .no-flicker .generating,
.mono-box.pending, .mono-box .pending, .mono-box .generating {
    opacity: 1 !important;
    filter: none !important;
    animation: none !important;
    border-color: transparent !important;
}
/* Indikátor odezvy: zelená -> žlutá -> červená -> ! (STOP) */
.resp-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    font-weight: 800;
    font-size: 12px;
    color: #ffffff;
    background: #22c55e;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.65);
    flex-shrink: 0;
}
.resp-badge::after {
    content: "●";
}
.hb-anim-a {
    animation: respAgeA 6.5s linear forwards !important;
}
.hb-anim-b {
    animation: respAgeB 6.5s linear forwards !important;
}
@keyframes respAgeA {
    0%, 22%   { background: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.7); content: "●"; }
    50%       { background: #eab308; box-shadow: 0 0 8px rgba(234, 179, 8, 0.7); }
    80%       { background: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.8); }
    95%, 100% { background: #dc2626; box-shadow: 0 0 12px rgba(220, 38, 38, 0.95); border-radius: 4px; }
}
@keyframes respAgeB {
    0%, 22%   { background: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.7); }
    50%       { background: #eab308; box-shadow: 0 0 8px rgba(234, 179, 8, 0.7); }
    80%       { background: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.8); }
    95%, 100% { background: #dc2626; box-shadow: 0 0 12px rgba(220, 38, 38, 0.95); border-radius: 4px; }
}
.hb-anim-a::after {
    animation: respSymA 6.5s steps(1, end) forwards !important;
}
.hb-anim-b::after {
    animation: respSymB 6.5s steps(1, end) forwards !important;
}
@keyframes respSymA {
    0%, 84%   { content: "●"; }
    85%, 100% { content: "!"; }
}
@keyframes respSymB {
    0%, 84%   { content: "●"; }
    85%, 100% { content: "!"; }
}
.expl-scroll {
    max-height: 195px !important;
    overflow-y: auto !important;
    padding: 6px 8px !important;
    border-top: 1px solid rgba(128, 128, 128, 0.2);
    font-size: 13px !important;
}
h1 { font-size: 1.35rem !important; margin-bottom: 2px !important; }
h3, h4 { margin-top: 2px !important; margin-bottom: 4px !important; }
"""


def model_choices() -> list[tuple[str, str]]:
    return [(MODEL_NOTES.get(m, m), m) for m in available_models()]


def music_files() -> list[str]:
    if not MUSIC_DIR.exists():
        return []
    return sorted(
        str(p) for p in MUSIC_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXT
    )


def music_dropdown_choices() -> list[tuple[str, str]]:
    return [(Path(p).name, p) for p in music_files()]


def default_songs() -> tuple[str | None, str | None]:
    saved = ui_state.load()
    files = music_files()
    a = saved.get("a_path") if saved.get("a_path") in files else (files[0] if files else None)
    b = saved.get("b_path") if saved.get("b_path") in files else (files[1] if len(files) > 1 else a)
    return a, b


def lora_choices() -> list[tuple[str, str]]:
    out = [("(žádná)", "(žádná)")]
    for p in sorted(LORAS.iterdir()) if LORAS.exists() else []:
        if p.is_dir() and not p.name.startswith("_"):
            ad = lora_mod.resolve_adapter(p)
            if ad:
                out.append((p.name, ad))
    return out


def result_choices() -> list[tuple[str, str]]:
    items = []
    for d in sorted(OUTPUTS.glob("*/out.flac"), reverse=True):
        try:
            info = json.loads((d.parent / "params.json").read_text(encoding="utf-8"))
            dur = info.get("duration_s", 0)
            m, s = divmod(int(round(dur)), 60)
            rec = info.get("recipe", "")
            bpm_info = f" · {info['plan']['bpm']:.0f} BPM" if "plan" in info else ""
            label = f"{d.parent.name}{bpm_info} · {m}:{s:02d}"
            if rec != "arrange" and info.get("strength") is not None:
                label += f" · s={info.get('strength')}"
        except (OSError, json.JSONDecodeError, KeyError):
            label = d.parent.name
        items.append((label, str(d.resolve())))
    return items


def _resolve_an(path: str | None, mult: str = "1x", nudge: float = 0.0) -> Analysis | None:
    if not path or not Path(path).exists():
        return None
    an = analyze(path)
    factor = {"1x": 1.0, "2x": 2.0, "0.5x": 0.5}.get(mult, 1.0)
    return retime(an, an.bpm * factor, float(nudge))


def song_info_and_plot(path: str | None, is_b: bool, mult: str, nudge: float,
                       r1_val: str | None = None, r2_val: str | None = None,
                       riff_bars: int = 4):
    """Vrátí (markdown karta, matplotlib graf) pro Song A nebo Song B."""
    an = _resolve_an(path, mult, nudge)
    if an is None:
        return "_nic nenahráno_", None
    cam = camelot(an.key_tonic, an.key_mode)
    md = (f"**{Path(path).stem[:44]}**  \n"
          f"`{an.bpm:.2f} BPM` · `{an.key}` (**{cam}**) · "
          f"`{int(an.duration // 60)}:{int(an.duration % 60):02d}` ({an.n_bars} taktů) · "
          f"1. doba `{an.downbeat0:.2f}s`")
    if not is_b:
        secs = scn_mod.structure_sections(an)
        fig = plots.song_plot(an, sections=secs)
    else:
        rlist = []
        for idx, val in ((1, r1_val), (2, r2_val)):
            bar = _parse_riff_choice(val)
            if bar is not None:
                rlist.append((bar, f"R{idx}"))
        if not rlist:
            rlist = [(an.best_window(int(riff_bars), align=4), "R1")]
        fig = plots.song_plot(an, riffs=rlist, riff_bars=int(riff_bars))
    return md, fig


def _parse_riff_choice(val: str | int | None) -> int | None:
    if val is None or val == "auto" or str(val).startswith("Auto"):
        return None
    try:
        return int(str(val).split(":")[0])
    except ValueError:
        return None


def riff_dropdown_choices(b_path: str | None, b_mult: str, b_nudge: float,
                          riff_bars: int) -> list[tuple[str, str]]:
    choices = [("Auto (nejlepší z B)", "auto")]
    an = _resolve_an(b_path, b_mult, b_nudge)
    if an is None:
        return choices
    if stems.have_stems(b_path):
        sb = stems.separate(b_path)
        cands = riffs.candidates(an, sb["other"], sb["bass"], int(riff_bars), top=8)
        for i, c in enumerate(cands):
            choices.append((riffs.label(c, i, int(riff_bars)), str(c["start_bar"])))
    else:
        for i in range(0, an.n_bars - int(riff_bars), max(4, an.n_bars // 8)):
            t = an.bar_time(i)
            choices.append((f"Takt {i + 1}-{i + int(riff_bars)} ({int(t // 60)}:{int(t % 60):02d})", str(i)))
    return choices


def on_scan_riffs(b_up, b_sel, b_mult, b_nudge, riff_bars):
    b_path = b_up or b_sel
    if not b_path:
        raise gr.Error("Nejdřív vyber Song B.")
    b = _resolve_an(b_path, b_mult, b_nudge)
    sb = stems.separate(b_path)
    cands = riffs.candidates(b, sb["other"], sb["bass"], int(riff_bars), top=8)
    choices = [("Auto (nejlepší z B)", "auto")] + [
        (riffs.label(c, i, int(riff_bars)), str(c["start_bar"])) for i, c in enumerate(cands)
    ]
    v1 = str(cands[0]["start_bar"]) if cands else "auto"
    v2 = str(cands[min(1, len(cands) - 1)]["start_bar"]) if cands else "auto"
    _, fig = song_info_and_plot(b_path, True, b_mult, b_nudge, v1, v2, int(riff_bars))
    return gr.update(choices=choices, value=v1), gr.update(choices=choices, value=v2), fig


def _build_settings(scenario, tempo_mode, bpm, key_mode, transpose, pitch_drums,
                    r1_val, r2_val, riff_bars, riff_mode,
                    acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
                    seed, duck, vox, roll, fx_level,
                    export_stems, export_midi, export_markers) -> arranger.ArrangeSettings:
    return arranger.ArrangeSettings(
        scenario=scenario or scn_mod.PRESETS["Klub ~6 min (128 taktů)"],
        tempo_mode=tempo_mode,
        bpm=float(bpm),
        key_mode=key_mode,
        transpose=int(transpose),
        pitch_drums=bool(pitch_drums),
        riff1=_parse_riff_choice(r1_val),
        riff2=_parse_riff_choice(r2_val),
        riff_bars=int(riff_bars),
        riff_mode=riff_mode,
        acid_density=float(acid_density),
        acid_cutoff=float(acid_cutoff),
        acid_reso=float(acid_reso),
        acid_decay=float(acid_decay),
        acid_octave=int(acid_octave),
        seed=int(seed),
        duck=float(duck),
        vox=bool(vox),
        roll=bool(roll),
        fx_level=float(fx_level),
        export_stems=bool(export_stems),
        export_midi=bool(export_midi),
        export_markers=bool(export_markers),
    )


def update_plan_and_timeline(a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
                             tempo_mode, bpm, key_mode, transpose, pitch_drums, scenario):
    a = _resolve_an(a_up or a_sel, a_mult, a_nudge)
    b = _resolve_an(b_up or b_sel, b_mult, b_nudge)
    s = arranger.ArrangeSettings(scenario=scenario or "", tempo_mode=tempo_mode, bpm=float(bpm),
                                 key_mode=key_mode, transpose=int(transpose),
                                 pitch_drums=bool(pitch_drums))
    if a and b:
        p = arranger.plan(a, b, s)
        cam_a, cam_b = camelot(a.key_tonic, a.key_mode), camelot(b.key_tonic, b.key_mode)
        rel = camelot_relation(cam_a, cam_b)
        warn_s = ("  \n⚠ " + " · ".join(p["warnings"])) if p["warnings"] else ""
        plan_md = (
            f"🎯 **Cíl:** `{p['bpm']:.1f} BPM` · `{p['key']}` (**{p['camelot']}**) &nbsp;│&nbsp; "
            f"**A:** `×{p['ratio_a']:.2f}` (`{p['semis_a']:+d}st`) &nbsp;│&nbsp; "
            f"**B:** `×{p['ratio_b']:.2f}` (`{p['semis_b']:+d}st`)  \n"
            f"🎼 **Harmonie A ({cam_a}) ↔ B ({cam_b}):** {rel}{warn_s}"
        )
        bar_len = p["bar_len"]
    else:
        plan_md = "_Vyber Song A a Song B._"
        bar_len = 240.0 / float(bpm or 128.0)

    try:
        secs = scn_mod.parse_scenario(scenario or "")
        nb = scn_mod.total_bars(secs)
        dur = nb * bar_len
        scn_info = f"✔ **{len(secs)} sekcí · {nb} taktů · `{int(dur // 60)}:{int(dur % 60):02d}`**"
        fig = plots.timeline_plot(secs, bar_len)
    except ValueError as e:
        scn_info = f"❌ **Chyba ve scénáři:** {e}"
        fig = None

    return plan_md, scn_info, fig


def on_director(wish, llm_model, a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge):
    """AI Režisér: načte LLM na GPU, vyplní celé rozhraní, vysvětlí co a proč a odpojí LLM z GPU."""
    a_path, b_path = a_up or a_sel, b_up or b_sel
    if not a_path or not b_path:
        raise gr.Error("Nejdřív vyber Song A i Song B.")
    a = _resolve_an(a_path, a_mult, a_nudge)
    b = _resolve_an(b_path, b_mult, b_nudge)

    res = director.direct_all(wish or "", a, b, a_path, b_path, model=llm_model)

    r1_str = str(res["riff1"]) if res["riff1"] is not None else "auto"
    r2_str = str(res["riff2"]) if res["riff2"] is not None else "auto"
    r_choices = riff_dropdown_choices(b_path, b_mult, b_nudge, res["riff_bars"])
    valid_vals = {v for _, v in r_choices}
    if r1_str not in valid_vals:
        r_choices.append((f"AI takt {r1_str}", r1_str))
    if r2_str not in valid_vals and r2_str != r1_str:
        r_choices.append((f"AI takt {r2_str}", r2_str))

    plan_md, scn_info, tl_fig = update_plan_and_timeline(
        a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
        res["tempo_mode"], res["bpm"], res["key_mode"], res["transpose"],
        res["pitch_drums"], res["scenario"],
    )
    _, b_fig = song_info_and_plot(b_path, True, b_mult, b_nudge, r1_str, r2_str, res["riff_bars"])

    ui_state.save({
        "a_path": a_path, "b_path": b_path, "wish": wish, "llm_model": llm_model,
        **{k: res[k] for k in ("tempo_mode", "bpm", "key_mode", "transpose", "pitch_drums",
                               "riff_mode", "riff_bars", "acid_density", "acid_cutoff",
                               "acid_reso", "acid_decay", "acid_octave", "duck", "vox",
                               "roll", "fx_level", "export_stems", "export_midi",
                               "export_markers", "seed", "scenario")}
    })

    mon_md, mon_log, mon_stream = monitor.status_view(force_all=True)
    return (
        res["explanation_md"],
        res["tempo_mode"],
        res["bpm"],
        res["key_mode"],
        res["transpose"],
        res["pitch_drums"],
        res["riff_mode"],
        res["riff_bars"],
        gr.update(choices=r_choices, value=r1_str),
        gr.update(choices=r_choices, value=r2_str),
        res["acid_density"],
        res["acid_cutoff"],
        res["acid_reso"],
        res["acid_decay"],
        res["acid_octave"],
        res["duck"],
        res["vox"],
        res["roll"],
        res["fx_level"],
        res["export_stems"],
        res["export_midi"],
        res["export_markers"],
        res["seed"],
        res["scenario"],
        plan_md,
        scn_info,
        tl_fig,
        b_fig,
        res["fusion_recipes"],
        res["fusion_command"],
        res["fusion_caption"],
        res["fusion_strength"],
        res["fusion_window_bars"],
        mon_md,
        mon_log,
        mon_stream,
    )


def on_from_structure(a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
                      tempo_mode, bpm, key_mode, transpose, pitch_drums):
    a = _resolve_an(a_up or a_sel, a_mult, a_nudge)
    if not a:
        raise gr.Error("Nejdřív vyber Song A.")
    txt = scn_mod.from_structure(a)
    plan_md, scn_info, fig = update_plan_and_timeline(
        a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
        tempo_mode, bpm, key_mode, transpose, pitch_drums, txt,
    )
    return txt, plan_md, scn_info, fig


def on_propose_ai_scn(a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
                      tempo_mode, bpm, key_mode, transpose, pitch_drums,
                      ai_wish, ai_mins, llm_model):
    a = _resolve_an(a_up or a_sel, a_mult, a_nudge)
    b = _resolve_an(b_up or b_sel, b_mult, b_nudge)
    if not a or not b:
        raise gr.Error("Nejdřív vyber Song A i Song B.")
    eff_bpm = {"A": a.bpm, "B": b.bpm}.get(tempo_mode, float(bpm or a.bpm))
    txt, _ = scn_mod.propose_ai(a, b, ai_wish or "", float(ai_mins), eff_bpm, model=llm_model)
    plan_md, scn_info, fig = update_plan_and_timeline(
        a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
        tempo_mode, bpm, key_mode, transpose, pitch_drums, txt,
    )
    mon_md, mon_log, mon_stream = monitor.status_view(force_all=True)
    return txt, plan_md, scn_info, fig, mon_md, mon_log, mon_stream


def on_preview_riff(which: int, a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
                    tempo_mode, bpm, key_mode, transpose, pitch_drums,
                    r1_val, r2_val, riff_bars, riff_mode,
                    acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
                    seed, duck, prev_kick):
    a_path, b_path = a_up or a_sel, b_up or b_sel
    if not a_path or not b_path:
        raise gr.Error("Vyber Song A i Song B.")
    a, b = _resolve_an(a_path, a_mult, a_nudge), _resolve_an(b_path, b_mult, b_nudge)
    s = _build_settings("", tempo_mode, bpm, key_mode, transpose, pitch_drums,
                        r1_val, r2_val, riff_bars, riff_mode,
                        acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
                        seed, duck, True, True, 1.0, False, False, False)
    out = arranger.preview_riff(a, b, b_path, s, which=which, with_kick=bool(prev_kick))
    return str(out.resolve())


def on_preview_acid(a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
                    tempo_mode, bpm, key_mode, transpose, pitch_drums,
                    r1_val, r2_val, riff_bars, riff_mode,
                    acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
                    seed, duck, prev_kick):
    a_path, b_path = a_up or a_sel, b_up or b_sel
    if not a_path or not b_path:
        raise gr.Error("Vyber Song A i Song B.")
    a, b = _resolve_an(a_path, a_mult, a_nudge), _resolve_an(b_path, b_mult, b_nudge)
    s = _build_settings("", tempo_mode, bpm, key_mode, transpose, pitch_drums,
                        r1_val, r2_val, riff_bars, riff_mode,
                        acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
                        seed, duck, True, True, 1.0, False, False, False)
    out = arranger.preview_acid(a, b, s, with_kick=bool(prev_kick))
    return str(out.resolve())


def on_arrange(a_up, a_sel, a_mult, a_nudge, b_up, b_sel, b_mult, b_nudge,
               scenario, tempo_mode, bpm, key_mode, transpose, pitch_drums,
               r1_val, r2_val, riff_bars, riff_mode,
               acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
               seed, duck, vox, roll, fx_level,
               export_stems, export_midi, export_markers,
               progress=gr.Progress()):
    a_path, b_path = a_up or a_sel, b_up or b_sel
    if not a_path or not b_path:
        raise gr.Error("Nahraj nebo vyber Song A i Song B.")
    secs = scn_mod.parse_scenario(scenario)
    a = _resolve_an(a_path, a_mult, a_nudge)
    b = _resolve_an(b_path, b_mult, b_nudge)
    s = _build_settings(scenario, tempo_mode, bpm, key_mode, transpose, pitch_drums,
                        r1_val, r2_val, riff_bars, riff_mode,
                        acid_density, acid_cutoff, acid_reso, acid_decay, acid_octave,
                        seed, duck, vox, roll, fx_level,
                        export_stems, export_midi, export_markers)
    ui_state.save({"a_path": a_path, "b_path": b_path, **arranger.asdict(s)})

    master, meta, extras = arranger.render(
        a, b, a_path, b_path, s,
        progress=lambda f, m: progress(min(max(f, 0.0), 1.0), desc=m),
    )
    out, files = arranger.save_render(master, meta, extras, s, {"a": a_path, "b": b_path})
    out_abs = str(out.resolve())

    lines = [
        f"{int(t['start_s'] // 60)}:{int(t['start_s'] % 60):02d}–"
        f"{int(t['end_s'] // 60)}:{int(t['end_s'] % 60):02d}  "
        f"{t['kind']:<6} {t['bars']:>3}t  {t['rms_db']:>5}dB  "
        f"[A:{t['a_src_bar'] + 1} R{t['riff']}] {' '.join(t['flags'])}"
        for t in meta["timeline"]
    ]
    dur = master.shape[1] / 48000
    lines.append(
        f"\n✔ Hotovo za {meta['render_seconds']} s → {out_abs} "
        f"({int(dur // 60)}:{int(dur % 60):02d} @ {meta['plan']['bpm']:.1f} BPM, {meta['plan']['key']})"
    )
    measured = [t["rms_db"] for t in meta["timeline"]]
    tl_fig = plots.timeline_plot(secs, meta["plan"]["bar_len"], measured_db=measured)
    params_txt = (out.parent / "params.json").read_text(encoding="utf-8")
    mon_md, mon_log, mon_stream = monitor.status_view(force_all=True)
    return (
        "\n".join(lines),
        tl_fig,
        gr.update(choices=result_choices(), value=out_abs),
        out_abs,
        [str(f.resolve()) for f in files],
        params_txt,
        mon_md,
        mon_log,
        mon_stream,
    )


def on_run_fusion(a_up, a_sel, b_up, b_sel, command, strength, recipes, mode, seed, steps, shift,
                  window_bars, chunk_bars, use_llm, caption_override, lora, lora_scale,
                  progress=gr.Progress()):
    a, b = a_up or a_sel, b_up or b_sel
    if not a or not b:
        raise gr.Error("Nahraj nebo vyber Song A i Song B.")
    if not recipes:
        raise gr.Error("Vyber aspoň jeden recept.")

    log: list[str] = []

    def say(frac: float, msg: str):
        progress(min(max(frac, 0.0), 1.0), desc=msg)

    ctx = Ctx(a_path=a, b_path=b, command=command or "", strength=float(strength), seed=int(seed),
              steps=int(steps), shift=float(shift), preview=(mode == "Náhled"),
              window_bars=int(window_bars), chunk_bars=int(chunk_bars), use_llm=use_llm,
              caption_override=caption_override or "",
              lora=None if lora in (None, "(žádná)") else lora, lora_scale=float(lora_scale),
              progress=say)
    last = None
    jobs = [r for r in recipes if r != SWEEP] + ([SWEEP] if SWEEP in recipes else [])
    for r in jobs:
        try:
            if r == SWEEP:
                results = run_strength_sweep(ctx)
                for p, info in results:
                    p_abs = str(p.resolve())
                    warn = "  ⚠ tempo ujelo" if info.get("tempo_warning") else ""
                    log.append(f"✔ sweep s={info['strength']} → {p_abs} ({info['seconds']} s){warn}")
                    last = p_abs
            else:
                p, info = run(r, ctx)
                p_abs = str(p.resolve())
                warn = (f"  ⚠ tempo ujelo (pulz {info['strongest_pulse_bpm']} BPM)"
                        if info.get("tempo_warning") else "")
                log.append(f"✔ {info['label']} → {p_abs} ({info['seconds']} s){warn}")
                last = p_abs
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log.append(f"✘ {r}: {e}")
        yield "\n".join(log), gr.update(choices=result_choices(), value=last), last


def on_caption(a_up, a_sel, b_up, b_sel, command, use_llm):
    a, b = a_up or a_sel, b_up or b_sel
    if not a or not b:
        raise gr.Error("Nejdřív vyber oba songy.")
    cap, src = build_caption(command or "", analyze(a), analyze(b), use_llm=use_llm)
    return cap, f"zdroj: `{src}`"


def on_train(song, name, caption, epochs, rank):
    if not song:
        raise gr.Error("Vyber song pro vibe pack.")
    engine.unload()
    monitor.unload_ollama(force=True)
    buf: list[str] = []
    for line in lora_mod.train(song, name or "vibe", caption or "techno", int(epochs), int(rank)):
        buf.append(line)
        yield "\n".join(buf[-60:]), gr.update()
    yield "\n".join(buf[-60:]), gr.update(choices=lora_choices())


def on_pick_result(path):
    if not path:
        return None, [], ""
    p = Path(path)
    folder = p.parent
    params_p = folder / "params.json"
    all_files = [str(f.resolve()) for f in sorted(folder.rglob("*")) if f.is_file()]
    return str(p.resolve()), all_files, (params_p.read_text(encoding="utf-8") if params_p.exists() else "")


def on_load_params_to_ui(path):
    if not path:
        raise gr.Error("Nejdřív vyber běh ve Výsledcích.")
    p = Path(path).parent / "params.json"
    if not p.exists():
        raise gr.Error("Soubor params.json u tohoto běhu neexistuje.")
    info = json.loads(p.read_text(encoding="utf-8"))
    st = info.get("settings")
    if not st:
        raise gr.Error("Tento běh nemá uložené ArrangeSettings (nejde o výstup Aranžéru).")
    s = arranger.ArrangeSettings.from_dict(st)
    r1 = str(s.riff1) if s.riff1 is not None else "auto"
    r2 = str(s.riff2) if s.riff2 is not None else "auto"
    return (
        s.tempo_mode, s.bpm, s.key_mode, s.transpose, s.pitch_drums,
        s.riff_mode, s.riff_bars, r1, r2,
        s.acid_density, s.acid_cutoff, s.acid_reso, s.acid_decay, s.acid_octave,
        s.duck, s.vox, s.roll, s.fx_level,
        s.export_stems, s.export_midi, s.export_markers,
        s.seed, s.scenario,
    )


def on_flush_gpu():
    """Okamžitě sestřelí běžící LLM runner (llama-server.exe) i ACE-Step mimo frontu Gradia."""
    engine.unload()
    monitor.unload_ollama(force=True)
    monitor.log_event("🧹", "system", "Okamžité vyčištění GPU VRAM (LLM runner + ACE-Step ukončeny)")
    return monitor.status_view(force_all=True)


def on_refresh_music(cur_a, cur_b, cur_lp, cur_res, cur_llm):
    files = music_files()
    dd_choices = music_dropdown_choices()
    res = result_choices()
    m_choices = model_choices()
    m_vals = [v for _, v in m_choices]
    loras = lora_choices()
    val_a = cur_a if cur_a in files else (files[0] if files else None)
    val_b = cur_b if cur_b in files else (files[1] if len(files) > 1 else val_a)
    val_lp = cur_lp if cur_lp in files else val_b
    val_llm = cur_llm if cur_llm in m_vals else (m_vals[0] if m_vals else None)
    monitor.log_event("🔄", "music", f"Obnovena složka Z:\\ai\\music ({len(files)} skladeb)")
    mon_md, mon_log, mon_stream = monitor.status_view(force_all=True)
    return (
        gr.update(choices=dd_choices, value=val_a),
        gr.update(choices=dd_choices, value=val_b),
        gr.update(choices=dd_choices, value=val_lp),
        gr.update(choices=res, value=cur_res),
        gr.update(choices=m_choices, value=val_llm),
        gr.update(choices=loras),
        f"📁 `Z:\\ai\\music`: **{len(files)}** skladeb",
        mon_md,
        mon_log,
        mon_stream,
    )


# --------------------------------------------------------------------------- sestavení Desktop UI

_SAVED = ui_state.load()
_DEF_A, _DEF_B = default_songs()
_M_CHOICES = model_choices()
_M_VALS = [v for _, v in _M_CHOICES]
_DEF_MODEL = _SAVED.get("llm_model") if _SAVED.get("llm_model") in _M_VALS else (
    "dolphin3:latest" if "dolphin3:latest" in _M_VALS else _M_VALS[0]
)
_S = arranger.ArrangeSettings.from_dict(_SAVED)

_INIT_STATUS_HTML, _INIT_LOG_TXT, _INIT_STREAM_TXT = monitor.status_view(force_all=True)

with gr.Blocks(title="Song Fusion & Aranžér") as demo:
    # ========================================================================= ŘÁDEK 1: STUDIO KOKPIT (3 SLOUPCE)
    with gr.Row(equal_height=True):
        # Sloupec 1: AI Režisér (Zadání)
        with gr.Column(scale=4, elem_classes=["studio-card"]):
            gr.Markdown("### 🎬 AI Režisér (Autopilot celého rozhraní)")
            director_wish = gr.Textbox(
                value=_SAVED.get("wish", ""),
                label="Popiš česky svou představu (vyplní celé UI níže + po sobě uvolní GPU)",
                lines=4,
                placeholder=(
                    "Např.: Temný hypnotický 6min klubový track na 128 BPM. Začni 16taktovým intrem "
                    "bez kicku s otevíráním filtru, v 1. dropu hlavní riff z B, po breakdownu s vokály "
                    "z A pusť 2. drop s jedovatým 303 acidem a basou z B..."
                ),
            )
            llm_dd = gr.Dropdown(
                choices=_M_CHOICES, value=_DEF_MODEL,
                label="Model LLM (gemma3/dolphin3 = rychlé 8–20s, qwen3:30b = hluboké ~60s)",
            )
            director_btn = gr.Button("✨ Navrhnout a vyplnit celé rozhraní (AI Režisér)",
                                     variant="primary", size="sm")

        # Sloupec 2: Živý token stream + Vysvětlení AI
        with gr.Column(scale=4, elem_classes=["studio-card"]):
            gr.Markdown("### 🧠 Živý tok tokenů LLM & Zdůvodnění návrhu")
            mon_stream = gr.Textbox(
                value=_INIT_STREAM_TXT,
                label="Právě generované tokeny / natahování modelu (live stream)",
                lines=5, max_lines=5, interactive=False,
                elem_classes=["mono-box", "no-flicker"],
            )
            director_expl = gr.Markdown(
                value="*Zde se po dokončení zobrazí české zdůvodnění AI Režiséra (co, jak a proč nastavil).*",
                elem_classes=["expl-scroll"],
            )

        # Sloupec 3: System & GPU Monitor + Log procesů
        with gr.Column(scale=4, elem_classes=["studio-card"]):
            with gr.Row(equal_height=True):
                gr.Markdown("### 🖥️ Monitoring GPU a procesů")
                flush_btn = gr.Button("🛑 Zastavit vše & Uvolnit GPU", size="sm", variant="stop")
            mon_status = gr.HTML(value=_INIT_STATUS_HTML, elem_classes=["no-flicker"])
            mon_log = gr.Textbox(
                value=_INIT_LOG_TXT,
                label="Historie procesů (ffmpeg, demucs, ollama, arranger, acestep)",
                lines=5, max_lines=5, interactive=False,
                elem_classes=["mono-box", "no-flicker"],
            )

    # Živý timer a tlačítko Uvolnit GPU běží MIMO frontu (queue=False) a BEZ progress baru (show_progress="hidden")!
    mon_timer = gr.Timer(1.5)
    mon_timer.tick(monitor.status_view, outputs=[mon_status, mon_log, mon_stream],
                   queue=False, show_progress="hidden")
    flush_btn.click(on_flush_gpu, outputs=[mon_status, mon_log, mon_stream],
                    queue=False, show_progress="hidden")

    # ========================================================================= ŘÁDEK 2: SKLADBY A + B + POSLECH (3 SLOUPCE)
    with gr.Row(equal_height=True):
        # Sloupec 1: Song A
        with gr.Column(scale=4, elem_classes=["studio-card"]):
            with gr.Row(equal_height=True):
                gr.Markdown("### 🅰️ Song A (kostra · bicí · struktura)")
                music_count_md = gr.Markdown(f"📁 `{len(music_files())}` skladeb")
                refresh_music_btn = gr.Button("🔄 Obnovit music", size="sm", variant="secondary")
            a_dd = gr.Dropdown(music_dropdown_choices(), value=_DEF_A,
                               label="Výběr ze složky Z:\\ai\\music", allow_custom_value=True)
            a_card = gr.Markdown()
            a_plot = gr.Plot(show_label=False)
            with gr.Accordion("➕ Nahrát vlastní soubor A / Opravit detekci gridu A", open=False):
                a_au = gr.Audio(type="filepath", label="Upload Song A", sources=["upload"])
                with gr.Row():
                    a_mult = gr.Radio(["1x", "2x", "0.5x"], value="1x", label="Násobek BPM")
                    a_nudge = gr.Slider(-2.0, 2.0, value=0.0, step=0.5, label="Posun 1. doby")

        # Sloupec 2: Song B
        with gr.Column(scale=4, elem_classes=["studio-card"]):
            gr.Markdown("### 🅱️ Song B (vibe · riffy · basové smyčky)")
            b_dd = gr.Dropdown(music_dropdown_choices(), value=_DEF_B,
                               label="Výběr ze složky Z:\\ai\\music", allow_custom_value=True)
            b_card = gr.Markdown()
            b_plot = gr.Plot(show_label=False)
            with gr.Accordion("➕ Nahrát vlastní soubor B / Opravit detekci gridu B", open=False):
                b_au = gr.Audio(type="filepath", label="Upload Song B", sources=["upload"])
                with gr.Row():
                    b_mult = gr.Radio(["1x", "2x", "0.5x"], value="1x", label="Násobek BPM")
                    b_nudge = gr.Slider(-2.0, 2.0, value=0.0, step=0.5, label="Posun 1. doby")

        # Sloupec 3: Master Přehrávač, Náhledy & Exporty (vždy po ruce)
        with gr.Column(scale=4, elem_classes=["studio-card"]):
            gr.Markdown("### 🎧 Master Přehrávač & Rychlý Poslech")
            res_dd = gr.Dropdown(result_choices(), label="Vyrenderované běhy (outputs/)")
            player = gr.Audio(label="Master přehrávač vybraného běhu", type="filepath", interactive=False)
            with gr.Row():
                load_back_btn = gr.Button("📥 Načíst parametry do UI", size="sm")
                use_as_a = gr.Button("🔁 Použít jako nový Song A", size="sm")
            prev_player = gr.Audio(label="Poslech 8taktového náhledu (Riff 1 / Riff 2 / 303 Acid)",
                                   type="filepath", interactive=False)
            with gr.Accordion("📦 Soubory běhu ke stažení (FLAC, MIDI, markery) & params.json", open=False):
                out_files = gr.File(label="Soubory", file_count="multiple", interactive=False)
                params_box = gr.Code(language="json", label="params.json", lines=8)

    # ========================================================================= ŘÁDEK 3: PRACOVNÍ STŮL (ZÁLOŽKY)
    with gr.Tabs():
        # --------------------------------------------------------------------- TAB 1: ARANŽÉR (DVOUSLOUPCOVÝ RACK)
        with gr.TabItem("🎛️ 1 · Aranžér (dlouhý mashup podle scénáře — drží 100% grid)"):
            with gr.Row(equal_height=False):
                # LEVÝ RACK: ZVUKOVÉ PARAMETRY (scale=5)
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.Markdown("#### 1 · Tempo a tónina (Arrange to BPM & Key)")
                        with gr.Row():
                            ar_tempo_mode = gr.Radio(
                                [("A", "A"), ("B", "B"), ("Vlastní", "vlastni")],
                                value=_S.tempo_mode, label="Režim tempa", scale=2,
                            )
                            ar_bpm = gr.Slider(60, 190, value=_S.bpm, step=0.5,
                                               label="Cílové BPM", scale=3)
                        with gr.Row():
                            ar_key_mode = gr.Radio(
                                [("Tónina A", "A"), ("Tónina B", "B")],
                                value=_S.key_mode, label="Tónina", scale=2,
                            )
                            ar_transpose = gr.Slider(-6, 6, value=_S.transpose, step=1,
                                                     label="Transpozice (st)", scale=2)
                            ar_pitch_drums = gr.Checkbox(_S.pitch_drums, label="Ladit i bicí", scale=1)
                        ar_plan_md = gr.Markdown()

                    with gr.Group(elem_classes=["studio-card"]):
                        gr.Markdown("#### 2 · Riffy ze Songu B & TB-303 Acid Synth")
                        with gr.Row():
                            ar_riff_mode = gr.Radio(
                                [("Riff B + Acid", "oba"), ("Jen Riff B", "b"), ("Jen 303 Acid", "acid")],
                                value=_S.riff_mode, label="Vrstvy", scale=3,
                            )
                            ar_riff_bars = gr.Radio([2, 4, 8], value=_S.riff_bars,
                                                    label="Délka (takty)", scale=2)
                        _INIT_R_CHOICES = riff_dropdown_choices(_DEF_B, "1x", 0.0, _S.riff_bars)
                        with gr.Row():
                            ar_r1 = gr.Dropdown(_INIT_R_CHOICES, value="auto",
                                                label="Riff 1 (hlavní)", allow_custom_value=True)
                            ar_r2 = gr.Dropdown(_INIT_R_CHOICES, value="auto",
                                                label="Riff 2 (druhý drop)", allow_custom_value=True)
                        with gr.Row():
                            scan_riffs_btn = gr.Button("🔍 Najít v B", size="sm")
                            prev_r1_btn = gr.Button("🔊 Riff 1", size="sm")
                            prev_r2_btn = gr.Button("🔊 Riff 2", size="sm")
                            prev_acid_btn = gr.Button("🔊 303 Acid", size="sm")
                            prev_kick = gr.Checkbox(True, label="S kickem")
                        with gr.Row():
                            ar_acid_dens = gr.Slider(0.2, 0.9, value=_S.acid_density, step=0.05,
                                                     label="303 Hustota")
                            ar_acid_cut = gr.Slider(150, 1200, value=_S.acid_cutoff, step=10,
                                                    label="303 Cutoff (Hz)")
                            ar_acid_reso = gr.Slider(1.0, 12.0, value=_S.acid_reso, step=0.5,
                                                     label="303 Rezonance")
                        with gr.Row():
                            ar_acid_dec = gr.Slider(0.05, 0.40, value=_S.acid_decay, step=0.01,
                                                    label="303 Decay (s)")
                            ar_acid_oct = gr.Slider(-1, 1, value=_S.acid_octave, step=1,
                                                    label="303 Oktáva")
                            ar_seed = gr.Number(_S.seed, label="Seed", precision=0)

                    with gr.Group(elem_classes=["studio-card"]):
                        gr.Markdown("#### 3 · Mix, FX a Export")
                        with gr.Row():
                            ar_duck = gr.Slider(0.0, 0.9, value=_S.duck, step=0.05,
                                                label="Sidechain pumpa")
                            ar_fx = gr.Slider(0.0, 1.5, value=_S.fx_level, step=0.05,
                                              label="Hlasitost FX")
                        with gr.Row():
                            ar_vox = gr.Checkbox(_S.vox, label="Vokály A ve zlomech")
                            ar_roll = gr.Checkbox(_S.roll, label="Snare roll")
                            ar_exp_stems = gr.Checkbox(_S.export_stems, label="Export stemů")
                            ar_exp_midi = gr.Checkbox(_S.export_midi, label="MIDI acidu")
                            ar_exp_markers = gr.Checkbox(_S.export_markers, label="Markery")

                # PRAVÝ RACK: SCÉNÁŘ, GRAF ČASOVÉ OSY A RENDER (scale=7)
                with gr.Column(scale=7, elem_classes=["studio-card"]):
                    with gr.Row(equal_height=True):
                        ar_preset = gr.Dropdown(list(scn_mod.PRESETS), value="Klub ~6 min (128 taktů)",
                                                label="Předloha scénáře", scale=3)
                        struct_btn = gr.Button("📐 Podle struktury A", size="sm", scale=2)
                        ar_go = gr.Button("🚀 VYRENDEROVAT ARANŽ", variant="primary", size="lg", scale=3)

                    with gr.Row(equal_height=True):
                        ai_scn_wish = gr.Textbox(
                            placeholder="Rychlé přání pouze pro scénář (např. 3 gradující dropy, dlouhé outro)...",
                            label="Zadání pro samostatný AI scénář", scale=4,
                        )
                        ai_scn_mins = gr.Slider(2, 12, value=6, step=0.5, label="Délka (min)", scale=2)
                        ai_scn_btn = gr.Button("🤖 Vymyslet scénář", size="sm", scale=2)

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            ar_scn = gr.Textbox(
                                _S.scenario, lines=14, max_lines=18,
                                label="Editor scénáře (druh takty [flagy] [db=-2] # komentář)",
                                elem_classes=["mono-box"],
                            )
                            gr.Markdown(
                                "<small>**Druhy:** `intro` `build` `drop` `break` `outro` &nbsp;│&nbsp; "
                                "**Flagy:** `acid` `noacid` `bassb` `bassa` `riff1` `riff2` `noriff` "
                                "`nokick` `filter` `roll` `noroll` `vox` `novox` `db=-3`</small>"
                            )
                        with gr.Column(scale=6):
                            ar_scn_info = gr.Markdown()
                            ar_tl_plot = gr.Plot(show_label=False)
                            ar_status = gr.Textbox(
                                label="Vyrenderovaná časová osa a RMS hlasitost sekcí",
                                lines=7, max_lines=10, interactive=False,
                                elem_classes=["mono-box"],
                            )

        # --------------------------------------------------------------------- TAB 2: AI FÚZE (ACE-STEP)
        with gr.TabItem("🧪 2 · Generativní AI Fúze (ACE-Step 1.5 recepty 1–7)"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["studio-card"]):
                    recipes = gr.CheckboxGroup(RECIPE_CHOICES, value=["skeleton"],
                                               label="Recepty fúze (Kostra A drží tempo nejlépe)")
                    command = gr.Textbox(
                        label="Příkaz pro zvuk", lines=2,
                        placeholder="např. temnější, víc acid, méně hi-hat, industriální textura",
                    )
                    strength = gr.Slider(
                        0.05, 1.0, value=0.5, step=0.05,
                        label="Kolik zůstane z A (1.0 = věrné, 0.2 = silná změna)",
                    )
                with gr.Column(scale=4, elem_classes=["studio-card"]):
                    with gr.Row():
                        seed = gr.Number(42, label="Seed", precision=0)
                        steps = gr.Slider(4, 20, value=8, step=1, label="Kroky (turbo: 8)")
                        shift = gr.Slider(1.0, 5.0, value=3.0, step=0.5, label="Shift")
                    with gr.Row():
                        window_bars = gr.Slider(8, 64, value=32, step=4, label="Náhled (takty)")
                        chunk_bars = gr.Slider(8, 64, value=32, step=4, label="Kus (takty)")
                    with gr.Row():
                        use_llm = gr.Checkbox(True, label="Caption přes LLM")
                        lora = gr.Dropdown(lora_choices(), value="(žádná)", label="LoRA")
                        lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Síla LoRA")
                    with gr.Row():
                        caption = gr.Textbox(label="Caption (prázdné = auto)", lines=2, scale=3)
                        cap_btn = gr.Button("Náhled captionu", size="sm", scale=1)
                    cap_src = gr.Markdown()
                with gr.Column(scale=3, elem_classes=["studio-card"]):
                    mode = gr.Radio(["Náhled", "Celá skladba"], value="Náhled", label="Rozsah")
                    go = gr.Button("▶ Spustit AI Fúzi", variant="primary", size="lg")
                    status = gr.Textbox(label="Průběh AI Fúze", lines=7, interactive=False,
                                        elem_classes=["mono-box"])

        # --------------------------------------------------------------------- TAB 3: LORA
        with gr.TabItem("🧬 3 · Vibe pack (LoRA trénink · experimentální na 6 GB)"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["studio-card"]):
                    gr.Markdown("Natrénuje malou LoRA z jednoho songu. Před spuštěním se automaticky uvolní VRAM.")
                    lp_song = gr.Dropdown(music_dropdown_choices(), value=_DEF_B,
                                          label="Song pro vibe pack", allow_custom_value=True)
                    with gr.Row():
                        lp_name = gr.Textbox("vibe", label="Název")
                        lp_epochs = gr.Slider(5, 100, value=30, step=5, label="Epochy")
                        lp_rank = gr.Slider(4, 64, value=16, step=4, label="Rank")
                    lp_cap = gr.Textbox(label="Popis zvuku (caption pro řezy)",
                                        placeholder="např. dark rolling techno, distorted kick, acid 303")
                    lp_go = gr.Button("Natrénovat LoRA", variant="primary")
                with gr.Column(scale=7, elem_classes=["studio-card"]):
                    lp_log = gr.Textbox(label="Log tréninku", lines=12, interactive=False,
                                        elem_classes=["mono-box"])

    # --------------------------------------------------------------------------- události (wiring)

    _PLAN_INPUTS = [
        a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
        ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums, ar_scn,
    ]
    _PLAN_OUTPUTS = [ar_plan_md, ar_scn_info, ar_tl_plot]

    def _refresh_a(up, sel, mult, nudge, b_up, b_sel, b_mult, b_nudge,
                   t_mode, bpm, k_mode, transp, p_drums, scn):
        md, fig = song_info_and_plot(up or sel, False, mult, nudge)
        p_md, s_info, tl_fig = update_plan_and_timeline(
            up, sel, mult, nudge, b_up, b_sel, b_mult, b_nudge,
            t_mode, bpm, k_mode, transp, p_drums, scn,
        )
        return md, fig, p_md, s_info, tl_fig

    def _refresh_b(a_up, a_sel, a_mult, a_nudge, up, sel, mult, nudge,
                   r1, r2, rbars, t_mode, bpm, k_mode, transp, p_drums, scn):
        path = up or sel
        md, fig = song_info_and_plot(path, True, mult, nudge, r1, r2, rbars)
        choices = riff_dropdown_choices(path, mult, nudge, rbars)
        p_md, s_info, tl_fig = update_plan_and_timeline(
            a_up, a_sel, a_mult, a_nudge, up, sel, mult, nudge,
            t_mode, bpm, k_mode, transp, p_drums, scn,
        )
        return md, fig, gr.update(choices=choices), gr.update(choices=choices), p_md, s_info, tl_fig

    for ctrl in (a_dd, a_au, a_mult, a_nudge):
        ctrl.change(
            _refresh_a,
            [a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
             ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums, ar_scn],
            [a_card, a_plot, ar_plan_md, ar_scn_info, ar_tl_plot],
        )

    for ctrl in (b_dd, b_au, b_mult, b_nudge, ar_riff_bars):
        ctrl.change(
            _refresh_b,
            [a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
             ar_r1, ar_r2, ar_riff_bars, ar_tempo_mode, ar_bpm, ar_key_mode,
             ar_transpose, ar_pitch_drums, ar_scn],
            [b_card, b_plot, ar_r1, ar_r2, ar_plan_md, ar_scn_info, ar_tl_plot],
        )

    for ctrl in (ar_r1, ar_r2):
        ctrl.change(
            lambda up, sel, mult, nudge, r1, r2, rbars: song_info_and_plot(
                up or sel, True, mult, nudge, r1, r2, rbars)[1],
            [b_au, b_dd, b_mult, b_nudge, ar_r1, ar_r2, ar_riff_bars],
            b_plot,
        )

    for ctrl in (ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums, ar_scn):
        ctrl.change(update_plan_and_timeline, _PLAN_INPUTS, _PLAN_OUTPUTS)

    ar_preset.change(lambda p: scn_mod.PRESETS[p], ar_preset, ar_scn)
    struct_btn.click(
        on_from_structure,
        [a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
         ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums],
        [ar_scn, ar_plan_md, ar_scn_info, ar_tl_plot],
    )
    ai_scn_btn.click(
        on_propose_ai_scn,
        [a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
         ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums,
         ai_scn_wish, ai_scn_mins, llm_dd],
        [ar_scn, ar_plan_md, ar_scn_info, ar_tl_plot, mon_status, mon_log, mon_stream],
    )
    scan_riffs_btn.click(
        on_scan_riffs,
        [b_au, b_dd, b_mult, b_nudge, ar_riff_bars],
        [ar_r1, ar_r2, b_plot],
    )

    _PREV_INPUTS = [
        a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
        ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums,
        ar_r1, ar_r2, ar_riff_bars, ar_riff_mode,
        ar_acid_dens, ar_acid_cut, ar_acid_reso, ar_acid_dec, ar_acid_oct,
        ar_seed, ar_duck, prev_kick,
    ]
    prev_r1_btn.click(lambda *args: on_preview_riff(1, *args), _PREV_INPUTS, prev_player)
    prev_r2_btn.click(lambda *args: on_preview_riff(2, *args), _PREV_INPUTS, prev_player)
    prev_acid_btn.click(on_preview_acid, _PREV_INPUTS, prev_player)

    director_btn.click(
        on_director,
        [director_wish, llm_dd, a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge],
        [
            director_expl,
            ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums,
            ar_riff_mode, ar_riff_bars, ar_r1, ar_r2,
            ar_acid_dens, ar_acid_cut, ar_acid_reso, ar_acid_dec, ar_acid_oct,
            ar_duck, ar_vox, ar_roll, ar_fx,
            ar_exp_stems, ar_exp_midi, ar_exp_markers,
            ar_seed, ar_scn,
            ar_plan_md, ar_scn_info, ar_tl_plot, b_plot,
            recipes, command, caption, strength, window_bars,
            mon_status, mon_log, mon_stream,
        ],
    )

    ar_go.click(
        on_arrange,
        [
            a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
            ar_scn, ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums,
            ar_r1, ar_r2, ar_riff_bars, ar_riff_mode,
            ar_acid_dens, ar_acid_cut, ar_acid_reso, ar_acid_dec, ar_acid_oct,
            ar_seed, ar_duck, ar_vox, ar_roll, ar_fx,
            ar_exp_stems, ar_exp_midi, ar_exp_markers,
        ],
        [ar_status, ar_tl_plot, res_dd, player, out_files, params_box, mon_status, mon_log, mon_stream],
    )

    cap_btn.click(on_caption, [a_au, a_dd, b_au, b_dd, command, use_llm], [caption, cap_src])
    go.click(
        on_run_fusion,
        [a_au, a_dd, b_au, b_dd, command, strength, recipes, mode, seed, steps,
         shift, window_bars, chunk_bars, use_llm, caption, lora, lora_scale],
        [status, res_dd, player],
    )
    lp_go.click(on_train, [lp_song, lp_name, lp_cap, lp_epochs, lp_rank], [lp_log, lora])

    res_dd.change(on_pick_result, res_dd, [player, out_files, params_box])
    load_back_btn.click(
        on_load_params_to_ui,
        res_dd,
        [
            ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums,
            ar_riff_mode, s_rb := ar_riff_bars, ar_r1, ar_r2,
            ar_acid_dens, ar_acid_cut, ar_acid_reso, ar_acid_dec, ar_acid_oct,
            ar_duck, ar_vox, ar_roll, ar_fx,
            ar_exp_stems, ar_exp_midi, ar_exp_markers,
            ar_seed, ar_scn,
        ],
    )
    use_as_a.click(lambda p: p or None, res_dd, a_au)
    refresh_music_btn.click(
        on_refresh_music,
        [a_dd, b_dd, lp_song, res_dd, llm_dd],
        [a_dd, b_dd, lp_song, res_dd, llm_dd, lora, music_count_md, mon_status, mon_log, mon_stream],
        queue=False,
    )

    demo.load(
        on_refresh_music,
        [a_dd, b_dd, lp_song, res_dd, llm_dd],
        [a_dd, b_dd, lp_song, res_dd, llm_dd, lora, music_count_md, mon_status, mon_log, mon_stream],
    )
    demo.load(
        _refresh_a,
        [a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
         ar_tempo_mode, ar_bpm, ar_key_mode, ar_transpose, ar_pitch_drums, ar_scn],
        [a_card, a_plot, ar_plan_md, ar_scn_info, ar_tl_plot],
    )
    demo.load(
        _refresh_b,
        [a_au, a_dd, a_mult, a_nudge, b_au, b_dd, b_mult, b_nudge,
         ar_r1, ar_r2, ar_riff_bars, ar_tempo_mode, ar_bpm, ar_key_mode,
         ar_transpose, ar_pitch_drums, ar_scn],
        [b_card, b_plot, ar_r1, ar_r2, ar_plan_md, ar_scn_info, ar_tl_plot],
    )

if __name__ == "__main__":
    monitor.ensure_ollama_server()
    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        css=DESKTOP_CSS,
        allowed_paths=[str(OUTPUTS.resolve()), str(MUSIC_DIR.resolve())],
    )
