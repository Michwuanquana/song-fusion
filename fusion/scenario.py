"""Scénář aranže: textový formát, předlohy, návrh podle struktury A a návrh přes lokální LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .analysis import Analysis

ALIASES = {
    "intro": "intro", "úvod": "intro", "uvod": "intro",
    "build": "build", "buildup": "build", "náběh": "build", "nabeh": "build",
    "drop": "drop",
    "break": "break", "breakdown": "break", "zlom": "break",
    "outro": "outro", "závěr": "outro", "zaver": "outro",
}
KINDS = ["intro", "build", "drop", "break", "outro"]
KIND_ENERGY = {"intro": 0.35, "build": 0.65, "drop": 1.0, "break": 0.45, "outro": 0.35}

FLAGS = {
    "acid": "přidá acid riff",
    "noacid": "vypne acid riff",
    "bassb": "bas ze songu B (smyčka z vybraného riffu)",
    "bassa": "bas ze songu A (výchozí)",
    "riff1": "použije riff 1",
    "riff2": "použije riff 2 (výchozí pro druhý a další drop)",
    "noriff": "bez riffu z B",
    "nokick": "bicí bez kicku (napětí)",
    "filter": "celá sekce se otevírá low-pass filtrem",
    "roll": "snare roll (v buildu výchozí)",
    "noroll": "bez snare rollu",
    "vox": "vokály A (ve zlomu výchozí)",
    "novox": "bez vokálů",
}

PRESETS = {
    "Krátký ~3 min (64 taktů)":
        "intro 8\nbuild 8\ndrop 16\nbreak 8\nbuild 4\ndrop 16 acid\noutro 4",
    "Klub ~6 min (128 taktů)":
        "intro 16\nbuild 8\ndrop 32\nbreak 16\nbuild 8\ndrop 32 acid bassb\noutro 16",
    "Dlouhý ~9 min (184 taktů)":
        "intro 16\nbuild 8\ndrop 32\nbreak 16\nbuild 8\ndrop 32 acid\nbreak 8\nbuild 8\n"
        "drop 32 acid bassb\noutro 24",
    "DJ tool (intro/outro 32 na mixování)":
        "intro 32 # dlouhé intro na namixování\nbuild 8\ndrop 32\nbreak 16 filter\nbuild 8\n"
        "drop 32 acid\noutro 32 # dlouhé outro",
    "Hypnotický (bez zlomu)":
        "intro 16\nbuild 4\ndrop 32\ndrop 32 acid riff2\ndrop 16 nokick filter\nbuild 4\n"
        "drop 32 acid bassb\noutro 16",
}


@dataclass
class Section:
    kind: str
    bars: int
    flags: set[str] = field(default_factory=set)
    db: float = 0.0
    comment: str = ""


def parse_scenario(text: str) -> list[Section]:
    """Řádky `druh takty [flagy] [db=-3] [# komentář]`. Chyba = ValueError s číslem řádku."""
    out: list[Section] = []
    for no, raw in enumerate(text.splitlines(), 1):
        body, _, comment = raw.partition("#")
        tok = body.replace(",", " ").split()
        if not tok:
            continue
        kind = ALIASES.get(tok[0].lower())
        if kind is None:
            raise ValueError(f"Řádek {no}: neznámý druh '{tok[0]}' (intro/build/drop/break/outro)")
        if len(tok) < 2 or not tok[1].isdigit() or int(tok[1]) < 1:
            raise ValueError(f"Řádek {no}: za druhem musí být počet taktů, např. '{kind} 16'")
        flags, db = set(), 0.0
        for t in tok[2:]:
            t = t.lower()
            if t.startswith("db="):
                try:
                    db = float(t[3:])
                except ValueError as e:
                    raise ValueError(f"Řádek {no}: špatná hodnota '{t}', např. db=-3") from e
            elif t in FLAGS:
                flags.add(t)
            else:
                raise ValueError(f"Řádek {no}: neznámý flag '{t}'. Povolené: {', '.join(FLAGS)}, db=X")
        out.append(Section(kind, int(tok[1]), flags, db, comment.strip()))
    if not out:
        raise ValueError("Scénář je prázdný.")
    return out


def total_bars(secs: list[Section]) -> int:
    return sum(s.bars for s in secs)


# --------------------------------------------------------------------------- podle struktury A

def structure_sections(a: Analysis, block: int = 8) -> list[tuple[str, int, int]]:
    """Odhad struktury A z energie po blocích: [(druh, start_takt, takty)]."""
    e = np.asarray(a.bar_energy)
    nblk = len(e) // block
    if nblk < 3:
        return [("drop", 0, len(e))]
    be = np.array([e[i * block:(i + 1) * block].mean() for i in range(nblk)])
    hi = np.percentile(be, 60)
    is_drop = be >= hi
    first, last = int(np.argmax(is_drop)), int(nblk - 1 - np.argmax(is_drop[::-1]))
    kinds = []
    for i in range(nblk):
        if i < first:
            kinds.append("intro")
        elif i > last:
            kinds.append("outro")
        else:
            kinds.append("drop" if is_drop[i] else "break")
    # jediný slabší blok mezi dvěma dropy = pokles uvnitř dropu (drop bez kicku), ne zlom
    for i in range(1, nblk - 1):
        if kinds[i] == "break" and kinds[i - 1] == "drop" and kinds[i + 1] == "drop":
            kinds[i] = "dip"
    for i in range(1, nblk):  # blok před každým dropem = build (jediný intro blok zůstane intrem)
        if kinds[i] == "drop" and kinds[i - 1] in ("break", "intro") \
                and not (kinds[i - 1] == "intro" and i - 1 == 0):
            kinds[i - 1] = "build"
    out: list[tuple[str, int, int]] = []
    for i, k in enumerate(kinds):
        if out and out[-1][0] == k:
            out[-1] = (k, out[-1][1], out[-1][2] + block)
        else:
            out.append((k, i * block, block))
    return out


def from_structure(a: Analysis) -> str:
    """Scénář kopírující dynamiku A; další dropy dostanou variace (acid, bas z B)."""
    lines = ["# Podle struktury A (energie po 8 taktech); uprav podle sebe"]
    drop_no, prev = 0, ""
    for kind, start, bars in structure_sections(a):
        where = f"  # A takty {start + 1}-{start + bars}"
        if kind == "dip":
            lines.append(f"drop {bars} nokick filter{where}, pokles energie")
        elif kind == "drop":
            if prev not in ("drop", "dip"):
                drop_no += 1
            flags = {1: "", 2: " acid"}.get(drop_no, " acid bassb")
            lines.append(f"drop {bars}{flags}{where}")
        else:
            lines.append(f"{kind} {bars}{where}")
        prev = kind
    return "\n".join(lines)


# --------------------------------------------------------------------------- AI návrh

AI_SYSTEM = """You are an experienced techno/house arranger. Write an arrangement scenario in a strict line format:
<kind> <bars> [flags] [# short comment in Czech]
kinds: intro, build, drop, break, outro
bars: multiples of 4. Typical lengths: intro 8-32, build 4-16, drop 16-32, break 8-32, outro 8-32.
flags (optional): acid (adds acid 303 riff), noacid, bassb (bass from song B), riff2 (alternate riff from B), noriff, nokick (drums without kick, tension), filter (slow low-pass opening), noroll (no snare roll), vox (vocals of A), novox, db=-3 (section gain in dB).
Rules: start with intro, end with outro, put a build before every drop, keep the total close to the target bars, make each drop different (acid, riff2, bassb, nokick passage), follow the energy profile of song A and the user's wish.
Output ONLY the scenario lines. No prose, no markdown, no code fences."""


def energy_profile(a: Analysis, block: int = 8) -> str:
    e = np.asarray(a.bar_energy)
    nblk = max(1, len(e) // block)
    be = np.array([e[i * block:(i + 1) * block].mean() for i in range(nblk)])
    return " ".join(str(int(round(v))) for v in be - be.max())


def _clean(text: str) -> str:
    """Z odpovědi LLM ponechá jen platné řádky; neznámé flagy zahodí, takty srovná na násobek 4."""
    keep = []
    for raw in text.replace("```", "\n").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if not line:
            continue
        if line.startswith("#"):
            keep.append(line)
            continue
        body, _, comment = line.partition("#")
        tok = body.replace(",", " ").split()
        if len(tok) < 2 or tok[0].lower() not in ALIASES:
            continue
        num = re.sub(r"\D", "", tok[1])
        if not num:
            continue
        bars = max(4, int(round(int(num) / 4)) * 4)
        flags = [t.lower() for t in tok[2:] if t.lower() in FLAGS or t.lower().startswith("db=")]
        out = " ".join([ALIASES[tok[0].lower()], str(bars), *flags])
        keep.append(out + (f"  # {comment.strip()}" if comment.strip() else ""))
    return "\n".join(keep)


def propose_ai(a: Analysis, b: Analysis, wish: str, minutes: float, bpm: float,
               model: str | None = None) -> tuple[str, str]:
    """Návrh scénáře lokálním LLM. Vrací (text, model). Při neplatném výstupu 1 oprava, pak ValueError."""
    from .prompting import chat

    bar_len = 240.0 / bpm
    target = max(16, int(round(minutes * 60 / bar_len / 8)) * 8)
    struct = ", ".join(f"{k} {n}" for k, _, n in structure_sections(a))
    user = (f"/no_think\nTarget length: {target} bars (about {minutes:.0f} min at {bpm:.1f} BPM).\n"
            f"Song A (backbone, drums): {a.summary()}\n"
            f"Song A energy per 8 bars in dB relative to loudest: {energy_profile(a)}\n"
            f"Song A detected structure: {struct}\n"
            f"Song B (riffs, vibe): {b.summary()}\n"
            f"User wish: {wish.strip() or 'none, make a club-ready arrangement'}\n")
    err = ""
    for _ in range(2):
        msg = user + (f"\nYour previous answer was invalid: {err}. Output corrected lines only."
                      if err else "")
        text, used_model = chat(AI_SYSTEM, msg, max_tokens=450, temperature=0.6,
                                model=model, use_gpu=True, unload_after=True)
        clean = _clean(text)
        try:
            secs = parse_scenario(clean)
            head = (f"# AI návrh ({used_model.split('/')[-1].split(':')[0]}), cíl {target} taktů, "
                    f"vyšlo {total_bars(secs)}")
            return head + "\n" + clean, used_model
        except ValueError as e:
            err = str(e)
    raise ValueError(f"LLM nevrátil platný scénář: {err}")
