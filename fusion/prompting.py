"""Příkaz + analýza A/B -> caption pro ACE-Step a streamované volání lokálního LLM přes Ollama."""
from __future__ import annotations

import json
import re
import threading
import time

import requests

from . import engine, monitor
from .analysis import Analysis
from .config import OLLAMA_MODELS, OLLAMA_URL

SYSTEM = (
    "You write captions for a text-to-music model (ACE-Step). Output ONE line, 8 to 14 "
    "comma-separated English tags, max 250 characters: genre, mood, 2-3 specific sounds, "
    "drum character, bass character, texture. Tags must be consistent with each other "
    "(never both 'tight' and 'loose'). Keep the track a club/techno style unless the command "
    "says otherwise. Instrumental only. Translate the command (may be Czech) into musical "
    "descriptors and weight it heavily. Do not mention BPM or key. No explanations, no quotes."
)

_CACHED_MODELS: set[str] = set()


def _installed() -> set[str]:
    global _CACHED_MODELS
    monitor.ensure_ollama_server()
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = {m["name"] for m in r.json().get("models", [])}
        if models:
            _CACHED_MODELS = models
        return models or _CACHED_MODELS
    except requests.RequestException:
        return _CACHED_MODELS


def available_models() -> list[str]:
    """Všechny stažené Ollama modely, seřazené podle OLLAMA_MODELS."""
    have = _installed()
    ordered = [m for m in OLLAMA_MODELS if m in have]
    rest = sorted(have - set(ordered))
    return ordered + rest if (ordered or rest) else list(OLLAMA_MODELS)


def chat(system: str, user: str, max_tokens: int = 300, temperature: float = 0.5,
         timeout: int = 1800, *, model: str | None = None, use_gpu: bool = True,
         unload_after: bool = True, json_mode: bool = False) -> tuple[str, str]:
    """Streamované volání lokálního LLM přes Ollamu se živým sledováním natahování modelu i tokenů.

    Pokud `use_gpu=True`, uvolní předem ACE-Step z VRAM a nechá Ollamu využít GPU+RAM.
    Pokud `unload_after=True`, po vygenerování okamžitě odpojí model z GPU/RAM (`keep_alive=0`).
    """
    monitor.ensure_ollama_server()
    have = _installed()
    if model:
        chosen = model
    elif have:
        chosen = next((m for m in OLLAMA_MODELS if m in have), sorted(have)[0])
    else:
        chosen = OLLAMA_MODELS[0]

    monitor.clear_abort()
    if use_gpu:
        engine.unload()
        try:
            ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=2.0).json().get("models", [])
            for m in ps:
                if m.get("name") != chosen or m.get("size_vram", 0) == 0:
                    monitor.unload_ollama(m.get("name"))
        except Exception:  # noqa: BLE001
            pass

    short = chosen.split("/")[-1]
    dev_label = "GPU+RAM" if use_gpu else "CPU"
    opts: dict = {"temperature": temperature, "num_ctx": 2048, "num_predict": max_tokens}
    if not use_gpu:
        opts["num_gpu"] = 0

    payload: dict = {
        "model": chosen,
        "stream": True,
        "think": False,
        "keep_alive": 0 if unload_after else "10m",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": opts,
    }
    if json_mode:
        payload["format"] = "json"

    buf: list[str] = []
    n_tok = 0
    t_start = time.time()
    t_first: float | None = None
    loading_done = threading.Event()

    monitor.set_llm_stream(
        f"⏳ Spouštím {short} ({dev_label}) — natahování vah modelu z disku Z:\\ai\\models…",
        "Čekám na inicializaci llama-server.exe a načtení modelu do VRAM/RAM…",
    )

    try:
        with monitor.track("ollama", f"{short} [{dev_label}]") as jid:
            monitor.update_job(jid, progress="spouští se llama-server.exe…")

            def _watch_loader():
                while not loading_done.wait(0.6):
                    if monitor.should_abort():
                        break
                    dt = time.time() - t_start
                    runner = monitor.find_llama_runner()
                    if runner:
                        monitor.update_job(
                            jid,
                            pid=runner["pid"],
                            progress=f"natahuje model do paměti: RAM {runner['ram_gb']:.2f} GB ({dt:.0f}s)",
                        )
                        monitor.set_llm_stream(
                            f"⏳ Natahování modelu {short} (PID {runner['pid']}) — {dt:.0f}s",
                            f"Proces llama-server.exe [PID {runner['pid']}] čte váhy ze Z:\\ai\\models...\n"
                            f"Aktuálně v RAM: {runner['ram_gb']:.2f} GB (prosím vyčkej, u velkých modelů to trvá ~20-40 s).",
                        )
                    else:
                        monitor.update_job(jid, progress=f"čeká na start runneru ({dt:.0f}s)…")

            watcher = threading.Thread(target=_watch_loader, daemon=True)
            watcher.start()

            try:
                # connect timeout 60s, read timeout `timeout` (1800s = 30 min pro pomalé načítání z disku)
                with requests.post(f"{OLLAMA_URL}/api/chat", timeout=(60, timeout), json=payload,
                                   stream=True) as r:
                    r.raise_for_status()
                    for raw_line in r.iter_lines():
                        if monitor.should_abort():
                            loading_done.set()
                            r.close()
                            raise RuntimeError("Generování LLM bylo ručně zastaveno uživatelem.")
                        if not raw_line:
                            continue
                        pkt = json.loads(raw_line)
                        piece = pkt.get("message", {}).get("content", "")
                        if piece:
                            if t_first is None:
                                t_first = time.time()
                                loading_done.set()
                                runner = monitor.find_llama_runner()
                                if runner:
                                    monitor.update_job(jid, pid=runner["pid"])
                            buf.append(piece)
                            n_tok += 1
                            elapsed = max(time.time() - t_first, 0.05)
                            tps = n_tok / elapsed
                            prog = f"streamuje {n_tok} tok ({tps:.1f} tok/s)"
                            monitor.update_job(jid, progress=prog)
                            cur_txt = "".join(buf)
                            monitor.set_llm_stream(
                                f"🟢 {short} píše: {n_tok} tok ({tps:.1f} tok/s)",
                                cur_txt[-1400:],
                            )
                        if pkt.get("done"):
                            loading_done.set()
                            break
            finally:
                loading_done.set()

            full = "".join(buf)
            text = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
            dt = max(time.time() - (t_first or t_start), 0.1)
            monitor.set_llm_stream(
                f"✔ {short} dokončil ({n_tok} tok za {dt:.1f}s, {n_tok / dt:.1f} tok/s) — odpojen z GPU",
                text[-1400:],
            )
        return text, chosen
    finally:
        if unload_after:
            monitor.unload_ollama(chosen)


def template_caption(command: str, a: Analysis | None, b: Analysis | None) -> str:
    """Bez LLM: základ pro techno + příkaz doslovně."""
    base = "driving techno, hypnotic, punchy kick, rolling bassline, tight hi-hats, club mix"
    return f"{base}, {command}".strip(", ")[:500] if command.strip() else base


def build_caption(command: str, a: Analysis | None = None, b: Analysis | None = None,
                  use_llm: bool = True) -> tuple[str, str]:
    """Vrací (caption, zdroj) kde zdroj je název modelu nebo 'šablona'."""
    if not use_llm:
        return template_caption(command, a, b), "šablona"
    user = (f"/no_think\nCommand: {command or 'keep the style, make it sound like a fused techno track'}\n"
            f"Song A: {a.summary() if a else 'n/a'}\nSong B (the vibe to add): "
            f"{b.summary() if b else 'n/a'}\nCaption:")
    try:
        text, model = chat(SYSTEM, user, max_tokens=140, temperature=0.4, timeout=600,
                           use_gpu=False, unload_after=True)
        line = text.splitlines()[-1].strip().strip('"')
        if len(line) < 10:
            raise ValueError("prázdná odpověď")
        return line[:400], model
    except Exception as e:  # noqa: BLE001
        return template_caption(command, a, b), f"šablona (LLM selhal: {type(e).__name__})"
