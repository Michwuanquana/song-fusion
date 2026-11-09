"""Centrální monitoring procesů (včetně OS procesů přes psutil), GPU/VRAM a živého streamu tokenů."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import requests

from .config import OLLAMA_URL

_LOCK = threading.Lock()
_SEQ = 0
_ACTIVE: dict[int, "Job"] = {}
_LOG: deque[str] = deque(maxlen=60)
_ABORT = False

_LLM_STREAM_HEADER: str = "Žádný model právě negeneruje."
_LLM_STREAM_TEXT: str = ""

# Pro potlačení zbytečného překreslování v Gradiu (gr.skip)
_LAST_LOG_TXT: str = ""
_LAST_STREAM_TXT: str = ""
_TICK_TOGGLE: int = 0

OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Programs" / "Ollama" / "ollama.exe"
MODELS_DIR = r"Z:\ai\models"


@dataclass
class Job:
    jid: int
    kind: str
    detail: str
    t0: float = field(default_factory=time.time)
    pid: int | None = None
    progress: str = ""


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def should_abort() -> bool:
    return _ABORT


def clear_abort() -> None:
    global _ABORT
    _ABORT = False


def request_abort() -> None:
    global _ABORT
    _ABORT = True


def set_llm_stream(header: str, text: str) -> None:
    global _LLM_STREAM_HEADER, _LLM_STREAM_TEXT
    with _LOCK:
        _LLM_STREAM_HEADER = header
        _LLM_STREAM_TEXT = text


def get_llm_stream() -> tuple[str, str]:
    with _LOCK:
        return _LLM_STREAM_HEADER, _LLM_STREAM_TEXT


def log_event(icon: str, kind: str, msg: str) -> None:
    with _LOCK:
        _LOG.appendleft(f"[{_stamp()}] {icon} {kind:<9} │ {msg}")


def start_job(kind: str, detail: str, pid: int | None = None) -> int:
    global _SEQ
    with _LOCK:
        _SEQ += 1
        jid = _SEQ
        _ACTIVE[jid] = Job(jid=jid, kind=kind, detail=detail, pid=pid)
        pid_str = f" (PID {pid})" if pid else ""
        _LOG.appendleft(f"[{_stamp()}] ▶ {kind:<9} │ {detail}{pid_str}")
        return jid


def update_job(jid: int, progress: str | None = None, pid: int | None = None,
               detail: str | None = None) -> None:
    with _LOCK:
        j = _ACTIVE.get(jid)
        if not j:
            return
        if progress is not None:
            j.progress = progress
        if pid is not None:
            j.pid = pid
        if detail is not None:
            j.detail = detail


def end_job(jid: int, ok: bool = True, extra: str = "") -> None:
    with _LOCK:
        j = _ACTIVE.pop(jid, None)
        if not j:
            return
        dt = time.time() - j.t0
        icon = "✔" if ok else "✘"
        pid_str = f" [PID {j.pid}]" if j.pid else ""
        tail = f" — {extra}" if extra else ""
        _LOG.appendleft(f"[{_stamp()}] {icon} {j.kind:<9} │ {j.detail}{pid_str} ({dt:.1f}s){tail}")


@contextmanager
def track(kind: str, detail: str):
    jid = start_job(kind, detail)
    try:
        yield jid
    except Exception as e:
        end_job(jid, ok=False, extra=f"{type(e).__name__}: {e}")
        raise
    else:
        end_job(jid, ok=True)


def run_proc(cmd: list[str], kind: str, detail: str, check: bool = True) -> bytes:
    """Spustí externí proces (např. ffmpeg), sleduje jeho PID v živém monitoru a vrátí stdout."""
    jid = start_job(kind, detail)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        update_job(jid, pid=proc.pid)
        out, err = proc.communicate()
        if check and proc.returncode != 0:
            msg = err.decode("utf-8", errors="replace").strip()[-240:]
            end_job(jid, ok=False, extra=f"exit {proc.returncode}: {msg}")
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=out, stderr=err)
        end_job(jid, ok=True)
        return out
    except Exception as e:
        if jid in _ACTIVE:
            end_job(jid, ok=False, extra=str(e)[:120])
        raise


def ensure_ollama_server() -> bool:
    """Zajistí, že běží `ollama serve` se správnou cestou `Z:\\ai\\models` a 30m load timeoutem."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4.0)
        if r.status_code == 200 and len(r.json().get("models", [])) > 0:
            return True
    except Exception:  # noqa: BLE001
        pass

    # Pokud běží llama-server.exe (model se právě natahuje z disku a blokuje /api/tags), nesestřelujeme ho!
    if find_llama_runner() is not None:
        return True

    if not OLLAMA_EXE.exists():
        log_event("✘", "ollama", f"Nenalezen {OLLAMA_EXE}")
        return False

    # Ukončíme případnou špatně spuštěnou instanci bez Z:\ai\models
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.info["name"] and p.info["name"].lower() in ("ollama.exe", "ollama app.exe"):
                p.kill()
        except Exception:  # noqa: BLE001
            pass
    time.sleep(0.4)

    env = {
        **os.environ,
        "OLLAMA_MODELS": MODELS_DIR,
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_LOAD_TIMEOUT": "30m",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        [str(OLLAMA_EXE), "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    log_event("▶", "ollama", f"Spuštěn server ollama serve (PID {proc.pid}, models={MODELS_DIR}, timeout=30m)")
    for _ in range(20):
        time.sleep(0.4)
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
            if r.status_code == 200:
                cnt = len(r.json().get("models", []))
                log_event("✔", "ollama", f"Ollama server připraven (PID {proc.pid}, {cnt} modelů)")
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def find_llama_runner() -> dict | None:
    """Najde běžící proces `llama-server.exe` (skutečný runner modelu v Ollamě) přes psutil."""
    for p in psutil.process_iter(["pid", "name", "memory_info", "create_time"]):
        try:
            name = (p.info["name"] or "").lower()
            if "llama-server" in name or "ollama_llama_server" in name:
                mem_gb = p.info["memory_info"].rss / (1024 ** 3)
                up_s = max(0.0, time.time() - (p.info["create_time"] or time.time()))
                return {"pid": p.info["pid"], "ram_gb": mem_gb, "uptime_s": up_s}
        except Exception:  # noqa: BLE001
            pass
    return None


def scan_external_processes() -> list[str]:
    """Vrátí seznam živých OS procesů souvisejících s aplikací (ollama, llama-server, ffmpeg)."""
    items = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "create_time"]):
        try:
            name = (p.info["name"] or "").lower()
            if name in ("llama-server.exe", "ffmpeg.exe", "ollama.exe"):
                mem_mb = p.info["memory_info"].rss / (1024 ** 2)
                up_s = max(0.0, time.time() - (p.info["create_time"] or time.time()))
                cmd = " ".join(p.info.get("cmdline") or [])
                if name == "llama-server.exe":
                    items.append(
                        f"🧠 `llama-server.exe` **PID {p.info['pid']}** · "
                        f"RAM `{mem_mb / 1024:.2f} GB` · běží `{up_s:.0f}s`"
                    )
                elif name == "ffmpeg.exe":
                    mode = "rubberband" if "rubberband" in cmd else "audio I/O"
                    items.append(
                        f"🎚️ `ffmpeg.exe ({mode})` **PID {p.info['pid']}** · "
                        f"RAM `{mem_mb:.0f} MB` · `{up_s:.1f}s`"
                    )
                elif name == "ollama.exe" and "serve" in cmd:
                    items.append(f"🟢 `ollama serve` **PID {p.info['pid']}** (`{mem_mb:.0f} MB`)")
        except Exception:  # noqa: BLE001
            pass
    return items


def _gpu_numbers() -> tuple[int, int, int]:
    """Vrátí (used_mb, total_mb, util_pct)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=0.8,
        )
        if r.returncode == 0 and r.stdout.strip():
            u, t, ut = [int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(",")]
            return u, t, ut
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (total - free) // (1024 * 1024), total // (1024 * 1024), 0
    except Exception:  # noqa: BLE001
        pass
    return 0, 6144, 0


def _ollama_info() -> tuple[str, list[str]]:
    runner = find_llama_runner()
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=0.6)
        models = r.json().get("models", [])
        if models:
            parts, names = [], []
            for m in models:
                name = m.get("name", "?")
                names.append(name)
                vram_gb = m.get("size_vram", 0) / (1024 ** 3)
                tot_gb = m.get("size", 0) / (1024 ** 3)
                short = name.split("/")[-1].split(":")[0]
                pid_s = f" PID {runner['pid']}" if runner else ""
                parts.append(f"🟠 **Ollama:** `{short}`{pid_s} (VRAM {vram_gb:.1f}G / {tot_gb:.1f}G)")
            return " · ".join(parts), names
        if runner:
            return (
                f"🟡 **Ollama natahuje model:** `llama-server.exe` PID {runner['pid']} "
                f"(RAM `{runner['ram_gb']:.1f} GB`, `{runner['uptime_s']:.0f}s`)",
                ["loading"],
            )
        return "🟢 **Ollama:** `0 MB` (odpojeno)", []
    except Exception:  # noqa: BLE001
        if runner:
            return (
                f"🟡 **Ollama natahuje model:** `llama-server.exe` PID {runner['pid']} "
                f"(RAM `{runner['ram_gb']:.1f} GB`, `{runner['uptime_s']:.0f}s`)",
                ["loading"],
            )
        return "⚪ **Ollama:** vypnutá (spustí se automaticky)", []


def unload_ollama(model: str | None = None, force: bool = False) -> list[str]:
    """Okamžitě odpojí model (nebo všechny načtené modely) z GPU a RAM v Ollamě."""
    unloaded = []
    if force:
        request_abort()
        killed = False
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if p.info["name"] and "llama-server" in p.info["name"].lower():
                    p.kill()
                    killed = True
            except Exception:  # noqa: BLE001
                pass
        if not killed:
            r = subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                               capture_output=True, text=True)
            killed = (r.returncode == 0)
        if killed:
            log_event("🛑", "ollama", "Runner llama-server.exe ukončen -> GPU VRAM okamžitě uvolněna")
            set_llm_stream("🛑 Generování zastaveno, GPU VRAM uvolněna.", "")
            unloaded.append("llama-server")
        with _LOCK:
            for jid in [k for k, j in _ACTIVE.items() if j.kind == "ollama"]:
                j = _ACTIVE.pop(jid)
                _LOG.appendleft(f"[{_stamp()}] 🛑 ollama    │ {j.detail} — zastaveno a uvolněno z GPU")

    _, loaded = _ollama_info()
    targets = [model] if model else [m for m in loaded if m != "loading"]
    if model and model not in targets:
        targets.append(model)
    for m in targets:
        if not m:
            continue
        try:
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": m, "keep_alive": 0}, timeout=2.0)
            unloaded.append(m)
            log_event("🧹", "ollama", f"Model {m.split('/')[-1]} odpojen z GPU/RAM (keep_alive=0)")
        except Exception:  # noqa: BLE001
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
            log_event("🧹", "ollama", f"Runner pro {m.split('/')[-1]} násilně odpojen z GPU")
    return unloaded


def status_html() -> str:
    """Vyrenderuje neblikající HTML status panel s ikonkou (zelená -> žlutá -> červená -> ! STOP)."""
    global _TICK_TOGGLE
    from . import engine

    _TICK_TOGGLE = 1 - _TICK_TOGGLE
    anim_cls = "hb-anim-a" if _TICK_TOGGLE == 0 else "hb-anim-b"

    used_mb, total_mb, util_pct = _gpu_numbers()
    ol_txt, _ = _ollama_info()
    ace_txt = "🟠 <b>ACE-Step:</b> v GPU" if engine.is_loaded() else "🟢 <b>ACE-Step:</b> vypnutý"

    with _LOCK:
        active = list(_ACTIVE.values())

    os_procs = scan_external_processes()

    if active:
        now = time.time()
        rows = []
        for j in active:
            dt = now - j.t0
            pid_s = f" <code>PID {j.pid}</code>" if j.pid else ""
            prog_s = f" — <b>{j.progress}</b>" if j.progress else ""
            # Barva úlohy podle trvání: zelená -> žlutá -> oranžová -> červená !
            if dt < 12:
                dot_c, sym = "#22c55e", "●"
            elif dt < 35:
                dot_c, sym = "#eab308", "●"
            elif dt < 75:
                dot_c, sym = "#f97316", "●"
            else:
                dot_c, sym = "#ef4444", "!"
            rows.append(
                f"<div style='margin-top:3px;'>"
                f"<span style='display:inline-block;width:18px;height:18px;line-height:18px;"
                f"text-align:center;border-radius:50%;background:{dot_c};color:#fff;"
                f"font-weight:bold;font-size:11px;margin-right:6px;'>{sym}</span>"
                f"<b>[{j.kind}]</b>{pid_s} {j.detail}{prog_s} "
                f"<span style='opacity:0.75;'>({dt:.1f}s)</span></div>"
            )
        act_html = "".join(rows)
    else:
        act_html = "<div style='margin-top:3px;opacity:0.85;'>✅ <b>Žádná úloha právě neběží</b></div>"

    os_html = (
        "<div style='margin-top:4px;font-size:12px;opacity:0.9;'>"
        + " &nbsp;│&nbsp; ".join(os_procs)
        + "</div>"
        if os_procs else ""
    )

    # Ikonka odezvy (Response Heartbeat): čerstvý tick = zelená, plynule žloutne -> červená -> ! (STOP)
    return f"""
    <div class="status-bar-wrap">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;">
        <span class="resp-badge {anim_cls}" title="Indikátor odezvy: zelená = živé spojení, postupně žloutne až zčervená a při výpadku ukáže ! (STOP)"></span>
        <span><b>VRAM:</b> <code>{used_mb}/{total_mb} MB</code> ({util_pct}%)</span>
        <span>│</span>
        <span>{ol_txt}</span>
        <span>│</span>
        <span>{ace_txt}</span>
      </div>
      {act_html}
      {os_html}
    </div>
    """


def status_view(force_all: bool = False):
    """Vrátí (html_status, log_txt_nebo_skip, stream_txt_nebo_skip) bez blikání v Gradiu."""
    global _LAST_LOG_TXT, _LAST_STREAM_TXT
    import gradio as gr

    html = status_html()
    with _LOCK:
        log_lines = list(_LOG)[:25]
        s_head = _LLM_STREAM_HEADER
        s_txt = _LLM_STREAM_TEXT

    log_txt = "\n".join(log_lines) if log_lines else "Zatím žádné spuštěné procesy."
    stream_out = f"[{s_head}]\n{s_txt}" if s_txt else s_head

    if not force_all:
        out_log = log_txt if log_txt != _LAST_LOG_TXT else gr.skip()
        out_stream = stream_out if stream_out != _LAST_STREAM_TXT else gr.skip()
    else:
        out_log = log_txt
        out_stream = stream_out

    _LAST_LOG_TXT = log_txt
    _LAST_STREAM_TXT = stream_out
    return html, out_log, out_stream
