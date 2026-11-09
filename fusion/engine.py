"""Obal nad ACE-Step 1.5 pro 6 GB VRAM: lazy init, cover / repaint, uvolnění paměti."""
from __future__ import annotations

import gc
import os
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio_ops as ao
from . import monitor
from .config import CHECKPOINTS, DIT_MODEL, QUANTIZATION, SR, VENDOR

_lock = threading.Lock()
_dit = None
_loaded_lora: str | None = None


def _ensure_path():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(VENDOR))
    os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", str(CHECKPOINTS))


def is_loaded() -> bool:
    return _dit is not None


def get_dit():
    """Inicializuje DiT (turbo, INT8, CPU offload) při prvním použití."""
    global _dit
    if _dit is not None:
        return _dit
    _ensure_path()
    from acestep.handler import AceStepHandler

    with monitor.track("acestep", f"načítání modelu {DIT_MODEL} ({QUANTIZATION})"):
        h = AceStepHandler()
        status, ok = h.initialize_service(
            project_root=str(VENDOR), config_path=DIT_MODEL, device="cuda",
            offload_to_cpu=True, offload_dit_to_cpu=True, quantization=QUANTIZATION,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step init selhal: {status}")
        _dit = h
    return h


def free_vram():
    """Uvolní GPU paměť (před demucsem)."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload():
    """Kompletně odpojí ACE-Step z paměti a vyčistí CUDA cache."""
    global _dit, _loaded_lora
    had = _dit is not None
    _dit = None
    _loaded_lora = None
    free_vram()
    if had:
        monitor.log_event("🧹", "acestep", "Model ACE-Step odpojen z GPU/RAM")


def set_lora(path: str | None, scale: float = 1.0):
    """Načte / vypne LoRA. path=None vypne."""
    global _loaded_lora
    h = get_dit()
    if not path:
        if _loaded_lora:
            h.set_use_lora(False)
        return
    if _loaded_lora != path:
        msg = h.load_lora(path)
        if not msg.startswith("✅"):
            raise RuntimeError(msg)
        _loaded_lora = path
    h.set_use_lora(True)
    h.set_lora_scale(scale)


def generate(*, src: np.ndarray | None, ref: np.ndarray | None, caption: str,
             task: str = "cover", strength: float = 0.5, bpm: float | None = None,
             key: str = "", seed: int = 42, steps: int = 8, shift: float = 3.0,
             repaint: tuple[float, float] | None = None, instrumental: bool = True,
             duration: float | None = None) -> np.ndarray:
    """Jedno volání ACE-Step. src/ref jsou (2, n) @ 48k. Vrací (2, n) @ 48k."""
    _ensure_path()
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    tmp_files: list[str] = []
    try:
        src_p = ao.tmp_wav(src) if src is not None else None
        ref_p = ao.tmp_wav(ref) if ref is not None else None
        tmp_files += [p for p in (src_p, ref_p) if p]
        p = GenerationParams(
            task_type=task, src_audio=src_p, reference_audio=ref_p, caption=caption,
            instrumental=instrumental, lyrics="[Instrumental]" if instrumental else "",
            bpm=int(round(bpm)) if bpm else None, keyscale=key, timesignature="4",
            duration=duration if duration else -1.0, inference_steps=steps, shift=shift,
            seed=seed, thinking=False, use_cot_caption=False, use_cot_metas=False,
            use_cot_language=False, audio_cover_strength=strength,
            enable_normalization=False,
        )
        if repaint:
            p.repainting_start, p.repainting_end = repaint
            p.chunk_mask_mode = "explicit"
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed],
                               audio_format="wav")
        with _lock:
            import tempfile
            outdir = tempfile.mkdtemp(prefix="fz_out_")
            with monitor.track("acestep", f"{task} · s={strength} · steps={steps}"):
                res = generate_music(get_dit(), None, p, cfg, save_dir=outdir)
        if not res.success:
            raise RuntimeError(f"generate_music: {res.error}")
        path = res.audios[0]["path"]
        a, sr = sf.read(path, dtype="float32")
        a = a.T if a.ndim == 2 else np.stack([a, a])
        if sr != SR:
            tmp_files.append(ao.tmp_wav(a, sr))
            a = ao.load(tmp_files[-1], SR)
        return a
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass
