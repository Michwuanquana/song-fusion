"""Dlouhé vstupy: rozsekání na kusy po taktech, generování a slepení crossfadem na gridu."""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import audio_ops as ao
from . import engine
from .config import SR

Progress = Callable[[float, str], None]


def bars_to_samples(n_bars: float, bar_len_s: float) -> int:
    return int(round(n_bars * bar_len_s * SR))


def generate_long(src: np.ndarray, bar_len_s: float, ref: np.ndarray | None, *,
                  chunk_bars: int = 32, overlap_bars: int = 2, progress: Progress | None = None,
                  **gen_kwargs) -> np.ndarray:
    """Přegeneruje `src` po kusech `chunk_bars` taktů s překryvem `overlap_bars`.

    Všechny kusy sdílejí seed, caption i referenci, takže styl drží. Kusy jsou zarovnané na
    takty (src musí začínat na downbeatu), crossfade probíhá přes celý překryv.
    """
    n_total = src.shape[1]
    chunk = bars_to_samples(chunk_bars, bar_len_s)
    ov = bars_to_samples(overlap_bars, bar_len_s)
    if n_total <= chunk * 1.25:
        out = engine.generate(src=src, ref=ref, duration=n_total / SR, **gen_kwargs)
        return ao.fit_length(out, n_total)

    stride = chunk - ov
    starts = list(range(0, max(n_total - ov, 1), stride))
    # poslední kus nesmí být drobek: pokud je kratší než překryv + 4 takty, sloučit s předchozím
    min_tail = ov + bars_to_samples(4, bar_len_s)
    if len(starts) > 1 and n_total - starts[-1] < min_tail:
        starts.pop()

    result: np.ndarray | None = None
    for i, s in enumerate(starts):
        e = n_total if i == len(starts) - 1 else min(s + chunk, n_total)
        piece = src[:, s:e]
        if progress:
            progress(i / len(starts), f"kus {i + 1}/{len(starts)}")
        gen = engine.generate(src=piece, ref=ref, duration=piece.shape[1] / SR, **gen_kwargs)
        gen = ao.fit_length(gen, piece.shape[1])
        result = gen if result is None else ao.crossfade_join(result, gen, ov)
    if progress:
        progress(1.0, "hotovo")
    return ao.fit_length(result, n_total)
