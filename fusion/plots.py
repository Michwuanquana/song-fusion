"""Grafy do UI: energie songu se strukturou / riffy a časová osa scénáře."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .analysis import Analysis
from .scenario import KIND_ENERGY, Section

COLORS = {"intro": "#5b8bd0", "build": "#d9a12b", "drop": "#d0453f", "break": "#3f9f6b",
          "outro": "#8a72c2", "dip": "#e08a3c"}
INK = "#8c8c8c"


def _style(ax):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(v % 60):02d}"))


def _fig(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), dpi=105)
    fig.patch.set_alpha(0)
    return fig


def song_plot(an: Analysis, sections: list[tuple[str, int, int]] | None = None,
              riffs: list[tuple[int, str]] | None = None, riff_bars: int = 4) -> Figure:
    """Energie po taktech; barevně odhad struktury (A) nebo vyznačení kandidátů riffu (B)."""
    fig = _fig(5.4, 1.35)
    ax = fig.add_subplot(111)
    _style(ax)
    e = np.asarray(an.bar_energy)
    t = an.downbeat0 + np.arange(len(e)) * an.bar_len
    lo = float(np.percentile(e, 2)) - 2 if len(e) else -60
    for kind, s, n in sections or []:
        ax.axvspan(an.bar_time(s), an.bar_time(s + n), color=COLORS[kind], alpha=0.22, lw=0)
        ax.text(an.bar_time(s) + 1, lo + 1, kind, color=COLORS[kind], fontsize=7, va="bottom")
    for bar, lab in riffs or []:
        ax.axvspan(an.bar_time(bar), an.bar_time(bar + riff_bars), color="#d0453f", alpha=0.4, lw=0)
        ax.text(an.bar_time(bar), e.max() + 0.5 if len(e) else 0, lab, color="#d0453f",
                fontsize=7, ha="left", va="bottom")
    ax.step(t, e, where="post", color=INK, lw=0.9)
    ax.set_ylim(lo, (e.max() if len(e) else 0) + 4)
    ax.set_xlim(0, an.duration)
    ax.set_ylabel("dB", color=INK, fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig


def timeline_plot(secs: list[Section], bar_len_s: float, measured_db: list[float] | None = None
                  ) -> Figure:
    """Aranž jako bloky: šířka = délka, výška = plánovaná energie; volitelně naměřené dB."""
    fig = _fig(6.2, 1.85)
    ax = fig.add_subplot(111)
    _style(ax)
    t = 0.0
    centers = []
    for s in secs:
        w = s.bars * bar_len_s
        h = KIND_ENERGY[s.kind] * 10 ** (s.db / 20)
        if "nokick" in s.flags:
            h *= 0.8
        ax.bar(t, h, width=w, align="edge", color=COLORS[s.kind], alpha=0.85,
               edgecolor="white", linewidth=0.6)
        tags = " ".join(sorted(f for f in s.flags if not f.startswith("no")))
        label = f"{s.kind}\n{s.bars}" + (f"\n{tags}" if tags and w > 18 else "")
        ax.text(t + w / 2, h + 0.03, label, ha="center", va="bottom", fontsize=6.5, color=INK)
        centers.append(t + w / 2)
        t += w
    ax.set_xlim(0, max(t, 1))
    ax.set_ylim(0, 1.55)
    ax.set_yticks([])
    if measured_db:
        ax2 = ax.twinx()
        ax2.plot(centers, measured_db, color=INK, marker="o", ms=3, lw=1)
        ax2.set_ylabel("naměřeno dB", color=INK, fontsize=7)
        ax2.tick_params(colors=INK, labelsize=7)
        for side in ("top", "left"):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_color(INK)
    fig.tight_layout(pad=0.3)
    return fig
