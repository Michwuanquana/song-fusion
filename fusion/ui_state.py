"""Pamatování posledního nastavení UI mezi spuštěními (cache/last_ui.json)."""
from __future__ import annotations

import json

from .config import CACHE

PATH = CACHE / "last_ui.json"


def load() -> dict:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save(values: dict) -> None:
    try:
        PATH.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
