"""Engine registry."""
from pathlib import Path

from .base import Engine
from .http import HttpEngine
from .identity import IdentityEngine
from .laya_text import LayaTextEngine
from .replay import ReplayEngine

ENGINE_NAMES = ("replay", "http", "laya_text", "identity")


def build(name: str, target: str | None) -> Engine:
    if name == "replay":
        return ReplayEngine(Path(target or ""))
    if name == "http":
        return HttpEngine(target or "")
    if name == "laya_text":
        return LayaTextEngine(target)
    if name == "identity":
        return IdentityEngine()
    raise ValueError(f"engine desconhecido: {name}")
