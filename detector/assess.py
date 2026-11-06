"""Turns detector boxes into the shadow verdict: only a box on the canhoto side of the cut line counts."""
from dataclasses import dataclass
from typing import Any

from . import cut_line


@dataclass(frozen=True)
class Box:
    score: float
    top: float  # fraction of the upright image height
    bottom: float


def assess(boxes: list[Box], cut: cut_line.CutLine | None) -> dict[str, Any]:
    side = cut_line.canhoto_side(cut) if cut else None
    kept = [b for b in boxes if cut is None or cut_line.on_canhoto_side((b.top + b.bottom) / 2, cut, side)]
    return {"score_bruto": round(max((b.score for b in boxes), default=0.0), 4),
            "score_canhoto": round(max((b.score for b in kept), default=0.0), 4),
            "linha_corte": cut is not None, "lado_canhoto": side, "caixas": len(boxes)}
