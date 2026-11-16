"""Takeover gate: an absolute accuracy floor per kind, apart from the improvement gate, before Laya decides alone."""
from dataclasses import dataclass

from .gate import HARD_ZERO_SAFETY
from .metrics import Metrics


@dataclass(frozen=True)
class Readiness:
    ready: bool
    accuracy: float
    minimum: float
    test_count: int | None
    min_test: int
    reasons: tuple[str, ...]


def evaluate(metrics: Metrics, minimum: float, min_test: int) -> Readiness:
    """A small test proves nothing: 90% of a handful of strips is not 90% in production."""
    accuracy = 1.0 - metrics.error
    reasons = [] if accuracy >= minimum else [f"acerto de {accuracy:.1%}, abaixo do mínimo de {minimum:.1%}"]
    if metrics.test_count is None or metrics.test_count < min_test:
        reasons.append(f"teste com {metrics.test_count or 0} amostras, abaixo do mínimo de {min_test}")
    reasons += [f"{key} deve ser zero e foi {metrics.safety.get(key)}"
                for key in HARD_ZERO_SAFETY if metrics.safety.get(key, 1) != 0]
    return Readiness(not reasons, accuracy, minimum, metrics.test_count, min_test, tuple(reasons))
