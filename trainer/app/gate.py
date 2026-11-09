"""Promotion gate: the error must fall enough on the frozen test and no safety metric may get worse."""
from dataclasses import dataclass

from .metrics import Metrics

HARD_ZERO_SAFETY = ("nao_assinado_liberado", "carimbo_liberado")
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Verdict:
    promote: bool
    baseline: bool
    reduction: float | None
    reasons: tuple[str, ...]


def reduction_of(base_error: float, error: float) -> float:
    """Relative error reduction; a base with no error leaves nothing to reduce."""
    if base_error == 0:
        return 0.0 if error == 0 else -1.0
    return (base_error - error) / base_error


def _reduction_reasons(reduction: float, minimum: float) -> list[str]:
    if reduction + TOLERANCE >= minimum:
        return []
    return [f"o erro caiu {reduction:.1%}, abaixo do mínimo de {minimum:.1%}"]


def _safety_reasons(metrics: Metrics) -> list[str]:
    reasons = [f"métrica de segurança ausente: {key}" for key in HARD_ZERO_SAFETY if key not in metrics.safety]
    reasons += [f"{key} deve ser zero e foi {metrics.safety[key]}"
                for key in HARD_ZERO_SAFETY if metrics.safety.get(key, 0) > 0]
    base = metrics.base_safety or {}
    reasons += [f"{key} piorou de {base[key]} para {value}"
                for key, value in metrics.safety.items() if key in base and value > base[key]]
    return reasons


def _rules_reasons(metrics: Metrics) -> list[str]:
    if metrics.rules_error is None or metrics.error <= metrics.rules_error:
        return []
    return ["o erro ficou acima do das regras"]


def evaluate(metrics: Metrics, min_reduction: float) -> Verdict:
    if metrics.base_error is None:
        return Verdict(False, True, None, ("sem versão anterior no mesmo teste: registrada como referência",))
    reduction = reduction_of(metrics.base_error, metrics.error)
    reasons = [*_reduction_reasons(reduction, min_reduction), *_safety_reasons(metrics), *_rules_reasons(metrics)]
    return Verdict(not reasons, False, reduction, tuple(reasons))
