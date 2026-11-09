"""The benchmark report: every number comes from the metrics, and the narrative text is checked against them."""
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .gate import Verdict
from .metrics import Metrics

REMAINING_ERROR = 0.05
PERCENT = 100
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
TOLERANCE = 0.051  # a percentage printed with one decimal


def build_report(metrics: Metrics, verdict: Verdict, min_reduction: float, in_progress: Sequence[str]) -> dict[str, Any]:
    classes = metrics.per_class
    return {
        "versao": metrics.version, "base": metrics.base, "teste": metrics.test,
        "antes": {"erro": metrics.base_error}, "depois": {"erro": metrics.error, "confianca": metrics.confidence},
        "reducao_relativa": verdict.reduction, "criterio_reducao": min_reduction, "seguranca": dict(metrics.safety),
        "melhorou": sorted(name for name, item in classes.items() if item.base is not None and item.current < item.base),
        "gaps": sorted(name for name, item in classes.items() if item.base is not None and item.base > REMAINING_ERROR),
        "falta": sorted(name for name, item in classes.items() if item.current > REMAINING_ERROR),
        "em_andamento": list(in_progress),
    }


def _collect(value: Any, found: list[float]) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        found.append(float(value))
    elif isinstance(value, str):
        found.extend(float(item.replace(",", ".")) for item in NUMBER.findall(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect(item, found)
    elif isinstance(value, (list, tuple)):
        found.append(float(len(value)))
        for item in value:
            _collect(item, found)


def allowed_numbers(report: Mapping[str, Any]) -> list[float]:
    found: list[float] = []
    _collect(report, found)
    return found


def unknown_numbers(text: str, report: Mapping[str, Any]) -> list[str]:
    """Numbers in the text that match nothing in the report, as a value or as a percentage of one."""
    allowed = allowed_numbers(report)
    unknown = []
    for raw in NUMBER.findall(text):
        number = float(raw.replace(",", "."))
        if not any(abs(number - value) <= TOLERANCE or abs(number - value * PERCENT) <= TOLERANCE for value in allowed):
            unknown.append(raw)
    return unknown


def _percent(value: float | None) -> str:
    return "n/d" if value is None else f"{value * PERCENT:.1f}%".replace(".", ",")


def _listed(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "nenhum"


def fallback_text(report: Mapping[str, Any]) -> str:
    """Plain text built by code, used when the written narrative cannot be trusted."""
    return "\n".join([
        f"Versão {report['versao']} promovida à sombra, medida no teste {report['teste']}.",
        f"Erro antes: {_percent(report['antes']['erro'])}. Erro agora: {_percent(report['depois']['erro'])}.",
        f"Redução relativa do erro: {_percent(report['reducao_relativa'])} (mínimo exigido {_percent(report['criterio_reducao'])}).",
        f"Onde melhorou: {_listed(report['melhorou'])}.",
        f"Gaps de antes: {_listed(report['gaps'])}.",
        f"O que falta melhorar: {_listed(report['falta'])}.",
        f"Em andamento: {_listed(report['em_andamento'])}.",
    ])
