"""Evaluation metrics of one trained version, measured on the frozen test set, and how a kernel output becomes them."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class MetricsError(ValueError):
    """The metrics are missing, out of range or from a different test set."""


@dataclass(frozen=True)
class ClassError:
    base: float | None
    current: float


@dataclass(frozen=True)
class Metrics:
    version: str
    base: str | None
    test: str
    error: float
    base_error: float | None
    rules_error: float | None
    safety: Mapping[str, int]
    base_safety: Mapping[str, int] | None
    confidence: float
    per_class: Mapping[str, ClassError]

    def to_dict(self) -> dict[str, Any]:
        classes = {name: {"base": item.base, "current": item.current} for name, item in self.per_class.items()}
        return {"version": self.version, "base": self.base, "test": self.test, "error": self.error,
                "base_error": self.base_error, "rules_error": self.rules_error, "safety": dict(self.safety),
                "base_safety": None if self.base_safety is None else dict(self.base_safety),
                "confidence": self.confidence, "per_class": classes}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Metrics":
        classes = {name: ClassError(item["base"], item["current"]) for name, item in raw["per_class"].items()}
        return cls(raw["version"], raw["base"], raw["test"], raw["error"], raw["base_error"], raw["rules_error"],
                   raw["safety"], raw["base_safety"], raw["confidence"], classes)


def _rate(raw: Mapping[str, Any], key: str, required: bool = True) -> float | None:
    value = raw.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise MetricsError(f"métrica {key} ausente ou fora de 0 a 1")
    return float(value)


def _counts(raw: Mapping[str, Any]) -> dict[str, int]:
    safety = raw.get("seguranca")
    if not isinstance(safety, dict) or any(not isinstance(v, int) or v < 0 for v in safety.values()):
        raise MetricsError("métricas de segurança ausentes ou inválidas")
    return dict(safety)


def from_kernel(raw: Mapping[str, Any], version: str, base_version: str | None, base: Metrics | None) -> Metrics:
    """The base only counts when it was measured on the same test set; otherwise this version becomes the new baseline."""
    test = str(raw.get("teste") or "")
    if not test:
        raise MetricsError("o kernel não informou o teste usado")
    comparable = base if base is not None and base.test == test else None
    classes = raw.get("por_classe")
    if not isinstance(classes, dict) or not classes:
        raise MetricsError("métricas por classe ausentes")
    per_class = {name: ClassError(comparable.per_class[name].current if comparable and name in comparable.per_class
                                  else None, _rate(classes, name)) for name in classes}
    return Metrics(version, base_version if comparable else None, test, _rate(raw, "erro"),
                   comparable.error if comparable else None, _rate(raw, "erro_regras", required=False),
                   _counts(raw), comparable.safety if comparable else None, _rate(raw, "confianca"), per_class)
