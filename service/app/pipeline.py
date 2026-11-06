"""One document end to end: the public entry point over the LangGraph orchestration."""
from typing import Any, Callable

from .config import Settings
from .document_graph import build_graph, initial_state
from .engines import Auditor, Classifier, Engines, Judge, SignatureModel, SignatureShadow
from .fields import extract_fields, laya_signals
from .judge_capacity import JudgeCapacity
from .vision.analyze import Evidence

__all__ = ["Pipeline", "extract_fields", "laya_signals"]


class Pipeline:
    def __init__(self, settings: Settings, analyze: Callable[[bytes], Evidence], laya: Classifier,
                 signature_model: SignatureModel | None = None, shadow: SignatureShadow | None = None,
                 hermes: Classifier | None = None, judge: Judge | None = None,
                 judge_capacity: JudgeCapacity | None = None, auditor: Auditor | None = None) -> None:
        self._graph = build_graph(Engines(settings, analyze, laya, signature_model, shadow, hermes, judge,
                                          judge_capacity, auditor))

    def run(self, data: bytes, context: dict[str, Any]) -> dict[str, Any]:
        return self._graph.invoke(initial_state(data, context))["envelope"]
