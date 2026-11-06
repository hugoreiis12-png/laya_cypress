"""LangGraph orchestration of one document: the steps, the routes between them and the contract at the end."""
import logging
import os
import time
from typing import Any, TypedDict

TRACING_SWITCHES = ("LANGSMITH_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
for switch in TRACING_SWITCHES:
    os.environ[switch] = "false"  # set before the import: the state holds customer photos and must never reach a tracer

from langgraph.graph import END, START, StateGraph  # noqa: E402

from . import audit, contract, signature_steps, type_steps  # noqa: E402
from .engines import MIN_REVIEW_SECONDS, Engines  # noqa: E402
from .fields import extract_fields  # noqa: E402
from .hermes_client import HermesError  # noqa: E402
from .signature_decision import INVALID_IMAGE, SIGNED, UNCERTAIN, UNSIGNED  # noqa: E402
from .vision.analyze import Evidence  # noqa: E402
from .vision.image_io import ImageError  # noqa: E402

log = logging.getLogger(__name__)
CONTRACT_VERSION = "v3"
MS_PER_SECOND = 1000
READ_DOCUMENT, UNREADABLE, CLASSIFY, TRANSCRIBE, READ_KEY = "ler_documento", "imagem_ilegivel", "classificar", "transcrever", "ler_chave"
SETTLE_TYPE, VOTE, RECORD_DETECTOR, BUILD, AUDIT = "consolidar_tipo", "votar_assinatura", "registrar_detector", "montar_resposta", "auditar"
CLOSE = "fechar"


class DocumentState(TypedDict, total=False):
    data: bytes
    started: float
    meta: dict[str, Any]
    evidence: Evidence
    fields: dict[str, Any]
    doc_type: str
    confidence: float | None
    reason: str | None
    decided: dict[str, Any] | None
    built: dict[str, Any]
    envelope: dict[str, Any]


def initial_state(data: bytes, context: dict[str, Any]) -> DocumentState:
    meta = {"contrato": CONTRACT_VERSION, "id": context.get("id"), "tentativas": []}
    return DocumentState(data=data, started=time.monotonic(), meta=meta, decided=None, confidence=None, reason=None)


def _finish(meta: dict[str, Any], started: float, evidence: Evidence | None = None) -> dict[str, Any]:
    meta["latencia_ms"] = int((time.monotonic() - started) * MS_PER_SECOND)
    if evidence:
        meta["rotacao"] = evidence.rotation
    return meta


def _inconclusive(engines: Engines, doc_type: str, meta: dict[str, Any]) -> bool:
    if doc_type not in engines.settings.conclusive_types or meta["auditoria"]["problemas"]:
        return True
    return doc_type == "canhoto" and meta.get("estado_assinatura") not in (SIGNED, UNSIGNED)


def _hermes_audit(engines: Engines, state: DocumentState) -> None:
    """Hermes is only recorded until it is measured; its failure never blocks the answer."""
    meta, evidence = state["meta"], state["evidence"]
    remaining = engines.seconds_left(state["started"])
    if engines.auditor is None or not evidence.text.strip() or remaining < MIN_REVIEW_SECONDS:
        return
    try:
        meta["auditoria"]["hermes"] = engines.auditor.audit(state["built"], evidence.text, remaining)
    except HermesError as exc:
        log.warning("auditoria do hermes indisponível: %s", exc)
        meta["auditoria"]["hermes"] = {"erro": "indisponivel"}


class _Steps:
    """One method per node: each reads the state and returns only what changed, always with the meta."""

    def __init__(self, engines: Engines) -> None:
        self.engines = engines

    def read_document(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        try:
            evidence = self.engines.analyze(state["data"])
        except ImageError as exc:
            meta["tentativas"].append({"motor": "visao", "erro": str(exc)})
            meta["estado_assinatura"] = INVALID_IMAGE
            meta["inconclusivo"] = True
            return {"meta": meta}
        fields = extract_fields(evidence, self.engines.settings.whitelist_cnpj)
        return {"meta": meta, "evidence": evidence, "fields": fields}

    def unreadable(self, state: DocumentState) -> DocumentState:
        return {"envelope": contract.envelope(contract.build("imagem_invalida"), _finish(state["meta"], state["started"]))}

    def classify(self, state: DocumentState) -> DocumentState:
        meta, evidence, fields = state["meta"], state["evidence"], state["fields"]
        doc_type, confidence = type_steps.decide(self.engines, evidence, fields, meta)
        reason = type_steps.judge_reason(self.engines, evidence, fields, doc_type, confidence)
        return {"meta": meta, "doc_type": doc_type, "confidence": confidence, "reason": reason}

    def transcribe(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        judged = type_steps.consult_judge(self.engines, state["data"], state["evidence"], state["reason"], meta)
        if judged is None:
            return {"meta": meta}
        evidence, fields, doc_type, confidence = judged
        return {"meta": meta, "evidence": evidence, "fields": fields, "doc_type": doc_type, "confidence": confidence}

    def read_key(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        fields = type_steps.read_key(self.engines, state["data"], state["evidence"], meta, state["started"])
        return {"meta": meta} if fields is None else {"meta": meta, "fields": fields, "doc_type": "canhoto", "confidence": None}

    def settle_type(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        doc_type = type_steps.settle_type(self.engines, state["doc_type"], state["confidence"], state["fields"], meta)
        if doc_type == "imagem_invalida":
            meta["estado_assinatura"] = UNCERTAIN if "tipo_em_duvida" in meta else INVALID_IMAGE
        return {"meta": meta, "doc_type": doc_type}

    def vote(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        verdict = signature_steps.decide(self.engines, state["data"], state["evidence"], state["fields"], state["started"], meta)
        meta["estado_assinatura"] = verdict.state
        return {"meta": meta, "decided": contract.signature(verdict.signed, verdict.confidence, "manuscrita")}

    def record_detector(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        signature_steps.observe_detector(self.engines, state["data"], state["evidence"].rotation,
                                         state["decided"]["possui_assinatura"], meta)
        return {"meta": meta}

    def build(self, state: DocumentState) -> DocumentState:
        return {"built": contract.build(state["doc_type"], state["decided"], state["fields"])}

    def audit(self, state: DocumentState) -> DocumentState:
        meta = state["meta"]
        meta["auditoria"] = {"problemas": audit.problems(state["doc_type"], state["fields"])}
        _hermes_audit(self.engines, state)
        return {"meta": meta}

    def close(self, state: DocumentState) -> DocumentState:
        meta, evidence, fields = state["meta"], state["evidence"], state["fields"]
        meta["leitura"] = audit.reading(evidence.text, evidence.access_key, fields)
        meta["inconclusivo"] = _inconclusive(self.engines, state["doc_type"], meta)
        return {"envelope": contract.envelope(state["built"], _finish(meta, state["started"], evidence))}


class _Routes:
    def __init__(self, engines: Engines) -> None:
        self.engines = engines

    @staticmethod
    def after_reading(state: DocumentState) -> str:
        return CLASSIFY if state.get("evidence") is not None else UNREADABLE

    def after_type(self, state: DocumentState) -> str:
        wanted = type_steps.wants_key(self.engines, state["evidence"], state["fields"], state["doc_type"])
        return READ_KEY if wanted else SETTLE_TYPE

    def after_classify(self, state: DocumentState) -> str:
        return TRANSCRIBE if state["reason"] else self.after_type(state)

    @staticmethod
    def after_settle(state: DocumentState) -> str:
        return VOTE if state["doc_type"] == "canhoto" else BUILD

    def after_vote(self, _state: DocumentState) -> str:
        return RECORD_DETECTOR if signature_steps.detector_only_observes(self.engines) else BUILD


def build_graph(engines: Engines) -> Any:
    """No checkpointer on purpose: nothing about a document, least of all the photo, may outlive its run."""
    steps, routes = _Steps(engines), _Routes(engines)
    graph = StateGraph(DocumentState)
    for name, step in ((READ_DOCUMENT, steps.read_document), (UNREADABLE, steps.unreadable), (CLASSIFY, steps.classify),
                       (TRANSCRIBE, steps.transcribe), (READ_KEY, steps.read_key), (SETTLE_TYPE, steps.settle_type),
                       (VOTE, steps.vote), (RECORD_DETECTOR, steps.record_detector), (BUILD, steps.build),
                       (AUDIT, steps.audit), (CLOSE, steps.close)):
        graph.add_node(name, step)
    graph.add_edge(START, READ_DOCUMENT)
    graph.add_conditional_edges(READ_DOCUMENT, routes.after_reading, [CLASSIFY, UNREADABLE])
    graph.add_edge(UNREADABLE, END)
    graph.add_conditional_edges(CLASSIFY, routes.after_classify, [TRANSCRIBE, READ_KEY, SETTLE_TYPE])
    graph.add_conditional_edges(TRANSCRIBE, routes.after_type, [READ_KEY, SETTLE_TYPE])
    graph.add_edge(READ_KEY, SETTLE_TYPE)
    graph.add_conditional_edges(SETTLE_TYPE, routes.after_settle, [VOTE, BUILD])
    graph.add_conditional_edges(VOTE, routes.after_vote, [RECORD_DETECTOR, BUILD])
    graph.add_edge(RECORD_DETECTOR, BUILD)
    graph.add_edge(BUILD, AUDIT)
    graph.add_edge(AUDIT, CLOSE)
    graph.add_edge(CLOSE, END)
    return graph.compile()
