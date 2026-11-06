"""Decides the signature of a canhoto under the configured engine; any failure leaves the case undecided."""
import logging
from dataclasses import asdict
from typing import Any

from . import signature_decision
from .config import SIGNATURE_ENGINE_JUDGE, SIGNATURE_ENGINE_TWO_VOTES
from .engines import MIN_REVIEW_SECONDS, Engines
from .fields import laya_signals
from .laya_client import state_for
from .signature_decision import UNCERTAIN, SignatureVerdict
from .signature_model import SignatureModelError
from .vision.analyze import Evidence
from .vision_judge import JudgeError, SignatureOpinion

log = logging.getLogger(__name__)


def _ask_signature(engines: Engines, data: bytes, started: float, meta: dict[str, Any]) -> SignatureOpinion | None:
    """One judge attempt inside the deadline and the capacity; the reason for any refusal stays on record."""
    if engines.judge is None:
        return None
    remaining = engines.seconds_left(started)
    if remaining < MIN_REVIEW_SECONDS:
        meta["revisao_assinatura"] = {"motor": "juiz", "resultado": "sem_prazo"}
        return None
    if not engines.enter_judge(meta):
        meta["revisao_assinatura"] = {"motor": "juiz", "resultado": "sem_capacidade"}
        return None
    try:
        return engines.judge.assess_signature(data, remaining)
    except JudgeError as exc:
        log.warning("juiz indisponível na revisão: %s", exc)
        meta["tentativas"].append({"motor": "juiz_assinatura", "erro": str(exc)})
        meta["revisao_assinatura"] = {"motor": "juiz", "resultado": "falha"}
        return None
    finally:
        engines.leave_judge()


def _record_votes(meta: dict[str, Any], detector: bool | None, opinion: SignatureOpinion | None) -> None:
    meta["votos"] = {"detector": detector, "juiz": opinion.signed if opinion else None,
                     "confianca_juiz": opinion.confidence if opinion else None}


def _detector_vote(engines: Engines, data: bytes, rotation: int, meta: dict[str, Any]) -> bool | None:
    if engines.shadow is None:
        return None
    observed = engines.shadow.observe(data, rotation)
    meta["sombra_assinatura"] = observed
    return observed.get("liberaria")


def _two_votes(engines: Engines, data: bytes, evidence: Evidence, started: float, meta: dict[str, Any]) -> SignatureVerdict:
    """The judge is asked only when the detector sees a signature, so its load stays on what could be released."""
    detector = _detector_vote(engines, data, evidence.rotation, meta)
    opinion = _ask_signature(engines, data, started, meta) if detector is True else None
    verdict = signature_decision.from_votes(detector, opinion, engines.settings.judge_min_vote_confidence)
    _record_votes(meta, detector, opinion)
    return verdict


def _judge_alone(engines: Engines, data: bytes, started: float, meta: dict[str, Any]) -> SignatureVerdict:
    opinion = _ask_signature(engines, data, started, meta)
    _record_votes(meta, None, opinion)
    return signature_decision.from_judge_alone(opinion, engines.settings.judge_min_vote_confidence)


def _model_signature(engines: Engines, evidence: Evidence, fields: dict[str, Any],
                     meta: dict[str, Any]) -> SignatureVerdict | None:
    if engines.signature_model is None or not evidence.text.strip():
        return None  # empty OCR never reached the model in training either
    try:
        decision = engines.signature_model.decide(state_for(evidence.text, laya_signals(fields), evidence.ink))
    except SignatureModelError as exc:
        log.warning("modelo de assinatura indisponível, usando regras: %s", exc)
        meta["tentativas"].append({"motor": "modelo_assinatura", "erro": str(exc)})
        return None
    meta["assinatura"] = {"motor": "modelo", "calibrado": True, "acao": decision.action,
                          "p_assinado": round(decision.probability_signed, 4)}
    return signature_decision.from_model(decision)


def _review(engines: Engines, data: bytes, verdict: SignatureVerdict, started: float, meta: dict[str, Any]) -> SignatureVerdict:
    opinion = _ask_signature(engines, data, started, meta)
    if opinion is None:
        return verdict
    reviewed = signature_decision.from_judge(opinion, engines.settings.judge_signature_max_confidence)
    meta["revisao_assinatura"] = {"motor": "juiz", "resultado": reviewed.state}
    return reviewed


def _current(engines: Engines, data: bytes, evidence: Evidence, fields: dict[str, Any], started: float,
             meta: dict[str, Any]) -> SignatureVerdict:
    verdict = _model_signature(engines, evidence, fields, meta)
    if verdict is None:
        verdict = signature_decision.from_ink(evidence.ink, engines.settings)
        meta["assinatura"] = {"motor": "regras", "calibrado": False, "atributos": asdict(evidence.ink)}
    return _review(engines, data, verdict, started, meta) if verdict.state == UNCERTAIN else verdict


def decide(engines: Engines, data: bytes, evidence: Evidence, fields: dict[str, Any], started: float,
           meta: dict[str, Any]) -> SignatureVerdict:
    engine = engines.settings.signature_engine
    if engine == SIGNATURE_ENGINE_JUDGE:
        return _judge_alone(engines, data, started, meta)
    if engine == SIGNATURE_ENGINE_TWO_VOTES:
        return _two_votes(engines, data, evidence, started, meta)
    return _current(engines, data, evidence, fields, started, meta)


def detector_only_observes(engines: Engines) -> bool:
    return engines.shadow is not None and engines.settings.signature_engine != SIGNATURE_ENGINE_TWO_VOTES


def observe_detector(engines: Engines, data: bytes, rotation: int, signed: bool, meta: dict[str, Any]) -> None:
    observed = engines.shadow.observe(data, rotation)
    if "liberaria" in observed:
        observed["concorda"] = observed["liberaria"] == signed
    meta["sombra_assinatura"] = observed
