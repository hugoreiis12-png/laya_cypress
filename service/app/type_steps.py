"""Decides the document type: rules or Laya, the optional judge reading, and the checks on the result."""
import logging
from dataclasses import replace
from typing import Any

from . import key_reading, rules
from .config import TYPE_ENGINE_RULES
from .engines import MIN_REVIEW_SECONDS, Engines
from .fields import extract_fields, laya_signals
from .hermes_client import HermesError
from .laya_client import LayaError
from .vision.analyze import Evidence
from .vision_judge import JudgeError

log = logging.getLogger(__name__)
MIN_OCR_CHARS = 40  # shorter OCR text means the photo was not read; the judge transcribes it


def _observe_laya(engines: Engines, evidence: Evidence, fields: dict[str, Any], meta: dict[str, Any]) -> None:
    """Laya answers for the record only; its failure never touches the decision."""
    try:
        decision = engines.laya.classify(evidence.text, laya_signals(fields))
    except LayaError as exc:
        log.warning("laya observado indisponível: %s", exc)
        meta["laya_observado"] = {"erro": "indisponivel"}
        return
    meta["laya_observado"] = {"tipo": decision.doc_type, "confianca": decision.confidence}


def _by_rules(evidence: Evidence, fields: dict[str, Any], meta: dict[str, Any]) -> tuple[str, None]:
    meta["motor_decisao"] = "regras"
    meta.pop("confianca_tipo", None)
    return rules.classify(evidence.text, fields["chave"], fields["comprovante"], fields["itens"]), None


def _classify(engines: Engines, evidence: Evidence, fields: dict[str, Any], meta: dict[str, Any]) -> tuple[str, float | None]:
    """Rules when configured; otherwise Laya, then Hermes as the fallback consultant, then the rules."""
    if engines.settings.type_engine == TYPE_ENGINE_RULES:
        _observe_laya(engines, evidence, fields, meta)
        return _by_rules(evidence, fields, meta)
    for name, engine, errors in (("laya", engines.laya, LayaError), ("hermes", engines.hermes, HermesError)):
        if engine is None:
            continue
        try:
            decision = engine.classify(evidence.text, laya_signals(fields))
        except errors as exc:
            log.warning("%s indisponível: %s", name, exc)
            meta["tentativas"].append({"motor": name, "erro": str(exc)})
            continue
        meta.update(motor_decisao=name, probabilidades=decision.probabilities, confidences=decision.confidences,
                    truncado=decision.truncated, confianca_tipo=decision.confidence)
        return decision.doc_type, decision.confidence
    return _by_rules(evidence, fields, meta)


def decide(engines: Engines, evidence: Evidence, fields: dict[str, Any], meta: dict[str, Any]) -> tuple[str, float | None]:
    """A valid access key is proof of a fiscal document, so it overrides any classifier."""
    doc_type, confidence = _classify(engines, evidence, fields, meta)
    if fields["chave"] and doc_type != "canhoto":
        meta["veto_tipo"] = {"de": doc_type, "motivo": "chave_valida"}
        return "canhoto", None
    return doc_type, confidence


def judge_reason(engines: Engines, evidence: Evidence, fields: dict[str, Any], doc_type: str,
                 confidence: float | None) -> str | None:
    settings = engines.settings
    if engines.judge is None or not settings.judge_transcribe:
        return None
    if len(evidence.text.strip()) < MIN_OCR_CHARS:
        return "ocr_curto"
    if doc_type == "imagem_invalida":
        return "reavaliar_invalida"
    if confidence is not None and confidence < settings.judge_threshold:
        return "baixa_confianca"
    if doc_type == "pedido" and not fields["itens"]:
        return "pedido_sem_itens"
    if doc_type == "comprovante" and not (fields["comprovante"] or {}).get("valor"):
        return "comprovante_sem_valor"
    return None


def consult_judge(engines: Engines, data: bytes, evidence: Evidence, reason: str,
                  meta: dict[str, Any]) -> tuple[Evidence, dict[str, Any], str, float | None] | None:
    """The judge only reads; the type is decided again, with its reading first because the encoder cuts from the end."""
    if not engines.enter_judge(meta):
        return None
    try:
        transcript = engines.judge.transcribe(data)
    except JudgeError as exc:
        log.warning("juiz indisponível: %s", exc)
        meta["tentativas"].append({"motor": "juiz", "erro": str(exc)})
        return None
    finally:
        engines.leave_judge()
    meta["juiz"] = {"motivo": reason, "caracteres": len(transcript)}
    judged = replace(evidence, text=f"{evidence.text}\n{transcript}".strip())
    fields = extract_fields(judged, engines.settings.whitelist_cnpj)
    reading = replace(evidence, text=f"{transcript}\n{evidence.text}".strip())
    doc_type, confidence = decide(engines, reading, fields, meta)
    return judged, fields, doc_type, confidence


def wants_key(engines: Engines, evidence: Evidence, fields: dict[str, Any], doc_type: str) -> bool:
    if engines.judge is None or not engines.settings.judge_read_key or fields["chave"]:
        return False
    return doc_type in ("imagem_invalida", "canhoto") or len(evidence.text.strip()) < MIN_OCR_CHARS


def read_key(engines: Engines, data: bytes, evidence: Evidence, meta: dict[str, Any],
             started: float) -> dict[str, Any] | None:
    """A valid whitelisted key from the judge proves a canhoto; the original evidence stays for the record."""
    settings, remaining = engines.settings, engines.seconds_left(started)
    if remaining < MIN_REVIEW_SECONDS:
        return None
    outcome, key = key_reading.read(engines.judge, engines.judge_capacity, data,
                                    min(remaining, settings.judge_read_timeout), settings.whitelist_cnpj)
    meta["leitura_juiz"] = outcome
    return extract_fields(replace(evidence, access_key=key), settings.whitelist_cnpj) if key else None


def settle_type(engines: Engines, doc_type: str, confidence: float | None, fields: dict[str, Any],
                meta: dict[str, Any]) -> str:
    """A type below the threshold is not an answer: the flow asks for a new photo and the case is queued."""
    undecided = confidence is not None and confidence < engines.settings.judge_threshold
    if not undecided or doc_type == "imagem_invalida" or fields["chave"]:
        return doc_type
    meta["tipo_em_duvida"] = {"sugerido": doc_type, "confianca": confidence}
    return "imagem_invalida"
