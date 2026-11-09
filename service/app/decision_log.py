"""Append-only JSONL of decisions and reference answers, joined by id. Never image, OCR text, phone or field values."""
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from .field_digest import comparable

log = logging.getLogger(__name__)
MAX_REASON_CHARS = 200
SHADOW_KEYS =("score_canhoto", "liberaria", "concorda", "versao", "erro")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class DecisionLog:
    def __init__(self, path: str) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def write(self, record: dict[str, Any]) -> None:
        """Never raises: a lost log line must not fail the request."""
        if self._path is None:
            return
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}, ensure_ascii=False)
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            log.warning("registro de decisões indisponível: %s", exc)


def _failure_reasons(attempts: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Why each engine failed, cut short: exception text names the cause, never the photo or what it read."""
    return [{"motor": attempt["motor"], "erro": str(attempt.get("erro", ""))[:MAX_REASON_CHARS]} for attempt in attempts]


def decision_record(envelope: dict[str, Any], digest_key: str = "") -> dict[str, Any]:
    body, meta = json.loads(envelope["output"]), envelope["_meta"]
    signature = body.get("assinatura") or {}
    shadow = meta.get("sombra_assinatura") or {}
    compared = {"campos": comparable(body, digest_key)} if digest_key else {}
    return {
        **compared,
        "evento": "decisao", "id": meta.get("id"), "tipo": body.get("tipo_documento"),
        "motor": meta.get("motor_decisao"), "confianca_tipo": meta.get("confianca_tipo"),
        "assinatura": {"possui": signature.get("possui_assinatura"), "confianca": signature.get("confianca")}
        if signature else None,
        "estado_assinatura": meta.get("estado_assinatura"),
        "revisao_assinatura": meta.get("revisao_assinatura"),
        "tipo_em_duvida": meta.get("tipo_em_duvida"), "veto_tipo": meta.get("veto_tipo"),
        "laya_observado": meta.get("laya_observado"), "capacidade_juiz": meta.get("capacidade_juiz"),
        "votos": meta.get("votos"), "inconclusivo": meta.get("inconclusivo"),
        "motivos_inconclusivo": meta.get("motivos_inconclusivo"), "auditoria": meta.get("auditoria"),
        "leitura": meta.get("leitura"), "leitura_juiz": meta.get("leitura_juiz"),
        "sombra": {key: shadow[key] for key in SHADOW_KEYS if key in shadow} or None,
        "juiz": meta.get("juiz"), "falhas": [attempt["motor"] for attempt in meta.get("tentativas", [])],
        "erros": _failure_reasons(meta.get("tentativas", [])),
        "latencia_ms": meta.get("latencia_ms"), "espera_fila_ms": meta.get("espera_fila_ms"), "rotacao": meta.get("rotacao"),
    }


def _as_dict(output: Any) -> dict[str, Any] | None:
    if isinstance(output, dict):
        return output
    match = JSON_OBJECT.search(str(output))
    try:
        parsed = json.loads(match.group(0)) if match else None
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def reference_record(record_id: str, source: str, output: Any, digest_key: str = "") -> dict[str, Any]:
    """Keeps the other engine's type and signature verdict; what it read only as keyed digests, if enabled."""
    record = {"evento": "referencia", "id": record_id, "origem": source}
    parsed = _as_dict(output)
    if parsed is None:
        return {**record, "tipo": None, "erro": "formato"}
    signature = parsed.get("assinatura") if isinstance(parsed.get("assinatura"), dict) else {}
    compared = {"campos": comparable(parsed, digest_key)} if digest_key else {}
    return {**record, "tipo": parsed.get("tipo_documento"), "possui_assinatura": signature.get("possui_assinatura"),
            **compared}
