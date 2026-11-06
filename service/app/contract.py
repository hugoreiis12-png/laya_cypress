"""Output contract the n8n flow consumes: {"output": "<json string>", "_meta": {...}}."""
import json
from typing import Any

from . import comprovante

DOC_TYPES = ("canhoto", "comprovante", "pedido", "imagem_invalida")
MAX_CONFIDENCE = 100
SIGNATURE_LABELS = {
    "manuscrita": "Assinatura manuscrita",
    "rubrica": "Rubrica",
    "nome_manuscrito": "Nome manuscrito",
    "carimbo": "Carimbo",
    "nenhuma": "",
}


def signature(signed: bool, confidence_percent: float, kind: str) -> dict[str, Any]:
    clamped = int(round(max(0, min(MAX_CONFIDENCE, confidence_percent))))
    return {"possui_assinatura": bool(signed), "confianca": clamped,
            "descricao": SIGNATURE_LABELS.get(kind, "") if signed else ""}


def _canhoto(signature_info: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "tipo_documento": "canhoto", "assinatura": signature_info,
        "emitente": {"cnpj": fields.get("cnpj", "")},
        "nota_fiscal": {"numero": fields.get("numero", "")},
        "qrcode": {"chave": fields.get("chave", "")},
    }


def _comprovante(fields: dict[str, Any]) -> dict[str, Any]:
    values = fields.get("comprovante") or {}
    return {"tipo_documento": "comprovante", "comprovante": {key: values.get(key) for key in comprovante.FIELDS}}


def build(doc_type: str, signature_info: dict[str, Any] | None = None, fields: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = fields or {}
    if doc_type == "canhoto":
        return _canhoto(signature_info or signature(False, 0, "nenhuma"), fields)
    if doc_type == "comprovante":
        return _comprovante(fields)
    if doc_type == "pedido":
        return {"tipo_documento": "pedido", "itens": fields.get("itens") or []}
    return {"tipo_documento": "imagem_invalida"}


def envelope(contract: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    return {"output": json.dumps(contract, ensure_ascii=False), "_meta": meta}
