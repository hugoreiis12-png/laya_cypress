"""Keyed digests of what two engines read, so the log can compare them without ever holding the values."""
import hashlib
import hmac
from decimal import Decimal, InvalidOperation
from typing import Any

from .cnpj_key import access_key_is_valid, digits

DIGEST_CHARS = 16
HOUR_MINUTE_CHARS = 5
CENTS = Decimal("0.01")


def _section(body: dict[str, Any], name: str) -> dict[str, Any]:
    value = body.get(name)
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _amount(value: Any) -> str:
    text = _text(value).replace(",", ".")
    try:
        return str(Decimal(text).quantize(CENTS))
    except InvalidOperation:
        return text


def _canhoto(body: dict[str, Any]) -> dict[str, str]:
    return {"chave": digits(_section(body, "qrcode").get("chave")),
            "cnpj": digits(_section(body, "emitente").get("cnpj")),
            "numero": digits(_section(body, "nota_fiscal").get("numero")).lstrip("0")}


def _comprovante(body: dict[str, Any]) -> dict[str, str]:
    fields = _section(body, "comprovante")
    return {"valor": _amount(fields.get("valor")), "data": _text(fields.get("data")),
            "hora": _text(fields.get("hora"))[:HOUR_MINUTE_CHARS],
            "identificacao": _text(fields.get("identificacao")).casefold()}


def _digest(name: str, value: str, key: str) -> str | None:
    if not value:
        return None
    return hmac.new(key.encode(), f"{name}:{value}".encode(), hashlib.sha256).hexdigest()[:DIGEST_CHARS]


def _order(body: dict[str, Any]) -> dict[str, Any]:
    found = body.get("itens")
    items = [item for item in found if isinstance(item, dict)] if isinstance(found, list) else []
    quantities = [item["qnt"] for item in items if isinstance(item.get("qnt"), (int, float))
                  and not isinstance(item.get("qnt"), bool)]
    return {"itens": len(items), "soma_qnt": sum(quantities)}


def comparable(body: dict[str, Any], key: str) -> dict[str, Any]:
    """Digests for values, plain counts for orders; names of people and banks stay out entirely."""
    doc_type = body.get("tipo_documento")
    if doc_type == "pedido":
        return _order(body)
    values = _canhoto(body) if doc_type == "canhoto" else _comprovante(body) if doc_type == "comprovante" else {}
    digests: dict[str, Any] = {name: _digest(name, value, key) for name, value in values.items()}
    if doc_type == "canhoto":
        digests["chave_valida"] = access_key_is_valid(values["chave"]) if values["chave"] else None
    return digests
