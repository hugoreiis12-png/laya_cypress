"""Deterministic consistency checks over what is about to be answered; a problem sends the case to the reserve."""
from typing import Any

from .cnpj_key import KEY_CNPJ, digits

MIN_PRODUCT_CHARS = 3
MAX_ITEM_QUANTITY = 100_000  # above this the "quantity" is a phone or document number the extractor misread


def _canhoto_problems(fields: dict[str, Any]) -> list[str]:
    problems = [] if fields.get("cnpj") else ["sem_cnpj"]
    if not fields.get("numero"):
        problems.append("sem_numero")
    key, cnpj = fields.get("chave"), digits(fields.get("cnpj"))
    if key and cnpj and key[KEY_CNPJ] != cnpj:
        problems.append("cnpj_diverge_da_chave")
    if len(fields.get("chaves") or []) > 1:
        problems.append("mais_de_uma_chave")
    return problems


def _implausible(item: dict[str, Any]) -> bool:
    quantity = item.get("qnt")
    return len(str(item.get("produto") or "").strip()) < MIN_PRODUCT_CHARS or (
        quantity is not None and quantity > MAX_ITEM_QUANTITY)


def _order_problems(fields: dict[str, Any]) -> list[str]:
    found = fields.get("itens") or []
    if not found:
        return ["sem_itens"]
    return ["itens_implausiveis"] if any(_implausible(item) for item in found) else []


def reading(text: str, key_from_code: str, fields: dict[str, Any]) -> dict[str, Any]:
    """How much the photo gave up, as counts and flags; it never carries what was read."""
    return {"caracteres_ocr": len(text.strip()), "chave_no_codigo": bool(key_from_code),
            "chaves": len(fields.get("chaves") or [])}


def problems(doc_type: str, fields: dict[str, Any]) -> list[str]:
    if doc_type == "canhoto":
        return _canhoto_problems(fields)
    return _order_problems(fields) if doc_type == "pedido" else []
