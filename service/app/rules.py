"""Deterministic document-type rules: the no-network fallback and the baseline Laya must beat."""
import re
from typing import Any

NF_KEYWORDS = ("danfe", "nf-e", "nota fiscal", "chave de acesso", "recebemos de", "recebedor")
RECEIPT_WORDS = re.compile(r"\b(pix|ted|doc|transfer[eê]ncia|comprovante)\b", re.IGNORECASE)
MIN_NF_KEYWORDS = 2
MIN_RECEIPT_FIELDS = 3
MIN_ORDER_ITEMS = 2


def classify(text: str, access_key: str, receipt_fields: dict[str, Any], order_items: list[Any]) -> str:
    lowered = text.casefold()
    invoice_hits = sum(word in lowered for word in NF_KEYWORDS)
    if access_key or invoice_hits >= MIN_NF_KEYWORDS:
        return "canhoto"
    filled = sum(1 for value in receipt_fields.values() if value)
    if filled >= MIN_RECEIPT_FIELDS and RECEIPT_WORDS.search(text):
        return "comprovante"
    is_order = len(order_items) >= MIN_ORDER_ITEMS and not invoice_hits  # one invoice marker: the "items" are header lines
    return "pedido" if is_order else "imagem_invalida"
