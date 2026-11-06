"""Deterministic fields shared by serving and training, so both see the same signals."""
from typing import Any

from . import cnpj_key, comprovante, items
from .vision.analyze import Evidence


def extract_fields(evidence: Evidence, whitelist: tuple[str, ...]) -> dict[str, Any]:
    text = evidence.text
    found = (evidence.access_key, *evidence.other_keys, *cnpj_key.find_access_keys(text))
    keys = list(dict.fromkeys(key for key in found if key))
    key = keys[0] if keys else ""
    return {
        "chave": key,
        "chaves": keys,
        "cnpj": cnpj_key.pick_issuer(text, whitelist, key),
        "numero": cnpj_key.invoice_number(text, key),
        "comprovante": comprovante.extract(text),
        "itens": items.extract(text),
    }


def laya_signals(fields: dict[str, Any]) -> dict[str, bool]:
    return {"chave_valida": bool(fields["chave"]), "cnpj_na_whitelist": bool(fields["cnpj"])}
