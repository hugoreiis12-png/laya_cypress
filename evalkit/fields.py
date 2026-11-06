"""Scalar and item field scoring against a gold record."""
from collections import Counter
from typing import Any, Callable

from . import normalize

Normalizer = Callable[[Any], str]
FieldSpec = tuple[str, tuple[str, ...], Normalizer]

# name -> (path in contract, normalizer, document type it applies to)
SCALAR_FIELDS: dict[str, tuple[tuple[str, ...], Normalizer, str]] = {
    "cnpj": (("emitente", "cnpj"), normalize.digits, "canhoto"),
    "nota_numero": (("nota_fiscal", "numero"), normalize.invoice_number, "canhoto"),
    "chave": (("qrcode", "chave"), normalize.digits, "canhoto"),
}
ITEM_NAME_KEYS = ("produto", "nome", "descricao")


def dig(data: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def outcome(gold: str, pred: str) -> str:
    if gold and pred:
        return "correct" if gold == pred else "wrong"
    if gold:
        return "missing"
    return "spurious" if pred else "empty_ok"


def field_specs(gold_record: dict[str, Any]) -> list[FieldSpec]:
    if gold_record.get("rotulo_parcial"):  # triagem: só tipo e assinatura foram rotulados
        return []
    doc_type = gold_record["tipo_documento"]
    specs = [(name, path, fn) for name, (path, fn, kind) in SCALAR_FIELDS.items() if kind == doc_type]
    if doc_type == "comprovante":
        for key in gold_record.get("comprovante") or {}:
            specs.append((f"comprovante.{key}", ("comprovante", key), normalize.text))
    return specs


def score_scalars(gold_record: dict[str, Any], pred: dict[str, Any]) -> dict[str, str]:
    return {
        name: outcome(fn(dig(gold_record, path)), fn(dig(pred, path)))
        for name, path, fn in field_specs(gold_record)
    }


def _item_name(item: dict[str, Any]) -> str:
    name = next((item[key] for key in ITEM_NAME_KEYS if item.get(key)), "")
    return normalize.text(name)


def item_keys(items: Any) -> Counter:
    valid = [item for item in items or [] if isinstance(item, dict)]
    return Counter((_item_name(item), normalize.quantity(item.get("qnt"))) for item in valid)


def score_items(gold_items: Any, pred_items: Any) -> tuple[int, int, int]:
    gold, pred = item_keys(gold_items), item_keys(pred_items)
    true_pos = sum((gold & pred).values())
    return true_pos, sum(pred.values()) - true_pos, sum(gold.values()) - true_pos
