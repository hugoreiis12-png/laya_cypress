"""Scores predictions against labeled gold records, field by field."""
from collections import Counter, defaultdict
from statistics import quantiles
from typing import Any

from . import classify, fields

P95 = 95


def _aggregate_scalars(per_record: list[dict[str, str]]) -> dict[str, Any]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for outcomes in per_record:
        for name, result in outcomes.items():
            counts[name][result] += 1
    report = {}
    for name, tally in sorted(counts.items()):
        expected = tally["correct"] + tally["wrong"] + tally["missing"]
        report[name] = {**tally, "accuracy": classify.ratio(tally["correct"], expected),
                        "hallucination_rate": classify.ratio(tally["spurious"], tally["spurious"] + tally["empty_ok"])}
    return report


def _signature_row(gold: dict[str, Any], pred: dict[str, Any]) -> tuple[bool, bool, float]:
    signed = fields.dig(pred, ("assinatura", "possui_assinatura")) is True
    confidence = fields.dig(pred, ("assinatura", "confianca"))
    return bool(gold["assinatura"]["possui_assinatura"]), signed, float(confidence or 0)


def _tipo_calibration(pairs: dict[str, list[tuple[float, bool]]]) -> dict[str, Any]:
    return {name: classify.calibration(rows) for name, rows in sorted(pairs.items())}


def _latency(values: list[float]) -> dict[str, float | None]:
    if len(values) < 2:
        return {"p50": values[0] if values else None, "p95": values[0] if values else None}
    cuts = quantiles(values, n=100)
    return {"p50": round(cuts[49], 1), "p95": round(cuts[P95 - 1], 1)}


def score(gold: list[dict[str, Any]], preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    doc_pairs, scalars, sig_rows, item_totals = [], [], [], [0, 0, 0]
    calib: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    latencies, errors = [], 0
    for rec in gold:
        entry = preds.get(rec["id"]) or {}
        pred = entry.get("prediction") or {}
        errors += 1 if entry.get("error") or not pred else 0
        if "latency_ms" in entry:
            latencies.append(entry["latency_ms"])
        predicted_type = pred.get("tipo_documento") or "none"
        doc_pairs.append((rec["tipo_documento"], predicted_type))
        for name, value in ((pred.get("_meta") or {}).get("confidences") or {}).items():
            calib[name].append((float(value), predicted_type == rec["tipo_documento"]))
        scalars.append(fields.score_scalars(rec, pred))
        if rec["tipo_documento"] == "canhoto":
            sig_rows.append(_signature_row(rec, pred))
        if rec["tipo_documento"] == "pedido":
            item_totals = [a + b for a, b in zip(item_totals, fields.score_items(rec.get("itens"), pred.get("itens")))]
    return {
        "n": len(gold), "prediction_errors": errors, "latency_ms": _latency(latencies),
        "tipo_documento": classify.doc_type_report(doc_pairs),
        "tipo_calibration": _tipo_calibration(calib),
        "fields": _aggregate_scalars(scalars),
        "assinatura": {**classify.signature_report(sig_rows), "sweep": classify.threshold_sweep(sig_rows)},
        "itens": classify.prf(*item_totals),
    }
