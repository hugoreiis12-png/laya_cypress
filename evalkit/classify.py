"""Classification, signature acceptance and calibration metrics."""
from collections import Counter
from typing import Any

CALIBRATION_BINS = 5
THRESHOLDS = tuple(range(50, 100, 5))


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def prf(true_pos: int, false_pos: int, false_neg: int) -> dict[str, float | None]:
    precision = ratio(true_pos, true_pos + false_pos)
    recall = ratio(true_pos, true_pos + false_neg)
    f1 = ratio(2 * precision * recall, precision + recall) if precision and recall else None
    return {"precision": precision, "recall": recall, "f1": f1}


def doc_type_report(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    confusion = Counter(pairs)
    classes = sorted({gold for gold, _ in pairs})
    per_class = {}
    for label in classes:
        true_pos = confusion[(label, label)]
        false_pos = sum(n for (g, p), n in confusion.items() if p == label and g != label)
        false_neg = sum(n for (g, p), n in confusion.items() if g == label and p != label)
        per_class[label] = {**prf(true_pos, false_pos, false_neg), "support": true_pos + false_neg}
    correct = sum(n for (g, p), n in confusion.items() if g == p)
    confusion_table = {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())}
    return {"accuracy": ratio(correct, len(pairs)), "per_class": per_class, "confusion": confusion_table}


def signature_report(rows: list[tuple[bool, bool, float]]) -> dict[str, Any]:
    """rows: (gold signed, predicted signed, predicted confidence 0-100)."""
    true_pos = sum(1 for g, p, _ in rows if g and p)
    false_pos = sum(1 for g, p, _ in rows if p and not g)
    false_neg = sum(1 for g, p, _ in rows if g and not p)
    return {**prf(true_pos, false_pos, false_neg), "n": len(rows), "false_accepts": false_pos}


def threshold_sweep(rows: list[tuple[bool, bool, float]]) -> list[dict[str, Any]]:
    """Mirrors If4: accept only when predicted signed AND confidence >= threshold."""
    signed_total = sum(1 for g, _, _ in rows if g)
    sweep = []
    for threshold in THRESHOLDS:
        accepted = [(g, p, c) for g, p, c in rows if p and c >= threshold]
        good = sum(1 for g, _, _ in accepted if g)
        sweep.append({
            "threshold": threshold,
            "accepted": len(accepted),
            "false_accepts": len(accepted) - good,
            "precision": ratio(good, len(accepted)),
            "recall": ratio(good, signed_total),
        })
    return sweep


def calibration(pairs: list[tuple[float, bool]]) -> dict[str, Any]:
    """pairs: (confidence 0-1, was correct). Equal-width bins plus ECE."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(CALIBRATION_BINS)]
    for conf, correct in pairs:
        bins[min(int(conf * CALIBRATION_BINS), CALIBRATION_BINS - 1)].append((conf, correct))
    rows, gap = [], 0.0
    for index, chunk in enumerate(bins):
        if not chunk:
            continue
        accuracy = sum(ok for _, ok in chunk) / len(chunk)
        mean_conf = sum(c for c, _ in chunk) / len(chunk)
        gap += len(chunk) * abs(accuracy - mean_conf)
        rows.append({"bin": f"{index / CALIBRATION_BINS:.1f}-{(index + 1) / CALIBRATION_BINS:.1f}",
                     "n": len(chunk), "accuracy": round(accuracy, 4), "mean_conf": round(mean_conf, 4)})
    return {"ece": ratio(gap, len(pairs)), "bins": rows}
