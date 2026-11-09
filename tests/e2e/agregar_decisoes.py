"""Aggregates the decision log into counts and latency percentiles; prints no id, text or field value.

Usage: python3 tests/e2e/agregar_decisoes.py /srv/dados/decisoes.jsonl [--desde 2026-11-09]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

EVENT_DECISION, EVENT_FULL_QUEUE = "decisao", "fila_cheia"
PERCENTILES = (50, 90, 99)
LATENCY_FIELDS = ("latencia_ms", "espera_fila_ms")
COUNTED = (("tipo", "tipo"), ("estado_assinatura", "estado da assinatura"), ("capacidade_juiz", "capacidade do juiz"),
           ("leitura_juiz", "leitura da chave pelo juiz"))


def read_records(path: Path, since: str) -> tuple[list[dict], int]:
    records: list[dict] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if isinstance(record, dict) and str(record.get("ts", "")) >= since:
            records.append(record)
    return records, malformed


def percentile(values: list[float], share: int) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * share / 100))]


def latency_lines(decisions: list[dict]) -> list[str]:
    lines = []
    for field in LATENCY_FIELDS:
        values = [record[field] for record in decisions if isinstance(record.get(field), (int, float))]
        if values:
            marks = ", ".join(f"p{share}={percentile(values, share):.0f}" for share in PERCENTILES)
            lines.append(f"{field}: {marks}, max={max(values):.0f} ({len(values)} casos)")
    return lines


def _tally(decisions: list[dict], getter) -> Counter:
    return Counter(str(value) for record in decisions for value in getter(record) if value is not None)


def counted_lines(decisions: list[dict]) -> list[str]:
    lines = []
    for key, title in COUNTED:
        counts = _tally(decisions, lambda record, key=key: [record.get(key)])
        if counts:
            lines.append(f"{title}: " + ", ".join(f"{name}={count}" for name, count in counts.most_common()))
    return lines


def refusal_lines(decisions: list[dict]) -> list[str]:
    review = _tally(decisions, lambda r: [(r.get("revisao_assinatura") or {}).get("resultado")])
    reasons = _tally(decisions, lambda r: r.get("motivos_inconclusivo") or [])
    engines = _tally(decisions, lambda r: [e.get("motor") for e in r.get("erros") or []])
    pending = sum(1 for r in decisions if r.get("inconclusivo"))
    return [f"revisão da assinatura: " + ", ".join(f"{n}={c}" for n, c in review.most_common()),
            f"inconclusivos: {pending} de {len(decisions)} ({100 * pending / len(decisions):.1f}%)",
            "motivos: " + ", ".join(f"{n}={c}" for n, c in reasons.most_common()),
            "falhas por motor: " + ", ".join(f"{n}={c}" for n, c in engines.most_common())]


def report(records: list[dict], malformed: int) -> list[str]:
    decisions = [r for r in records if r.get("evento") == EVENT_DECISION]
    full = sum(1 for r in records if r.get("evento") == EVENT_FULL_QUEUE)
    if not decisions:
        return [f"nenhuma decisão no período; fila cheia: {full}"]
    return [f"decisões: {len(decisions)}; fila cheia (503): {full}; linhas inválidas: {malformed}",
            *latency_lines(decisions), *counted_lines(decisions), *refusal_lines(decisions)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registro", type=Path)
    parser.add_argument("--desde", default="", help="data inicial, ex.: 2026-11-09")
    args = parser.parse_args()
    records, malformed = read_records(args.registro, args.desde)
    print("\n".join(report(records, malformed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
