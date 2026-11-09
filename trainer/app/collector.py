"""Reads the service decision log (read only) and joins the votes of each photo by id."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .consensus import SOURCE_GEMINI, SOURCE_VLLM, Sample
from .type_consensus import SOURCE_GEMINI as TYPE_GEMINI, SOURCE_RULES, TypeSample

MIN_JUDGE_CONFIDENCE = 80.0  # same minimum the service asks of the judge's vote
EVENT_DECISION, EVENT_REFERENCE = "decisao", "referencia"
GEMINI_ORIGIN = "gemini"
RULES_ENGINE = "regras"  # only this engine is independent of Laya


@dataclass(frozen=True)
class Collected:
    samples: tuple[Sample, ...]
    malformed: int


def _judge_vote(record: dict[str, Any]) -> bool | None:
    votes = record.get("votos") or {}
    vote, confidence = votes.get("juiz"), votes.get("confianca_juiz")
    if not isinstance(vote, bool):
        return None
    return vote if confidence is None or confidence >= MIN_JUDGE_CONFIDENCE else None


def _merge(table: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    item_id = record.get("id")
    if not item_id:
        return
    entry = table.setdefault(str(item_id), {"votes": {}, "rotation": 0})
    if record.get("evento") == EVENT_DECISION:
        entry["votes"][SOURCE_VLLM] = _judge_vote(record)
        entry["rotation"] = int(record.get("rotacao") or 0)
    elif record.get("evento") == EVENT_REFERENCE and record.get("origem") == GEMINI_ORIGIN:
        signed = record.get("possui_assinatura")
        entry["votes"][SOURCE_GEMINI] = signed if isinstance(signed, bool) else None


def _records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
    return records, malformed


def load_samples(path: Path) -> Collected:
    table: dict[str, dict[str, Any]] = {}
    records, malformed = _records(path)
    for record in records:
        _merge(table, record)
    samples = tuple(Sample(item_id, entry["votes"], entry["rotation"]) for item_id, entry in table.items())
    return Collected(samples, malformed)


def _merge_type(table: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    item_id = record.get("id")
    if not item_id:
        return
    entry = table.setdefault(str(item_id), {"votes": {}, "rotation": 0})
    if record.get("evento") == EVENT_DECISION:
        entry["votes"][SOURCE_RULES] = record.get("tipo") if record.get("motor") == RULES_ENGINE else None
        entry["rotation"] = int(record.get("rotacao") or 0)
    elif record.get("evento") == EVENT_REFERENCE and record.get("origem") == GEMINI_ORIGIN:
        entry["votes"][TYPE_GEMINI] = record.get("tipo")


def load_type_samples(path: Path) -> tuple[TypeSample, ...]:
    table: dict[str, dict[str, Any]] = {}
    for record in _records(path)[0]:
        _merge_type(table, record)
    return tuple(TypeSample(item_id, entry["votes"], entry["rotation"]) for item_id, entry in table.items())
