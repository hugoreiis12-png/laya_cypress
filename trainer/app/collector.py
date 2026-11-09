"""Reads the service decision log (read only) and joins the votes of each photo by id."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .consensus import SOURCE_GEMINI, SOURCE_VLLM, Sample

MIN_JUDGE_CONFIDENCE = 80.0  # same minimum the service asks of the judge's vote
EVENT_DECISION, EVENT_REFERENCE = "decisao", "referencia"
GEMINI_ORIGIN = "gemini"


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


def load_samples(path: Path) -> Collected:
    table: dict[str, dict[str, Any]] = {}
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(record, dict):
                _merge(table, record)
    samples = tuple(Sample(item_id, entry["votes"], entry["rotation"]) for item_id, entry in table.items())
    return Collected(samples, malformed)
