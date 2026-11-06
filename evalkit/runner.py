"""Runs an engine over labeled gold records and records predictions with latency."""
import json
import time
from pathlib import Path
from typing import Any

from .engines.base import Engine


def run_one(engine: Engine, record: dict[str, Any], root: Path) -> dict[str, Any]:
    started = time.monotonic()
    try:
        prediction = engine.predict(record, root / record["image"])
        error = None
    except Exception as exc:  # noqa: BLE001 - one bad document must not stop the run
        prediction, error = {}, f"{type(exc).__name__}: {exc}"
    latency = round((time.monotonic() - started) * 1000, 1)
    return {"id": record["id"], "engine": engine.name, "prediction": prediction,
            "latency_ms": latency, "error": error}


def run(engine: Engine, records: list[dict[str, Any]], root: Path, out: Path) -> list[dict[str, Any]]:
    rows = [run_one(engine, record, root) for record in records]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {row["id"]: row for row in map(json.loads, filter(str.strip, lines))}
