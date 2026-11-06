"""Replays predictions captured elsewhere (e.g. the current Gemini flow) as a baseline."""
import json
from pathlib import Path
from typing import Any


class ReplayEngine:
    name = "replay"

    def __init__(self, source: Path) -> None:
        lines = source.read_text(encoding="utf-8").splitlines()
        self._by_id = {row["id"]: row["prediction"] for row in map(json.loads, filter(str.strip, lines))}

    def predict(self, record: dict[str, Any], image_path: Path) -> dict[str, Any]:
        if record["id"] not in self._by_id:
            raise KeyError(f"sem predição gravada para {record['id']}")
        return self._by_id[record["id"]]
