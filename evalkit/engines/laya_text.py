"""Laya alone, fed with the gold `ocr_text`: calibrates tipo_documento before any vision exists."""
import os
from pathlib import Path
from typing import Any

import httpx

from service.app.laya_questions import TIPO_DOCUMENTO

TIMEOUT_SECONDS = 30.0
DEFAULT_URL = "http://192.168.0.140:8010"
CONFIDENCE_KEYS = ("answer_confidence", "confidence", "x_jev_confidence")
QUESTION = {"tipo_documento": TIPO_DOCUMENTO}


class LayaTextEngine:
    name = "laya_text"

    def __init__(self, base_url: str | None = None) -> None:
        self._url = (base_url or os.environ.get("LAYA_URL", DEFAULT_URL)).rstrip("/") + "/v1/systemone"
        key = os.environ.get("LAYA_API_KEY", "")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}

    def predict(self, record: dict[str, Any], image_path: Path) -> dict[str, Any]:
        if not record.get("ocr_text"):
            raise ValueError("registro sem ocr_text; este engine só avalia texto")
        body = {"model": "multilingual", "state": {"documento": {"texto_ocr": record["ocr_text"]}}, "questions": QUESTION}
        response = httpx.post(self._url, json=body, headers=self._headers, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        answer = response.json()["answers"]["tipo_documento"]
        confidences = {key: answer[key] for key in CONFIDENCE_KEYS if key in answer}
        return {"tipo_documento": answer["choice"], "_meta": {"confidences": confidences}}
