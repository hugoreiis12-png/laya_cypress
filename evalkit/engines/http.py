"""Calls the extraction microservice exactly as n8n does (POST /v1/extrair-documento)."""
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

TIMEOUT_SECONDS = 120.0
ENDPOINT = "/v1/extrair-documento"


class HttpEngine:
    name = "http"

    def __init__(self, base_url: str) -> None:
        self._url = base_url.rstrip("/") + ENDPOINT
        key = os.environ.get("EVAL_API_KEY", "")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}

    def predict(self, record: dict[str, Any], image_path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        body = {"imagem_base64": base64.b64encode(image_path.read_bytes()).decode(), "mime_type": mime,
                "contexto": {"id": record["id"], "origem": "evalkit"}}
        response = httpx.post(self._url, json=body, headers=self._headers, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        envelope = response.json()
        prediction = json.loads(envelope["output"])
        prediction["_meta"] = envelope.get("_meta") or {}
        return prediction
