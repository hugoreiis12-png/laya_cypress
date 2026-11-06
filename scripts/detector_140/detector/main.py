"""Shadow signature detector for the .140: POST /v1/assinatura (base64 image + OCR rotation) and GET /healthz."""
import base64
import binascii
import hmac
import os
import threading
from typing import Any, Protocol

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from service.app.vision.image_io import ImageError, decode, rotate

from . import cut_line
from .assess import Box, assess

VERSION = "rfdetr-corte-v1"
MAX_BASE64_CHARS = 20_000_000


class BoxSource(Protocol):
    def boxes(self, rgb: np.ndarray) -> list[Box]: ...


class ScoreRequest(BaseModel):
    imagem_base64: str
    rotacao: int = 0


def _upright_rgb(payload: str, rotation: int) -> np.ndarray:
    if len(payload) > MAX_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="imagem acima do limite")
    try:
        image = rotate(decode(base64.b64decode(payload, validate=True)), rotation)
    except (binascii.Error, ValueError, ImageError) as exc:
        raise HTTPException(status_code=422, detail="imagem inválida") from exc
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # same pixels the spike scored


def create_app(detector: BoxSource | None = None, api_key: str | None = None) -> FastAPI:
    api_key = os.environ.get("DETECTOR_API_KEY", "") if api_key is None else api_key
    if not api_key:
        raise RuntimeError("DETECTOR_API_KEY ausente")
    if detector is None:
        from .model import Detector
        detector = Detector()
    lock = threading.Lock()  # one inference at a time: the CPU is shared with Laya
    app = FastAPI(title="Detector de assinatura (sombra)", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "versao": VERSION}

    @app.post("/v1/assinatura")
    def score(req: ScoreRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        if not hmac.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(status_code=401, detail="credencial inválida")
        rgb = _upright_rgb(req.imagem_base64, req.rotacao)
        with lock:
            boxes = detector.boxes(rgb)
        return {"versao": VERSION, **assess(boxes, cut_line.find(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))}

    return app
