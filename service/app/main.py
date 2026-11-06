"""HTTP surface: POST /v1/extrair-documento (the contract n8n consumes), POST /v1/referencia, GET /healthz."""
import base64
import binascii
import hmac
import logging
import threading
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import config
from .decision_log import DecisionLog, decision_record, reference_record
from .hermes_client import HermesClient
from .judge_capacity import JudgeCapacity
from .laya_client import LayaClient
from .pipeline import Pipeline
from .signature_model import SignatureModelClient
from .signature_shadow import ShadowDetector
from .vision.analyze import analyze
from .vision_judge import VisionJudge

MAX_BASE64_CHARS = 20_000_000
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")


class Context(BaseModel):
    id: str | None = None
    telefone: str | None = None
    timestamp: str | None = None
    origem: str | None = None


class ExtractionRequest(BaseModel):
    imagem_base64: str
    mime_type: str = "image/jpeg"
    contexto: Context | None = None


class Reference(BaseModel):
    id: str
    origem: str = "gemini"
    output: Any = None


def _authorize(header: str | None, api_key: str) -> None:
    if not hmac.compare_digest(header or "", f"Bearer {api_key}"):
        raise HTTPException(status_code=401, detail="credencial inválida")


def _decode(payload: str) -> bytes:
    if len(payload) > MAX_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="imagem acima do limite")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="imagem_base64 inválido") from exc


def _pipeline(settings: config.Settings) -> Pipeline:
    signature_model = SignatureModelClient(settings) if settings.signature_model_url else None
    shadow = ShadowDetector(settings) if settings.signature_shadow_url else None
    hermes = HermesClient(settings) if settings.hermes_url else None
    judge = VisionJudge(settings) if settings.judge_url and settings.judge_model else None
    capacity = JudgeCapacity(settings) if judge else None
    auditor = hermes if settings.hermes_audit else None
    return Pipeline(settings, analyze, LayaClient(settings), signature_model, shadow, hermes=hermes, judge=judge,
                    judge_capacity=capacity, auditor=auditor)


def create_app(settings: config.Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or config.load()
    pipeline = pipeline or _pipeline(settings)
    workers = threading.BoundedSemaphore(settings.max_concurrency)
    decisions = DecisionLog(settings.decision_log_path)
    app = FastAPI(title="Comprover · Extração de Documentos", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/extrair-documento")
    def extract(req: ExtractionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _authorize(authorization, settings.api_key)
        data = _decode(req.imagem_base64)
        if not workers.acquire(timeout=settings.queue_timeout):
            raise HTTPException(status_code=503, detail="serviço ocupado; tente novamente")
        try:
            result = pipeline.run(data, req.contexto.model_dump() if req.contexto else {})
        finally:
            workers.release()
        decisions.write(decision_record(result, settings.decision_digest_key))
        return result

    @app.post("/v1/referencia")
    def reference(ref: Reference, authorization: str | None = Header(default=None)) -> dict[str, bool]:
        """Another engine's answer for the same photo (Gemini during the shadow), joined later by id."""
        _authorize(authorization, settings.api_key)
        decisions.write(reference_record(ref.id, ref.origem, ref.output, settings.decision_digest_key))
        return {"registrado": decisions.enabled}

    return app
