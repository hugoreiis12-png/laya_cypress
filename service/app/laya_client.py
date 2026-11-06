"""Laya client: classifies the document type from OCR text. Every failure becomes LayaError."""
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from . import laya_questions
from .config import Settings
from .contract import DOC_TYPES
from .vision.types import InkFeatures

MODEL = "multilingual"
ENDPOINT = "/v1/systemone"
TEXT_CHAR_BUDGET = 12000
INK_DECIMALS = 3
CONFIDENCE_KEYS = ("answer_confidence", "confidence", "x_jev_confidence")


class LayaError(RuntimeError):
    """The decision engine is unavailable or answered something unusable."""


@dataclass(frozen=True)
class TypeDecision:
    doc_type: str
    probabilities: dict[str, float]
    confidences: dict[str, float]
    confidence: float
    truncated: bool


def ink_evidence(ink: InkFeatures) -> dict[str, Any]:
    """Ink features as the signature model reads them; training rounds the same way."""
    return {key: round(value, INK_DECIMALS) if isinstance(value, float) else value for key, value in asdict(ink).items()}


def state_for(ocr_text: str, extracted: dict[str, Any], ink: InkFeatures | None = None) -> dict[str, Any]:
    """The state Laya reads at serving time; training uses the same shape."""
    document: dict[str, Any] = {} if ink is None else {"evidencia_tinta": ink_evidence(ink)}
    document["campos_extraidos"] = extracted
    document["texto_ocr"] = ocr_text[:TEXT_CHAR_BUDGET]  # last: the encoder truncates from the end
    return {"documento": document}


def build_request(ocr_text: str, extracted: dict[str, Any]) -> dict[str, Any]:
    return {"model": MODEL, "state": state_for(ocr_text, extracted),
            "questions": {"tipo_documento": laya_questions.TIPO_DOCUMENTO}}


def parse(body: dict[str, Any], confidence_field: str) -> TypeDecision:
    try:
        answer = body["answers"]["tipo_documento"]
        confidences = {key: float(answer[key]) for key in CONFIDENCE_KEYS if key in answer}
        decision = TypeDecision(
            doc_type=answer["choice"],
            probabilities={label: float(p) for label, p in answer["probabilities"].items()},
            confidences=confidences,
            confidence=confidences[confidence_field],
            truncated=bool(body.get("usage", {}).get("truncated")),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LayaError(f"resposta do Laya fora do formato esperado: {exc!r}") from exc
    if decision.doc_type not in DOC_TYPES:
        raise LayaError(f"tipo de documento desconhecido: {decision.doc_type}")
    return decision


class LayaClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http = http or httpx.Client()
        auth = {"Authorization": f"Bearer {settings.laya_api_key}"} if settings.laya_api_key else {}
        self._headers = {"Content-Type": "application/json", **auth}

    def classify(self, ocr_text: str, extracted: dict[str, Any]) -> TypeDecision:
        if not ocr_text.strip():
            raise LayaError("texto OCR vazio; nada para classificar")
        try:
            response = self._http.post(
                self._settings.laya_url + ENDPOINT, json=build_request(ocr_text, extracted),
                headers=self._headers, timeout=self._settings.laya_timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LayaError(f"falha ao chamar o Laya: {exc!r}") from exc
        return parse(body, self._settings.laya_confidence_field)
