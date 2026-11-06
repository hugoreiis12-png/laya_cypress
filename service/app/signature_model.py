"""Client for the fine-tuned signature model (ITS-FINE package behind serve.py). Every failure becomes SignatureModelError."""
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings

ENDPOINT = "/v1/decide"
QUESTION = "possui_assinatura"
ACTION_UNSIGNED = "positiva"  # the policy's positive option is false: the alarm is the unsigned canhoto
ACTION_SIGNED = "negativa"
ACTION_REVIEW = "revisar"
ACTIONS = (ACTION_UNSIGNED, ACTION_SIGNED, ACTION_REVIEW)


class SignatureModelError(RuntimeError):
    """The signature model is unavailable or answered something unusable."""


@dataclass(frozen=True)
class SignatureDecision:
    action: str
    probability_signed: float


def parse(body: dict[str, Any]) -> SignatureDecision:
    try:
        decision = SignatureDecision(body["action"]["acao"], float(body["answers"][QUESTION]["noul"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SignatureModelError(f"resposta do modelo de assinatura fora do formato: {exc!r}") from exc
    if decision.action not in ACTIONS:
        raise SignatureModelError(f"ação desconhecida: {decision.action}")
    return decision


class SignatureModelClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http = http or httpx.Client()

    def decide(self, state: dict[str, Any]) -> SignatureDecision:
        try:
            response = self._http.post(self._settings.signature_model_url + ENDPOINT, json={"state": state},
                                       timeout=self._settings.signature_model_timeout)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SignatureModelError(f"falha ao chamar o modelo de assinatura: {exc!r}") from exc
        return parse(body)
