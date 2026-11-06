"""Shadow signature detector client: records what RF-DETR on the canhoto side would decide, never decides."""
import base64
from typing import Any

import httpx

from .config import Settings

ENDPOINT = "/v1/assinatura"


class ShadowDetector:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http = http or httpx.Client()

    def observe(self, data: bytes, rotation: int) -> dict[str, Any]:
        """Never raises: a shadow failure must not touch the answer n8n gets."""
        settings = self._settings
        try:
            response = self._http.post(settings.signature_shadow_url + ENDPOINT,
                                       json={"imagem_base64": base64.b64encode(data).decode(), "rotacao": rotation},
                                       headers={"Authorization": f"Bearer {settings.signature_shadow_api_key}"},
                                       timeout=settings.signature_shadow_timeout)
            response.raise_for_status()
            body = response.json()
            released = float(body["score_canhoto"]) >= settings.signature_shadow_threshold
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            return {"erro": type(exc).__name__}
        return {**body, "liberaria": released}
