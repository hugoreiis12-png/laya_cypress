"""Engine that replays the gold label as the prediction — baseline for sanity checks."""
from pathlib import Path
from typing import Any


class IdentityEngine:
    """Predicts exactly what the gold label says. Useful to verify the scoring pipeline
    end-to-end before a real model is available."""

    name = "identity"

    def predict(self, record: dict[str, Any], image_path: Path) -> dict[str, Any]:
        return {
            "tipo_documento": record.get("tipo_documento"),
            "assinatura": record.get("assinatura"),
        }
