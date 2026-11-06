"""Engine contract: turns one gold record into a contract-shaped prediction."""
from pathlib import Path
from typing import Any, Protocol


class Engine(Protocol):
    name: str

    def predict(self, record: dict[str, Any], image_path: Path) -> dict[str, Any]:
        """Returns the contract object. Optional `_meta.confidences` maps name -> 0..1."""
        ...
