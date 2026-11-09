"""Persistent scheduler state: survives restarts and never holds photos, text or labels."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json

KIND_LAYA, KIND_VISION = "laya", "visao"
KINDS = (KIND_LAYA, KIND_VISION)
IDLE, TRAINING, PAUSED, SCHEDULED = "aguardando", "treinando", "pausa", "agendado"


class StateError(RuntimeError):
    """The state file is unreadable or the transition is invalid."""


@dataclass(frozen=True)
class TrainerState:
    phase: str = IDLE
    next_kind: str = KIND_VISION
    wait_until: float = 0.0
    retries: int = 0
    running: dict[str, Any] | None = None
    trained_samples: dict[str, int] = field(default_factory=dict)
    notified: tuple[str, ...] = ()


def load_state(path: Path) -> TrainerState:
    try:
        raw = read_json(path, None)
        if raw is None:
            return TrainerState()
        return TrainerState(**{**raw, "notified": tuple(raw.get("notified", ()))})
    except (ValueError, TypeError) as exc:
        raise StateError(f"estado ilegível em {path}: {exc}") from exc


def save_state(path: Path, state: TrainerState) -> None:
    write_json(path, asdict(state))
