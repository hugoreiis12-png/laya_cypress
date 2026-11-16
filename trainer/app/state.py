"""Persistent scheduler state: survives restarts and never holds photos, text or labels."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json

KIND_LAYA, KIND_VISION = "laya", "visao"
KINDS = (KIND_LAYA, KIND_VISION)
IDLE, TRAINING, PAUSED, SCHEDULED = "aguardando", "treinando", "pausa", "agendado"
HALTED = "parado_por_falha"


class StateError(RuntimeError):
    """The state file is unreadable or the transition is invalid."""


@dataclass(frozen=True)
class TrainerState:
    phase: str = IDLE
    next_kind: str = KIND_VISION
    wait_until: float = 0.0
    running: dict[str, Any] | None = None
    trained_samples: dict[str, int] = field(default_factory=dict)
    notified: tuple[str, ...] = ()
    halted: dict[str, Any] | None = None  # the failed job and where its backup is; nothing runs until it is resumed
    resume_from: str | None = None  # backup folder whose package the next job reuses
    mail_retry_at: float = 0.0


def load_state(path: Path) -> TrainerState:
    """Keys of older versions (such as the removed retry counter) are ignored, so an upgrade keeps the state."""
    try:
        raw = read_json(path, None)
        if raw is None:
            return TrainerState()
        known = {item.name for item in fields(TrainerState)}
        values = {key: value for key, value in raw.items() if key in known}
        return TrainerState(**{**values, "notified": tuple(values.get("notified", ()))})
    except (ValueError, TypeError, AttributeError) as exc:
        raise StateError(f"estado ilegível em {path}: {exc}") from exc


def save_state(path: Path, state: TrainerState) -> None:
    write_json(path, asdict(state))
