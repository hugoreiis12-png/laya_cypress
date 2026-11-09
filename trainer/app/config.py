"""Trainer settings from the environment; secrets have no default, so the trainer refuses to start without them."""
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .state import KIND_VISION, KINDS

DEFAULT_STATE_DIR = "/srv/trainer"
DEFAULT_DECISIONS_PATH = "/srv/dados/decisoes.jsonl"
DEFAULT_PAUSE_SECONDS = 240.0
MIN_PAUSE_SECONDS = 180.0
MAX_PAUSE_SECONDS = 300.0
DEFAULT_MIN_NEW_SAMPLES = 200
DEFAULT_MAX_RETRIES = 3
DEFAULT_WINDOW = "22:00-06:00"
DEFAULT_QUOTA_WAIT_HOURS = 24.0
DEFAULT_TEST_PERCENT = 15
DEFAULT_MIN_ERROR_REDUCTION = 0.20
DEFAULT_TICK_SECONDS = 30.0
DEFAULT_HERMES_MODEL = "hermes3:8b"
DEFAULT_HERMES_TIMEOUT = 60.0
DEFAULT_SMTP_PORT = 465
MIN_SECRET_CHARS = 16
MINUTES_PER_HOUR = 60
MAX_PERCENT = 100
WINDOW_PATTERN = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


class ConfigError(RuntimeError):
    """Raised at startup when required configuration is missing or invalid."""


@dataclass(frozen=True)
class MailSettings:
    host: str = ""
    port: int = DEFAULT_SMTP_PORT
    user: str = ""
    password: str = ""
    sender: str = ""
    recipients: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.sender and self.recipients)


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    decisions_path: Path
    id_secret: str
    kaggle_username: str
    kinds: tuple[str, ...]
    pause_seconds: float
    min_new_samples: int
    max_retries: int
    window: tuple[int, int]
    quota_wait_hours: float
    test_percent: int
    min_error_reduction: float
    tick_seconds: float
    hermes_url: str
    hermes_model: str
    hermes_timeout: float
    mail: MailSettings


def parse_window(raw: str) -> tuple[int, int]:
    match = WINDOW_PATTERN.match(raw)
    if not match:
        raise ConfigError(f"TRAINER_WINDOW inválida: {raw}")
    start_hour, start_minute, end_hour, end_minute = (int(group) for group in match.groups())
    if max(start_hour, end_hour) > 23 or max(start_minute, end_minute) > 59:
        raise ConfigError(f"TRAINER_WINDOW fora do relógio: {raw}")
    return start_hour * MINUTES_PER_HOUR + start_minute, end_hour * MINUTES_PER_HOUR + end_minute


def _kinds(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _mail(env: Mapping[str, str]) -> MailSettings:
    recipients = tuple(item.strip() for item in env.get("REPORT_TO", "").split(",") if item.strip())
    return MailSettings(host=env.get("SMTP_HOST", ""), port=int(env.get("SMTP_PORT", DEFAULT_SMTP_PORT)),
                        user=env.get("SMTP_USER", ""), password=env.get("SMTP_PASSWORD", ""),
                        sender=env.get("SMTP_FROM", ""), recipients=recipients)


def _problems(env: Mapping[str, str]) -> list[str]:
    problems = []
    if len(env.get("TRAINER_ID_SECRET", "")) < MIN_SECRET_CHARS:
        problems.append(f"TRAINER_ID_SECRET ausente ou com menos de {MIN_SECRET_CHARS} caracteres")
    if not env.get("KAGGLE_USERNAME"):
        problems.append("KAGGLE_USERNAME ausente")
    pause = float(env.get("TRAINER_PAUSE_SECONDS", DEFAULT_PAUSE_SECONDS))
    if not MIN_PAUSE_SECONDS <= pause <= MAX_PAUSE_SECONDS:
        problems.append(f"TRAINER_PAUSE_SECONDS fora de {MIN_PAUSE_SECONDS:.0f} a {MAX_PAUSE_SECONDS:.0f}")
    kinds = _kinds(env.get("TRAINER_KINDS", KIND_VISION))
    if not kinds or any(kind not in KINDS for kind in kinds):
        problems.append(f"TRAINER_KINDS inválido: use {', '.join(KINDS)}")
    percent = int(env.get("TRAINER_TEST_PERCENT", DEFAULT_TEST_PERCENT))
    if not 0 < percent < MAX_PERCENT:
        problems.append("TRAINER_TEST_PERCENT precisa estar entre 1 e 99")
    return problems


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    problems = _problems(env)
    window = parse_window(env.get("TRAINER_WINDOW", DEFAULT_WINDOW))
    if problems:
        raise ConfigError("; ".join(problems))
    return Settings(
        state_dir=Path(env.get("TRAINER_STATE_DIR", DEFAULT_STATE_DIR)),
        decisions_path=Path(env.get("DECISIONS_PATH", DEFAULT_DECISIONS_PATH)),
        id_secret=env["TRAINER_ID_SECRET"], kaggle_username=env["KAGGLE_USERNAME"],
        kinds=_kinds(env.get("TRAINER_KINDS", KIND_VISION)),
        pause_seconds=float(env.get("TRAINER_PAUSE_SECONDS", DEFAULT_PAUSE_SECONDS)),
        min_new_samples=int(env.get("TRAINER_MIN_NEW_SAMPLES", DEFAULT_MIN_NEW_SAMPLES)),
        max_retries=int(env.get("TRAINER_MAX_RETRIES", DEFAULT_MAX_RETRIES)), window=window,
        quota_wait_hours=float(env.get("TRAINER_QUOTA_WAIT_HOURS", DEFAULT_QUOTA_WAIT_HOURS)),
        test_percent=int(env.get("TRAINER_TEST_PERCENT", DEFAULT_TEST_PERCENT)),
        min_error_reduction=float(env.get("TRAINER_MIN_ERROR_REDUCTION", DEFAULT_MIN_ERROR_REDUCTION)),
        tick_seconds=float(env.get("TRAINER_TICK_SECONDS", DEFAULT_TICK_SECONDS)),
        hermes_url=env.get("HERMES_URL", "").rstrip("/"), hermes_model=env.get("HERMES_MODEL", DEFAULT_HERMES_MODEL),
        hermes_timeout=float(env.get("HERMES_TIMEOUT", DEFAULT_HERMES_TIMEOUT)), mail=_mail(env))
