from dataclasses import replace
from datetime import datetime

from trainer.app import config
from trainer.app.config import Settings

ENV = {"TRAINER_ID_SECRET": "x" * 24, "KAGGLE_USERNAME": "laya", "TRAINER_KINDS": "laya,visao",
       "TRAINER_STATE_DIR": "/tmp/trainer-test"}


def settings(**overrides) -> Settings:
    return replace(config.load(ENV), **overrides)


def at(hour: int, minute: int = 0, day: int = 6) -> float:
    return datetime(2026, 11, day, hour, minute).timestamp()
