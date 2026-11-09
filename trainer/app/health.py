"""Container healthcheck: healthy while the loop keeps writing its heartbeat."""
import os
import sys
import time
from pathlib import Path

DEFAULT_STATE_DIR = "/srv/trainer"
MAX_AGE_TICKS = 4


def is_alive(heartbeat: Path, tick_seconds: float, now: float) -> bool:
    try:
        beat = float(heartbeat.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return now - beat <= tick_seconds * MAX_AGE_TICKS


if __name__ == "__main__":
    state_dir = Path(os.environ.get("TRAINER_STATE_DIR", DEFAULT_STATE_DIR))
    tick = float(os.environ.get("TRAINER_TICK_SECONDS", "30"))
    sys.exit(0 if is_alive(state_dir / "heartbeat", tick, time.time()) else 1)
