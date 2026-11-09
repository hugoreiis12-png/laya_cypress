"""Pure scheduling rules: one job at a time, a pause between jobs, a night window and a quota-aware retry."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from .config import MINUTES_PER_HOUR, Settings
from .state import PAUSED, SCHEDULED, TRAINING, StateError, TrainerState

RUN, POLL, WAIT, IDLE_ACTION = "executar", "acompanhar", "esperar", "ocioso"
SECONDS_PER_HOUR = 3600
BACKOFF_BASE = 2


@dataclass(frozen=True)
class Action:
    name: str
    kind: str | None = None
    until: float = 0.0


def _minute_of_day(now: float) -> int:
    moment = datetime.fromtimestamp(now)
    return moment.hour * MINUTES_PER_HOUR + moment.minute


def in_window(now: float, window: tuple[int, int]) -> bool:
    start, end = window
    minute = _minute_of_day(now)
    if start == end:
        return True
    return start <= minute < end if start < end else minute >= start or minute < end


def next_window_start(now: float, window: tuple[int, int]) -> float:
    start = window[0]
    candidate = datetime.fromtimestamp(now).replace(hour=start // MINUTES_PER_HOUR, minute=start % MINUTES_PER_HOUR,
                                                    second=0, microsecond=0)
    return (candidate + timedelta(days=1)).timestamp() if candidate.timestamp() <= now else candidate.timestamp()


def following(kind: str, kinds: tuple[str, ...]) -> str:
    index = kinds.index(kind) if kind in kinds else -1
    return kinds[(index + 1) % len(kinds)]


def _pick_kind(preferred: str, new_samples: Mapping[str, int], settings: Settings) -> str | None:
    kinds = settings.kinds
    start = kinds.index(preferred) if preferred in kinds else 0
    for kind in kinds[start:] + kinds[:start]:
        if new_samples.get(kind, 0) >= settings.min_new_samples:
            return kind
    return None


def next_action(state: TrainerState, now: float, new_samples: Callable[[], Mapping[str, int]],
                settings: Settings) -> Action:
    """The sample count is lazy: counting reads the decision log, so it only runs when a job could start."""
    if state.running:
        return Action(POLL)
    if now < state.wait_until:
        return Action(WAIT, until=state.wait_until)
    if not in_window(now, settings.window):
        return Action(WAIT, until=next_window_start(now, settings.window))
    kind = _pick_kind(state.next_kind, new_samples(), settings)
    return Action(RUN, kind=kind) if kind else Action(IDLE_ACTION)


def start_job(state: TrainerState, kind: str, job: Mapping[str, Any], now: float) -> TrainerState:
    return replace(state, phase=TRAINING, running={"kind": kind, **job, "started_at": now})


def finish_job(state: TrainerState, kind: str, now: float, settings: Settings, samples: int) -> TrainerState:
    return replace(state, phase=PAUSED, running=None, retries=0, next_kind=following(kind, settings.kinds),
                   wait_until=now + settings.pause_seconds, trained_samples={**state.trained_samples, kind: samples})


def fail_job(state: TrainerState, kind: str, now: float, settings: Settings, quota: bool) -> TrainerState:
    """Quota exhaustion waits for the next window; any other failure retries with backoff, then moves on."""
    if quota:
        wait = now + settings.quota_wait_hours * SECONDS_PER_HOUR
        return replace(state, phase=SCHEDULED, running=None, retries=0, next_kind=kind, wait_until=wait)
    retries = state.retries + 1
    if retries > settings.max_retries:
        return replace(state, phase=PAUSED, running=None, retries=0, next_kind=following(kind, settings.kinds),
                       wait_until=now + settings.pause_seconds)
    backoff = settings.pause_seconds * BACKOFF_BASE ** retries
    return replace(state, phase=PAUSED, running=None, retries=retries, next_kind=kind, wait_until=now + backoff)


def running_kind(state: TrainerState) -> str:
    if not state.running:
        raise StateError("nenhum job em andamento")
    return str(state.running["kind"])
