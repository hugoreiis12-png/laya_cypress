"""Admits a judge call only while the vLLM has room, because it also serves other work. Waits only for this service's own slot, briefly."""
import re
import threading

import httpx

from .config import Settings

METRICS_ENDPOINT = "/metrics"
METRICS_TIMEOUT = 1.0
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
CACHE_NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")  # the name depends on the vLLM version
PERCENT_SCALE = 100.0
METRIC_LINE = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.MULTILINE)
BUSY_HERE = "ocupado_no_servico"
METRICS_DOWN = "metricas_indisponiveis"
OVER_RUNNING = "muitas_em_execucao"
OVER_WAITING = "fila_no_vllm"
OVER_CACHE = "cache_alto"


def read_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in METRIC_LINE.findall(text):
        values[name] = values.get(name, 0.0) + float(value)
    return values


def refusal_for(metrics: dict[str, float], settings: Settings) -> str | None:
    if RUNNING not in metrics or WAITING not in metrics:
        return METRICS_DOWN
    if metrics[WAITING] > settings.judge_max_waiting:
        return OVER_WAITING
    if metrics[RUNNING] >= settings.judge_max_running:
        return OVER_RUNNING
    cache = next((metrics[name] for name in CACHE_NAMES if name in metrics), 0.0)
    fraction = cache / PERCENT_SCALE if cache > 1 else cache
    return OVER_CACHE if fraction > settings.judge_max_cache else None


class JudgeCapacity:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http = http or httpx.Client()
        self._slots = threading.BoundedSemaphore(settings.judge_max_in_flight)

    def _load_refusal(self) -> str | None:
        """Unknown load counts as no room: the case goes to the reserve instead of pressing the vLLM."""
        try:
            response = self._http.get(self._settings.judge_url + METRICS_ENDPOINT, timeout=METRICS_TIMEOUT)
            response.raise_for_status()
        except httpx.HTTPError:
            return METRICS_DOWN
        return refusal_for(read_metrics(response.text), self._settings)

    def enter(self) -> str | None:
        """Returns None when admitted, otherwise why not; an admitted caller must call leave."""
        if not self._slots.acquire(timeout=self._settings.judge_wait_seconds):
            return BUSY_HERE
        refusal = self._load_refusal()
        if refusal:
            self._slots.release()
        return refusal

    def leave(self) -> None:
        self._slots.release()
