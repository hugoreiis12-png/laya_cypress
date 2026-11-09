"""One scheduler tick: decides what comes next and drives a training job from package to registered version."""
import json
import logging
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from . import gate, scheduler
from .config import Settings
from .kaggle_client import COMPLETE, FAILED, JobGateway, KaggleError, QuotaExceeded
from .mailing import MailError
from .metrics import Metrics, MetricsError, from_kernel
from .registry import Registry, RegistryError
from .report import build_report
from .state import KIND_LAYA, KIND_VISION, TrainerState, load_state, save_state

log = logging.getLogger(__name__)
KERNEL_TIMEOUT_SECONDS = 10 * 3600  # Kaggle stops a run at about 9 hours
KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"
KERNEL_SCRIPTS = {KIND_VISION: KERNELS_DIR / "train_vision.py", KIND_LAYA: KERNELS_DIR / "train_laya.py"}
METRICS_FILE, WEIGHTS_FILE = "metricas.json", "pesos.bin"


class DataError(RuntimeError):
    """The training data could not be prepared."""


class TrainingData(Protocol):
    def new_samples(self, trained: Mapping[str, int]) -> Mapping[str, int]: ...
    def prepare(self, kind: str, directory: Path) -> int: ...


@dataclass
class Cycle:
    settings: Settings
    gateway: JobGateway
    data: TrainingData
    registry: Registry
    notify: Callable[[dict[str, Any]], None] | None = None
    clock: Callable[[], float] = time.time

    @property
    def _state_path(self) -> Path:
        return self.settings.state_dir / "estado.json"

    def tick(self) -> str:
        state, now = load_state(self._state_path), self.clock()
        action = scheduler.next_action(state, now, lambda: self.data.new_samples(state.trained_samples), self.settings)
        if action.name == scheduler.RUN and action.kind:
            self._start(state, action.kind, now)
        elif action.name == scheduler.POLL:
            self._poll(state, now)
        return action.name

    def _start(self, state: TrainerState, kind: str, now: float) -> None:
        slug, workdir = f"laya-{kind}-{int(now)}", self.settings.state_dir / "jobs" / f"{kind}-{int(now)}"
        try:
            samples = self.data.prepare(kind, workdir / "pacote")
            self.gateway.create_dataset(workdir / "pacote", slug)
            kernel = self.gateway.push_kernel(workdir, slug, _script(kind))
        except (KaggleError, DataError) as exc:
            self._abort(state, kind, now, {"dataset": slug, "workdir": str(workdir)}, exc)
            return
        job = {"kernel": kernel, "dataset": slug, "samples": samples, "workdir": str(workdir)}
        save_state(self._state_path, scheduler.start_job(state, kind, job, now))

    def _poll(self, state: TrainerState, now: float) -> None:
        job = state.running or {}
        try:
            status = self.gateway.status(job["kernel"])
        except KaggleError as exc:
            log.warning("não foi possível consultar o kernel, tentando de novo: %s", exc)
            return
        if status == COMPLETE:
            self._complete(state, job, now)
        elif status == FAILED or now - job["started_at"] > KERNEL_TIMEOUT_SECONDS:
            self._abort(state, job["kind"], now, job, KaggleError("o kernel falhou ou passou do tempo"))

    def _complete(self, state: TrainerState, job: dict[str, Any], now: float) -> None:
        try:
            self.gateway.download(job["kernel"], Path(job["workdir"]) / "saida")
            sent = self._conclude(job["kind"], Path(job["workdir"]) / "saida", state)
        except (KaggleError, MetricsError, RegistryError, OSError, ValueError) as exc:
            self._abort(state, job["kind"], now, job, exc)
            return
        self._discard(job["dataset"])
        done = scheduler.finish_job(state, job["kind"], now, self.settings, job["samples"])
        save_state(self._state_path, replace(done, notified=(*done.notified, *sent)))
        shutil.rmtree(job["workdir"], ignore_errors=True)

    def _conclude(self, kind: str, output: Path, state: TrainerState) -> tuple[str, ...]:
        base_version = self.registry.current(kind)
        version = self.registry.next_version(kind)
        raw = json.loads((output / METRICS_FILE).read_text(encoding="utf-8"))
        metrics = from_kernel(raw, version, base_version, self.registry.metrics_of(base_version))
        self.registry.register(kind, version, output / WEIGHTS_FILE, metrics)
        verdict = gate.evaluate(metrics, self.settings.min_error_reduction)
        if verdict.baseline:
            self.registry.mark_baseline(kind, version)
        elif verdict.promote:
            self.registry.promote_to_shadow(kind, version)
            return self._announce(metrics, verdict, state)
        else:
            log.info("versão %s não promovida: %s", version, "; ".join(verdict.reasons))
        return ()

    def _announce(self, metrics: Metrics, verdict: gate.Verdict, state: TrainerState) -> tuple[str, ...]:
        if self.notify is None or metrics.version in state.notified:
            return ()
        report = build_report(metrics, verdict, self.settings.min_error_reduction, self.settings.kinds)
        try:
            self.notify(report)
        except MailError as exc:
            log.error("relatório da versão %s não enviado: %s", metrics.version, exc)
            return ()
        return (metrics.version,)

    def _abort(self, state: TrainerState, kind: str, now: float, job: Mapping[str, Any], exc: Exception) -> None:
        log.warning("job %s interrompido: %s", kind, exc)
        self._discard(job["dataset"])
        shutil.rmtree(job["workdir"], ignore_errors=True)
        failed = scheduler.fail_job(state, kind, now, self.settings, quota=isinstance(exc, QuotaExceeded))
        save_state(self._state_path, failed)

    def _discard(self, slug: str) -> None:
        try:
            self.gateway.delete_dataset(slug)
        except KaggleError as exc:
            log.warning("o dataset %s continua na conta do Kaggle, apague pelo site: %s", slug, exc)


def _script(kind: str) -> Path:
    if kind not in KERNEL_SCRIPTS:
        raise KaggleError(f"treino de {kind} ainda não tem script de kernel")
    return KERNEL_SCRIPTS[kind]
