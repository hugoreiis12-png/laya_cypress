"""One scheduler tick: decides what comes next and drives a training job from package to registered version."""
import json
import logging
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from . import backup, gate, scheduler, takeover
from .alerts import halt_message
from .config import Settings
from .kaggle_client import COMPLETE, FAILED, JobGateway, KaggleError, QuotaExceeded, kernel_error
from .mailing import MailError
from .metrics import Metrics, MetricsError, from_kernel
from .registry import Registry, RegistryError
from .report import build_report
from .state import KIND_LAYA, KIND_VISION, TrainerState, load_state, save_state

log = logging.getLogger(__name__)
KERNEL_TIMEOUT_SECONDS = 10 * 3600  # Kaggle stops a run at about 9 hours
KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"
KERNEL_SCRIPTS = {KIND_VISION: KERNELS_DIR / "train_vision.py", KIND_LAYA: KERNELS_DIR / "train_laya.py"}
METRICS_FILE = "metricas.json"
WEIGHTS_FILES = {KIND_VISION: "pesos.onnx", KIND_LAYA: "pesos.bin"}  # the vision weights are served as ONNX
MAIL_RETRY_SECONDS = 3600  # an unsent report is tried again hourly, not on every tick
JOB_ERRORS = (KaggleError, MetricsError, RegistryError, OSError, ValueError)


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
    alert: Callable[[str, str], None] | None = None
    clock: Callable[[], float] = time.time

    @property
    def _state_path(self) -> Path:
        return self.settings.state_dir / "estado.json"

    def halted(self) -> dict[str, Any] | None:
        return load_state(self._state_path).halted

    def resume_if_due(self) -> bool:
        """A halt resumes on its own a limited number of times, each after a pause; then it waits for an operator."""
        state = load_state(self._state_path)
        if not scheduler.auto_resume_due(state, self.clock(), self.settings):
            return False
        attempt = state.auto_resumes + 1
        log.warning("retomada automática %d de %d após falha em %s", attempt, self.settings.auto_resume_max,
                    (state.halted or {}).get("tipo"))
        save_state(self._state_path, replace(scheduler.resume(state), auto_resumes=attempt))
        return True

    def tick(self) -> str:
        now = self.clock()
        self._announce_pending(now)
        state = load_state(self._state_path)
        action = scheduler.next_action(state, now, lambda: self.data.new_samples(state.trained_samples), self.settings)
        if action.name == scheduler.RUN and action.kind:
            self._start(state, action.kind, now)
        elif action.name == scheduler.POLL:
            self._poll(state, now)
        return action.name

    def _package(self, state: TrainerState, kind: str, target: Path) -> int:
        """A resumed job reuses the package of the failed one; otherwise the data source builds a new one."""
        saved = backup.reusable_package(state.resume_from)
        if saved is None:
            return self.data.prepare(kind, target)
        backup.restore(saved[0], target)
        return saved[1]

    def _start(self, state: TrainerState, kind: str, now: float) -> None:
        slug, workdir = f"laya-{kind}-{int(now)}", self.settings.state_dir / "jobs" / f"{kind}-{int(now)}"
        job: dict[str, Any] = {"dataset": slug, "workdir": str(workdir)}
        try:
            job["samples"] = self._package(state, kind, workdir / "pacote")
            self.gateway.create_dataset(workdir / "pacote", slug)
            job["kernel"] = self.gateway.push_kernel(workdir, slug, _script(kind))
        except (KaggleError, DataError, OSError) as exc:
            self._abort(state, kind, now, job, exc)
            return
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
        elif status == FAILED:
            self._abort(state, job["kind"], now, job, KaggleError(self._kernel_failure(job)))
        elif now - job["started_at"] > KERNEL_TIMEOUT_SECONDS:
            self._abort(state, job["kind"], now, job, KaggleError("o kernel passou do tempo"))

    def _kernel_failure(self, job: Mapping[str, Any]) -> str:
        """Keeps the kernel log in the job folder, which becomes the backup, and names the error it ended with."""
        output = Path(job["workdir"]) / "saida"
        try:
            self.gateway.download(job["kernel"], output)
        except KaggleError as exc:
            log.warning("log do kernel %s indisponível: %s", job["kernel"], exc)
            return "o kernel falhou (log indisponível)"
        error = kernel_error(output)
        return f"o kernel falhou: {error}" if error else "o kernel falhou (sem exceção no log)"

    def _complete(self, state: TrainerState, job: dict[str, Any], now: float) -> None:
        try:
            self.gateway.download(job["kernel"], Path(job["workdir"]) / "saida")
            self._conclude(job["kind"], Path(job["workdir"]) / "saida")
        except JOB_ERRORS as exc:
            self._abort(state, job["kind"], now, job, exc)
            return
        self._discard(job["dataset"])
        backup.discard(state.resume_from)
        save_state(self._state_path, scheduler.finish_job(state, job["kind"], now, self.settings, job["samples"]))
        shutil.rmtree(job["workdir"], ignore_errors=True)
        self._announce_pending(now)

    def _conclude(self, kind: str, output: Path) -> None:
        base_version = self.registry.current(kind)
        version = self.registry.next_version(kind)
        raw = json.loads((output / METRICS_FILE).read_text(encoding="utf-8"))
        metrics = from_kernel(raw, version, base_version, self.registry.metrics_of(base_version))
        ready = self._readiness(metrics).ready
        self.registry.register(kind, version, output / WEIGHTS_FILES[kind], metrics, raw.get("classes"), ready)
        verdict = gate.evaluate(metrics, self.settings.min_error_reduction)
        if verdict.baseline:
            self.registry.mark_baseline(kind, version)
        elif verdict.promote:
            self.registry.promote_to_shadow(kind, version)
        else:
            log.info("versão %s não promovida: %s", version, "; ".join(verdict.reasons))

    def _readiness(self, metrics: Metrics) -> takeover.Readiness:
        return takeover.evaluate(metrics, self.settings.takeover_accuracy, self.settings.takeover_min_test)

    def _report(self, status: str, metrics: Metrics) -> dict[str, Any]:
        verdict = gate.evaluate(metrics, self.settings.min_error_reduction)
        return build_report(metrics, verdict, self.settings.min_error_reduction, self.settings.kinds, status,
                            self._readiness(metrics))

    def _announce_pending(self, now: float) -> None:
        """Every baseline and shadow version gets its report once; a failed send is retried later, not lost."""
        state = load_state(self._state_path)
        if self.notify is None or now < state.mail_retry_at:
            return
        sent: list[str] = []
        retry_at = 0.0
        for status, metrics in self.registry.announceable():
            if metrics.version in state.notified:
                continue
            try:
                self.notify(self._report(status, metrics))
            except MailError as exc:
                log.error("relatório da versão %s não enviado, nova tentativa em 1 h: %s", metrics.version, exc)
                retry_at = now + MAIL_RETRY_SECONDS
                break
            sent.append(metrics.version)
        if sent or retry_at:
            save_state(self._state_path, replace(state, notified=(*state.notified, *sent), mail_retry_at=retry_at))

    def _abort(self, state: TrainerState, kind: str, now: float, job: Mapping[str, Any], exc: Exception) -> None:
        """A spent quota waits for the next window; any other failure halts everything and keeps a backup."""
        self._discard(job["dataset"])
        if isinstance(exc, QuotaExceeded):
            log.warning("cota do Kaggle esgotada no job %s: %s", kind, exc)
            shutil.rmtree(job["workdir"], ignore_errors=True)
            save_state(self._state_path, scheduler.wait_for_quota(state, kind, now, self.settings))
            return
        log.error("job %s falhou, trainer parado: %s", kind, exc)
        failure = backup.save(self.settings.state_dir, kind, job, str(exc), now)
        save_state(self._state_path, scheduler.halt(state, kind, failure, now + self.settings.auto_resume_seconds))
        self._warn(failure)

    def _warn(self, failure: Mapping[str, Any]) -> None:
        if self.alert is None:
            return
        try:
            self.alert(*halt_message(failure))
        except MailError as exc:
            log.error("alerta de parada não enviado: %s", exc)

    def _discard(self, slug: str) -> None:
        try:
            self.gateway.delete_dataset(slug)
        except KaggleError as exc:
            log.warning("o dataset %s continua na conta do Kaggle, apague pelo site: %s", slug, exc)


def _script(kind: str) -> Path:
    if kind not in KERNEL_SCRIPTS:
        raise KaggleError(f"treino de {kind} ainda não tem script de kernel")
    return KERNEL_SCRIPTS[kind]
