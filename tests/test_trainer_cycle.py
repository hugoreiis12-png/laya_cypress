import json
from pathlib import Path

from tests.trainer_support import at, settings
from trainer.app import scheduler
from trainer.app.cycle import Cycle, DataError
from trainer.app.kaggle_client import COMPLETE, FAILED, RUNNING, KaggleError, QuotaExceeded
from trainer.app.registry import BASELINE, SHADOW, Registry
from trainer.app.mailing import MailError
from trainer.app.report import fallback_text
from trainer.app.state import HALTED, KIND_VISION, SCHEDULED, load_state, save_state

NIGHT = at(23)
GOOD = {"teste": "teste-p15", "erro": 0.08, "por_classe": {"assinado": 0.03, "nao_assinado": 0.15},
        "seguranca": {"nao_assinado_liberado": 0, "carimbo_liberado": 0}, "confianca": 0.9}
SENT_BOTH = ["visao-0001", "visao-0002"]


class FakeData:
    def __init__(self, samples=500, fail=False):
        self.samples, self.fail = samples, fail

    def new_samples(self, trained):
        return {KIND_VISION: self.samples}

    def prepare(self, kind, directory):
        if self.fail:
            raise DataError("sem fotos")
        directory.mkdir(parents=True, exist_ok=True)
        return self.samples


class FakeGateway:
    def __init__(self, status=RUNNING, quota=False, metrics=None):
        self.state, self.quota, self.metrics = status, quota, metrics or GOOD
        self.created, self.deleted, self.pushed = [], [], []
        self.log: list[dict] | None = None

    def create_dataset(self, folder, slug):
        if self.quota:
            raise QuotaExceeded("cota")
        self.created.append(slug)

    def push_kernel(self, folder, slug, script):
        self.pushed.append(script.name)
        return f"laya/{slug}"

    def status(self, kernel):
        return self.state

    def download(self, kernel, destination):
        destination.mkdir(parents=True, exist_ok=True)
        if self.log is not None:
            (destination / "treino.log").write_text(json.dumps(self.log), encoding="utf-8")
            return
        (destination / "metricas.json").write_text(json.dumps(self.metrics), encoding="utf-8")
        (destination / "pesos.onnx").write_bytes(b"pesos")

    def delete_dataset(self, slug):
        self.deleted.append(slug)


def build(tmp_path, gateway=None, data=None, notify=None, alert=None):
    cfg = settings(state_dir=tmp_path, kinds=(KIND_VISION,), min_new_samples=100)
    clock = {"now": NIGHT}
    cycle = Cycle(cfg, gateway or FakeGateway(), data or FakeData(), Registry(tmp_path / "registro"), notify,
                  alert, clock=lambda: clock["now"])
    return cycle, clock


def test_a_tick_starts_one_job_and_records_it(tmp_path):
    gateway = FakeGateway()
    cycle, _ = build(tmp_path, gateway)
    assert cycle.tick() == scheduler.RUN
    state = load_state(tmp_path / "estado.json")
    assert state.running["kind"] == KIND_VISION and gateway.pushed == ["train_vision.py"]
    assert cycle.tick() == scheduler.POLL


def test_first_finished_job_becomes_the_baseline_and_the_dataset_is_deleted(tmp_path):
    gateway = FakeGateway()
    cycle, clock = build(tmp_path, gateway)
    cycle.tick()
    gateway.state, clock["now"] = COMPLETE, NIGHT + 600
    cycle.tick()
    state = load_state(tmp_path / "estado.json")
    assert state.running is None and state.wait_until == clock["now"] + cycle.settings.pause_seconds
    assert gateway.deleted == gateway.created and Registry(tmp_path / "registro").current(KIND_VISION) == "visao-0001"


def run_two_jobs(tmp_path, second_metrics, notify=None):
    gateway = FakeGateway()
    cycle, clock = build(tmp_path, gateway, notify=notify)
    for metrics in (GOOD, second_metrics):
        gateway.metrics, gateway.state = metrics, RUNNING
        cycle.tick()
        gateway.state, clock["now"] = COMPLETE, clock["now"] + 600
        cycle.tick()
        clock["now"] += cycle.settings.pause_seconds + 1
    return cycle, Registry(tmp_path / "registro")


def test_a_clearly_better_version_goes_to_shadow_and_is_announced_once(tmp_path):
    sent = []
    better = {**GOOD, "erro": 0.05, "por_classe": {"assinado": 0.02, "nao_assinado": 0.1}}
    cycle, registry = run_two_jobs(tmp_path, better, notify=sent.append)
    assert registry.current(KIND_VISION) == "visao-0002" and [report["versao"] for report in sent] == SENT_BOTH
    assert sent[1]["status"] == SHADOW and load_state(tmp_path / "estado.json").notified == tuple(SENT_BOTH)


def test_the_first_version_is_reported_as_the_baseline(tmp_path):
    sent = []
    run_two_jobs(tmp_path, GOOD, notify=sent.append)
    assert sent[0]["versao"] == "visao-0001" and sent[0]["status"] == BASELINE
    assert "referência" in fallback_text(sent[0]) and sent[0]["assumir"]["pronta"] is False


def test_a_version_that_releases_an_unsigned_canhoto_is_not_promoted(tmp_path):
    unsafe = {**GOOD, "erro": 0.01, "seguranca": {"nao_assinado_liberado": 1, "carimbo_liberado": 0}}
    sent = []
    _, registry = run_two_jobs(tmp_path, unsafe, notify=sent.append)
    assert registry.current(KIND_VISION) == "visao-0001" and [report["versao"] for report in sent] == ["visao-0001"]


def test_a_version_above_the_floor_on_a_large_test_is_marked_ready_for_the_service(tmp_path):
    accurate = {**GOOD, "erro": 0.04, "amostras_teste": 150, "classes": ["assinado", "nao_assinado"]}
    _, registry = run_two_jobs(tmp_path, accurate)
    record = registry.versions()["visao-0002"]
    assert record["pronta"] is True and record["classes"] == ["assinado", "nao_assinado"]
    assert record["file"].endswith(".onnx") and registry.versions()["visao-0001"]["pronta"] is False


def test_high_accuracy_on_a_small_test_is_not_ready(tmp_path):
    _, registry = run_two_jobs(tmp_path, {**GOOD, "erro": 0.0, "amostras_teste": 5})
    assert registry.versions()["visao-0002"]["pronta"] is False


def test_quota_exhaustion_schedules_the_next_window_and_cleans_up(tmp_path):
    gateway = FakeGateway(quota=True)
    cycle, _ = build(tmp_path, gateway)
    cycle.tick()
    state = load_state(tmp_path / "estado.json")
    assert state.phase == SCHEDULED and state.running is None and state.next_kind == KIND_VISION
    assert state.wait_until == NIGHT + cycle.settings.quota_wait_hours * 3600 and len(gateway.deleted) == 1


def test_a_data_failure_halts_the_trainer_and_alerts(tmp_path):
    alerts = []
    cycle, _ = build(tmp_path, data=FakeData(fail=True), alert=lambda subject, body: alerts.append((subject, body)))
    cycle.tick()
    state = load_state(tmp_path / "estado.json")
    assert state.phase == HALTED and state.halted["motivo"] == "sem fotos" and cycle.halted() is not None
    assert cycle.tick() == scheduler.HALTED_ACTION and len(alerts) == 1 and "retomar" in alerts[0][1]


def test_a_failed_kernel_keeps_its_package_as_a_backup_on_the_volume(tmp_path):
    gateway = FakeGateway()
    cycle, clock = build(tmp_path, gateway)
    cycle.tick()
    gateway.state, clock["now"] = FAILED, NIGHT + 60
    cycle.tick()
    failure = load_state(tmp_path / "estado.json").halted
    saved = tmp_path / "backups" / f"visao-{int(NIGHT)}"
    assert failure["backup"] == str(saved) and (saved / "falha.json").is_file() and (saved / "pacote").is_dir()
    assert not (tmp_path / "jobs" / f"visao-{int(NIGHT)}").exists() and gateway.deleted == gateway.created


KERNEL_LOG = [{"stream_name": "stdout", "time": 1.0, "data": "Starting training for 30 epochs...\n"},
              {"stream_name": "stderr", "time": 2.0, "data": "Traceback (most recent call last):\n"},
              {"stream_name": "stderr", "time": 2.1, "data": "AttributeError: module 'ray' has no attribute 'x'\n"}]


def fail_running_job(tmp_path, gateway=None):
    gateway = gateway or FakeGateway()
    cycle, clock = build(tmp_path, gateway)
    cycle.tick()
    gateway.state, gateway.log, clock["now"] = FAILED, KERNEL_LOG, NIGHT + 60
    cycle.tick()
    return cycle, clock


def test_a_failed_kernel_names_its_error_and_keeps_its_log_in_the_backup(tmp_path):
    fail_running_job(tmp_path)
    failure = load_state(tmp_path / "estado.json").halted
    assert failure["motivo"] == "o kernel falhou: AttributeError: module 'ray' has no attribute 'x'"
    assert (Path(failure["backup"]) / "saida" / "treino.log").is_file()


def test_a_halt_resumes_on_its_own_after_the_pause_a_limited_number_of_times(tmp_path):
    cycle, clock = fail_running_job(tmp_path)
    assert cycle.resume_if_due() is False
    for attempt in range(1, cycle.settings.auto_resume_max + 1):
        clock["now"] = load_state(tmp_path / "estado.json").wait_until
        assert cycle.resume_if_due() is True
        state = load_state(tmp_path / "estado.json")
        assert state.halted is None and state.auto_resumes == attempt and state.resume_from
        cycle.tick()
        clock["now"] += 60
        cycle.tick()
    clock["now"] = load_state(tmp_path / "estado.json").wait_until
    assert cycle.resume_if_due() is False and cycle.halted() is not None


def test_a_resumed_job_reuses_the_backed_up_package_and_drops_the_backup_when_done(tmp_path):
    gateway = FakeGateway()
    cycle, clock = build(tmp_path, gateway)
    cycle.tick()
    (tmp_path / "jobs" / f"visao-{int(NIGHT)}" / "pacote" / "manifest.json").write_text("{}", encoding="utf-8")
    gateway.state, clock["now"] = FAILED, NIGHT + 60
    cycle.tick()
    saved = load_state(tmp_path / "estado.json").halted["backup"]
    save_state(tmp_path / "estado.json", scheduler.resume(load_state(tmp_path / "estado.json")))
    cycle.data = FakeData(fail=True)
    gateway.state, clock["now"] = RUNNING, NIGHT + 120
    assert cycle.tick() == scheduler.RUN and load_state(tmp_path / "estado.json").running["samples"] == 500
    gateway.state, clock["now"] = COMPLETE, NIGHT + 600
    cycle.tick()
    assert load_state(tmp_path / "estado.json").resume_from is None and not Path(saved).exists()


def test_an_unsent_report_is_retried_an_hour_later_not_lost(tmp_path):
    attempts = []

    def flaky(report):
        attempts.append(report["versao"])
        if len(attempts) == 1:
            raise MailError("smtp fora")

    better = {**GOOD, "erro": 0.05, "por_classe": {"assinado": 0.02, "nao_assinado": 0.1}}
    cycle, _ = run_two_jobs(tmp_path, better, notify=flaky)
    assert load_state(tmp_path / "estado.json").notified == ()
    clock_now = load_state(tmp_path / "estado.json").mail_retry_at
    cycle.clock = lambda: clock_now
    cycle.tick()
    assert attempts == ["visao-0001", *SENT_BOTH] and load_state(tmp_path / "estado.json").notified == tuple(SENT_BOTH)


def test_status_lookup_errors_keep_the_job_for_the_next_tick(tmp_path):
    gateway = FakeGateway()
    cycle, _ = build(tmp_path, gateway)
    cycle.tick()

    def broken(kernel):
        raise KaggleError("rede")

    gateway.status = broken
    cycle.tick()
    assert load_state(tmp_path / "estado.json").running is not None


def test_data_source_without_samples_keeps_the_loop_idle(tmp_path):
    from trainer.app.dataset import InboxData

    cycle, _ = build(tmp_path, data=InboxData(settings(state_dir=tmp_path / "s", decisions_path=tmp_path / "x.jsonl")))
    assert cycle.tick() == scheduler.IDLE_ACTION
