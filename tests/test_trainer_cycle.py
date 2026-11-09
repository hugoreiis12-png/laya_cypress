import json

from tests.trainer_support import at, settings
from trainer.app import scheduler
from trainer.app.cycle import Cycle, DataError
from trainer.app.kaggle_client import COMPLETE, FAILED, RUNNING, KaggleError, QuotaExceeded
from trainer.app.registry import Registry
from trainer.app.state import KIND_VISION, SCHEDULED, load_state

NIGHT = at(23)
GOOD = {"teste": "teste-p15", "erro": 0.08, "por_classe": {"assinado": 0.03, "nao_assinado": 0.15},
        "seguranca": {"nao_assinado_liberado": 0, "carimbo_liberado": 0}, "confianca": 0.9}


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
        (destination / "metricas.json").write_text(json.dumps(self.metrics), encoding="utf-8")
        (destination / "pesos.bin").write_bytes(b"pesos")

    def delete_dataset(self, slug):
        self.deleted.append(slug)


def build(tmp_path, gateway=None, data=None, notify=None):
    cfg = settings(state_dir=tmp_path, kinds=(KIND_VISION,), min_new_samples=100)
    clock = {"now": NIGHT}
    cycle = Cycle(cfg, gateway or FakeGateway(), data or FakeData(), Registry(tmp_path / "registro"), notify,
                  clock=lambda: clock["now"])
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
    assert registry.current(KIND_VISION) == "visao-0002" and len(sent) == 1
    assert sent[0]["versao"] == "visao-0002" and load_state(tmp_path / "estado.json").notified == ("visao-0002",)


def test_a_version_that_releases_an_unsigned_canhoto_is_not_promoted(tmp_path):
    unsafe = {**GOOD, "erro": 0.01, "seguranca": {"nao_assinado_liberado": 1, "carimbo_liberado": 0}}
    sent = []
    _, registry = run_two_jobs(tmp_path, unsafe, notify=sent.append)
    assert registry.current(KIND_VISION) == "visao-0001" and sent == []


def test_quota_exhaustion_schedules_the_next_window_and_cleans_up(tmp_path):
    gateway = FakeGateway(quota=True)
    cycle, _ = build(tmp_path, gateway)
    cycle.tick()
    state = load_state(tmp_path / "estado.json")
    assert state.phase == SCHEDULED and state.running is None and state.next_kind == KIND_VISION
    assert state.wait_until == NIGHT + cycle.settings.quota_wait_hours * 3600 and len(gateway.deleted) == 1


def test_data_failure_backs_off_instead_of_looping(tmp_path):
    cycle, clock = build(tmp_path, data=FakeData(fail=True))
    cycle.tick()
    state = load_state(tmp_path / "estado.json")
    assert state.retries == 1 and state.wait_until > clock["now"] and cycle.tick() == scheduler.WAIT


def test_failed_kernel_discards_the_dataset_and_retries_later(tmp_path):
    gateway = FakeGateway()
    cycle, clock = build(tmp_path, gateway)
    cycle.tick()
    gateway.state, clock["now"] = FAILED, NIGHT + 60
    cycle.tick()
    state = load_state(tmp_path / "estado.json")
    assert state.running is None and state.retries == 1 and gateway.deleted == gateway.created


def test_status_lookup_errors_keep_the_job_for_the_next_tick(tmp_path):
    gateway = FakeGateway()
    cycle, _ = build(tmp_path, gateway)
    cycle.tick()

    def broken(kernel):
        raise KaggleError("rede")

    gateway.status = broken
    cycle.tick()
    assert load_state(tmp_path / "estado.json").running is not None


def test_unconfigured_data_source_keeps_the_loop_idle(tmp_path):
    from trainer.app.main import UnconfiguredData

    cycle, _ = build(tmp_path, data=UnconfiguredData())
    assert cycle.tick() == scheduler.IDLE_ACTION
