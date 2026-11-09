import json
import threading
import time
from dataclasses import replace

from fastapi.testclient import TestClient

from service.app.decision_log import decision_record
from service.app.document_graph import initial_state
from service.app.engines import Engines
from service.app.main import create_app
from tests.test_service_pipeline import SETTINGS, FakeLaya
from tests.test_key_reading import run

SECONDS_WAITED = 10.0


def test_the_wait_before_the_run_is_already_spent_from_the_deadline():
    state = initial_state(b"x", {}, arrived=time.monotonic() - SECONDS_WAITED)
    engines = Engines(SETTINGS, lambda data: None, FakeLaya())
    assert state["meta"]["espera_fila_ms"] >= SECONDS_WAITED * 1000
    assert engines.seconds_left(state["started"]) <= SETTINGS.review_deadline - SECONDS_WAITED


def test_without_an_arrival_nothing_has_been_waited():
    state = initial_state(b"x", {})
    assert state["meta"]["espera_fila_ms"] < 1000


def test_an_arrival_from_the_future_never_extends_the_deadline():
    state = initial_state(b"x", {}, arrived=time.monotonic() + 100)
    assert state["meta"]["espera_fila_ms"] == 0 and state["started"] <= time.monotonic()


def test_the_decision_log_keeps_the_queue_wait():
    _, _, envelope = run(None)
    assert "espera_fila_ms" in decision_record(envelope)


def test_a_full_queue_is_refused_and_recorded(tmp_path):
    path = tmp_path / "decisoes.jsonl"
    settings = replace(SETTINGS, decision_log_path=str(path), max_concurrency=1, queue_timeout=0.05)
    release, started = threading.Event(), threading.Event()

    class SlowPipeline:
        def run(self, data, context, arrived=None):
            started.set()
            release.wait(5)
            return {"output": json.dumps({"tipo_documento": "imagem_invalida"}), "_meta": {"id": "a"}}

    client = TestClient(create_app(settings, SlowPipeline()))
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    body = {"imagem_base64": "eA==", "contexto": {"id": "fila-1"}}
    first = threading.Thread(target=lambda: client.post("/v1/extrair-documento", json=body, headers=headers))
    first.start()
    started.wait(5)
    refused = client.post("/v1/extrair-documento", json=body, headers=headers)
    release.set()
    first.join()
    assert refused.status_code == 503
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {"evento": "fila_cheia", "id": "fila-1"}.items() <= next(r for r in records if r["evento"] == "fila_cheia").items()
