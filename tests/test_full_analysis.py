from dataclasses import replace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from service.app import config, main
from service.app.pipeline import Pipeline

from service.app.full_analysis import ORIGIN, FullAnalysis, Job, prompt_for
from service.app.vision_judge import JudgeError, parse_document
from tests.test_service_engines import BASE_ENV, canhoto
from tests.test_service_pipeline import SETTINGS, FakeLaya

CANHOTO = {"tipo_documento": "canhoto", "assinatura": {"possui_assinatura": True, "confianca": 92}}


class FakeAnalyst:
    def __init__(self, answer=CANHOTO, fail=False):
        self.answer, self.fail, self.seen = answer, fail, None

    def analyze_document(self, data, prompt, timeout):
        self.seen = data
        if self.fail:
            raise JudgeError("fora do ar")
        return self.answer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def build(analyst=None, idle=lambda: True, queue_size=None):
    settings = SETTINGS if queue_size is None else replace(SETTINGS, full_analysis_queue=queue_size)
    written, clock = [], Clock()
    full = FullAnalysis(settings, analyst or FakeAnalyst(), idle, written.append, clock, clock.sleep)
    return full, written, clock


def test_the_answer_is_kept_as_a_vllm_reference_with_its_latency():
    full, written, _ = build()
    full.process(Job("ID1", b"x", 0, 0.0))
    assert written[0]["evento"] == "referencia" and written[0]["origem"] == ORIGIN
    assert written[0]["possui_assinatura"] is True and written[0]["confianca_assinatura"] == 92
    assert "latencia_ms" in written[0]


def test_the_vllm_reads_the_photo_upright():
    photo = cv2.imencode(".jpg", np.zeros((120, 80, 3), np.uint8))[1].tobytes()
    analyst = FakeAnalyst()
    full, _, _ = build(analyst)
    full.process(Job("ID1", photo, 90, 0.0))
    assert cv2.imdecode(np.frombuffer(analyst.seen, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (80, 120)


def test_a_failure_is_logged_without_a_reference():
    full, written, _ = build(FakeAnalyst(fail=True))
    full.process(Job("ID1", b"x", 0, 0.0))
    assert written == [{"evento": "analise_completa_falhou", "id": "ID1", "erro": "fora do ar"}]


def test_a_busy_vllm_is_waited_for_and_then_given_up():
    full, written, clock = build(idle=lambda: False)
    full.process(Job("ID1", b"x", 0, 0.0))
    assert written[0] == {"evento": "analise_completa_descartada", "id": "ID1", "motivo": "vllm_ocupado"}
    assert clock.now > 600


def test_a_full_queue_drops_without_blocking_and_a_missing_id_is_ignored():
    full, written, _ = build(queue_size=1)
    full.submit(None, b"x", 0)
    full.submit("ID1", b"x", 0)
    full.submit("ID2", b"x", 0)
    assert written == [{"evento": "analise_completa_descartada", "id": "ID2", "motivo": "fila_cheia"}]


def test_the_prompt_lists_the_whitelisted_issuers_formatted():
    assert "36.895.856/0001-10" in prompt_for(("36895856000110",))


def test_an_answer_without_a_type_is_rejected():
    assert parse_document('```json\n{"tipo_documento": "pedido", "itens": []}\n```')["tipo_documento"] == "pedido"
    for text in ("sem json", '{"itens": []}', "[1, 2]"):
        try:
            parse_document(text)
        except JudgeError:
            continue
        raise AssertionError(text)


class RecordingFull:
    def __init__(self):
        self.calls = []

    def submit(self, record_id, data, rotation):
        self.calls.append((record_id, rotation))


def test_the_endpoint_hands_the_photo_over_after_answering():
    settings = config.load(BASE_ENV)
    full = RecordingFull()
    client = TestClient(main.create_app(settings, Pipeline(settings, lambda data: canhoto(), FakeLaya()), full))
    photo = {"imagem_base64": "eA==", "contexto": {"id": "abc"}}
    assert client.post("/v1/extrair-documento", json=photo, headers={"Authorization": "Bearer k"}).status_code == 200
    assert full.calls == [("abc", 0)]


def test_the_full_analysis_is_off_unless_asked():
    assert config.load(BASE_ENV).full_analysis is False
    assert config.load({**BASE_ENV, "FULL_ANALYSIS": "1"}).full_analysis is True
