import json

import httpx
import pytest
from fastapi.testclient import TestClient

from service.app import config, main
from service.app.decision_log import DecisionLog, decision_record, reference_record
from service.app.hermes_client import HermesClient, HermesError
from service.app.laya_client import LayaError, TypeDecision
from service.app.pipeline import Pipeline
from service.app.vision.analyze import Evidence
from service.app.vision_judge import JudgeError, VisionJudge
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import BLANK, DANFE_TEXT, SIGNED, FakeLaya, parse

BASE_ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10"}
SETTINGS = config.load({**BASE_ENV, "HERMES_URL": "http://hermes", "JUDGE_URL": "http://judge", "JUDGE_MODEL": "vlm"})
ORDER = "Alface 10\nRúcula 5\nCheiro verde 20"


class FakeHermes:
    def __init__(self, fail=False):
        self.fail = fail

    def classify(self, ocr_text, extracted):
        if self.fail:
            raise HermesError("fora do ar")
        return TypeDecision("canhoto", {"canhoto": 0.8}, {"hermes": 0.8}, 0.8, False)


class FakeJudge:
    def __init__(self, transcript=ORDER, fail=False):
        self.transcript, self.fail, self.calls = transcript, fail, 0

    def transcribe(self, data):
        self.calls += 1
        if self.fail:
            raise JudgeError("fora do ar")
        return self.transcript


class ReadingLaya:
    """Raises on empty text like the real client; sees an order only when the text shows one."""

    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def classify(self, ocr_text, extracted):
        if not ocr_text.strip():
            raise LayaError("texto OCR vazio")
        doc_type = "pedido" if "Alface" in ocr_text else "canhoto"
        return TypeDecision(doc_type, {doc_type: self.confidence}, {"answer_confidence": self.confidence},
                            self.confidence, False)


def run(evidence, laya, hermes=None, judge=None):
    engine = Pipeline(SETTINGS, lambda data: evidence, laya, hermes=hermes, judge=judge)
    return parse(engine.run(b"x", {"id": "abc"}))


def canhoto():
    return Evidence(DANFE_TEXT, make_key(), SIGNED, 0)


def http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_hermes_decides_when_laya_fails():
    body, meta = run(canhoto(), FakeLaya(fail=True), FakeHermes())
    assert body["tipo_documento"] == "canhoto" and meta["motor_decisao"] == "hermes"
    assert [attempt["motor"] for attempt in meta["tentativas"]] == ["laya"]


def test_rules_answer_when_laya_and_hermes_fail():
    body, meta = run(canhoto(), FakeLaya(fail=True), FakeHermes(fail=True))
    assert body["tipo_documento"] == "canhoto" and meta["motor_decisao"] == "regras"
    assert "confianca_tipo" not in meta


def test_judge_reads_what_ocr_missed_and_laya_decides_again():
    body, meta = run(Evidence("", "", BLANK, 0), ReadingLaya(), judge=FakeJudge())
    assert body["tipo_documento"] == "pedido" and [i["produto"] for i in body["itens"]][:2] == ["Alface", "Rúcula"]
    assert meta["juiz"] == {"motivo": "ocr_curto", "caracteres": len(ORDER)} and meta["motor_decisao"] == "laya"


def test_low_confidence_goes_to_the_judge_once():
    judge = FakeJudge(transcript="IDENTIFICAÇÃO E ASSINATURA DO RECEBEDOR")
    _, meta = run(canhoto(), ReadingLaya(confidence=0.4), judge=judge)
    assert meta["juiz"]["motivo"] == "baixa_confianca" and judge.calls == 1


def test_confident_decision_never_calls_the_judge():
    judge = FakeJudge()
    _, meta = run(canhoto(), ReadingLaya(), judge=judge)
    assert judge.calls == 0 and "juiz" not in meta


def test_judge_failure_keeps_the_first_answer():
    body, meta = run(Evidence("", "", BLANK, 0), ReadingLaya(), judge=FakeJudge(fail=True))
    assert body["tipo_documento"] == "imagem_invalida" and meta["tentativas"][-1]["motor"] == "juiz"


def test_hermes_client_asks_ollama_for_json():
    def handler(request):
        sent = json.loads(request.content)
        assert request.url.path == "/api/generate" and sent["format"] == "json" and sent["stream"] is False
        return httpx.Response(200, json={"response": '{"tipo_documento": "pedido", "confianca": 0.7}'})

    decision = HermesClient(SETTINGS, http(handler)).classify(ORDER, {})
    assert decision.doc_type == "pedido" and decision.confidence == 0.7


@pytest.mark.parametrize("answer", ['{"tipo_documento": "boleto"}', "não sei", '{"confianca": 1}'])
def test_hermes_client_rejects_unusable_answers(answer):
    def handler(request):
        return httpx.Response(200, json={"response": answer})

    with pytest.raises(HermesError):
        HermesClient(SETTINGS, http(handler)).classify(ORDER, {})


def test_vision_judge_sends_the_image_and_drops_thinking():
    def handler(request):
        content = json.loads(request.content)["messages"][0]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>\nAlface 10"}}]})

    assert VisionJudge(SETTINGS, http(handler)).transcribe(b"\x89PNG fake") == "Alface 10"


def test_vision_judge_empty_answer_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    with pytest.raises(JudgeError):
        VisionJudge(SETTINGS, http(handler)).transcribe(b"jpeg")


def test_decision_log_keeps_the_decision_but_no_document_content(tmp_path):
    envelope = Pipeline(SETTINGS, lambda data: canhoto(), FakeLaya()).run(b"x", {"id": "abc", "telefone": "5521999999999"})
    DecisionLog(str(tmp_path / "d.jsonl")).write(decision_record(envelope))
    line = (tmp_path / "d.jsonl").read_text(encoding="utf-8")
    record = json.loads(line)
    assert record["id"] == "abc" and record["tipo"] == "canhoto" and record["assinatura"]["possui"] is True
    assert "5521999999999" not in line and "36.895.856" not in line and "DANFE" not in line


def test_reference_keeps_only_type_and_signature_verdict():
    gemini = json.dumps({"tipo_documento": "canhoto", "assinatura": {"possui_assinatura": False},
                         "emitente": {"cnpj": "36.895.856/0001-10"}})
    assert reference_record("abc", "gemini", f"```json\n{gemini}\n```") == {
        "evento": "referencia", "id": "abc", "origem": "gemini", "tipo": "canhoto", "possui_assinatura": False}


def test_unparseable_reference_is_recorded_without_content():
    assert reference_record("abc", "gemini", "oops")["erro"] == "formato"


def test_endpoints_write_decisions_and_references(tmp_path):
    settings = config.load({**BASE_ENV, "DECISION_LOG_PATH": str(tmp_path / "d.jsonl")})
    client = TestClient(main.create_app(settings, Pipeline(settings, lambda data: canhoto(), FakeLaya())))
    auth = {"Authorization": "Bearer k"}
    reference = {"id": "abc", "output": '{"tipo_documento": "pedido"}'}
    assert client.post("/v1/referencia", json=reference).status_code == 401
    assert client.post("/v1/referencia", json=reference, headers=auth).json() == {"registrado": True}
    photo = {"imagem_base64": "eA==", "contexto": {"id": "abc"}}
    assert client.post("/v1/extrair-documento", json=photo, headers=auth).status_code == 200
    events = [json.loads(line)["evento"] for line in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["referencia", "decisao"]
