import json
import threading
from dataclasses import replace

import httpx
import pytest

from service.app import audit, config
from service.app.decision_log import decision_record
from service.app.hermes_client import HermesClient, HermesError, parse_audit
from service.app.judge_capacity import BUSY_HERE, METRICS_DOWN, OVER_CACHE, OVER_RUNNING, OVER_WAITING, JudgeCapacity, read_metrics
from service.app.pipeline import Pipeline
from service.app.signature_decision import SIGNED, TWO_VOTES_CONFIDENCE, UNCERTAIN, from_votes
from service.app.vision.analyze import Evidence
from service.app.vision.image_io import ImageError
from service.app.vision_judge import JudgeError, SignatureOpinion
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED as SIGNED_INK, FakeLaya

ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10", "JUDGE_URL": "http://judge", "JUDGE_MODEL": "vlm"}
PROTOTYPE = replace(SETTINGS, signature_engine=config.SIGNATURE_ENGINE_TWO_VOTES, type_engine=config.TYPE_ENGINE_RULES)
SIGNED_OPINION = SignatureOpinion(True, 95.0)
IDLE = 'vllm:num_requests_running{model_name="m"} 0.0\nvllm:num_requests_waiting{model_name="m"} 0.0\n'


class FakeDetector:
    def __init__(self, released=True, fail=False):
        self.released, self.fail, self.calls = released, fail, 0

    def observe(self, data, rotation):
        self.calls += 1
        return {"erro": "ConnectError"} if self.fail else {"score_canhoto": 0.9, "liberaria": self.released}


class FakeJudge:
    def __init__(self, opinion=SIGNED_OPINION, fail=False):
        self.opinion, self.fail, self.calls = opinion, fail, 0

    def transcribe(self, data):
        raise AssertionError("o voto de assinatura não transcreve")

    def assess_signature(self, data, timeout):
        self.calls += 1
        if self.fail:
            raise JudgeError("fora do ar")
        return self.opinion


class FakeCapacity:
    def __init__(self, refusal=None):
        self.refusal, self.entered, self.left = refusal, 0, 0

    def enter(self):
        self.entered += 1
        return self.refusal

    def leave(self):
        self.left += 1


class FakeAuditor:
    def __init__(self, fail=False):
        self.fail, self.answers = fail, []

    def audit(self, answer, ocr_text, timeout):
        self.answers.append(answer)
        if self.fail:
            raise HermesError("fora do ar")
        return {"consistente": False, "motivo": "campo_contraditorio"}


def run(detector=None, judge=None, capacity=None, auditor=None, settings=PROTOTYPE, key=None):
    detector = detector or FakeDetector()
    judge = judge if judge is not None else FakeJudge()
    evidence = Evidence(DANFE_TEXT, make_key() if key is None else key, SIGNED_INK, 0)
    engine = Pipeline(settings, lambda data: evidence, FakeLaya(), shadow=detector, judge=judge,
                      judge_capacity=capacity, auditor=auditor)
    envelope = engine.run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


def capacity_for(metrics, settings=None):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=metrics))
    return JudgeCapacity(settings or config.load(ENV), httpx.Client(transport=transport))


def test_two_votes_release_the_canhoto_above_the_cutoff():
    body, meta, _ = run()
    assert body["assinatura"]["possui_assinatura"] is True and body["assinatura"]["confianca"] == TWO_VOTES_CONFIDENCE
    assert meta["estado_assinatura"] == SIGNED and meta["inconclusivo"] is False
    assert meta["votos"] == {"detector": True, "juiz": True, "confianca_juiz": 95.0}


@pytest.mark.parametrize("detector,opinion", [
    (True, SignatureOpinion(False, 95.0)), (True, SignatureOpinion(None, 95.0)), (True, SignatureOpinion(True, 79.0)),
    (True, None), (False, SIGNED_OPINION), (None, SIGNED_OPINION), (None, None)])
def test_anything_short_of_two_confident_votes_stays_undecided(detector, opinion):
    verdict = from_votes(detector, opinion, 80.0)
    assert verdict.state == UNCERTAIN and verdict.confidence == 0


def test_judge_is_not_asked_when_the_detector_sees_no_signature():
    judge = FakeJudge()
    body, meta, _ = run(FakeDetector(released=False), judge)
    assert judge.calls == 0 and body["assinatura"]["possui_assinatura"] is False
    assert meta["estado_assinatura"] == UNCERTAIN and meta["inconclusivo"] is True


def test_detector_failure_goes_to_the_reserve_without_the_judge():
    judge = FakeJudge()
    body, meta, _ = run(FakeDetector(fail=True), judge)
    assert judge.calls == 0 and body["assinatura"]["confianca"] == 0 and meta["inconclusivo"] is True


def test_judge_failure_goes_to_the_reserve():
    body, meta, _ = run(judge=FakeJudge(fail=True))
    assert body["assinatura"]["confianca"] == 0 and meta["inconclusivo"] is True
    assert meta["revisao_assinatura"]["resultado"] == "falha"


def test_no_capacity_skips_the_judge_and_goes_to_the_reserve():
    judge, capacity = FakeJudge(), FakeCapacity(refusal=OVER_RUNNING)
    _, meta, _ = run(judge=judge, capacity=capacity)
    assert judge.calls == 0 and capacity.left == 0 and meta["inconclusivo"] is True
    assert meta["capacidade_juiz"] == OVER_RUNNING and meta["revisao_assinatura"]["resultado"] == "sem_capacidade"


@pytest.mark.parametrize("fail", [False, True])
def test_capacity_slot_is_always_returned(fail):
    capacity = FakeCapacity()
    run(judge=FakeJudge(fail=fail), capacity=capacity)
    assert capacity.entered == 1 and capacity.left == 1


def test_spent_deadline_skips_the_judge():
    judge = FakeJudge()
    _, meta, _ = run(judge=judge, settings=replace(PROTOTYPE, review_deadline=0.0))
    assert judge.calls == 0 and meta["revisao_assinatura"]["resultado"] == "sem_prazo"


def test_detector_is_called_once():
    detector = FakeDetector()
    run(detector)
    assert detector.calls == 1


def test_read_metrics_sums_every_model():
    text = '# HELP x\nvllm:num_requests_running{model_name="a"} 1.0\nvllm:num_requests_running{model_name="b"} 2.0\n'
    assert read_metrics(text) == {"vllm:num_requests_running": 3.0}


@pytest.mark.parametrize("metrics,expected", [
    (IDLE, None),
    ('vllm:num_requests_running{model_name="m"} 2.0\nvllm:num_requests_waiting{model_name="m"} 0.0\n', OVER_RUNNING),
    ('vllm:num_requests_running{model_name="m"} 0.0\nvllm:num_requests_waiting{model_name="m"} 1.0\n', OVER_WAITING),
    (IDLE + 'vllm:kv_cache_usage_perc{model_name="m"} 0.9\n', OVER_CACHE),
    (IDLE + 'vllm:gpu_cache_usage_perc{model_name="m"} 90.0\n', OVER_CACHE),
    (IDLE + 'vllm:kv_cache_usage_perc{model_name="m"} 0.5\n', None),
    ("sem metricas", METRICS_DOWN)])
def test_capacity_follows_the_vllm_state(metrics, expected):
    assert capacity_for(metrics).enter() == expected


def test_unreachable_metrics_count_as_no_room():
    def broken(request):
        raise httpx.ConnectError("fora")

    capacity = JudgeCapacity(config.load(ENV), httpx.Client(transport=httpx.MockTransport(broken)))
    assert capacity.enter() == METRICS_DOWN


def test_only_one_judge_call_at_a_time_by_default():
    capacity = capacity_for(IDLE, config.load({**ENV, "JUDGE_WAIT_SECONDS": "0"}))
    assert capacity.enter() is None and capacity.enter() == BUSY_HERE
    capacity.leave()
    assert capacity.enter() is None


def test_a_case_waits_for_the_slot_instead_of_going_to_the_reserve():
    capacity = capacity_for(IDLE, config.load({**ENV, "JUDGE_WAIT_SECONDS": "2"}))
    assert capacity.enter() is None
    threading.Timer(0.2, capacity.leave).start()
    assert capacity.enter() is None


def test_the_wait_for_the_slot_is_bounded():
    capacity = capacity_for(IDLE, config.load({**ENV, "JUDGE_WAIT_SECONDS": "0.1"}))
    assert capacity.enter() is None and capacity.enter() == BUSY_HERE


def test_refused_call_gives_its_slot_back():
    capacity = capacity_for("sem metricas")
    assert capacity.enter() == METRICS_DOWN and capacity.enter() == METRICS_DOWN


def test_key_with_another_issuer_is_a_problem():
    fields = {"chave": make_key(cnpj="22089653000393"), "cnpj": "36.895.856/0001-10", "numero": "123"}
    assert audit.problems("canhoto", fields) == ["cnpj_diverge_da_chave"]


def test_canhoto_missing_fields_are_problems():
    assert audit.problems("canhoto", {"chave": "", "cnpj": "", "numero": ""}) == ["sem_cnpj", "sem_numero"]


@pytest.mark.parametrize("found,expected", [
    ([], ["sem_itens"]),
    ([{"produto": "F", "qnt": 1}], ["itens_implausiveis"]),
    ([{"produto": "Alface", "qnt": 283397}], ["itens_implausiveis"]),
    ([{"produto": "Alface", "qnt": None}], []),
    ([{"produto": "Alface", "qnt": 10}], [])])
def test_order_items_are_checked(found, expected):
    assert audit.problems("pedido", {"itens": found}) == expected


def test_audit_problem_sends_a_released_canhoto_to_the_reserve():
    _, meta, _ = run(key=make_key(cnpj="22089653000393"))
    assert meta["estado_assinatura"] == SIGNED and meta["inconclusivo"] is True
    assert "sem_cnpj" in meta["auditoria"]["problemas"]


def test_other_types_go_to_the_reserve_until_measured():
    engine = Pipeline(PROTOTYPE, lambda data: Evidence("Alface 10\nRúcula 5", "", SIGNED_INK, 0), FakeLaya())
    assert engine.run(b"x", {})["_meta"]["inconclusivo"] is True


def test_conclusive_types_come_from_configuration():
    settings = config.load({**ENV, "CONCLUSIVE_TYPES": "canhoto, pedido"})
    assert settings.conclusive_types == ("canhoto", "pedido")


def test_unreadable_image_is_inconclusive():
    def broken(data):
        raise ImageError("imagem ilegível")

    assert Pipeline(PROTOTYPE, broken, FakeLaya()).run(b"x", {})["_meta"]["inconclusivo"] is True


def test_hermes_opinion_is_recorded_but_does_not_change_the_answer():
    auditor = FakeAuditor()
    _, meta, _ = run(auditor=auditor)
    assert meta["auditoria"]["hermes"]["consistente"] is False and meta["inconclusivo"] is False
    assert auditor.answers[0]["tipo_documento"] == "canhoto"


def test_hermes_failure_never_blocks():
    _, meta, _ = run(auditor=FakeAuditor(fail=True))
    assert meta["auditoria"]["hermes"] == {"erro": "indisponivel"} and meta["inconclusivo"] is False


def test_hermes_client_audits_with_the_time_left():
    seen = {}

    def handler(request):
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, json={"response": '{"consistente": true, "motivo": "ok"}'})

    client = HermesClient(config.load({**ENV, "HERMES_URL": "http://hermes"}), httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.audit({"tipo_documento": "canhoto"}, "texto", 3.0) == {"consistente": True, "motivo": "ok"}
    assert seen["read"] == 3.0


def test_unknown_audit_reason_never_reaches_the_record():
    assert parse_audit('{"consistente": false, "motivo": "CPF 123"}')["motivo"] == "outro"


@pytest.mark.parametrize("raw", ["não sei", "{}", '{"consistente": "sim"}'])
def test_unusable_audit_is_an_error(raw):
    with pytest.raises(HermesError):
        parse_audit(raw)


def test_decision_log_keeps_votes_without_content():
    _, _, envelope = run(capacity=FakeCapacity())
    record = decision_record(envelope)
    assert record["votos"]["detector"] is True and record["inconclusivo"] is False and record["capacidade_juiz"] == "ok"
    assert "36.895.856" not in json.dumps(record) and "DANFE" not in json.dumps(record)
