import json
from dataclasses import replace

import httpx
import pytest

from service.app.config import load
from service.app.decision_log import decision_record
from service.app.pipeline import Pipeline
from service.app.signature_decision import SIGNED, UNCERTAIN, UNSIGNED
from service.app.vision.analyze import Evidence
from service.app.vision_judge import JudgeError, SignatureOpinion, VisionJudge, parse_signature
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import BLANK, DANFE_TEXT, SETTINGS, SIGNED as SIGNED_INK, FakeLaya

REGION_NOT_FOUND = replace(BLANK, region_found=False)
ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10", "JUDGE_URL": "http://judge", "JUDGE_MODEL": "vlm"}
UNSIGNED_CONTRACT = {"possui_assinatura": False, "confianca": 0, "descricao": ""}


class FakeJudge:
    def __init__(self, opinion=None, fail=False):
        self.opinion, self.fail, self.timeouts = opinion, fail, []

    def transcribe(self, data):
        raise AssertionError("a revisão de assinatura não transcreve")

    def assess_signature(self, data, timeout):
        self.timeouts.append(timeout)
        if self.fail:
            raise JudgeError("fora do ar")
        return self.opinion


def run(judge, ink=REGION_NOT_FOUND, settings=SETTINGS):
    evidence = Evidence(DANFE_TEXT, make_key(), ink, 0)
    envelope = Pipeline(settings, lambda data: evidence, FakeLaya(), judge=judge).run(b"x", {})
    return json.loads(envelope["output"])["assinatura"], envelope["_meta"], envelope


def client(content):
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}))
    return VisionJudge(load(ENV), httpx.Client(transport=transport))


def test_uncertain_case_goes_to_the_judge_once():
    judge = FakeJudge(SignatureOpinion(False, 80.0))
    signature, meta, _ = run(judge)
    assert len(judge.timeouts) == 1 and meta["estado_assinatura"] == UNSIGNED
    assert signature["possui_assinatura"] is False and meta["revisao_assinatura"]["resultado"] == UNSIGNED


def test_judge_cannot_clear_the_n8n_release_cutoff_until_measured():
    signature, meta, _ = run(FakeJudge(SignatureOpinion(True, 100.0)))
    assert meta["estado_assinatura"] == SIGNED and signature["possui_assinatura"] is True
    assert signature["confianca"] == 89


def test_cap_is_raised_only_by_configuration():
    settings = load({**ENV, "JUDGE_SIGNATURE_MAX_CONFIDENCE": "95"})
    signature, _, _ = run(FakeJudge(SignatureOpinion(True, 100.0)), settings=settings)
    assert signature["confianca"] == 95


def test_abstaining_judge_stays_uncertain():
    signature, meta, _ = run(FakeJudge(SignatureOpinion(None, 50.0)))
    assert signature == UNSIGNED_CONTRACT and meta["estado_assinatura"] == UNCERTAIN


def test_judge_failure_stays_uncertain_and_is_reported():
    signature, meta, _ = run(FakeJudge(fail=True))
    assert signature == UNSIGNED_CONTRACT and meta["estado_assinatura"] == UNCERTAIN
    assert meta["revisao_assinatura"]["resultado"] == "falha" and meta["tentativas"][-1]["motor"] == "juiz_assinatura"


def test_spent_deadline_skips_the_judge():
    judge = FakeJudge(SignatureOpinion(True, 100.0))
    _, meta, _ = run(judge, settings=replace(SETTINGS, review_deadline=0.0))
    assert judge.timeouts == [] and meta["revisao_assinatura"]["resultado"] == "sem_prazo"


def test_judge_receives_only_the_time_left():
    judge = FakeJudge(SignatureOpinion(False, 80.0))
    run(judge, settings=replace(SETTINGS, review_deadline=10.0))
    assert 0 < judge.timeouts[0] <= 10.0


@pytest.mark.parametrize("ink", [SIGNED_INK, BLANK])
def test_confirmed_states_never_call_the_judge(ink):
    judge = FakeJudge(SignatureOpinion(True, 100.0))
    _, meta, _ = run(judge, ink=ink)
    assert judge.timeouts == [] and "revisao_assinatura" not in meta


def test_no_judge_keeps_the_case_uncertain():
    _, meta, _ = run(None)
    assert meta["estado_assinatura"] == UNCERTAIN and "revisao_assinatura" not in meta


def test_unresolved_case_is_visible_in_the_decision_log():
    _, _, envelope = run(FakeJudge(fail=True))
    record = decision_record(envelope)
    assert record["estado_assinatura"] == UNCERTAIN and record["revisao_assinatura"]["resultado"] == "falha"


def test_decision_log_keeps_why_the_judge_failed():
    _, _, envelope = run(FakeJudge(fail=True))
    record = decision_record(envelope)
    assert record["falhas"] == ["juiz_assinatura"]
    assert record["erros"] == [{"motor": "juiz_assinatura", "erro": "fora do ar"}]


def test_decision_log_cuts_long_failure_reasons():
    envelope = {"output": json.dumps({"tipo_documento": "canhoto"}), "_meta": {"tentativas": [{"motor": "juiz", "erro": "x" * 500}]}}
    assert len(decision_record(envelope)["erros"][0]["erro"]) == 200


def test_judge_prompt_does_not_count_a_stamp_alone_as_signed():
    sent = []

    def answer(request):
        sent.append(json.loads(request.content)["messages"][0]["content"][0]["text"])
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"possui_assinatura": false, "confianca": 90}'}}]})

    VisionJudge(load(ENV), httpx.Client(transport=httpx.MockTransport(answer))).assess_signature(b"jpeg", 5.0)
    assert "Carimbo sozinho" in sent[0] and "NÃO conta" in sent[0] and "carimbo preenchido" not in sent[0]


def test_client_reads_the_opinion_and_drops_thinking():
    judge = client('<think>hmm</think>{"possui_assinatura": false, "confianca": 70}')
    assert judge.assess_signature(b"jpeg", 5.0) == SignatureOpinion(False, 70.0)


def test_client_limits_the_timeout_to_the_time_left():
    seen = {}

    def handler(request):
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"possui_assinatura": null, "confianca": 0}'}}]})

    VisionJudge(load(ENV), httpx.Client(transport=httpx.MockTransport(handler))).assess_signature(b"jpeg", 3.0)
    assert seen["read"] == 3.0


def test_confidence_is_clamped():
    assert parse_signature('{"possui_assinatura": true, "confianca": 140}').confidence == 100.0


@pytest.mark.parametrize("answer", [
    "não sei", "{}", '{"possui_assinatura": true}', '{"possui_assinatura": "sim", "confianca": 90}',
    '{"possui_assinatura": true, "confianca": "alta"}', '{"possui_assinatura": true, "confianca": true}',
    '{"possui_assinatura": true, "confianca": NaN}', "[1]"])
def test_unusable_opinions_are_errors(answer):
    with pytest.raises(JudgeError):
        parse_signature(answer)
