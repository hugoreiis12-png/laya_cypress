import json
from dataclasses import replace

import pytest

from service.app import config
from service.app.judge_capacity import OVER_RUNNING
from service.app.pipeline import Pipeline
from service.app.signature_decision import SIGNED, UNCERTAIN
from service.app.vision.analyze import Evidence
from service.app.vision_judge import JudgeError, SignatureOpinion
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED as SIGNED_INK, FakeLaya

ENGINE = replace(SETTINGS, signature_engine=config.SIGNATURE_ENGINE_JUDGE, type_engine=config.TYPE_ENGINE_RULES,
                 judge_transcribe=False)
CONFIDENT = SignatureOpinion(True, 85.0)


class FakeDetector:
    def __init__(self, released=False, fail=False):
        self.released, self.fail, self.calls = released, fail, 0

    def observe(self, data, rotation):
        self.calls += 1
        return {"erro": "ConnectError"} if self.fail else {"score_canhoto": 0.1, "liberaria": self.released}


class FakeJudge:
    def __init__(self, opinion=CONFIDENT, fail=False):
        self.opinion, self.fail, self.calls = opinion, fail, 0

    def assess_signature(self, data, timeout):
        self.calls += 1
        if self.fail:
            raise JudgeError("fora do ar")
        return self.opinion


class FakeCapacity:
    def __init__(self, refusal=None):
        self.refusal = refusal

    def enter(self):
        return self.refusal

    def leave(self):
        pass


def run(judge=None, detector=None, capacity=None, key=None):
    evidence = Evidence(DANFE_TEXT, key or make_key(), SIGNED_INK, 0)
    engine = Pipeline(ENGINE, lambda data: evidence, FakeLaya(), shadow=detector or FakeDetector(),
                      judge=judge or FakeJudge(), judge_capacity=capacity)
    envelope = engine.run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"]


def test_confident_judge_releases_even_if_the_detector_disagrees():
    detector = FakeDetector(released=False)
    body, meta = run(detector=detector)
    assert body["assinatura"] == {"possui_assinatura": True, "confianca": 95, "descricao": "Assinatura manuscrita"}
    assert meta["estado_assinatura"] == SIGNED and meta["inconclusivo"] is False
    assert detector.calls == 1 and meta["sombra_assinatura"]["concorda"] is False


@pytest.mark.parametrize("opinion", [SignatureOpinion(False, 95.0), SignatureOpinion(None, 95.0), SignatureOpinion(True, 79.0)])
def test_anything_but_a_confident_yes_goes_to_the_reserve(opinion):
    body, meta = run(FakeJudge(opinion))
    assert body["assinatura"]["confianca"] == 0 and meta["estado_assinatura"] == UNCERTAIN and meta["inconclusivo"] is True


def test_judge_failure_goes_to_the_reserve():
    _, meta = run(FakeJudge(fail=True))
    assert meta["inconclusivo"] is True and meta["revisao_assinatura"]["resultado"] == "falha"


def test_no_capacity_goes_to_the_reserve_without_calling_the_judge():
    judge = FakeJudge()
    _, meta = run(judge, capacity=FakeCapacity(OVER_RUNNING))
    assert judge.calls == 0 and meta["inconclusivo"] is True and meta["capacidade_juiz"] == OVER_RUNNING


def test_detector_failure_never_blocks_the_release():
    _, meta = run(detector=FakeDetector(fail=True))
    assert meta["estado_assinatura"] == SIGNED and meta["inconclusivo"] is False and "erro" in meta["sombra_assinatura"]


def test_audit_problem_still_goes_to_the_reserve():
    _, meta = run(key=make_key(cnpj="22089653000393"))
    assert meta["estado_assinatura"] == SIGNED and meta["inconclusivo"] is True


def test_configuration_accepts_the_engine():
    env = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10", "SIGNATURE_ENGINE": "juiz"}
    assert config.load(env).signature_engine == config.SIGNATURE_ENGINE_JUDGE
