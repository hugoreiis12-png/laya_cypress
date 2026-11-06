import json
from dataclasses import replace

from service.app import config
from service.app.pipeline import Pipeline
from service.app.vision.analyze import Evidence
from service.app.vision_judge import SignatureOpinion
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED, FakeLaya

MAIN = replace(SETTINGS, signature_engine=config.SIGNATURE_ENGINE_TWO_VOTES, type_engine=config.TYPE_ENGINE_RULES,
               judge_transcribe=False)
KEY = make_key(number="000812839")


class FakeDetector:
    def __init__(self, released=True):
        self.released = released

    def observe(self, data, rotation):
        return {"score_canhoto": 0.9, "liberaria": self.released}


class FakeJudge:
    def assess_signature(self, data, timeout):
        return SignatureOpinion(True, 95.0)


def answer(detector=None):
    engine = Pipeline(MAIN, lambda data: Evidence(DANFE_TEXT, KEY, SIGNED, 0), FakeLaya(), shadow=detector or FakeDetector(),
                      judge=FakeJudge())
    return engine.run(b"x", {})


def test_released_canhoto_has_exactly_the_shape_the_flow_expects():
    assert json.loads(answer()["output"]) == {
        "tipo_documento": "canhoto",
        "assinatura": {"possui_assinatura": True, "confianca": 95, "descricao": "Assinatura manuscrita"},
        "emitente": {"cnpj": "36.895.856/0001-10"},
        "nota_fiscal": {"numero": "812839"},
        "qrcode": {"chave": KEY},
    }


def test_released_canhoto_clears_the_release_cutoff_and_stays_out_of_the_reserve():
    result = answer()
    assert json.loads(result["output"])["assinatura"]["confianca"] >= 90 and result["_meta"]["inconclusivo"] is False


def test_anything_short_of_two_votes_goes_to_the_reserve():
    assert answer(FakeDetector(released=False))["_meta"]["inconclusivo"] is True
