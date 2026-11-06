import json

import pytest

from service.app.laya_client import state_for
from service.app.pipeline import Pipeline, laya_signals
from service.app.signature_model import SignatureDecision, SignatureModelError, parse
from service.app.vision.analyze import Evidence
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import BLANK, DANFE_TEXT, SETTINGS, SIGNED, FakeLaya

PACKAGE_ANSWER = {"answers": {"possui_assinatura": {"type": "noul", "noul": 0.97, "confidence": 0.97}},
                  "action": {"saida": "possui_assinatura", "acao": "negativa", "p": 0.03}}


class FakeModel:
    def __init__(self, action="negativa", p_signed=0.97, fail=False):
        self.decision, self.fail, self.states = SignatureDecision(action, p_signed), fail, []

    def decide(self, state):
        self.states.append(state)
        if self.fail:
            raise SignatureModelError("fora do ar")
        return self.decision


def run(evidence, model):
    envelope = Pipeline(SETTINGS, lambda data: evidence, FakeLaya(), model).run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"]


def canhoto(ink=SIGNED, text=DANFE_TEXT):
    return Evidence(text, make_key(), ink, 0)


def test_package_answer_is_parsed():
    assert parse(PACKAGE_ANSWER) == SignatureDecision("negativa", 0.97)


@pytest.mark.parametrize("body", [{}, {"action": {"acao": "talvez"}, "answers": {"possui_assinatura": {"noul": 0.5}}}])
def test_unusable_answer_raises(body):
    with pytest.raises(SignatureModelError):
        parse(body)


def test_model_signed_decision_reaches_the_contract():
    body, meta = run(canhoto(BLANK), FakeModel("negativa", 0.97))
    assert body["assinatura"]["possui_assinatura"] is True and body["assinatura"]["confianca"] == 97
    assert meta["assinatura"]["motor"] == "modelo" and meta["assinatura"]["acao"] == "negativa"


def test_model_unsigned_decision_reaches_the_contract():
    body, _ = run(canhoto(SIGNED), FakeModel("positiva", 0.1))
    assert body["assinatura"]["possui_assinatura"] is False and body["assinatura"]["confianca"] == 90


def test_review_is_never_released_as_signed():
    body, meta = run(canhoto(SIGNED), FakeModel("revisar", 0.6))
    assert body["assinatura"] == {"possui_assinatura": False, "confianca": 0, "descricao": ""}
    assert meta["assinatura"]["acao"] == "revisar"


def test_model_failure_falls_back_to_the_ink_rule():
    body, meta = run(canhoto(SIGNED), FakeModel(fail=True))
    assert body["assinatura"]["possui_assinatura"] is True and meta["assinatura"]["motor"] == "regras"
    assert meta["tentativas"][-1]["motor"] == "modelo_assinatura"


def test_empty_ocr_skips_the_model_like_training_did():
    model = FakeModel()
    run(canhoto(SIGNED, text=" "), model)
    assert model.states == []


def test_model_reads_the_training_state_shape():
    model = FakeModel()
    run(canhoto(SIGNED), model)
    document = model.states[0]["documento"]
    assert set(document) == {"texto_ocr", "campos_extraidos", "evidencia_tinta"}
    assert document["evidencia_tinta"]["density_relative"] == 4.0 and document["evidencia_tinta"]["region_found"] is True


def test_state_without_ink_is_unchanged_for_the_type_question():
    assert state_for("abc", laya_signals({"chave": "", "cnpj": ""})) == {
        "documento": {"texto_ocr": "abc", "campos_extraidos": {"chave_valida": False, "cnpj_na_whitelist": False}}}

def test_text_goes_last_so_truncation_never_drops_the_ink():
    model = FakeModel()
    run(canhoto(SIGNED), model)
    assert list(model.states[0]["documento"]) == ["evidencia_tinta", "campos_extraidos", "texto_ocr"]
