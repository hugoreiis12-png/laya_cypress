import json

from service.app.decision_log import decision_record
from service.app.laya_client import LayaError, TypeDecision
from service.app.pipeline import Pipeline
from service.app.signature_decision import INVALID_IMAGE, UNCERTAIN
from service.app.vision.analyze import Evidence
from service.app.vision_judge import JudgeError
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED

FISCAL_TEXT = "INSCRIÇÃO ESTADUAL SÉRIE 1 ENTRADA SAÍDA SUPERMERCADO EXEMPLO LTDA"


class ScriptedLaya:
    """Answers each call in order and keeps the text it read."""

    def __init__(self, *answers, fail=False):
        self.answers, self.fail, self.texts = list(answers), fail, []

    def classify(self, ocr_text, extracted):
        self.texts.append(ocr_text)
        if self.fail:
            raise LayaError("fora do ar")
        doc_type, confidence = self.answers.pop(0)
        return TypeDecision(doc_type, {doc_type: confidence}, {"answer_confidence": confidence}, confidence, False)


class FakeJudge:
    def __init__(self, transcript="SÉRIE 1\nEntrada 0", fail=False):
        self.transcript, self.fail, self.calls = transcript, fail, 0

    def transcribe(self, data):
        self.calls += 1
        if self.fail:
            raise JudgeError("fora do ar")
        return self.transcript


def run(text, laya, judge=None, key=""):
    evidence = Evidence(text, key, SIGNED, 0)
    envelope = Pipeline(SETTINGS, lambda data: evidence, laya, judge=judge).run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


def test_low_confidence_after_the_judge_is_not_accepted_as_an_order():
    body, meta, _ = run(FISCAL_TEXT, ScriptedLaya(("comprovante", 0.9), ("pedido", 0.58)), FakeJudge())
    assert body == {"tipo_documento": "imagem_invalida"}
    assert meta["juiz"]["motivo"] == "comprovante_sem_valor"
    assert meta["tipo_em_duvida"] == {"sugerido": "pedido", "confianca": 0.58}
    assert meta["estado_assinatura"] == UNCERTAIN


def test_confident_decision_after_the_judge_is_accepted():
    body, meta, _ = run(FISCAL_TEXT, ScriptedLaya(("comprovante", 0.9), ("pedido", 0.8)), FakeJudge())
    assert body["tipo_documento"] == "pedido" and "tipo_em_duvida" not in meta


def test_low_confidence_with_a_failed_judge_stays_in_doubt():
    body, meta, _ = run("Alface 10", ScriptedLaya(("pedido", 0.4)), FakeJudge(fail=True))
    assert body == {"tipo_documento": "imagem_invalida"} and meta["tipo_em_duvida"]["sugerido"] == "pedido"
    assert meta["tentativas"][-1]["motor"] == "juiz"


def test_low_confidence_invalid_image_is_not_a_doubt():
    _, meta, _ = run(FISCAL_TEXT, ScriptedLaya(("imagem_invalida", 0.3), ("imagem_invalida", 0.3)), FakeJudge())
    assert "tipo_em_duvida" not in meta and meta["estado_assinatura"] == INVALID_IMAGE


def test_deterministic_rules_have_no_confidence_to_doubt():
    body, meta, _ = run(DANFE_TEXT, ScriptedLaya(fail=True))
    assert body["tipo_documento"] == "canhoto" and "tipo_em_duvida" not in meta


def test_judge_reading_comes_first_for_laya_but_ocr_keeps_priority_in_the_fields():
    laya = ScriptedLaya(("canhoto", 0.4), ("canhoto", 0.9))
    body, _, _ = run("DANFE NF-e RECEBEMOS DE Nº 111", laya, FakeJudge("Nº 222"))
    assert laya.texts[1].startswith("Nº 222") and body["nota_fiscal"]["numero"] == "111"


def test_valid_key_vetoes_another_type_without_calling_the_judge():
    judge = FakeJudge()
    body, meta, _ = run(FISCAL_TEXT, ScriptedLaya(("pedido", 0.9)), judge, key=make_key())
    assert body["tipo_documento"] == "canhoto" and body["qrcode"]["chave"] == make_key()
    assert meta["veto_tipo"] == {"de": "pedido", "motivo": "chave_valida"} and judge.calls == 0
    assert "assinatura" in body


def test_key_read_by_the_judge_also_vetoes():
    body, meta, _ = run(FISCAL_TEXT, ScriptedLaya(("pedido", 0.4), ("pedido", 0.9)), FakeJudge(make_key()))
    assert body["tipo_documento"] == "canhoto" and meta["veto_tipo"]["de"] == "pedido"


def test_canhoto_with_a_valid_key_is_not_doubted_at_low_confidence():
    body, meta, _ = run(DANFE_TEXT, ScriptedLaya(("canhoto", 0.4), ("canhoto", 0.4)), FakeJudge(), key=make_key())
    assert body["tipo_documento"] == "canhoto" and "tipo_em_duvida" not in meta and "veto_tipo" not in meta


def test_doubt_and_veto_reach_the_decision_log_without_content():
    _, _, doubted = run(FISCAL_TEXT, ScriptedLaya(("comprovante", 0.9), ("pedido", 0.58)), FakeJudge())
    _, _, vetoed = run(FISCAL_TEXT, ScriptedLaya(("pedido", 0.9)), FakeJudge(), key=make_key())
    assert decision_record(doubted)["tipo_em_duvida"]["sugerido"] == "pedido"
    assert decision_record(vetoed)["veto_tipo"]["de"] == "pedido"
    assert "SUPERMERCADO" not in json.dumps(decision_record(doubted), ensure_ascii=False)
