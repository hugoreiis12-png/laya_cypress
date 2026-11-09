from service.app.decision_log import decision_record
from service.app.pipeline import Pipeline
from service.app.vision.image_io import ImageError
from service.app.vision_judge import SignatureOpinion
from tests.test_key_reading import BAD_CHECK_DIGIT, ENGINE, KEY, FakeJudge, run
from tests.test_prototype_decision import make_key
from tests.test_service_pipeline import FakeLaya


def test_a_case_the_service_answers_alone_carries_no_reasons():
    _, meta, _ = run(FakeJudge())
    assert meta["inconclusivo"] is False and "motivos_inconclusivo" not in meta


def test_a_type_without_its_own_answer_says_so():
    _, meta, _ = run(FakeJudge(BAD_CHECK_DIGIT))
    assert meta["inconclusivo"] is True and meta["motivos_inconclusivo"] == ["tipo_sem_resposta_propria"]


class AbstainingJudge(FakeJudge):
    def assess_signature(self, data, timeout):
        return SignatureOpinion(None, 0.0)


def test_an_unconfirmed_signature_is_a_reason_of_its_own():
    _, meta, _ = run(AbstainingJudge(), barcode_key=KEY)
    assert meta["inconclusivo"] is True and meta["motivos_inconclusivo"] == ["assinatura_nao_confirmada"]


def test_an_audit_problem_is_a_reason_of_its_own():
    _, meta, _ = run(FakeJudge(), barcode_key=make_key(cnpj="22089653000393"))
    assert "auditoria" in meta["motivos_inconclusivo"]


def test_an_unreadable_image_is_a_reason_of_its_own():
    def broken(data):
        raise ImageError("imagem ilegível")

    meta = Pipeline(ENGINE, broken, FakeLaya()).run(b"x", {})["_meta"]
    assert meta["motivos_inconclusivo"] == ["imagem_ilegivel"]


def test_the_decision_log_keeps_the_reasons():
    _, meta, envelope = run(FakeJudge(BAD_CHECK_DIGIT))
    assert decision_record(envelope)["motivos_inconclusivo"] == meta["motivos_inconclusivo"]
