import json
from dataclasses import replace

from service.app import audit, config
from service.app.decision_log import decision_record
from service.app.pipeline import Pipeline
from service.app.vision import analyze as analyze_module
from service.app.vision.analyze import Evidence
from service.app.vision.types import OcrResult
from service.app.vision_judge import SignatureOpinion
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED, FakeLaya

PROTOTYPE = replace(SETTINGS, signature_engine=config.SIGNATURE_ENGINE_TWO_VOTES, type_engine=config.TYPE_ENGINE_RULES,
                    judge_transcribe=False)
FIRST, SECOND = make_key(), make_key(number="000654321")


class FakeDetector:
    def observe(self, data, rotation):
        return {"score_canhoto": 0.9, "liberaria": True}


class FakeJudge:
    def assess_signature(self, data, timeout):
        return SignatureOpinion(True, 95.0)


def run(evidence):
    engine = Pipeline(PROTOTYPE, lambda data: evidence, FakeLaya(), shadow=FakeDetector(), judge=FakeJudge())
    envelope = engine.run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


def test_two_keys_in_the_barcode_go_to_the_reserve_even_with_two_votes():
    body, meta, _ = run(Evidence(DANFE_TEXT, FIRST, SIGNED, 0, (SECOND,)))
    assert body["assinatura"]["possui_assinatura"] is True and meta["inconclusivo"] is True
    assert "mais_de_uma_chave" in meta["auditoria"]["problemas"]


def test_a_second_key_in_the_ocr_text_also_goes_to_the_reserve():
    _, meta, _ = run(Evidence(f"{DANFE_TEXT}\n{SECOND}", FIRST, SIGNED, 0))
    assert "mais_de_uma_chave" in meta["auditoria"]["problemas"] and meta["leitura"]["chaves"] == 2


def test_the_same_key_read_twice_is_one_document():
    _, meta, _ = run(Evidence(f"{DANFE_TEXT}\n{FIRST}", FIRST, SIGNED, 0))
    assert meta["auditoria"]["problemas"] == [] and meta["inconclusivo"] is False and meta["leitura"]["chaves"] == 1


def test_reading_tells_unread_from_read_without_a_rule():
    _, unread, _ = run(Evidence("", "", SIGNED, 0))
    _, read, _ = run(Evidence("texto que nenhuma regra reconhece", "", SIGNED, 0))
    assert unread["leitura"] == {"caracteres_ocr": 0, "chave_no_codigo": False, "chaves": 0}
    assert read["leitura"]["caracteres_ocr"] == 33


def test_reading_reaches_the_log_without_content():
    _, _, envelope = run(Evidence(DANFE_TEXT, FIRST, SIGNED, 0))
    record = decision_record(envelope)
    assert record["leitura"]["chave_no_codigo"] is True and FIRST not in json.dumps(record["leitura"])


def test_audit_accepts_fields_without_the_key_list():
    assert audit.reading("abc", "", {})["chaves"] == 0


def test_analyze_keeps_every_key_of_the_photo(monkeypatch):
    image = object()
    monkeypatch.setattr(analyze_module, "decode", lambda data: image)
    monkeypatch.setattr(analyze_module, "to_gray", lambda picture: picture)
    monkeypatch.setattr(analyze_module.ocr, "run", lambda picture: (OcrResult("", (), 0), _Upright()))
    monkeypatch.setattr(analyze_module.barcode, "read_codes", lambda picture: [FIRST, SECOND])
    monkeypatch.setattr(analyze_module.region, "locate", lambda words, width, height: None)
    monkeypatch.setattr(analyze_module.ink, "measure", lambda gray, words, box: SIGNED)
    evidence = analyze_module.analyze(b"x")
    assert evidence.access_key == FIRST and evidence.other_keys == (SECOND,)


class _Upright:
    shape = (100, 100, 3)
