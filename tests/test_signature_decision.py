import json
from dataclasses import replace

import pytest

from service.app.decision_log import decision_record
from service.app.pipeline import Pipeline
from service.app.signature_decision import INVALID_IMAGE, SIGNED, UNCERTAIN, UNSIGNED, from_ink, from_model
from service.app.signature_model import SignatureDecision
from service.app.vision.analyze import Evidence
from service.app.vision.image_io import ImageError
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import BLANK, DANFE_TEXT, SETTINGS, SIGNED as SIGNED_INK, FakeLaya

REGION_NOT_FOUND = replace(BLANK, region_found=False)
UNSIGNED_CONTRACT = {"possui_assinatura": False, "confianca": 0, "descricao": ""}


def run(ink=SIGNED_INK, doc_type="canhoto", model=None):
    evidence = Evidence(DANFE_TEXT, make_key() if doc_type == "canhoto" else "", ink, 0)
    envelope = Pipeline(SETTINGS, lambda data: evidence, FakeLaya(doc_type), model).run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


def test_ink_in_the_receiver_field_is_signed():
    assert from_ink(SIGNED_INK, SETTINGS).state == SIGNED


def test_blank_located_field_is_a_confirmed_absence():
    assert from_ink(BLANK, SETTINGS).state == UNSIGNED


def test_field_not_found_is_uncertain_not_unsigned():
    verdict = from_ink(REGION_NOT_FOUND, SETTINGS)
    assert verdict.state == UNCERTAIN and verdict.confidence == 0 and not verdict.signed


@pytest.mark.parametrize("action,state,confidence", [
    ("negativa", SIGNED, 90), ("positiva", UNSIGNED, 10), ("revisar", UNCERTAIN, 0)])
def test_model_action_maps_to_state(action, state, confidence):
    verdict = from_model(SignatureDecision(action, 0.9))
    assert verdict.state == state and verdict.confidence == pytest.approx(confidence)


def test_uncertain_keeps_the_contract_the_n8n_flow_reads():
    body, meta, _ = run(REGION_NOT_FOUND)
    assert body["assinatura"] == UNSIGNED_CONTRACT and meta["estado_assinatura"] == UNCERTAIN


def test_contract_exposes_no_new_field_to_the_flow():
    body, _, _ = run()
    assert set(body["assinatura"]) == {"possui_assinatura", "confianca", "descricao"}


def test_unreadable_image_reports_invalid_state():
    def broken(data):
        raise ImageError("imagem ilegível")

    meta = Pipeline(SETTINGS, broken, FakeLaya()).run(b"junk", {})["_meta"]
    assert meta["estado_assinatura"] == INVALID_IMAGE


def test_document_classified_invalid_reports_invalid_state():
    _, meta, _ = run(doc_type="imagem_invalida")
    assert meta["estado_assinatura"] == INVALID_IMAGE


def test_order_has_no_signature_state():
    _, meta, _ = run(doc_type="pedido")
    assert "estado_assinatura" not in meta


def test_uncertain_case_can_be_queued_from_the_decision_log():
    _, _, envelope = run(REGION_NOT_FOUND)
    assert decision_record(envelope)["estado_assinatura"] == UNCERTAIN
