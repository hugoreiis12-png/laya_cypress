import base64
import json

import pytest
from fastapi.testclient import TestClient

from service.app import config, main
from service.app.laya_client import LayaError, TypeDecision
from service.app.pipeline import Pipeline
from service.app.vision.analyze import Evidence
from service.app.vision.image_io import ImageError
from service.app.vision.types import InkFeatures
from tests.test_service_extractors import ISSUER, make_key

SETTINGS = config.load({"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10"})
SIGNED = InkFeatures(True, 0.04, 0.01, 4.0, 2000, 3, 0.7, 0.3)
BLANK = InkFeatures(True, 0.0, 0.01, 0.0, 0, 0, 0.0, 0.0)
DANFE_TEXT = "DANFE NF-e RECEBEMOS DE NATURAL FARMS IDENTIFICAÇÃO E ASSINATURA DO RECEBEDOR"


class FakeLaya:
    def __init__(self, doc_type="canhoto", fail=False):
        self.doc_type, self.fail, self.calls = doc_type, fail, []

    def classify(self, ocr_text, extracted):
        self.calls.append(extracted)
        if self.fail:
            raise LayaError("fora do ar")
        return TypeDecision(self.doc_type, {self.doc_type: 0.9}, {"answer_confidence": 0.9}, 0.9, False)


def pipeline(evidence, laya):
    return Pipeline(SETTINGS, lambda data: evidence, laya)


def parse(envelope):
    return json.loads(envelope["output"]), envelope["_meta"]


def test_signed_canhoto_fills_every_field_the_flow_reads():
    key = make_key()
    evidence = Evidence(DANFE_TEXT, key, SIGNED, 0)
    body, meta = parse(pipeline(evidence, FakeLaya()).run(b"x", {"id": "abc"}))
    assert body["tipo_documento"] == "canhoto" and body["qrcode"]["chave"] == key
    assert body["emitente"]["cnpj"] == "36.895.856/0001-10" and body["nota_fiscal"]["numero"] == "123456"
    assert body["assinatura"]["possui_assinatura"] is True and body["assinatura"]["confianca"] >= 50
    assert meta["motor_decisao"] == "laya" and meta["id"] == "abc" and meta["assinatura"]["calibrado"] is False


def test_blank_receiver_region_is_not_signed():
    body, _ = parse(pipeline(Evidence(DANFE_TEXT, make_key(), BLANK, 0), FakeLaya()).run(b"x", {}))
    assert body["assinatura"]["possui_assinatura"] is False


def test_laya_failure_degrades_to_rules_and_is_reported():
    fake = FakeLaya(fail=True)
    body, meta = parse(pipeline(Evidence(DANFE_TEXT, make_key(), SIGNED, 0), fake).run(b"x", {}))
    assert body["tipo_documento"] == "canhoto"
    assert meta["motor_decisao"] == "regras" and meta["tentativas"][0]["motor"] == "laya"


def test_laya_receives_validated_signals_not_raw_guesses():
    fake = FakeLaya()
    pipeline(Evidence(DANFE_TEXT, make_key(), SIGNED, 0), fake).run(b"x", {})
    assert fake.calls == [{"chave_valida": True, "cnpj_na_whitelist": True}]


def test_unreadable_image_becomes_invalid_document_not_an_error():
    def broken(data):
        raise ImageError("imagem ilegível")

    body, meta = parse(Pipeline(SETTINGS, broken, FakeLaya()).run(b"junk", {}))
    assert body == {"tipo_documento": "imagem_invalida"} and meta["tentativas"][0]["motor"] == "visao"


def test_order_text_becomes_items_and_skips_signature():
    evidence = Evidence("Alface 10\nRúcula 5", "", BLANK, 0)
    body, meta = parse(pipeline(evidence, FakeLaya("pedido")).run(b"x", {}))
    assert [i["produto"] for i in body["itens"]] == ["Alface", "Rúcula"] and "assinatura" not in meta


def test_non_whitelisted_issuer_is_dropped():
    key = make_key(cnpj="22089653000393")
    body, _ = parse(pipeline(Evidence(DANFE_TEXT, key, SIGNED, 0), FakeLaya()).run(b"x", {}))
    assert body["emitente"]["cnpj"] == ""


@pytest.fixture
def client():
    app = main.create_app(SETTINGS, pipeline(Evidence(DANFE_TEXT, make_key(), SIGNED, 0), FakeLaya()))
    return TestClient(app)


def post(client, token="k", image=b"img"):
    body = {"imagem_base64": base64.b64encode(image).decode(), "contexto": {"id": "z"}}
    return client.post("/v1/extrair-documento", json=body, headers={"Authorization": f"Bearer {token}"})


def test_endpoint_returns_the_n8n_envelope(client):
    response = post(client)
    assert response.status_code == 200
    assert json.loads(response.json()["output"])["tipo_documento"] == "canhoto"
    assert response.json()["_meta"]["id"] == "z"


def test_endpoint_rejects_wrong_or_missing_credentials(client):
    assert post(client, token="wrong").status_code == 401
    assert client.post("/v1/extrair-documento", json={"imagem_base64": "AAAA"}).status_code == 401


def test_endpoint_rejects_malformed_base64_visibly(client):
    body = {"imagem_base64": "***not-base64***"}
    response = client.post("/v1/extrair-documento", json=body, headers={"Authorization": "Bearer k"})
    assert response.status_code == 422


def test_healthz_is_open_and_leaks_nothing(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 404
