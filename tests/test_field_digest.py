import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from service.app import config
from service.app.decision_log import decision_record, reference_record
from service.app.field_digest import comparable
from service.app.main import create_app
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import SETTINGS

KEY = "chave-de-teste"
CHAVE = make_key()


def canhoto(chave=CHAVE, cnpj="36.895.856/0001-10", numero="123456"):
    return {"tipo_documento": "canhoto", "assinatura": {"possui_assinatura": True, "confianca": 95},
            "emitente": {"cnpj": cnpj}, "nota_fiscal": {"numero": numero}, "qrcode": {"chave": chave}}


def comprovante(valor="100.00", hora="10:30:00", favorecido="Fulana de Tal"):
    return {"tipo_documento": "comprovante", "comprovante": {
        "instituicao": "Banco", "valor": valor, "data": "2026-06-16", "hora": hora,
        "favorecido": favorecido, "pagador": "Beltrano", "identificacao": "ABC123"}}


def test_same_values_in_different_formats_match():
    first = comparable(canhoto(numero="000123456"), KEY)
    second = comparable(canhoto(cnpj="36895856000110", numero="123456"), KEY)
    assert first == second and all(first.values())


def test_different_key_does_not_match():
    first, second = comparable(canhoto(), KEY), comparable(canhoto(chave=make_key(number="000654321")), KEY)
    assert first["chave"] != second["chave"] and first["cnpj"] == second["cnpj"]


@pytest.mark.parametrize("value", ["", None])
def test_empty_and_null_are_both_absent(value):
    assert comparable(canhoto(chave=value), KEY)["chave"] is None


@pytest.mark.parametrize("chave,expected", [(CHAVE, True), (CHAVE + "123", False), (CHAVE[:-1] + str((int(CHAVE[-1]) + 1) % 10), False), ("", None)])
def test_key_validity_is_recorded_as_a_flag(chave, expected):
    assert comparable(canhoto(chave=chave), KEY)["chave_valida"] is expected


def test_receipt_amount_and_time_formats_match():
    assert comparable(comprovante("100,0", "10:30"), KEY) == comparable(comprovante("100.00", "10:30:45"), KEY)


def test_unparseable_amount_does_not_raise():
    assert comparable(comprovante("cem reais"), KEY)["valor"]


def test_digest_depends_on_the_secret():
    assert comparable(canhoto(), KEY)["chave"] != comparable(canhoto(), "outra")["chave"]


def test_names_and_values_never_appear():
    saved = json.dumps([comparable(canhoto(), KEY), comparable(comprovante(), KEY)], ensure_ascii=False)
    for value in (CHAVE, "36895856000110", "123456", "100.00", "2026-06-16", "ABC123", "Fulana", "Beltrano", "Banco"):
        assert value not in saved


def test_orders_are_plain_counts_only():
    body = {"tipo_documento": "pedido", "itens": [{"produto": "Alface", "qnt": 2}, {"produto": "Rúcula", "qnt": None},
                                                  {"produto": "Couve", "qnt": True}, "lixo"]}
    assert comparable(body, KEY) == {"itens": 3, "soma_qnt": 2}


@pytest.mark.parametrize("body", [{}, {"tipo_documento": "imagem_invalida"}, {"tipo_documento": "canhoto", "qrcode": "x"},
                                  {"tipo_documento": "pedido", "itens": "x"}, {"tipo_documento": "comprovante", "comprovante": []}])
def test_malformed_answers_do_not_raise(body):
    comparable(body, KEY)


def test_without_a_secret_the_records_stay_as_they_were():
    envelope = {"output": json.dumps(canhoto()), "_meta": {}}
    assert "campos" not in decision_record(envelope)
    assert "campos" not in reference_record("a", "gemini", canhoto())


def test_the_service_logs_digests_both_engines_can_be_joined_on(tmp_path):
    path = tmp_path / "d.jsonl"
    settings = replace(SETTINGS, decision_log_path=str(path), decision_digest_key=KEY)

    class FakePipeline:
        def run(self, data, context):
            return {"output": json.dumps(canhoto()), "_meta": {"id": "abc"}}

    client = TestClient(create_app(settings, FakePipeline()))
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    assert client.post("/v1/extrair-documento", json={"imagem_base64": "eA=="}, headers=headers).status_code == 200
    gemini = json.dumps(canhoto(numero="000123456"))
    assert client.post("/v1/referencia", json={"id": "abc", "output": gemini}, headers=headers).status_code == 200
    decision, reference = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert decision["campos"] == reference["campos"] and decision["campos"]["chave"]
    assert CHAVE not in path.read_text(encoding="utf-8")


def test_config_reads_the_secret_and_defaults_to_off():
    env = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10"}
    assert config.load(env).decision_digest_key == ""
    assert config.load({**env, "DECISION_DIGEST_KEY": "x"}).decision_digest_key == "x"
