import json

import pytest

from service.app import config, contract

ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10, 22.089.653/0003-93"}


def test_envelope_output_is_a_json_string_n8n_can_parse():
    body = contract.envelope(contract.build("imagem_invalida"), {"motor": "x"})
    assert isinstance(body["output"], str)
    assert json.loads(body["output"]) == {"tipo_documento": "imagem_invalida"}
    assert body["_meta"] == {"motor": "x"}


def test_canhoto_has_every_field_the_flow_reads():
    sig = contract.signature(True, 97.4, "rubrica")
    built = contract.build("canhoto", sig, {"cnpj": "36.895.856/0001-10", "numero": "123", "chave": "1" * 44})
    assert built["assinatura"] == {"possui_assinatura": True, "confianca": 97, "descricao": "Rubrica"}
    assert built["emitente"]["cnpj"] and built["nota_fiscal"]["numero"] and built["qrcode"]["chave"]


def test_unsigned_signature_has_no_description():
    assert contract.signature(False, 20, "manuscrita")["descricao"] == ""


def test_confidence_is_clamped_to_0_100():
    assert contract.signature(True, 140, "rubrica")["confianca"] == 100
    assert contract.signature(True, -5, "rubrica")["confianca"] == 0


def test_comprovante_always_returns_the_seven_keys():
    built = contract.build("comprovante", fields={"comprovante": {"valor": "10.00"}})
    assert len(built["comprovante"]) == 7 and built["comprovante"]["pagador"] is None


def test_unknown_type_degrades_to_invalid_image():
    assert contract.build("outro") == {"tipo_documento": "imagem_invalida"}


def test_config_loads_and_normalizes_whitelist():
    settings = config.load(ENV)
    assert settings.whitelist_cnpj == ("36895856000110", "22089653000393")


@pytest.mark.parametrize("env,expected", [
    ({"WHITELIST_CNPJ": "36.895.856/0001-10"}, "API_KEY"),
    ({"API_KEY": "k"}, "WHITELIST_CNPJ"),
    ({"API_KEY": "k", "WHITELIST_CNPJ": "04.972.092/00031-48"}, "inválido"),
])
def test_config_fails_closed(env, expected):
    with pytest.raises(config.ConfigError, match=expected):
        config.load(env)
