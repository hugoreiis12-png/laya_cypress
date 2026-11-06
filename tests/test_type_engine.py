import json
from dataclasses import replace

import pytest

from service.app import config
from service.app.decision_log import decision_record
from service.app.pipeline import Pipeline
from service.app.vision.analyze import Evidence
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED, FakeLaya

RULES_SETTINGS = replace(SETTINGS, type_engine=config.TYPE_ENGINE_RULES)
OTHER_TEXT = "Texto qualquer sem relação com documentos fiscais e com mais de quarenta caracteres"
ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10"}


def run(evidence, laya):
    envelope = Pipeline(RULES_SETTINGS, lambda data: evidence, laya).run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


def test_rules_decide_and_laya_is_only_observed():
    body, meta, _ = run(Evidence(DANFE_TEXT, "", SIGNED, 0), FakeLaya("pedido"))
    assert body["tipo_documento"] == "canhoto" and meta["motor_decisao"] == "regras"
    assert meta["laya_observado"] == {"tipo": "pedido", "confianca": 0.9} and "confianca_tipo" not in meta


def test_laya_failure_never_changes_a_rules_decision():
    body, meta, _ = run(Evidence(DANFE_TEXT, "", SIGNED, 0), FakeLaya(fail=True))
    assert body["tipo_documento"] == "canhoto" and meta["laya_observado"] == {"erro": "indisponivel"}
    assert meta["tentativas"] == []


def test_valid_key_still_defines_canhoto_with_rules():
    body, _, _ = run(Evidence(OTHER_TEXT, make_key(), SIGNED, 0), FakeLaya("pedido"))
    assert body["tipo_documento"] == "canhoto"


def test_decision_log_keeps_the_observation():
    _, _, envelope = run(Evidence(DANFE_TEXT, "", SIGNED, 0), FakeLaya("pedido"))
    assert decision_record(envelope)["laya_observado"]["tipo"] == "pedido"


def test_defaults_keep_the_current_engines():
    settings = config.load(ENV)
    assert settings.type_engine == config.TYPE_ENGINE_LAYA
    assert settings.signature_engine == config.SIGNATURE_ENGINE_CURRENT and settings.hermes_audit is False


@pytest.mark.parametrize("variable", ["TYPE_ENGINE", "SIGNATURE_ENGINE"])
def test_invalid_engine_fails_closed(variable):
    with pytest.raises(config.ConfigError, match=variable):
        config.load({**ENV, variable: "gemini"})
