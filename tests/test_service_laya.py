import json

import httpx
import pytest

from service.app import config
from service.app.laya_client import LayaClient, LayaError, build_request, parse

SETTINGS = config.load({
    "API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10",
    "LAYA_API_KEY": "secret", "LAYA_URL": "http://laya.test",
})

# shape captured from the real server (routing/usage trimmed)
REAL_ANSWER = {
    "model": "laya-rl-agent",
    "answers": {"tipo_documento": {
        "type": "choice", "choice": "canhoto",
        "probabilities": {"canhoto": 0.9511, "comprovante": 0.0134, "pedido": 0.0221, "imagem_invalida": 0.0134},
        "confidence": 0.8213, "answer_confidence": 0.9511, "x_jev_confidence": 0.9348,
        "action": {"act_probability": 1.0},
    }},
    "usage": {"truncated": False},
}


def client_for(handler):
    return LayaClient(SETTINGS, httpx.Client(transport=httpx.MockTransport(handler)))


def ok(request):
    return httpx.Response(200, json=REAL_ANSWER)


def test_parses_the_real_server_response():
    decision = parse(REAL_ANSWER, "answer_confidence")
    assert decision.doc_type == "canhoto" and decision.confidence == 0.9511
    assert decision.confidences["confidence"] == 0.8213 and not decision.truncated


def test_confidence_field_is_configurable():
    assert parse(REAL_ANSWER, "confidence").confidence == 0.8213


def test_request_carries_auth_model_and_truncated_text():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=REAL_ANSWER)

    client_for(handler).classify("x" * 20000, {"chave": "1"})
    assert seen["auth"] == "Bearer secret" and seen["body"]["model"] == "multilingual"
    assert len(seen["body"]["state"]["documento"]["texto_ocr"]) == 12000
    assert list(seen["body"]["questions"]) == ["tipo_documento"]


def test_request_contains_only_the_document_type_question():
    assert list(build_request("texto", {})["questions"]) == ["tipo_documento"]


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(401, json={"detail": "unauthorized"}),
    lambda r: httpx.Response(200, text="not json"),
    lambda r: httpx.Response(200, json={"answers": {}}),
    lambda r: (_ for _ in ()).throw(httpx.ConnectTimeout("slow")),
])
def test_every_failure_becomes_laya_error(handler):
    with pytest.raises(LayaError):
        client_for(handler).classify("DANFE", {})


def test_unknown_document_type_is_rejected():
    bad = json.loads(json.dumps(REAL_ANSWER))
    bad["answers"]["tipo_documento"]["choice"] = "recibo"
    with pytest.raises(LayaError, match="desconhecido"):
        parse(bad, "answer_confidence")


def test_blank_ocr_never_reaches_the_network():
    def boom(request):
        raise AssertionError("não deveria chamar o Laya")

    with pytest.raises(LayaError, match="vazio"):
        client_for(boom).classify("   \n ", {})
