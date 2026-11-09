import json
from dataclasses import replace

import httpx

from service.app import vision_judge
from service.app.config import load
from service.app.vision_judge import VisionJudge

ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10", "JUDGE_URL": "http://judge", "JUDGE_MODEL": "vlm"}
KEY = "33261115268427000142550010008179371192540761"


def judge(content, disable=False):
    sent = []

    def answer(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = replace(load(ENV), judge_disable_thinking=disable)
    return VisionJudge(settings, httpx.Client(transport=httpx.MockTransport(answer))), sent


def test_reasoning_is_left_alone_by_default():
    client, sent = judge('{"possui_assinatura": true, "confianca": 90}')
    client.assess_signature(b"jpeg", 5.0)
    assert "chat_template_kwargs" not in sent[0] and sent[0]["max_tokens"] == vision_judge.MAX_TOKENS


def test_disabling_reasoning_is_asked_per_request_with_a_short_answer_budget():
    client, sent = judge('{"possui_assinatura": false, "confianca": 80}', disable=True)
    assert client.assess_signature(b"jpeg", 5.0).signed is False
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent[0]["max_tokens"] == vision_judge.SHORT_ANSWER_TOKENS


def test_the_key_question_also_gets_the_short_budget():
    client, sent = judge(json.dumps({"chave": KEY}), disable=True)
    assert client.read_access_key(b"jpeg", 5.0) == KEY and sent[0]["max_tokens"] == vision_judge.SHORT_ANSWER_TOKENS


def test_transcription_keeps_the_long_budget_even_without_reasoning():
    client, sent = judge("texto transcrito", disable=True)
    assert client.transcribe(b"jpeg") == "texto transcrito"
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False} and sent[0]["max_tokens"] == vision_judge.MAX_TOKENS


def test_the_flag_comes_from_the_environment_and_is_off_without_it():
    assert load({**ENV, "JUDGE_DISABLE_THINKING": "1"}).judge_disable_thinking is True
    assert load(ENV).judge_disable_thinking is False
    assert load({**ENV, "JUDGE_DISABLE_THINKING": "0"}).judge_disable_thinking is False
