import json
from dataclasses import replace

import pytest

from service.app import config, key_reading
from service.app.decision_log import decision_record
from service.app.judge_capacity import OVER_RUNNING
from service.app.pipeline import Pipeline
from service.app.signature_decision import SIGNED
from service.app.vision.analyze import Evidence
from service.app.vision_judge import JudgeError, SignatureOpinion, parse_key
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import SETTINGS, SIGNED as SIGNED_INK, FakeLaya

ENGINE = replace(SETTINGS, signature_engine=config.SIGNATURE_ENGINE_JUDGE, type_engine=config.TYPE_ENGINE_RULES,
                 judge_transcribe=False, judge_read_key=True)
KEY = make_key()
BAD_CHECK_DIGIT = KEY[:-1] + str((int(KEY[-1]) + 1) % 10)


class FakeDetector:
    def observe(self, data, rotation):
        return {"score_canhoto": 0.1, "liberaria": False}


class FakeJudge:
    def __init__(self, read=KEY, fail=False):
        self.read, self.fail, self.reads, self.votes, self.timeout = read, fail, 0, 0, None

    def read_access_key(self, data, timeout):
        self.reads, self.timeout = self.reads + 1, timeout
        if self.fail:
            raise JudgeError("fora do ar")
        return self.read

    def assess_signature(self, data, timeout):
        self.votes += 1
        return SignatureOpinion(True, 90.0)


class FakeCapacity:
    def __init__(self, refusal=None):
        self.refusal, self.left = refusal, 0

    def enter(self):
        return self.refusal

    def leave(self):
        self.left += 1


def run(judge, text="", settings=ENGINE, barcode_key="", capacity=None):
    evidence = Evidence(text, barcode_key, SIGNED_INK, 0)
    engine = Pipeline(settings, lambda data: evidence, FakeLaya(), shadow=FakeDetector(), judge=judge, judge_capacity=capacity)
    envelope = engine.run(b"x", {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


def test_valid_key_turns_an_unread_photo_into_a_canhoto_that_still_needs_the_vote():
    judge = FakeJudge()
    body, meta, _ = run(judge)
    assert body["tipo_documento"] == "canhoto" and body["qrcode"]["chave"] == KEY and judge.votes == 1
    assert meta["estado_assinatura"] == SIGNED and meta["inconclusivo"] is False and meta["leitura_juiz"] == key_reading.FOUND


@pytest.mark.parametrize("read,expected", [(BAD_CHECK_DIGIT, key_reading.INVALID), (KEY + "123", key_reading.INVALID),
                                           (make_key(cnpj="22089653000393"), key_reading.OUTSIDE_WHITELIST),
                                           ("", key_reading.NOT_READ)])
def test_a_key_that_fails_any_check_is_not_used(read, expected):
    judge = FakeJudge(read)
    body, meta, _ = run(judge)
    assert body["tipo_documento"] == "imagem_invalida" and judge.votes == 0
    assert meta["inconclusivo"] is True and meta["leitura_juiz"] == expected


def test_it_stays_off_by_default():
    judge = FakeJudge()
    body, _, _ = run(judge, settings=replace(ENGINE, judge_read_key=False))
    assert judge.reads == 0 and body["tipo_documento"] == "imagem_invalida"
    assert config.load({"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10"}).judge_read_key is False


def test_key_from_the_barcode_skips_the_question():
    judge = FakeJudge()
    run(judge, barcode_key=KEY)
    assert judge.reads == 0


def test_a_readable_order_or_receipt_is_not_questioned():
    judge = FakeJudge()
    run(judge, text="Alface 10\nRúcula 5\nCheiro verde 20\nCouve 7\nTomate 3\nCebola 9")
    assert judge.reads == 0


def test_failure_and_no_capacity_go_to_the_reserve():
    capacity = FakeCapacity()
    _, failed, _ = run(FakeJudge(fail=True), capacity=capacity)
    judge = FakeJudge()
    _, refused, _ = run(judge, capacity=FakeCapacity(OVER_RUNNING))
    assert failed["leitura_juiz"] == key_reading.FAILED and failed["inconclusivo"] is True and capacity.left == 1
    assert refused["leitura_juiz"] == key_reading.NO_CAPACITY and judge.reads == 0


def test_question_time_is_capped_so_the_vote_still_fits():
    judge = FakeJudge()
    run(judge)
    assert judge.timeout <= ENGINE.judge_read_timeout


def test_the_record_shows_the_key_came_from_the_judge_without_holding_it():
    _, meta, envelope = run(FakeJudge())
    record = decision_record(envelope)
    assert record["leitura_juiz"] == key_reading.FOUND and meta["leitura"]["chave_no_codigo"] is False
    assert KEY not in json.dumps(record)


@pytest.mark.parametrize("text,expected", [('{"chave": "1234 5678"}', "12345678"), ('{"chave": null}', ""), ("<think></think>{\"chave\": \"1\"}", "1")])
def test_parse_key_keeps_only_digits(text, expected):
    assert parse_key(text) == expected


@pytest.mark.parametrize("text", ["não sei", "{}"])
def test_unusable_answer_is_an_error(text):
    with pytest.raises(JudgeError):
        parse_key(text)
