import json
from dataclasses import replace

import cv2
import numpy as np

from service.app.pipeline import Pipeline
from service.app.signature_decision import SIGNED, UNCERTAIN
from service.app.vision.analyze import Evidence
from service.app.vision_judge import SignatureOpinion
from tests.test_judge_engine import ENGINE, FakeDetector
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SIGNED as SIGNED_INK, FakeLaya

NO, YES = SignatureOpinion(False, 95.0), SignatureOpinion(True, 95.0)
PHOTO = cv2.imencode(".jpg", np.zeros((400, 300, 3), np.uint8))[1].tobytes()


class ScriptedJudge:
    def __init__(self, *opinions):
        self.opinions, self.seen = list(opinions), []

    def assess_signature(self, data, timeout):
        self.seen.append(data)
        return self.opinions.pop(0)


def run(judge, second_look=True):
    settings = replace(ENGINE, judge_second_look=second_look)
    evidence = Evidence(DANFE_TEXT, make_key(), SIGNED_INK, 90)
    engine = Pipeline(settings, lambda data: evidence, FakeLaya(), shadow=FakeDetector(), judge=judge)
    envelope = engine.run(PHOTO, {})
    return json.loads(envelope["output"]), envelope["_meta"]


def _shape(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2]


def test_a_no_on_the_whole_photo_is_released_by_a_closer_yes():
    judge = ScriptedJudge(NO, YES)
    body, meta = run(judge)
    assert body["assinatura"]["confianca"] == 95 and meta["estado_assinatura"] == SIGNED
    assert meta["votos"]["juiz"] is True and meta["segunda_olhada"] == {"juiz": True, "confianca": 95.0}
    assert _shape(judge.seen[0]) == (300, 400) and _shape(judge.seen[1]) == (300, 800)


def test_two_noes_still_go_to_the_reserve():
    judge = ScriptedJudge(NO, NO)
    _, meta = run(judge)
    assert meta["estado_assinatura"] == UNCERTAIN and meta["votos"]["juiz"] is False and len(judge.seen) == 2


def test_a_yes_is_never_asked_twice():
    judge = ScriptedJudge(YES)
    _, meta = run(judge)
    assert meta["estado_assinatura"] == SIGNED and len(judge.seen) == 1 and "segunda_olhada" not in meta


def test_the_second_look_is_off_unless_asked():
    judge = ScriptedJudge(NO)
    _, meta = run(judge, second_look=False)
    assert meta["estado_assinatura"] == UNCERTAIN and len(judge.seen) == 1
