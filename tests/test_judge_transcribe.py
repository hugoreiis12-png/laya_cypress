from dataclasses import replace

from service.app import config
from service.app.pipeline import Pipeline
from service.app.vision.analyze import Evidence
from service.app.vision_judge import SignatureOpinion
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import SETTINGS, SIGNED, FakeLaya

ENV = {"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10"}
OFF = replace(SETTINGS, judge_transcribe=False, signature_engine=config.SIGNATURE_ENGINE_TWO_VOTES,
              type_engine=config.TYPE_ENGINE_RULES)


class FakeDetector:
    def observe(self, data, rotation):
        return {"score_canhoto": 0.9, "liberaria": True}


class FakeJudge:
    def __init__(self):
        self.readings, self.votes = 0, 0

    def transcribe(self, data):
        self.readings += 1
        return "texto"

    def assess_signature(self, data, timeout):
        self.votes += 1
        return SignatureOpinion(True, 95.0)


def run(evidence, judge):
    return Pipeline(OFF, lambda data: evidence, FakeLaya(), shadow=FakeDetector(), judge=judge).run(b"x", {})["_meta"]


def test_unread_photo_goes_to_the_reserve_without_transcription():
    judge = FakeJudge()
    meta = run(Evidence("curto", "", SIGNED, 0), judge)
    assert judge.readings == 0 and judge.votes == 0 and meta["inconclusivo"] is True and "juiz" not in meta


def test_signature_vote_still_runs_without_transcription():
    judge = FakeJudge()
    meta = run(Evidence("curto", make_key(), SIGNED, 0), judge)
    assert judge.readings == 0 and judge.votes == 1 and meta["votos"]["juiz"] is True


def test_only_zero_or_other_values_turn_it_off():
    assert config.load(ENV).judge_transcribe is True
    assert config.load({**ENV, "JUDGE_TRANSCRIBE": "1"}).judge_transcribe is True
    assert config.load({**ENV, "JUDGE_TRANSCRIBE": "0"}).judge_transcribe is False
