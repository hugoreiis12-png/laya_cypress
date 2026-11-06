import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from service.app import config, document_graph
from service.app.document_graph import build_graph, initial_state
from service.app.engines import Engines
from service.app.pipeline import Pipeline
from service.app.vision.analyze import Evidence
from service.app.vision.image_io import ImageError
from service.app.vision_judge import SignatureOpinion
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SETTINGS, SIGNED as SIGNED_INK, FakeLaya

MAIN = replace(SETTINGS, signature_engine=config.SIGNATURE_ENGINE_JUDGE, type_engine=config.TYPE_ENGINE_RULES,
               judge_transcribe=False)


class FakeDetector:
    def observe(self, data, rotation):
        return {"score_canhoto": 0.9, "liberaria": True}


class FakeJudge:
    def assess_signature(self, data, timeout):
        return SignatureOpinion(True, 90.0)


def engines(analyze):
    return Engines(MAIN, analyze, FakeLaya(), shadow=FakeDetector(), judge=FakeJudge())


def visited(analyze):
    graph = build_graph(engines(analyze))
    return [name for chunk in graph.stream(initial_state(b"x", {"id": "abc"}), stream_mode="updates") for name in chunk]


def canhoto(data):
    return Evidence(DANFE_TEXT, make_key(), SIGNED_INK, 0)


def test_a_canhoto_walks_the_whole_graph_in_order():
    assert visited(canhoto) == [document_graph.READ_DOCUMENT, document_graph.CLASSIFY, document_graph.SETTLE_TYPE,
                                document_graph.VOTE, document_graph.RECORD_DETECTOR, document_graph.BUILD,
                                document_graph.AUDIT, document_graph.CLOSE]


def test_an_unreadable_image_stops_right_after_the_reading():
    def broken(data):
        raise ImageError("imagem ilegível")

    assert visited(broken) == [document_graph.READ_DOCUMENT, document_graph.UNREADABLE]


def test_a_document_without_a_signature_skips_the_vote():
    path = visited(lambda data: Evidence("texto que nenhuma regra reconhece", "", SIGNED_INK, 0))
    assert document_graph.VOTE not in path and path[-1] == document_graph.CLOSE


def test_runs_keep_no_state_between_them():
    assert build_graph(engines(canhoto)).checkpointer is None


def test_external_tracing_stays_off_because_the_state_holds_photos():
    assert all(os.environ[switch] == "false" for switch in document_graph.TRACING_SWITCHES)


def test_the_contract_is_only_the_envelope():
    result = Pipeline(MAIN, canhoto, FakeLaya(), shadow=FakeDetector(), judge=FakeJudge()).run(b"x", {"id": "abc"})
    assert set(result) == {"output", "_meta"} and json.loads(result["output"])["tipo_documento"] == "canhoto"
    assert result["_meta"]["id"] == "abc" and "latencia_ms" in result["_meta"]


def test_concurrent_documents_do_not_share_meta():
    pipeline = Pipeline(MAIN, canhoto, FakeLaya(), shadow=FakeDetector(), judge=FakeJudge())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda number: pipeline.run(b"x", {"id": str(number)}), range(8)))
    assert [result["_meta"]["id"] for result in results] == [str(number) for number in range(8)]
