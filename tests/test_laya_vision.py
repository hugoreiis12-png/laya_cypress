import hashlib
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from service.app.decision_log import decision_record
from service.app.laya_vision import LayaVision, current_record
from service.app.pipeline import Pipeline
from service.app.signature_decision import SIGNED, UNCERTAIN, from_laya_vision
from service.app.vision.analyze import Evidence
from tests.test_judge_engine import ENGINE, FakeDetector
from tests.test_judge_second_look import NO, PHOTO, YES, ScriptedJudge
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import DANFE_TEXT, SIGNED as SIGNED_INK, FakeLaya

CLASSES = ["assinado", "nao_assinado"]


class FixedSession:
    def __init__(self, probabilities):
        self.probabilities, self.feeds = probabilities, []

    def run(self, outputs, feed):
        self.feeds.append(feed)
        return [np.array([self.probabilities], np.float32)]


def write_registry(root, versions, shadow=(), baseline="visao-0001", weights=b"onnx"):
    digest = hashlib.sha256(weights).hexdigest()
    (root / "pesos").mkdir(parents=True, exist_ok=True)
    (root / "pesos" / f"{digest}.onnx").write_bytes(weights)
    records = {name: {"kind": "visao", "sha256": digest, "file": f"{digest}.onnx", "classes": CLASSES,
                      "pronta": ready, "status": "sombra"} for name, ready in versions.items()}
    registry = {"versions": records, "shadow": {"visao": list(shadow)}, "baseline": {"visao": baseline}}
    (root / "versoes.json").write_text(json.dumps(registry), encoding="utf-8")


def vision(tmp_path, probabilities=(0.97, 0.03), **registry):
    write_registry(tmp_path, **registry)
    opened = []

    def open_session(path):
        opened.append(path.name)
        return FixedSession(list(probabilities)), "images"
    return LayaVision(str(tmp_path), open_session), opened


def test_the_current_version_is_the_last_shadow_else_the_baseline():
    registry = {"versions": {"visao-0001": {"file": "a"}, "visao-0002": {"file": "b"}}, "shadow": {"visao": ["visao-0002"]},
                "baseline": {"visao": "visao-0001"}}
    assert current_record(registry)[0] == "visao-0002"
    assert current_record({**registry, "shadow": {}})[0] == "visao-0001" and current_record({}) is None


def test_it_reads_the_label_of_the_current_version_from_the_strip(tmp_path):
    model, _ = vision(tmp_path, versions={"visao-0001": True})
    observed = model.observe(PHOTO, 90)
    assert observed == {"versao": "visao-0001", "rotulo": "assinado", "confianca": 0.97, "pronta": True}


def test_the_model_is_loaded_once_and_again_only_when_the_registry_changes(tmp_path):
    model, opened = vision(tmp_path, versions={"visao-0001": False})
    model.observe(PHOTO, 0)
    model.observe(PHOTO, 0)
    write_registry(tmp_path, versions={"visao-0001": False, "visao-0002": True}, shadow=["visao-0002"])
    assert model.observe(PHOTO, 0)["versao"] == "visao-0002" and len(opened) == 2


def test_weights_that_do_not_match_the_registry_are_never_run(tmp_path):
    model, opened = vision(tmp_path, versions={"visao-0001": True})
    next((tmp_path / "pesos").iterdir()).write_bytes(b"trocado")
    assert "erro" in model.observe(PHOTO, 0) and opened == []


def test_without_a_registry_it_reports_an_error_instead_of_raising(tmp_path):
    assert LayaVision(str(tmp_path / "ausente")).observe(PHOTO, 0) == {"erro": "sem_registro"}


def test_a_real_onnx_model_runs_on_the_classifier_input(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    weights = helper.make_tensor("w", TensorProto.FLOAT, [3, 2], [1, 0, 1, 0, 1, 0])
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["images"], ["pooled"]), helper.make_node("Flatten", ["pooled"], ["flat"]),
         helper.make_node("MatMul", ["flat", "w"], ["logits"]), helper.make_node("Softmax", ["logits"], ["probs"])],
        "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 2])], [weights])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    write_registry(tmp_path, versions={"visao-0001": True}, weights=model.SerializeToString())
    white = cv2.imencode(".jpg", np.full((400, 300, 3), 255, np.uint8))[1].tobytes()
    observed = LayaVision(str(tmp_path)).observe(white, 0)
    assert observed["rotulo"] == "assinado" and observed["confianca"] == pytest.approx(0.9526, abs=1e-3)
    assert onnx.checker.check_model(model) is None


@pytest.mark.parametrize("observed, takeover, released", [
    ({"rotulo": "assinado", "confianca": 0.95, "pronta": True}, True, True),
    ({"rotulo": "assinado", "confianca": 0.95, "pronta": True}, False, False),
    ({"rotulo": "assinado", "confianca": 0.95, "pronta": False}, True, False),
    ({"rotulo": "assinado", "confianca": 0.85, "pronta": True}, True, False),
    ({"rotulo": "nao_assinado", "confianca": 0.99, "pronta": True}, True, False),
    ({"erro": "sem_versao"}, True, False),
])
def test_laya_releases_only_a_ready_confident_signed(observed, takeover, released):
    verdict = from_laya_vision(observed, takeover, 0.90)
    assert (verdict is not None and verdict.state == SIGNED) == released


class FakeVision:
    def __init__(self, observed):
        self.observed = observed

    def observe(self, data, rotation):
        return dict(self.observed)


def run(observed, judge, takeover=True):
    settings = replace(ENGINE, laya_vision_takeover=takeover)
    evidence = Evidence(DANFE_TEXT, make_key(), SIGNED_INK, 90)
    engine = Pipeline(settings, lambda data: evidence, FakeLaya(), shadow=FakeDetector(), judge=judge,
                      laya_vision=FakeVision(observed))
    envelope = engine.run(PHOTO, {})
    return json.loads(envelope["output"]), envelope["_meta"], envelope


READY = {"versao": "visao-0003", "rotulo": "assinado", "confianca": 0.96, "pronta": True}


def test_a_ready_version_takes_over_without_asking_the_judge():
    judge = ScriptedJudge()
    body, meta, envelope = run(READY, judge)
    assert meta["estado_assinatura"] == SIGNED and body["assinatura"]["confianca"] == 96 and judge.seen == []
    assert decision_record(envelope)["laya_visao"] == {**READY, "assumiu": True}


def test_with_the_takeover_off_the_judge_decides_and_laya_is_recorded():
    judge = ScriptedJudge(YES)
    _, meta, _ = run(READY, judge, takeover=False)
    assert len(judge.seen) == 1 and meta["laya_visao"]["assumiu"] is False and meta["votos"]["juiz"] is True


def test_a_laya_no_never_blocks_the_judge():
    judge = ScriptedJudge(NO)
    _, meta, _ = run({**READY, "rotulo": "nao_assinado"}, judge)
    assert meta["estado_assinatura"] == UNCERTAIN and len(judge.seen) == 1
