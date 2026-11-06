import base64
import json

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from detector import cut_line
from detector.assess import Box, assess
from detector.main import create_app
from service.app import config
from service.app.pipeline import Pipeline
from service.app.signature_shadow import ShadowDetector
from service.app.vision.analyze import Evidence
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import BLANK, DANFE_TEXT, SETTINGS, SIGNED, FakeLaya

WIDTH, HEIGHT = 800, 600


def danfe_page(cut_y=150, canhoto_rules=(40, 100), body_rules=(260, 400)) -> np.ndarray:
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)
    for y in canhoto_rules:
        cv2.line(page, (20, y), (WIDTH - 20, y), 0, 2)
    for x in range(20, WIDTH - 20, 12):
        cv2.line(page, (x, cut_y), (x + 6, cut_y), 0, 1)
    for y in body_rules:
        for start in range(20, WIDTH - 20, 200):  # body rules are cut by cells
            cv2.line(page, (start, y), (start + 150, y), 0, 2)
    return page


def test_canhoto_above_the_cut_line_is_found():
    cut = cut_line.find(danfe_page())
    assert cut is not None and abs(cut.y - 150 / HEIGHT) < 0.02
    assert cut_line.canhoto_side(cut) == cut_line.ABOVE


def test_logo_below_the_cut_line_does_not_count_but_canhoto_ink_does():
    cut = cut_line.find(danfe_page())
    logo, ink = Box(0.88, 0.45, 0.6), Box(0.4, 0.1, 0.2)
    verdict = assess([logo, ink], cut)
    assert verdict["score_bruto"] == 0.88 and verdict["score_canhoto"] == 0.4


def test_without_cut_line_every_box_counts():
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)
    assert cut_line.find(page) is None
    assert assess([Box(0.7, 0.5, 0.6)], None)["score_canhoto"] == 0.7


class FakeBoxes:
    def __init__(self):
        self.shapes = []

    def boxes(self, rgb):
        self.shapes.append(rgb.shape)
        return [Box(0.9, 0.1, 0.2)]


def jpeg(page: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", page)[1].tobytes()).decode()


def test_detector_endpoint_rotates_scores_and_requires_the_key():
    fake = FakeBoxes()
    client = TestClient(create_app(fake, api_key="s"))
    payload = {"imagem_base64": jpeg(danfe_page()), "rotacao": 90}
    assert client.post("/v1/assinatura", json=payload).status_code == 401
    body = client.post("/v1/assinatura", json=payload, headers={"Authorization": "Bearer s"}).json()
    assert fake.shapes == [(WIDTH, HEIGHT, 3)] and body["score_bruto"] == 0.9 and body["versao"]


def test_detector_rejects_bytes_that_are_not_an_image():
    client = TestClient(create_app(FakeBoxes(), api_key="s"))
    response = client.post("/v1/assinatura", json={"imagem_base64": base64.b64encode(b"x").decode()},
                           headers={"Authorization": "Bearer s"})
    assert response.status_code == 422


SHADOW_SETTINGS = config.load({"API_KEY": "k", "WHITELIST_CNPJ": "36.895.856/0001-10",
                               "SIGNATURE_SHADOW_URL": "http://detector", "SIGNATURE_SHADOW_API_KEY": "s"})


def shadow_client(handler) -> ShadowDetector:
    return ShadowDetector(SHADOW_SETTINGS, httpx.Client(transport=httpx.MockTransport(handler)))


def answering(score):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer s"
        return httpx.Response(200, json={"score_canhoto": score, "score_bruto": score})
    return handler


def run(evidence, shadow):
    envelope = Pipeline(SHADOW_SETTINGS, lambda data: evidence, FakeLaya(), None, shadow).run(b"img", {})
    return json.loads(envelope["output"]), envelope["_meta"]


def test_shadow_is_recorded_and_never_changes_the_answer():
    body, meta = run(Evidence(DANFE_TEXT, make_key(), BLANK, 0), shadow_client(answering(0.9)))
    assert body["assinatura"]["possui_assinatura"] is False
    assert meta["sombra_assinatura"]["liberaria"] is True and meta["sombra_assinatura"]["concorda"] is False


def test_shadow_below_threshold_would_not_release():
    _, meta = run(Evidence(DANFE_TEXT, make_key(), SIGNED, 0), shadow_client(answering(0.3)))
    assert meta["sombra_assinatura"]["liberaria"] is False


def test_shadow_failure_is_reported_not_raised():
    body, meta = run(Evidence(DANFE_TEXT, make_key(), SIGNED, 0),
                     shadow_client(lambda request: httpx.Response(503)))
    assert body["assinatura"]["possui_assinatura"] is True and meta["sombra_assinatura"] == {"erro": "HTTPStatusError"}


def test_shadow_skips_documents_that_are_not_canhotos():
    calls = []
    shadow = shadow_client(lambda request: calls.append(1) or httpx.Response(200, json={"score_canhoto": 1}))
    envelope = Pipeline(SHADOW_SETTINGS, lambda data: Evidence("PIX", "", BLANK, 0), FakeLaya("comprovante"),
                        None, shadow).run(b"img", {})
    assert calls == [] and "sombra_assinatura" not in envelope["_meta"]
