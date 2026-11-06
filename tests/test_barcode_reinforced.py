import numpy as np

from service.app.vision import analyze as analyze_module
from service.app.vision import barcode
from service.app.vision.types import OcrResult
from tests.test_service_extractors import make_key
from tests.test_service_pipeline import SIGNED

KEY = make_key()
PICTURE = np.full((200, 300), 255, dtype=np.uint8)


def reader_that_succeeds_on(call, payload=KEY):
    calls = []

    def read(view):
        calls.append(view)
        return [payload] if len(calls) == call else []

    return read, calls


def test_stops_at_the_first_view_with_a_valid_key(monkeypatch):
    read, calls = reader_that_succeeds_on(3)
    monkeypatch.setattr(barcode, "_zbar", read)
    assert barcode.reinforced_access_keys(PICTURE) == [KEY] and len(calls) == 3


def test_every_view_is_tried_before_giving_up(monkeypatch):
    read, calls = reader_that_succeeds_on(-1)
    monkeypatch.setattr(barcode, "_zbar", read)
    assert barcode.reinforced_access_keys(PICTURE) == []
    assert len(calls) == 12 + barcode.BANDS + len(barcode.TILTS)


def test_a_code_without_a_valid_key_is_not_an_answer(monkeypatch):
    monkeypatch.setattr(barcode, "_zbar", lambda view: ["1" * 44])
    assert barcode.reinforced_access_keys(PICTURE) == []


def test_budget_stops_the_search(monkeypatch):
    read, calls = reader_that_succeeds_on(-1)
    monkeypatch.setattr(barcode, "_zbar", read)
    ticks = iter(range(100))
    assert barcode.reinforced_access_keys(PICTURE, budget=2, clock=lambda: next(ticks)) == []
    assert len(calls) == 2


def test_views_keep_the_gray_format_zbar_needs():
    views = list(barcode._views(PICTURE))
    assert all(view.ndim == 2 and view.dtype == np.uint8 and view.size for view in views)


def run_analyze(monkeypatch, plain, reinforced):
    monkeypatch.setattr(analyze_module, "decode", lambda data: PICTURE)
    monkeypatch.setattr(analyze_module.ocr, "run", lambda picture: (OcrResult("", (), 0), PICTURE))
    monkeypatch.setattr(analyze_module.barcode, "read_codes", lambda picture: plain)
    monkeypatch.setattr(analyze_module.barcode, "reinforced_access_keys", reinforced)
    monkeypatch.setattr(analyze_module.region, "locate", lambda words, width, height: None)
    monkeypatch.setattr(analyze_module.ink, "measure", lambda gray, words, box: SIGNED)
    return analyze_module.analyze(b"x")


def test_analyze_falls_back_to_the_reinforced_read(monkeypatch):
    assert run_analyze(monkeypatch, [], lambda picture: [KEY]).access_key == KEY


def test_analyze_skips_the_reinforced_read_when_the_plain_read_works(monkeypatch):
    def must_not_run(picture):
        raise AssertionError("leitura reforçada desnecessária")

    assert run_analyze(monkeypatch, [KEY], must_not_run).access_key == KEY
