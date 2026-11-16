import sys
from types import SimpleNamespace

import numpy as np
import pytest

from service.app.vision import ocr

HEIGHT, WIDTH = 100, 200
WORDS_WHEN_READABLE = 30
DATA_KEYS = ("text", "conf", "left", "top", "width", "height", "block_num", "par_num", "line_num")


def photo_readable_only_at_270() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[0, WIDTH - 1] = 255  # after a 270 rotation this pixel is the top-left corner
    return image


class FakeTesseract:
    """Reads text only from an image whose top-left pixel is lit, so only one orientation is readable."""

    Output = SimpleNamespace(DICT="dict")
    TesseractError = RuntimeError

    def __init__(self, osd_rotation: int) -> None:
        self.osd_rotation, self.reads = osd_rotation, 0

    def image_to_osd(self, gray, output_type=None):
        return {"rotate": self.osd_rotation}

    def image_to_data(self, gray, lang=None, output_type=None):
        self.reads += 1
        count = WORDS_WHEN_READABLE if gray[0, 0] == 255 else 0
        data = {key: [0] * count for key in DATA_KEYS}
        data["text"] = ["palavra"] * count
        data["conf"] = [90.0] * count
        return data


@pytest.fixture
def tesseract(monkeypatch):
    def install(osd_rotation: int) -> FakeTesseract:
        fake = FakeTesseract(osd_rotation)
        monkeypatch.setitem(sys.modules, "pytesseract", fake)
        return fake
    return install


def test_wrong_osd_rotation_falls_back_to_the_orientation_that_reads(tesseract):
    tesseract(osd_rotation=180)
    result, upright = ocr.run(photo_readable_only_at_270())
    assert result.rotation == 270 and len(result.text) >= ocr.MIN_READABLE_CHARS
    assert upright[0, 0].any()


def test_readable_osd_rotation_is_kept_without_extra_reads(tesseract):
    fake = tesseract(osd_rotation=270)
    result, _ = ocr.run(photo_readable_only_at_270())
    assert result.rotation == 270 and fake.reads == 1


def test_unreadable_photo_keeps_the_osd_rotation(tesseract):
    fake = tesseract(osd_rotation=180)
    result, _ = ocr.run(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    assert result.rotation == 180 and result.text == "" and fake.reads == 2 * len(ocr.CANDIDATE_ROTATIONS)


class SmallPrintTesseract(FakeTesseract):
    """Reads nothing until the image is enlarged; its word boxes then refer to the enlarged image."""

    def image_to_data(self, gray, lang=None, output_type=None):
        self.reads += 1
        count = WORDS_WHEN_READABLE if gray.shape[1] >= ocr.UPSCALE * WIDTH else 0
        data = {key: [0] * count for key in DATA_KEYS}
        data["text"], data["conf"] = ["palavra"] * count, [90.0] * count
        data["left"], data["width"] = [100] * count, [40] * count
        return data


def test_small_print_is_read_after_a_zoom_with_boxes_in_the_original_scale(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", SmallPrintTesseract(osd_rotation=0))
    result, upright = ocr.run(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    assert len(result.text) >= ocr.MIN_READABLE_CHARS and upright.shape[1] == WIDTH
    assert result.words[0].box.left == 50 and result.words[0].box.width == 20


def test_a_readable_photo_is_never_zoomed(tesseract):
    fake = tesseract(osd_rotation=270)
    ocr.run(photo_readable_only_at_270())
    assert fake.reads == 1


class UpsideDownTesseract(FakeTesseract):
    """OSD is 180 degrees off: that orientation yields plenty of characters, but all low confidence."""

    def image_to_data(self, gray, lang=None, output_type=None):
        self.reads += 1
        readable = gray[0, 0] == 255
        data = {key: [0] * WORDS_WHEN_READABLE for key in DATA_KEYS}
        data["text"] = ["palavra"] * WORDS_WHEN_READABLE
        data["conf"] = [90.0 if readable else 25.0] * WORDS_WHEN_READABLE
        return data


def test_a_long_but_low_confidence_read_does_not_stop_the_search_for_the_right_orientation(monkeypatch):
    fake = UpsideDownTesseract(osd_rotation=180)
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    result, _ = ocr.run(photo_readable_only_at_270())
    assert result.rotation == 270 and fake.reads == len(ocr.CANDIDATE_ROTATIONS)
    assert all(word.confidence >= ocr.CONFIDENT_WORD for word in result.words)


def test_a_confident_read_of_few_words_still_tries_the_other_orientations(tesseract):
    fake = tesseract(osd_rotation=180)
    ocr.run(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    assert fake.reads > 1


class ZoomedStripTesseract(FakeTesseract):
    """A strip printed sideways: unenlarged, 0 degrees gives the most noise; only 270 enlarged reads."""

    def image_to_data(self, gray, lang=None, output_type=None):
        self.reads += 1
        enlarged = max(gray.shape) >= ocr.UPSCALE * WIDTH
        count = WORDS_WHEN_READABLE if enlarged and gray[0, 0] == 255 else 5
        confidence = 90.0 if count == WORDS_WHEN_READABLE else (80.0 if gray.shape[1] == WIDTH else 10.0)
        data = {key: [0] * count for key in DATA_KEYS}
        data["text"], data["conf"] = ["palavra"] * count, [confidence] * count
        return data


def test_a_strip_readable_only_enlarged_is_read_in_its_own_orientation(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", ZoomedStripTesseract(osd_rotation=180))
    result, _ = ocr.run(photo_readable_only_at_270())
    assert result.rotation == 270 and len(result.words) == WORDS_WHEN_READABLE


class PrintSizeTesseract(FakeTesseract):
    """Always readable; enlarged reads yield more words, and the boxes carry the given print height."""

    def __init__(self, osd_rotation: int, height: int) -> None:
        super().__init__(osd_rotation)
        self.height = height

    def image_to_data(self, gray, lang=None, output_type=None):
        self.reads += 1
        count = WORDS_WHEN_READABLE * (2 if gray.shape[1] > WIDTH else 1)
        data = {key: [0] * count for key in DATA_KEYS}
        data["text"], data["conf"], data["height"] = ["palavra"] * count, [90.0] * count, [self.height] * count
        return data


def test_a_readable_page_in_small_print_keeps_the_better_zoomed_read(monkeypatch):
    fake = PrintSizeTesseract(osd_rotation=0, height=ocr.SMALL_PRINT_HEIGHT - 3)
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    result, upright = ocr.run(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    assert fake.reads == 2 and len(result.words) == 2 * WORDS_WHEN_READABLE and upright.shape[1] == WIDTH


def test_a_readable_page_in_normal_print_is_not_zoomed(monkeypatch):
    fake = PrintSizeTesseract(osd_rotation=0, height=ocr.SMALL_PRINT_HEIGHT * 2)
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    ocr.run(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    assert fake.reads == 1


class StrongZoomTesseract(FakeTesseract):
    """Unreadable unenlarged; every enlarged orientation reads strongly, so the first one ends the search."""

    def image_to_data(self, gray, lang=None, output_type=None):
        self.reads += 1
        count = ocr.STRONG_CONFIDENT_WORDS if gray.shape[1] > WIDTH else 0
        data = {key: [0] * count for key in DATA_KEYS}
        data["text"], data["conf"] = ["palavra"] * count, [90.0] * count
        return data


def test_a_strong_zoomed_read_ends_the_search(monkeypatch):
    fake = StrongZoomTesseract(osd_rotation=0)
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    result, _ = ocr.run(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    assert result.rotation == 0 and fake.reads == len(ocr.CANDIDATE_ROTATIONS) + 1


def test_the_zoomed_search_stops_when_its_budget_runs_out(tesseract):
    fake = tesseract(osd_rotation=0)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    unread = ocr._read(image, 0)
    ticks = iter([0.0, ocr.ZOOM_BUDGET_SECONDS])
    ocr._zoomed_search(image, unread, clock=lambda: next(ticks))
    assert fake.reads == 2
