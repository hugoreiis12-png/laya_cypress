import cv2
import numpy as np
import pytest

from service.app.vision import barcode, image_io, ink, ocr, region
from service.app.vision.types import Box, Word
from tests.test_service_extractors import make_key

PAGE_W, PAGE_H = 1000, 800
REGION = Box(200, 300, 600, 250)


def blank_page():
    return np.full((PAGE_H, PAGE_W), 255, np.uint8)


def with_form_line(page):
    cv2.line(page, (100, 520), (900, 520), 0, 2)
    return page


def with_scribble(page):
    points = np.array([[250, 450], [300, 380], [350, 470], [410, 360], [470, 480], [540, 390], [600, 460]], np.int32)
    cv2.polylines(page, [points], False, 0, 5)
    return page


def with_speckle(page, count=40):
    rng = np.random.default_rng(7)
    for x, y in zip(rng.integers(220, 780, count), rng.integers(320, 530, count)):
        page[y, x] = 0
    return page


def test_printed_form_line_alone_is_not_ink():
    features = ink.measure(with_form_line(blank_page()), (), REGION)
    assert features.region_found and features.ink_pixels == 0 and features.component_count == 0


def test_scribble_is_detected_even_next_to_a_form_line():
    features = ink.measure(with_scribble(with_form_line(blank_page())), (), REGION)
    assert features.component_count >= 1 and features.ink_pixels > 500
    assert features.density_relative > 1 and 0 < features.extent_ratio <= 1


def test_isolated_speckle_noise_is_filtered_out():
    assert ink.measure(with_speckle(blank_page()), (), REGION).component_count == 0


def test_confidently_read_printed_words_do_not_count_as_ink():
    page = blank_page()
    cv2.putText(page, "RECEBEDOR", (260, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    printed = (Word("RECEBEDOR", Box(250, 360, 330, 60), 95.0),)
    assert ink.measure(page, printed, REGION).ink_pixels == 0
    assert ink.measure(page, (), REGION).ink_pixels > 0


def test_missing_region_reports_not_found():
    features = ink.measure(with_scribble(blank_page()), (), None)
    assert not features.region_found and features.ink_pixels == 0 and features.density_page > 0


def label(text, left, top, height=20):
    return Word(text, Box(left, top, 80, height), 90.0)


def test_region_is_found_from_label_words_and_clamped_to_the_image():
    words = (label("IDENTIFICAÇÃO", 100, 500), label("E", 190, 502), label("ASSINATURA", 230, 501), label("RECEBEDOR", 330, 500))
    found = region.locate(words, PAGE_W, PAGE_H)
    assert found == Box(0, 200, 710, 600)  # left/bottom clamped; right = label end (410) + 300


def test_a_lone_label_word_is_not_enough():
    assert region.locate((label("ASSINATURA", 100, 500),), PAGE_W, PAGE_H) is None


def test_label_words_on_different_lines_do_not_combine():
    words = (label("ASSINATURA", 100, 100), label("RECEBEDOR", 100, 700))
    assert region.locate(words, PAGE_W, PAGE_H) is None


def test_tesseract_data_is_grouped_into_lines_and_blank_words_dropped():
    data = {
        "text": ["NF-e", "", "Nº", "123"], "conf": ["90", "-1", "88", "75"],
        "left": [10, 0, 10, 60], "top": [10, 0, 40, 40], "width": [40, 0, 20, 30], "height": [12, 0, 12, 12],
        "block_num": [1, 1, 1, 1], "par_num": [1, 1, 1, 1], "line_num": [1, 1, 2, 2],
    }
    result = ocr.from_tesseract_data(data, rotation=90)
    assert result.text == "NF-e\nNº 123" and len(result.words) == 3 and result.rotation == 90


def qr_image(payload):
    qr = cv2.QRCodeEncoder.create().encode(payload)
    big = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)


def test_access_key_is_read_from_a_qr_payload():
    key = make_key()
    codes = barcode.read_codes(qr_image(f"https://nfe.example/qr?p={key}|2|1"))
    assert barcode.access_key(codes) == key


def test_payload_without_a_valid_key_yields_nothing():
    assert barcode.access_key(["https://example.com/?p=" + "1" * 44]) == ""


def test_decode_roundtrip_and_garbage():
    ok, encoded = cv2.imencode(".png", with_scribble(blank_page()))
    assert image_io.decode(encoded.tobytes()).shape[:2] == (PAGE_H, PAGE_W)
    with pytest.raises(image_io.ImageError):
        image_io.decode(b"not an image")


def test_tiny_image_is_rejected_not_crashed():
    ok, encoded = cv2.imencode(".png", np.full((1, 400), 255, np.uint8))
    with pytest.raises(image_io.ImageError):
        image_io.decode(encoded.tobytes())


def test_large_photos_are_downscaled():
    huge = np.full((4800, 3600, 3), 255, np.uint8)
    assert max(image_io.limit_size(huge).shape[:2]) == image_io.MAX_SIDE


def test_rotation_quarter_turns():
    page = np.zeros((10, 20), np.uint8)
    assert image_io.rotate(page, 90).shape == (20, 10) and image_io.rotate(page, 180).shape == (10, 20)
    assert image_io.rotate(page, 0) is page
