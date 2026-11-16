import cv2
import numpy as np

from service.app.vision.image_io import upright_jpeg


def _photo(height=120, width=80) -> bytes:
    image = np.zeros((height, width, 3), np.uint8)
    image[:20, :] = 255  # a bright band marks the top before turning
    return cv2.imencode(".jpg", image)[1].tobytes()


def _shape(data: bytes) -> tuple[int, int]:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2]


def test_an_upright_photo_is_sent_untouched():
    photo = _photo()
    assert upright_jpeg(photo, 0) is photo


def test_a_turned_photo_is_rotated_by_the_ocr_rotation():
    assert _shape(upright_jpeg(_photo(), 90)) == (80, 120)
    assert _shape(upright_jpeg(_photo(), 180)) == (120, 80)
