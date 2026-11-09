"""Cuts the receiver strip (stamp and signature field) out of a canhoto photo; re-encoding drops the EXIF."""
import cv2
import numpy as np

RECEIVER_TOP_FRACTION = 0.22  # the receiver field sits in the top band of an upright DANFE
MIN_STRIP_PIXELS = 32
JPEG_QUALITY = 90
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class CropError(ValueError):
    """The photo cannot yield a usable strip; the item is dropped, never guessed."""


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CropError("imagem ilegível ou formato não suportado")
    return image


def receiver_strip(data: bytes, rotation: int) -> bytes:
    image = _decode(data)
    upright = cv2.rotate(image, ROTATIONS[rotation]) if rotation in ROTATIONS else image
    strip = upright[: int(upright.shape[0] * RECEIVER_TOP_FRACTION)]
    if strip.shape[0] < MIN_STRIP_PIXELS:
        raise CropError("faixa do recebedor pequena demais")
    done, encoded = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not done:
        raise CropError("falha ao codificar a faixa")
    return encoded.tobytes()
