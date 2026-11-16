"""Cuts the receiver strip (stamp and signature field) out of a canhoto photo; re-encoding drops the EXIF."""
import cv2
import numpy as np

from service.app.vision.strip import STRIP_JPEG_QUALITY, upright_top_half

MIN_STRIP_PIXELS = 32


class CropError(ValueError):
    """The photo cannot yield a usable strip; the item is dropped, never guessed."""


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CropError("imagem ilegível ou formato não suportado")
    return image


def receiver_strip(data: bytes, rotation: int) -> bytes:
    strip = upright_top_half(_decode(data), rotation)
    if strip.shape[0] < MIN_STRIP_PIXELS:
        raise CropError("faixa do recebedor pequena demais")
    done, encoded = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, STRIP_JPEG_QUALITY])
    if not done:
        raise CropError("falha ao codificar a faixa")
    return encoded.tobytes()
