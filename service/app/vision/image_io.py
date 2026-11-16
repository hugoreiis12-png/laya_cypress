"""Image decoding and basic geometry. cv2.imdecode already applies EXIF orientation."""
import cv2
import numpy as np

MAX_SIDE = 2400
JPEG_QUALITY = 92
MIN_SIDE = 64  # below this no field is readable; also keeps half-scale resizes non-empty
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class ImageError(ValueError):
    """The bytes are not a decodable image."""


def limit_size(image: np.ndarray) -> np.ndarray:
    scale = MAX_SIDE / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageError("imagem ilegível ou formato não suportado")
    if min(image.shape[:2]) < MIN_SIDE:
        raise ImageError("imagem pequena demais para leitura")
    return limit_size(image)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    return cv2.rotate(image, ROTATIONS[degrees]) if degrees in ROTATIONS else image


def upright_jpeg(data: bytes, rotation: int) -> bytes:
    """The photo turned by the rotation the OCR found, so a vision model reads it the right way up."""
    if rotation not in ROTATIONS:
        return data
    encoded, buffer = cv2.imencode(".jpg", rotate(decode(data), rotation), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes() if encoded else data
