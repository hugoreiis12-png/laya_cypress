"""Reads QR codes and 1D barcodes. zbar reads DANFE Code128 and QR; OpenCV is a QR-only fallback."""
import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from ..cnpj_key import find_access_keys
from .image_io import to_gray

HALF_SCALE = 0.5
REINFORCED_BUDGET_SECONDS = 6.0  # slowest photo measured on the gold set took 5.02 s on one CPU
ENLARGEMENTS = (1.5, 2.0)
BANDS = 4
TILTS = (-10, -5, 5, 10)
CLAHE_LIMIT = 2.0
CLAHE_GRID = (8, 8)
ADAPTIVE_BLOCK = 31
ADAPTIVE_OFFSET = 10
WHITE = 255


def _zbar(gray: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar
    except ImportError:
        return []
    symbols = [pyzbar.ZBarSymbol.CODE128, pyzbar.ZBarSymbol.QRCODE]
    return [item.data.decode("ascii", "ignore") for item in pyzbar.decode(gray, symbols=symbols)]


def _opencv_qr(image: np.ndarray) -> list[str]:
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    return [text] if text else []


def read_codes(image: np.ndarray) -> list[str]:
    gray = to_gray(image)
    half = cv2.resize(gray, None, fx=HALF_SCALE, fy=HALF_SCALE, interpolation=cv2.INTER_AREA)
    return _zbar(gray) or _zbar(half) or _opencv_qr(image)


def access_keys(codes: list[str]) -> list[str]:
    """Every distinct valid 44-digit NF-e access key in the decoded payloads."""
    return find_access_keys("\n".join(codes))


def access_key(codes: list[str]) -> str:
    keys = access_keys(codes)
    return keys[0] if keys else ""


def _enlarged(gray: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _filtered(gray: np.ndarray) -> Iterator[np.ndarray]:
    yield gray
    yield cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=CLAHE_GRID).apply(gray)
    yield cv2.threshold(gray, 0, WHITE, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield cv2.adaptiveThreshold(gray, WHITE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                ADAPTIVE_BLOCK, ADAPTIVE_OFFSET)


def _bands(gray: np.ndarray) -> Iterator[np.ndarray]:
    height = gray.shape[0]
    step = height // BANDS
    for start in range(0, height - step + 1, step):
        yield gray[start:min(height, start + 2 * step)]


def _tilted(gray: np.ndarray, degrees: int) -> np.ndarray:
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), borderValue=WHITE)


def _views(gray: np.ndarray) -> Iterator[np.ndarray]:
    for view in _filtered(gray):
        yield view
        for scale in ENLARGEMENTS:
            yield _enlarged(view, scale)
    for band in _bands(gray):
        yield _enlarged(band, ENLARGEMENTS[-1])
    for degrees in TILTS:
        yield _tilted(gray, degrees)


def reinforced_access_keys(image: np.ndarray, budget: float = REINFORCED_BUDGET_SECONDS,
                           clock: Callable[[], float] = time.monotonic) -> list[str]:
    """Second pass for photos the plain read missed; it stops at the first key or when the budget runs out."""
    deadline = clock() + budget
    for view in _views(to_gray(image)):
        keys = access_keys(_zbar(view))
        if keys:
            return keys
        if clock() >= deadline:
            break
    return []
