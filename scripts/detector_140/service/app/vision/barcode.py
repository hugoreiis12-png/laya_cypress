"""Reads QR codes and 1D barcodes. zbar reads DANFE Code128 and QR; OpenCV is a QR-only fallback."""
import cv2
import numpy as np

from ..cnpj_key import find_access_keys
from .image_io import to_gray

HALF_SCALE = 0.5


def _zbar(gray: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar
    except ImportError:
        return []
    return [item.data.decode("ascii", "ignore") for item in pyzbar.decode(gray)]


def _opencv_qr(image: np.ndarray) -> list[str]:
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    return [text] if text else []


def read_codes(image: np.ndarray) -> list[str]:
    gray = to_gray(image)
    half = cv2.resize(gray, None, fx=HALF_SCALE, fy=HALF_SCALE, interpolation=cv2.INTER_AREA)
    return _zbar(gray) or _zbar(half) or _opencv_qr(image)


def access_key(codes: list[str]) -> str:
    """First valid 44-digit NF-e access key found in the decoded payloads, or ''."""
    keys = find_access_keys("\n".join(codes))
    return keys[0] if keys else ""
