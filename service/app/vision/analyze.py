"""Runs every vision tool on one image and returns the evidence the pipeline needs."""
from dataclasses import dataclass

from . import barcode, ink, ocr, region
from .image_io import decode, to_gray
from .types import InkFeatures


@dataclass(frozen=True)
class Evidence:
    text: str
    access_key: str
    ink: InkFeatures
    rotation: int
    other_keys: tuple[str, ...] = ()  # more than one key means more than one document in the photo


def analyze(data: bytes) -> Evidence:
    image = decode(data)
    result, upright = ocr.run(image)
    keys = (barcode.access_keys(barcode.read_codes(image)) or barcode.access_keys(barcode.read_codes(upright))
            or barcode.reinforced_access_keys(upright))
    height, width = upright.shape[:2]
    box = region.locate(result.words, width, height)
    features = ink.measure(to_gray(upright), result.words, box)
    return Evidence(result.text, keys[0] if keys else "", features, result.rotation, tuple(keys[1:]))
