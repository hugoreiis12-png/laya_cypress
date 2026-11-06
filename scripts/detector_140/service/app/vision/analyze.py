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


def analyze(data: bytes) -> Evidence:
    image = decode(data)
    result, upright = ocr.run(image)
    key = barcode.access_key(barcode.read_codes(image)) or barcode.access_key(barcode.read_codes(upright))
    height, width = upright.shape[:2]
    box = region.locate(result.words, width, height)
    features = ink.measure(to_gray(upright), result.words, box)
    return Evidence(result.text, key, features, result.rotation)
